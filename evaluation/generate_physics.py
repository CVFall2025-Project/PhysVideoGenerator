"""
generate_physics.py

Generates 16-frame videos for all 25 evaluation prompts using the
physics-conditioned Latte model (LatteTransformer3DModelWithPhysics loaded
from a training checkpoint). Mirrors generate_baseline.py layout and seeds
so pairwise comparison is direct:

    evaluation/baseline_videos/01_a_person_pours_water_...mp4   (base Latte-1)
    evaluation/physics_videos/01_a_person_pours_water_...mp4    (this script)

Seeds match generate_baseline.py exactly (prompt_id * 1000), so any per-prompt
differences between the two folders are attributable to the physics head.

Output layout:
    evaluation/physics_videos/<id>_<slug>.mp4

VRAM (A100 40 GB assumed):
    T5-XXL fp16       : ~22 GB
    VAE bf16          : ~0.4 GB
    Latte + physics   : ~4 GB
    Peak (with T5)    : ~27 GB — fits with headroom.

Usage:
    # From project root
    python evaluation/generate_physics.py \
        --checkpoint checkpoints_run1/checkpoint_epoch_5.pt

    # One prompt only (fast smoke test)
    python evaluation/generate_physics.py \
        --checkpoint checkpoints_run1/checkpoint_epoch_5.pt --prompt_id 1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import torch

# make src/ importable so latte_physics + encoders resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusers import DDIMScheduler
from transformers import T5EncoderModel, T5Tokenizer

from src.encoders.vae_encoder_decoder import VAEEncoder
from src.latte_physics import LatteTransformer3DModelWithPhysics


PROMPTS_CSV    = Path(__file__).parent / "eval_prompts.csv"
OUTPUT_DIR     = Path(__file__).parent / "physics_videos"

MODEL_ID       = "maxin-cn/Latte-1"
TEXT_MODEL_ID  = "google/t5-v1_1-xxl"

NUM_FRAMES     = 16
NUM_STEPS      = 50
GUIDANCE_SCALE = 7.5
HEIGHT         = 512
WIDTH          = 512
PLAYBACK_FPS   = 8

NEGATIVE_PROMPT = "blurry, low quality, static, no motion"


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len]


def save_video(frames: np.ndarray, path: Path, fps: int = PLAYBACK_FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def build_physics_model(checkpoint_path: str, device: str) -> LatteTransformer3DModelWithPhysics:
    """Instantiate the model with the same config used during training and load
    the checkpoint's state dict."""
    model = LatteTransformer3DModelWithPhysics(
        num_attention_heads=16,
        attention_head_dim=72,
        in_channels=4,
        out_channels=4,
        num_layers=28,
        dropout=0.0,
        cross_attention_dim=1152,
        attention_bias=False,
        sample_size=64,
        patch_size=2,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=4096,
        video_length=NUM_FRAMES,
        predictor_hidden_dim=512,
        vjepa_dim=1408,
        vjepa_seq_len=2048,
        use_predictor=True,
    ).to(device, dtype=torch.bfloat16)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint (strict): {checkpoint_path}")
    except RuntimeError as e:
        print(f"Strict load failed, falling back to strict=False:\n  {e}")
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def encode_prompt(tokenizer, text_model, prompt: str, device: str, dtype) -> torch.Tensor:
    """Return CFG-batched text embeddings [2, 226, 4096] as (uncond, cond)."""
    def enc(text: str) -> torch.Tensor:
        inputs = tokenizer(
            text, return_tensors="pt", padding="max_length",
            max_length=226, truncation=True,
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        with torch.inference_mode():
            out = text_model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state.to(dtype)

    uncond = enc(NEGATIVE_PROMPT)
    cond = enc(prompt)
    return torch.cat([uncond, cond], dim=0)


def denoise(model, scheduler, encoder_hidden_states, generator, device,
            num_steps: int, guidance_scale: float) -> torch.Tensor:
    """CFG-DDIM loop. Returns denoised latents in [1, T, C, H, W] layout."""
    latent_h = HEIGHT // 8
    latent_w = WIDTH // 8
    latents = torch.randn(
        1, NUM_FRAMES, 4, latent_h, latent_w,
        generator=generator, device=device, dtype=torch.bfloat16,
    )

    scheduler.set_timesteps(num_steps, device=device)

    for t in scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2, dim=0)          # [2, T, C, H, W]
        latent_model_input = latent_model_input.permute(0, 2, 1, 3, 4) # [2, C, T, H, W] — model layout

        timestep = torch.tensor([t, t], device=device, dtype=torch.long)

        noise_pred, _ = model(
            latent_model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            ground_truth_vjepa=None,
            enable_temporal_attentions=True,
        )
        noise_pred = noise_pred.sample                         # [2, C, T, H, W]

        # CFG in [B, T, C, H, W]
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)         # [2, T, C, H, W]
        uncond, cond = noise_pred.chunk(2, dim=0)
        noise_pred = uncond + guidance_scale * (cond - uncond) # [1, T, C, H, W]

        # scheduler.step expects same layout as latents_for_step
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)          # [1, C, T, H, W]
        latents_for_step = latents.permute(0, 2, 1, 3, 4)       # [1, C, T, H, W]

        latents_stepped = scheduler.step(noise_pred, t, latents_for_step).prev_sample
        latents = latents_stepped.permute(0, 2, 1, 3, 4)        # [1, T, C, H, W]

    return latents


def decode_latents_to_frames(vae, latents) -> np.ndarray:
    """Decode [1, T, C, H, W] latents -> [T, H, W, 3] uint8 frames."""
    latents = latents.permute(0, 2, 1, 3, 4)                     # [1, C, T, H, W]
    latents = latents / vae.model.config.scaling_factor

    latents_frames = latents.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [T, C, H, W]
    latents_frames = latents_frames.to(vae.torch_dtype)

    with torch.inference_mode():
        video_frames = vae.model.decode(latents_frames).sample     # [T, 3, H, W] in [-1, 1]

    frames = video_frames.permute(0, 2, 3, 1).cpu().float().numpy()
    frames = ((frames + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return frames


def generate(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU found. Generation will be extremely slow.")

    df = pd.read_csv(PROMPTS_CSV)
    if args.prompt_id is not None:
        df = df[df["id"] == args.prompt_id]
        if df.empty:
            raise ValueError(f"No prompt with id={args.prompt_id} in {PROMPTS_CSV}")

    print(f"Loading T5-XXL text encoder in fp16 ...")
    tokenizer = T5Tokenizer.from_pretrained(TEXT_MODEL_ID)
    text_model = T5EncoderModel.from_pretrained(
        TEXT_MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    text_model.eval()

    print(f"Loading VAE (bf16) and DDIM scheduler from {MODEL_ID} ...")
    vae = VAEEncoder(MODEL_ID, torch_dtype=torch.bfloat16, device=device)
    scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    print(f"Loading physics-trained transformer from {args.checkpoint} ...")
    model = build_physics_model(args.checkpoint, device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        prompt_id = int(row["id"])
        prompt    = row["prompt"]
        slug      = slugify(prompt)
        out_path  = OUTPUT_DIR / f"{prompt_id:02d}_{slug}.mp4"

        # Seed matches generate_baseline.py so pairs are directly comparable
        generator = torch.Generator(device=device).manual_seed(prompt_id * 1000)

        print(f"[{prompt_id:02d}] Generating (seed={prompt_id * 1000}): {prompt[:80]}")

        with torch.inference_mode():
            encoder_hidden_states = encode_prompt(
                tokenizer, text_model, prompt, device, dtype=torch.bfloat16
            )
            latents = denoise(
                model, scheduler, encoder_hidden_states, generator, device,
                num_steps=args.num_steps, guidance_scale=args.guidance_scale,
            )
            frames_np = decode_latents_to_frames(vae, latents)

        save_video(frames_np, out_path)
        print(f"  Saved ({NUM_FRAMES} frames @ {PLAYBACK_FPS}fps → "
              f"{NUM_FRAMES/PLAYBACK_FPS:.1f}s): {out_path}")

    print(f"\nDone. Videos saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate physics-conditioned videos for eval prompts"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--prompt_id", type=int, default=None,
                        help="Generate only this prompt id (for testing)")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="DDIM denoising steps (default: 50)")
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE,
                        help="CFG scale (default: 7.5)")
    args = parser.parse_args()
    generate(args)
