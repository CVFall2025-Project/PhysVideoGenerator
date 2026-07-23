"""
generate_physics_v2.py

Same as generate_physics.py but uses LattePipeline for all the plumbing
(text encoding, denoising loop, VAE decode) and only swaps in the physics
transformer as pipe.transformer. This bypasses the custom denoising loop
in v1 that was producing incoherent output.

Usage:
    python evaluation/generate_physics_v2.py \
        --checkpoint checkpoints_run1/checkpoint_epoch_5.pt \
        --prompt_id 1
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusers import LattePipeline

from src.latte_physics import LatteTransformer3DModelWithPhysics


PROMPTS_CSV    = Path(__file__).parent / "eval_prompts.csv"
OUTPUT_DIR     = Path(__file__).parent / "physics_videos"

MODEL_ID       = "maxin-cn/Latte-1"

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


def build_physics_model(checkpoint_path: str, device: str, dtype=torch.float16):
    """Instantiate the physics model and load the trained checkpoint."""
    model = LatteTransformer3DModelWithPhysics(
        num_attention_heads=16,
        attention_head_dim=72,
        in_channels=4,
        out_channels=8,
        num_layers=28,
        dropout=0.0,
        cross_attention_dim=1152,
        attention_bias=True,
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
    ).to(device, dtype=dtype)

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


class PhysicsPipelineAdapter(torch.nn.Module):
    """Makes LatteTransformer3DModelWithPhysics look like a vanilla
    LatteTransformer3DModel to LattePipeline.

    - Drops physics-training-only kwargs (ground_truth_vjepa, teacher_force_prob).
    - Ignores added_cond_kwargs from the pipeline; our model builds its own.
    - Unwraps the (Transformer2DModelOutput, predicted_vjepa) tuple return so
      the pipeline sees just the transformer output with .sample.
    """
    def __init__(self, physics_model):
        super().__init__()
        self.physics_model = physics_model
        # LattePipeline reads these off transformer directly
        self.config = physics_model.config
        self.dtype = next(physics_model.parameters()).dtype
        self.device = next(physics_model.parameters()).device

    def forward(
        self,
        hidden_states,
        timestep=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        added_cond_kwargs=None,
        enable_temporal_attentions=True,
        return_dict=True,
        **kwargs,
    ):
        # Physics model builds added_cond_kwargs internally; discard pipeline's
        _ = added_cond_kwargs
        kwargs.pop("ground_truth_vjepa", None)
        kwargs.pop("teacher_force_prob", None)

        result = self.physics_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            enable_temporal_attentions=enable_temporal_attentions,
            return_dict=return_dict,
        )
        # Physics model returns:
        #   return_dict=True  -> (Transformer2DModelOutput, predicted_vjepa)
        #   return_dict=False -> (output_tensor, predicted_vjepa)
        first = result[0] if isinstance(result, tuple) else result

        # Model natively outputs 8 channels (noise + learned sigma). LattePipeline
        # calls chunk(2, dim=1)[0] downstream to take the noise half.
        if return_dict:
            return first
        else:
            return (first,)


def generate(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(PROMPTS_CSV)
    if args.prompt_id is not None:
        df = df[df["id"] == args.prompt_id]
        if df.empty:
            raise ValueError(f"No prompt with id={args.prompt_id}")

    print(f"Loading LattePipeline from {MODEL_ID} (fp16) ...")
    pipe = LattePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)

    # Match training-time T5 sequence length (226) instead of the pipeline's
    # default (120), so caption_projection sees the same shape.
    pipe.tokenizer.model_max_length = 226
    print(f"Tokenizer max length forced to: {pipe.tokenizer.model_max_length}")

    print(f"Loading physics-trained transformer (fp16) from {args.checkpoint} ...")
    physics_model = build_physics_model(args.checkpoint, device, dtype=torch.float16)

    # Swap the pipeline's transformer with the physics-conditioned model
    pipe.transformer = PhysicsPipelineAdapter(physics_model)
    print("Swapped in physics transformer.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        prompt_id = int(row["id"])
        prompt    = row["prompt"]
        slug      = slugify(prompt)
        out_path  = OUTPUT_DIR / f"{prompt_id:02d}_{slug}.mp4"

        generator = torch.Generator(device=device).manual_seed(prompt_id * 1000)

        print(f"[{prompt_id:02d}] Generating (seed={prompt_id * 1000}): {prompt[:80]}")

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                video_length=NUM_FRAMES,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )

        pil_frames = result.frames[0]
        frames_np  = np.stack([np.array(f) for f in pil_frames])
        save_video(frames_np, out_path)
        print(f"  Saved ({NUM_FRAMES} frames @ {PLAYBACK_FPS}fps): {out_path}")

    print(f"\nDone. Videos saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate physics-conditioned videos via LattePipeline + swapped transformer"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--prompt_id", type=int, default=None,
                        help="Generate only this prompt id (for testing)")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    args = parser.parse_args()
    generate(args)
