import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F  # only needed if you add metrics
import imageio
from datasets import load_dataset
from diffusers import AutoencoderKLCogVideoX

# If you don't use these, you can safely remove:
# from huggingface_hub import hf_hub_download
# import pandas as pd

#expected to run from the project root as such:
#python src/infer_videophy2.py --checkpoint /path/to/checkpoint_epochX_stepY.pt --hard_only


# ------------------------------------------------------------------------
# Local imports – assumes this file lives in src/ and training file renamed
# from 02_cogvideox_physics_with_lora.py -> cogvideox_physics_with_lora.py
# ------------------------------------------------------------------------
from 02_cogvideox_physics_with_lora import (  # type: ignore
    Config,
    CogVideoXWithPhysics,
    PredictorP,
    make_beta_schedule,
    sinusoidal_timestep_embedding,
)

# Match the dataset pipeline usage exactly
# (see 01_prepare_video_dataset.py / _streaming.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.encoders.text_caption_enocder import TextEncoder  # type: ignore

# ------------------------------------------------------------------------
# Utility: load prompts (VideoPhy-2 upsampled prompts)
# ------------------------------------------------------------------------
def load_prompts_videophy2_from_datasets(
    split: str = "test",
    use_upsampled: bool = True,
    hard_only: bool = False,
    dedup: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load prompts from the official VideoPhy-2 upsampled prompts dataset:
        videophysics/videophy2_upsampled_prompts

    Returns a list of dicts:
        {
            "id":       str,
            "prompt":   str,
            "action":   Optional[str],
            "category": Optional[str],
            "is_hard":  int (0/1),
        }
    """
    ds = load_dataset("videophysics/videophy2_upsampled_prompts", split=split)

    text_col = "upsampled_caption" if use_upsampled and "upsampled_caption" in ds.column_names else "caption"
    if text_col not in ds.column_names:
        raise ValueError(f"Text column {text_col} not found. Available: {ds.column_names}")

    prompts: List[Dict[str, Any]] = []
    seen_texts = set()

    for idx, row in enumerate(ds):
        prompt = row[text_col]

        if prompt is None:
            continue
        if isinstance(prompt, float) and math.isnan(prompt):
            continue

        is_hard = int(row.get("is_hard", 0)) if "is_hard" in row else 0
        if hard_only and not is_hard:
            continue

        prompt_str = str(prompt).strip()
        if not prompt_str:
            continue

        if dedup:
            if prompt_str in seen_texts:
                continue
            seen_texts.add(prompt_str)

        action = row.get("action", None)
        category = row.get("category", None)

        action_prefix = action if (action is not None and action != "") else "videophy2"
        pid = f"{action_prefix}_{idx}"

        prompts.append(
            {
                "id": pid,
                "prompt": prompt_str,
                "action": action,
                "category": category,
                "is_hard": is_hard,
            }
        )

    print(
        f"Loaded {len(prompts)} prompts from split='{split}', "
        f"hard_only={hard_only}, dedup={dedup}."
    )
    return prompts


# ------------------------------------------------------------------------
# Load trained model + predictor for inference
# ------------------------------------------------------------------------
def load_models_for_inference(config: Config, ckpt_path: str, device: str):
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # 1) Instantiate base CogVideoX+Physics and PredictorP
    vdm = CogVideoXWithPhysics(config).to(device)
    predictor = PredictorP(config).half().to(device)

    # 2) Load physics modules + predictor weights
    print(f"Loading physics checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    physics_state_dicts = ckpt.get("physics_attns", None)
    if physics_state_dicts is not None:
        if len(physics_state_dicts) != len(vdm.physics_attns):
            print(
                f"Warning: checkpoint has {len(physics_state_dicts)} physics_attns, "
                f"model has {len(vdm.physics_attns)}."
            )
        for attn_module, attn_state in zip(vdm.physics_attns, physics_state_dicts):
            attn_module.load_state_dict(attn_state, strict=False)

    predictor.load_state_dict(ckpt["predictor"], strict=False)

    # 3) Load LoRA weights via PEFT
    name = ckpt_path.stem  # e.g., "checkpoint_epoch2_step500"
    parts = name.split("_")
    epoch = None
    step = None
    for p in parts:
        if p.startswith("epoch"):
            epoch = p.replace("epoch", "")
        if p.startswith("step"):
            step = p.replace("step", "")
    if epoch is None or step is None:
        raise ValueError(
            f"Could not parse epoch/step from checkpoint filename: {ckpt_path}"
        )

    lora_dir = config.CHECKPOINT_DIR / f"lora_epoch{epoch}_step{step}"
    if not lora_dir.exists():
        raise FileNotFoundError(
            f"Expected LoRA directory {lora_dir} not found. "
            f"Make sure you saved LoRA with save_checkpoint()."
        )

    print(f"Loading LoRA weights from {lora_dir}")
    base_model = vdm.transformer.base_model.model  # underlying CogVideoX transformer
    from peft import PeftModel

    vdm.transformer = PeftModel.from_pretrained(
        base_model,
        lora_dir,
        is_trainable=False,
    ).to(device)

    vdm.eval()
    predictor.eval()
    return vdm, predictor


# ------------------------------------------------------------------------
# Reverse diffusion sampler in latent space
# ------------------------------------------------------------------------
def ddpm_sample(
    vdm: CogVideoXWithPhysics,
    predictor: PredictorP,
    config: Config,
    text_tokens: torch.Tensor,
    device: str,
    num_steps: Optional[int] = None,
) -> torch.Tensor:
    """
    Run DDPM sampling in latent space.

    text_tokens: [1, L, TEXT_DIM]
    Returns: final latent z_0 of shape [1, C, T, H, W]
    """
    if num_steps is None:
        num_steps = config.T_STEPS

    # Use same dtype as VDM
    dtype = next(vdm.parameters()).dtype

    # Diffusion schedule (same as training)
    betas, alphas, alpha_bar = make_beta_schedule(config.T_STEPS, "linear")
    betas = betas.to(device=device, dtype=dtype)
    alphas = alphas.to(device=device, dtype=dtype)
    alpha_bar = alpha_bar.to(device=device, dtype=dtype)

    # We will sample all steps T-1 -> 0
    # (For now ignore num_steps < T_STEPS; keep them equal)
    assert num_steps == config.T_STEPS, "For now, set num_steps == config.T_STEPS for consistency."

    # Init latent with Gaussian noise
    z = torch.randn(
        1,
        config.LATENT_C,
        config.LATENT_T,
        config.LATENT_H,
        config.LATENT_W,
        device=device,
        dtype=dtype,
    )

    vdm.eval()
    predictor.eval()

    with torch.no_grad():
        for t_int in reversed(range(num_steps)):
            t = torch.full((1,), t_int, dtype=torch.long, device=device)

            # t embedding (same as training)
            t_emb = sinusoidal_timestep_embedding(t, config.DIM_T).to(device=device, dtype=dtype)

            # (z_t, text_tokens, t_emb) -> predicted VFM tokens
            predicted_vfm = predictor(z, text_tokens, t_emb)

            # VDM forward: predicts eps_hat (noise) in latent space
            eps_hat = vdm(
                hidden_states=z,
                encoder_hidden_states=text_tokens,
                timestep=t,
                predicted_vfm=predicted_vfm,
            ).sample  # [1, C, T, H, W]

            beta_t = betas[t_int]
            alpha_t = alphas[t_int]
            alpha_bar_t = alpha_bar[t_int]

            # DDPM posterior mean (Ho et al., 2020) – training used eps-prediction
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            mean = coef1 * (z - coef2 * eps_hat)

            if t_int > 0:
                noise = torch.randn_like(z)
                sigma_t = torch.sqrt(beta_t)
                z = mean + sigma_t * noise
            else:
                z = mean

    return z


# ------------------------------------------------------------------------
# Decode latents to video with CogVideoX VAE
# ------------------------------------------------------------------------
def decode_latents_to_video(
    latents: torch.Tensor,
    vae: AutoencoderKLCogVideoX,
    device: str,
    fps: int = 12,
) -> np.ndarray:
    """
    latents: [1, C, T, H, W]
    returns video: [T, H, W, 3] uint8 in [0, 255]
    """
    latents = latents.to(device)

    scaling = getattr(vae.config, "scaling_factor", 0.18215)
    latents = latents / scaling

    with torch.no_grad():
        decoded = vae.decode(latents).sample  # [B, C, T, H, W] in [-1, 1]

    decoded = (decoded.clamp(-1, 1) + 1.0) / 2.0
    decoded = decoded[0]  # [C, T, H, W]
    decoded = decoded.permute(1, 2, 3, 0)  # [T, H, W, C]
    decoded = (decoded * 255.0).cpu().numpy().astype(np.uint8)
    return decoded


def save_video(video: np.ndarray, path: str, fps: int = 12):
    """
    video: [T, H, W, 3] uint8
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, video, fps=fps, codec="libx264")


# ------------------------------------------------------------------------
# Main entry: generate videos for a list of prompts
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inference script for CogVideoX + Physics (VideoPhy-2 style)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint_epoch*_step*.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="samples_videophy2",
    )
    parser.add_argument(
        "--hard_only",
        action="store_true",
        help="If set, only use the 'hard' subset of VideoPhy-2 prompts.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of DDPM steps (default: use config.T_STEPS).",
    )
    args = parser.parse_args()

    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.num_steps is None:
        num_steps = config.T_STEPS
    else:
        num_steps = args.num_steps

    # 1) Load models
    vdm, predictor = load_models_for_inference(config, args.checkpoint, device)
    dtype = next(vdm.parameters()).dtype

    # 2) Load VAE for decoding
    print(f"Loading VAE from {config.MODEL_NAME}")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        config.MODEL_NAME,
        subfolder="vae",
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    vae.eval()

    # 3) Text encoder – MUST match training (t5-v1_1-xxl, hidden dim 4096)
    print("Loading text encoder (google/t5-v1_1-xxl).")
    text_encoder = TextEncoder(
        model_name="google/t5-v1_1-xxl",
        device=device,
    )

    # 4) Load prompts
    prompts = load_prompts_videophy2_from_datasets(
        split="test",
        use_upsampled=True,
        hard_only=args.hard_only,
        dedup=False,
    )
    print(f"Loaded {len(prompts)} prompts.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)

    # 5) Generate videos
    for i, item in enumerate(prompts):
        pid = item["id"]
        prompt = item["prompt"]
        print(f"\n[{i + 1}/{len(prompts)}] Generating for id={pid} | prompt={prompt}")

        # Encode text -> [1, L, TEXT_DIM]
        text_tokens = text_encoder.encode(
            prompt,
            max_length=config.TEXT_SEQ_LEN,
        )
        # Move to device & match dtype
        text_tokens = text_tokens.to(device=device, dtype=dtype)

        # DDPM sampling in latent space
        latents = ddpm_sample(
            vdm,
            predictor,
            config,
            text_tokens,
            device,
            num_steps=num_steps,
        )

        # Decode to video and save
        video = decode_latents_to_video(latents, vae, device, fps=12)
        out_path = out_dir / f"{pid}.mp4"
        save_video(video, out_path, fps=12)
        print(f"Saved video to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
