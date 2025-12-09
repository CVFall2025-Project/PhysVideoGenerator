import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
import imageio

from diffusers import AutoencoderKLCogVideoX
from peft import PeftModel

# ------------------------------------------------------------------------
# Import your training definitions
#   >>> IMPORTANT: change this import to match your actual module name <<<
# ------------------------------------------------------------------------
from cogvideox_physics_train import (   # was "02_cogvideox_physics_with_lora"
    Config,
    CogVideoXWithPhysics,
    PredictorP,
    make_beta_schedule,
    sinusoidal_timestep_embedding,
)

# If you have a separate TextEncoder helper (recommended)
from text_caption_enocder import TextEncoder   # adjust path if needed


# ------------------------------------------------------------------------
# Utility: load prompts (VideoPhy-2 style or simple list)
# ------------------------------------------------------------------------
def load_prompts(path: str) -> List[dict]:
    """
    Returns a list of dicts: {"id": str, "prompt": str}

    Supports:
      - JSON list of {"id": ..., "prompt": ...}
      - JSON list of {"scene_id": ..., "text": ...}  (adapt to VideoPhy-2)
      - Plain text file: one prompt per line
    """
    path = Path(path)
    if path.suffix == ".txt":
        prompts = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                prompts.append({"id": f"line_{i}", "prompt": line})
        return prompts

    # JSON
    with open(path, "r") as f:
        data = json.load(f)

    prompts = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            prompts.append({"id": f"item_{i}", "prompt": item})
        else:
            # Try common key names – adapt if your VideoPhy-2 json is different
            pid = str(item.get("id", item.get("scene_id", i)))
            ptxt = item.get("prompt", item.get("text", None))
            if ptxt is None:
                raise ValueError(f"Cannot find prompt text in item {i}: {item}")
            prompts.append({"id": pid, "prompt": ptxt})
    return prompts


# ------------------------------------------------------------------------
# Load trained model + predictor for inference
# ------------------------------------------------------------------------
def load_models_for_inference(config: Config, ckpt_path: str, device: str):
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # 1) Instantiate base CogVideoX+Physics and PredictorP
    vdm = CogVideoXWithPhysics(config).to(device)
    predictor = PredictorP(config).to(device)

    # 2) Load physics modules + predictor weights
    print(f"Loading physics checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # physics_attns: list of state_dicts, one per injected PhysicsCrossAttention
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
    #    We expect checkpoint file name: checkpoint_epoch{e}_step{s}.pt
    #    and LoRA dir: checkpoints/lora_epoch{e}_step{s}
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
    # The training path: vdm.transformer = get_peft_model(...)
    # For inference, wrap base model again with PEFT, then load pretrained LoRA
    base_model = vdm.transformer.base_model.model  # underlying CogVideoX transformer
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
    num_steps: int = None,
) -> torch.Tensor:
    """
    Run DDPM sampling:
      - Start from Gaussian noise z_T
      - At each step t, use PredictorP + CogVideoXWithPhysics
        to predict eps_hat and step to z_{t-1}.

    text_tokens: [1, L, TEXT_DIM]
    Returns: final latent z_0 of shape [1, C, T, H, W]
    """
    if num_steps is None:
        num_steps = config.T_STEPS

    # Diffusion schedule (same as training)
    betas, alphas, alpha_bar = make_beta_schedule(config.T_STEPS, "linear")
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)

    alpha_bar_prev = torch.cat(
        [torch.tensor([1.0], device=device), alpha_bar[:-1]], dim=0
    )

    # Init latent with Gaussian noise
    z = torch.randn(
        1,
        config.LATENT_C,
        config.LATENT_T,
        config.LATENT_H,
        config.LATENT_W,
        device=device,
    )

    # Sampling from T-1 -> 0
    vdm.eval()
    predictor.eval()

    with torch.no_grad():
        for i, t_int in enumerate(reversed(range(num_steps))):
            t = torch.full((1,), t_int, dtype=torch.long, device=device)

            # t embedding (same as training)
            t_emb = sinusoidal_timestep_embedding(t, config.DIM_T)

            # (z_t, text_tokens, t_emb) -> predicted VFM tokens
            predicted_vfm = predictor(z, text_tokens, t_emb)

            # VDM forward: predicts eps_hat (noise) in latent space
            eps_hat = vdm(z, t, text_tokens, predicted_vfm)  # [1, C, T, H, W]

            beta_t = betas[t_int]
            alpha_t = alphas[t_int]
            alpha_bar_t = alpha_bar[t_int]
            alpha_bar_prev_t = alpha_bar_prev[t_int]

            # DDPM posterior mean (Ho et al., 2020)
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mean = coef1 * (z - coef2 * eps_hat)

            if t_int > 0:
                noise = torch.randn_like(z)
                # Simple choice sigma_t = sqrt(beta_t)
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

    # CogVideoX VAE uses scaling_factor similar to Stable Diffusion
    scaling = getattr(vae.config, "scaling_factor", 0.18215)
    latents = latents / scaling

    with torch.no_grad():
        # decode returns [B, C, T, H, W] in [-1, 1]
        decoded = vae.decode(latents).sample

    # [-1, 1] -> [0, 1]
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
        help="Path to checkpoint_epoch*_step*.pt (physics + predictor).",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to prompts file (JSON or TXT).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="samples_videophy2",
        help="Where to save generated videos.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of diffusion steps to run (default: Config.T_STEPS).",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="cuda or cpu (default: auto)"
    )
    args = parser.parse_args()

    config = Config()

    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # 1) Load models
    vdm, predictor = load_models_for_inference(config, args.checkpoint, device)

    # 2) Load VAE for decoding
    print(f"Loading VAE from {config.MODEL_NAME}")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        config.MODEL_NAME,
        subfolder="vae",
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    vae.eval()

    # 3) Text encoder (T5-large, same as dataset pipeline)
    print("Loading text encoder (T5-large).")
    text_encoder = TextEncoder(
        device=device,
        model_name="t5-large",  # must match dataset encoding
        max_length=config.TEXT_SEQ_LEN,
    )

    # 4) Load prompts
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) Generate videos
    torch.set_grad_enabled(False)

    for i, item in enumerate(prompts):
        pid = item["id"]
        prompt = item["prompt"]
        print(f"\n[{i+1}/{len(prompts)}] Generating for id={pid} | prompt={prompt}")

        # Encode text -> [L, TEXT_DIM] -> [1, L, TEXT_DIM]
        text_tokens_np = text_encoder.encode(prompt)  # np.ndarray
        text_tokens = (
            torch.from_numpy(text_tokens_np)
            .unsqueeze(0)
            .to(device)
            .float()
        )

        # DDPM sampling in latent space
        latents = ddpm_sample(
            vdm,
            predictor,
            config,
            text_tokens,
            device,
            num_steps=args.num_steps,
        )

        # Decode to video and save
        video = decode_latents_to_video(latents, vae, device, fps=12)
        out_path = out_dir / f"{pid}.mp4"
        save_video(video, out_path, fps=12)
        print(f"Saved video to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
