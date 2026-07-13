"""
generate_baseline.py

Generates 16-frame videos for all 25 evaluation prompts using the unmodified
base Latte-1 model (no physics conditioning). These serve as the baseline for
comparison against the physics-trained model.

Output layout:
    evaluation/baseline_videos/<id>_<slug>.mp4   — one file per prompt

Playback:
    Videos are saved at 8 fps (16 frames = 2 seconds). This is intentionally
    slower than the training fps of 12 so physics motion is visually inspectable.
    Both baseline and physics-trained videos must use the same playback fps for
    a fair side-by-side comparison.

VRAM requirements:
    T5-XXL encoder  : ~22 GB (float16)
    Latte transformer: ~3 GB (float16)
    Pipeline peak   : ~26 GB total — requires A100 (40GB) or equivalent.

    On a 24 GB GPU: use --sequential_offload flag (loads text encoder then
    unloads before running the diffusion transformer). Slower but fits.

Usage:
    # From project root
    python evaluation/generate_baseline.py

    # With sequential CPU offload for 24 GB GPUs
    python evaluation/generate_baseline.py --sequential_offload

    # Generate a single prompt by id (for testing)
    python evaluation/generate_baseline.py --prompt_id 1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import pandas as pd
import imageio
import numpy as np

PROMPTS_CSV    = Path(__file__).parent / "eval_prompts.csv"
OUTPUT_DIR     = Path(__file__).parent / "baseline_videos"
MODEL_ID       = "maxin-cn/Latte-1"
NUM_FRAMES     = 16       # Fixed by our latent shape [4, 16, 32, 32]
NUM_STEPS      = 50       # DDIM steps; matches generate_latte_physics.py
GUIDANCE_SCALE = 7.5
HEIGHT         = 256      # Matches training resolution
WIDTH          = 256
PLAYBACK_FPS   = 8        # Slower than training fps (12) for visual inspection


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len]


def tensor_to_uint8(frames: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert [T, H, W, C] float in [-1,1] or [0,1] to uint8."""
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    if frames.max() <= 1.01:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    else:
        frames = frames.clip(0, 255).astype(np.uint8)
    return frames


def save_video(frames: np.ndarray, path: Path, fps: int = PLAYBACK_FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def generate(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU found. Generation will be extremely slow (~30-60 min/video).")

    # Load prompts
    df = pd.read_csv(PROMPTS_CSV)
    if args.prompt_id is not None:
        df = df[df["id"] == args.prompt_id]
        if df.empty:
            raise ValueError(f"No prompt with id={args.prompt_id}")

    # Build pipeline
    print(f"Loading Latte-1 pipeline from {MODEL_ID} ...")
    from diffusers import LattePipeline

    pipe = LattePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)

    if args.sequential_offload:
        # Moves each submodule to GPU only when needed — fits 24 GB GPUs
        pipe.enable_sequential_cpu_offload()
        print("Sequential CPU offload enabled.")
    else:
        pipe = pipe.to(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        prompt_id = int(row["id"])
        prompt    = row["prompt"]
        slug      = slugify(prompt)
        out_path  = OUTPUT_DIR / f"{prompt_id:02d}_{slug}.mp4"

        if out_path.exists() and not args.overwrite:
            print(f"[{prompt_id:02d}] Skipping (exists): {out_path.name}")
            continue

        print(f"[{prompt_id:02d}] Generating: {prompt[:80]}")

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, static, no motion",
                video_length=NUM_FRAMES,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            )

        # diffusers LattePipeline returns result.frames: list[list[PIL.Image]]
        # shape: [num_videos][num_frames] PIL images
        pil_frames = result.frames[0]
        frames_np  = np.stack([np.array(f) for f in pil_frames])  # [T, H, W, 3]

        save_video(frames_np, out_path)
        print(f"  Saved ({NUM_FRAMES} frames @ {PLAYBACK_FPS}fps → {NUM_FRAMES/PLAYBACK_FPS:.1f}s): {out_path}")

    print(f"\nDone. Videos saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate baseline Latte-1 videos for eval prompts")
    parser.add_argument("--prompt_id",        type=int, default=None,
                        help="Generate only this prompt id (for testing)")
    parser.add_argument("--sequential_offload", action="store_true",
                        help="Enable sequential CPU offload (for 24 GB GPUs)")
    parser.add_argument("--overwrite",        action="store_true",
                        help="Regenerate even if output file already exists")
    args = parser.parse_args()
    generate(args)
