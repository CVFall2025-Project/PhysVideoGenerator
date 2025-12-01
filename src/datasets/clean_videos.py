"""clean_videos.py

Convert raw videos into cleaned, fixed-size .npz clips for VAE encoding.

Functions:
- load_video(path, target_fps): loads frames using Decord and resamples to target_fps
- clean_frames(frames, target_size, max_frames): resize + center crop + normalize
- process_video(path, out_path, ...): process a single video and save .npz
- process_videos(input_dir, output_dir, ...): batch-process all mp4s in a folder

CLI usage:
    python -m src.datasets.clean_videos

"""

from __future__ import annotations

import os
from typing import Tuple, Dict, List

import numpy as np
import cv2
from tqdm import tqdm

try:
    import decord
    decord.bridge.set_bridge("native")
except Exception:
    decord = None


def load_video(path: str, target_fps: int) -> np.ndarray:
    """Load video using Decord and normalize to `target_fps`.

    Returns a numpy array of frames with shape (T, H, W, 3) and dtype uint8.
    """
    if decord is None:
        raise RuntimeError("Decord is not available. Install decord or ensure it's importable.")

    vr = decord.VideoReader(path)

    orig_fps = vr.get_avg_fps()
    if orig_fps is None or orig_fps == 0:
        orig_fps = target_fps

    step = float(orig_fps) / float(target_fps)
    if step <= 0:
        step = 1.0

    # sample indices at the target fps rate
    idxs = (np.arange(0, len(vr), step)).astype(int)
    idxs = idxs[idxs < len(vr)]

    frames = vr.get_batch(idxs).asnumpy()  # (T, H, W, 3) uint8
    return frames


def clean_frames(frames: np.ndarray, target_size: int, max_frames: int) -> np.ndarray:
    """Resize, center-crop, limit frames, and normalize to float32 in [0,1].

    Args:
        frames: (T, H, W, C) uint8 numpy array
        target_size: desired height/width after crop
        max_frames: maximum number of frames to keep

    Returns:
        cleaned: (T', target_size, target_size, 3) float32 array with values in [0,1]
    """
    cleaned: List[np.ndarray] = []

    for i, f in enumerate(frames):
        if i >= max_frames:
            break

        h, w = f.shape[:2]
        # scale shorter edge to target_size
        scale = float(target_size) / float(min(h, w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        f_resized = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # center crop to (target_size, target_size)
        h2, w2 = f_resized.shape[:2]
        top = max(0, (h2 - target_size) // 2)
        left = max(0, (w2 - target_size) // 2)
        f_cropped = f_resized[top:top + target_size, left:left + target_size]

        # If cropping produced smaller shape for edge cases, pad with black
        ch, cw = f_cropped.shape[:2]
        if ch != target_size or cw != target_size:
            pad_h = target_size - ch
            pad_w = target_size - cw
            f_padded = cv2.copyMakeBorder(
                f_cropped,
                0,
                pad_h,
                0,
                pad_w,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            f_final = f_padded
        else:
            f_final = f_cropped

        # Convert to float32 normalized
        f_final = f_final.astype(np.float32) / 255.0
        cleaned.append(f_final)

    if len(cleaned) == 0:
        # return empty array with expected dims
        return np.zeros((0, target_size, target_size, 3), dtype=np.float32)

    cleaned_arr = np.stack(cleaned, axis=0)
    return cleaned_arr


def process_video(
    path: str,
    out_path: str,
    target_fps: int = 12,
    target_size: int = 256,
    max_frames: int = 16,
) -> bool:
    """Process a single video into a compressed .npz containing `frames`.

    Returns True on success, False on error.
    """
    try:
        frames = load_video(path, target_fps)
        frames = clean_frames(frames, target_size, max_frames)
        np.savez_compressed(out_path, frames)
        return True

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return False


def process_videos(
    input_dir: str,
    output_dir: str,
    target_fps: int = 12,
    target_size: int = 256,
    max_frames: int = 16,
    limit: int | None = None,
) -> Dict[str, int]:
    """Process all .mp4 files in `input_dir` and save .npz to `output_dir`.

    Returns a summary dict: {"processed": n, "failed": m, "total": t}
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".mp4"))]
    if limit is not None:
        video_files = video_files[:limit]

    total = len(video_files)
    processed = 0
    failed = 0

    for v in tqdm(video_files, desc="Processing videos"):
        in_path = os.path.join(input_dir, v)
        out_path = os.path.join(output_dir, f"{os.path.splitext(v)[0]}.npz")
        ok = process_video(in_path, out_path, target_fps, target_size, max_frames)
        if ok:
            processed += 1
        else:
            failed += 1

    summary = {"processed": processed, "failed": failed, "total": total}
    print(f"Done. Processed {processed}/{total}, failed {failed}.")
    return summary


if __name__ == "__main__":
    # Simple CLI example
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    INPUT_DIR_DEFAULT = os.path.join(project_root, "data", "video")
    OUTPUT_DIR_DEFAULT = os.path.join(project_root, "data", "clean_video_npz")

    parser = argparse.ArgumentParser(description="Process raw videos into cleaned .npz clips")
    parser.add_argument("--input_dir", default=INPUT_DIR_DEFAULT, help="Directory with raw videos")
    parser.add_argument("--output_dir", default=OUTPUT_DIR_DEFAULT, help="Directory to save .npz files")
    parser.add_argument("--fps", type=int, default=12, help="Target FPS for clips")
    parser.add_argument("--size", type=int, default=256, help="Target spatial size (height)")
    parser.add_argument("--max_frames", type=int, default=16, help="Max frames per clip")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
    
    args = parser.parse_args()

    process_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_fps=args.fps,
        target_size=args.size,
        max_frames=args.max_frames,
        limit=args.limit,
    )
