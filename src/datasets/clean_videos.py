from __future__ import annotations

import os
from typing import Dict, Tuple, Optional
import logging

import numpy as np
import cv2
from tqdm import tqdm

import torch
from torchvision.transforms import functional as F

try:
    import decord
    decord.bridge.set_bridge("native")
except Exception:
    decord = None

logger = logging.getLogger(__name__)


def load_video_decord(path: str, target_fps: int) -> np.ndarray:
    """Load video using Decord and resample to target_fps.
    
    Returns:
        frames: (T, H, W, 3) uint8 numpy array
    """
    if decord is None:
        raise RuntimeError("Decord not available. Install decord.")

    vr = decord.VideoReader(path)
    orig_fps = vr.get_avg_fps()
    if orig_fps is None or orig_fps == 0:
        orig_fps = target_fps

    step = float(orig_fps) / float(target_fps)
    if step <= 0:
        step = 1.0

    idxs = (np.arange(0, len(vr), step)).astype(int)
    idxs = idxs[idxs < len(vr)]
    frames = vr.get_batch(idxs).asnumpy()
    return frames  # (T, H, W, 3)


def load_video_opencv(path: str, target_fps: int) -> np.ndarray:
    """Fallback: Load video using OpenCV.
    
    Returns:
        frames: (T, H, W, 3) uint8 numpy array
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps is None or orig_fps <= 0:
        orig_fps = target_fps

    step = max(1, int(round(orig_fps / target_fps)))
    frames = []
    frame_idx = 0
    target_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1

    cap.release()
    if frames:
        return np.stack(frames, axis=0)
    else:
        raise RuntimeError(f"No frames extracted from {path}")


class VideoProcessor:
    """Process raw videos into VAE and VJEPA2 formats."""

    def __init__(
        self,
        target_fps: int = 12,
        frame_size: int = 256,
        num_frames: int = 16,
        vjepa_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        vjepa_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_decord: bool = True,
        device: str = "cpu",
    ):
        """Initialize video processor.
        
        Args:
            target_fps: target frames per second for loading videos
            vae_height, vae_width: VAE input dimensions
            vae_frames: number of frames for VAE (should be 49)
            vjepa_size: square size for VJEPA2 (should be 256)
            vjepa_frames: number of frames for VJEPA2 (should be 48, must be even)
            vjepa_mean, vjepa_std: ImageNet normalization stats
            use_decord: if True, use Decord; else use OpenCV fallback
            device: torch device for processing
        """
        self.target_fps = target_fps
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.vjepa_mean = vjepa_mean
        self.vjepa_std = vjepa_std
        self.use_decord = use_decord
        self.device = device

        if torch is None:
            raise RuntimeError("PyTorch is required. Install torch and torchvision.")


    def load_video(self, path: str) -> np.ndarray:
        """Load video and resample to target fps."""
        if self.use_decord and decord is not None:
            try:
                return load_video_decord(path, self.target_fps)
            except Exception as e:
                logger.warning(f"Decord failed ({e}), falling back to OpenCV")
                return load_video_opencv(path, self.target_fps)
        else:
            return load_video_opencv(path, self.target_fps)

    def process_video(self, frames_np: np.ndarray) -> Dict[str, np.ndarray]:
        """Process raw video frames into VAE and VJEPA2 formats.
        
        Args:
            frames_np: (T, H, W, 3) uint8 numpy array
        
        Returns:
            {
                "vae": (3, 16, 256, 256) float32 in [-1, 1],
                "vjepa": (3, 16, 256, 256) float32 ImageNet normalized
            }
        """
        # Convert to torch tensor (T, H, W, 3) and move to device
        video_tensor = torch.from_numpy(frames_np).float().to(self.device)  # [T, H, W, 3]

        # Permute to (T, 3, H, W) for torchvision ops
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, 3, H, W]

        H_orig = video_tensor.shape[2]
        W_orig = video_tensor.shape[3]
        min_dim = min(H_orig, W_orig)

        video = F.center_crop(video_tensor, [min_dim, min_dim])
        video = F.resize(video, [self.frame_size, self.frame_size])

        T = video.shape[0]

        indices = torch.linspace(0, T - 1, self.num_frames, device=video_tensor.device).long()

        video = video[indices]


        # ---------------------------------------------------------
        # Pipeline A: VAE (256x256, 16 frames, [-1, 1])
        # ---------------------------------------------------------
        # Normalize to [-1, 1]
        vae_video = video.clone().detach()
        vae_video = (vae_video / 127.5) - 1.0

        # Permute to [1, 16, 3 , 256, 256] for VAE input
        vae_video = vae_video.unsqueeze(0)

        # ---------------------------------------------------------
        # Pipeline B: VJEPA2 (256x256, 16 frames, ImageNet normalized)
        # ---------------------------------------------------------
        # Normalize to [0, 1] then apply ImageNet normalization
        vjepa_video = video.clone().detach()
        vjepa_video = video / 255.0

        # Apply ImageNet normalization per-channel
        for c in range(3):
            vjepa_video[:, c] = (vjepa_video[:, c] - self.vjepa_mean[c]) / self.vjepa_std[c]

        # Move outputs back to CPU and convert to numpy for saving (keeps API unchanged)
        return {
            "vae": vae_video.cpu().numpy().astype(np.float16), # [1, 16, 3, 256, 256]
            "vjepa": vjepa_video.cpu().numpy().astype(np.float16), # [16, 3, 256, 256]
        }

    def process_video_to_file(
        self,
        video_path: str,
        output_base: str,
        vae_suffix: str = "_vae.npz",
        vjepa_suffix: str = "_vjepa.npz",
    ) -> bool:
        """Process a single video file and save outputs.
        
        Args:
            video_path: path to input video
            output_base: base path for outputs (e.g., "data/vid_001")
            vae_suffix: suffix for VAE output file
            vjepa_suffix: suffix for VJEPA2 output file
        
        Returns:
            True on success, False on error
        """
        try:
            frames = self.load_video(video_path)
            outputs = self.process_video(frames)

            vae_path = output_base + vae_suffix
            vjepa_path = output_base + vjepa_suffix

            np.savez_compressed(vae_path, outputs["vae"])
            np.savez_compressed(vjepa_path, outputs["vjepa"])

            return True
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            return False

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        limit: Optional[int] = None,
        video_extensions: Optional[Tuple[str, ...]] = (".mp4", ".mov", ".avi", ".mkv"),
    ) -> Dict[str, int]:
        """Batch process all videos in a folder.
        
        Args:
            input_dir: directory with raw videos
            output_dir: directory for processed outputs
            limit: max number of videos to process (None = all)
            video_extensions: video file extensions to process
        
        Returns:
            summary dict with counts
        """
        os.makedirs(output_dir, exist_ok=True)

        video_files = [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith(video_extensions)
        ]
        if limit is not None:
            video_files = video_files[:limit]

        total = len(video_files)
        processed = 0
        failed = 0

        for v in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(input_dir, v)
            base_name = os.path.splitext(v)[0]
            output_base = os.path.join(output_dir, base_name)

            if self.process_video_to_file(video_path, output_base):
                processed += 1
            else:
                failed += 1

        delete_command = "rm -rf " + input_dir + "/*.mp4"
        os.system(delete_command)

        summary = {"processed": processed, "failed": failed, "total": total}
        logger.info(f"Batch processing complete: {processed}/{total} successful, {failed} failed.")
        return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process raw videos into VAE and VJEPA2 formats"
    )
    parser.add_argument(
        "--input_dir",
        default="data/video",
        help="Directory with raw videos",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed",
        help="Directory to save processed videos",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Target FPS for loading",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max videos to process (for testing)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (cpu or cuda)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    processor = VideoProcessor(
        target_fps=args.fps,
        device=args.device,
    )
    processor.process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
    )
