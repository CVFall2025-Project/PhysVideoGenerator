"""
Video Evaluation Script - CORRECTED for Frame Sampling

Handles the case where training videos had 16 frames sampled from longer videos.
Compares generated videos with the EXACT SAME 16 frames from ground truth.

Usage:
    python evaluate_videos_corrected.py \
        --generated_dir ./test_outputs \
        --test_index ./data/test_index.json \
        --output_file metrics_results.json
"""

import torch
import numpy as np
import json
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Metrics
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def load_video_frames(
    video_path: str,
    frame_indices: Optional[List[int]] = None,
    num_frames: int = 16,
    size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Load specific frames from video.
    
    Args:
        video_path: Path to video file
        frame_indices: Specific frame indices to load (e.g., [0, 5, 10, ...])
                       If None, uniformly sample num_frames
        num_frames: Number of frames to load (if frame_indices is None)
        size: Resize to (H, W)
    
    Returns:
        Video as numpy array [T, H, W, C] in range [0, 255]
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine which frames to extract
    if frame_indices is None:
        # Uniform sampling
        if total_frames >= num_frames:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        else:
            # Video shorter than requested frames - take all and pad
            frame_indices = list(range(total_frames))
    
    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {idx} from {video_path}")
            # Use last valid frame or black frame
            if frames:
                frame = frames[-1].copy()
            else:
                frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Resize
        frame = cv2.resize(frame, size)
        
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frames.append(frame)
    
    cap.release()
    
    # Pad if necessary
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((size[1], size[0], 3), dtype=np.uint8))
    
    return np.array(frames[:num_frames])  # [T, H, W, C]


def load_generated_video(video_path: str, num_frames: int = 16, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Load all frames from generated video (should already be exactly num_frames).
    
    Args:
        video_path: Path to generated video
        num_frames: Expected number of frames
        size: Expected size (H, W)
    
    Returns:
        Video as numpy array [T, H, W, C] in range [0, 255]
    """
    return load_video_frames(video_path, frame_indices=None, num_frames=num_frames, size=size)


def compute_lpips(gen_video: np.ndarray, gt_video: np.ndarray, device: str = 'cuda') -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).
    
    Args:
        gen_video: Generated video [T, H, W, C] in range [0, 255]
        gt_video: Ground truth video [T, H, W, C] in range [0, 255]
        device: Device to use
    
    Returns:
        Average LPIPS score across all frames
    """
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Convert to torch tensors [T, C, H, W] in range [-1, 1]
    gen_tensor = torch.from_numpy(gen_video).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    gt_tensor = torch.from_numpy(gt_video).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    
    gen_tensor = gen_tensor.to(device)
    gt_tensor = gt_tensor.to(device)
    
    # Compute LPIPS for each frame
    lpips_scores = []
    with torch.no_grad():
        for i in range(gen_tensor.shape[0]):
            score = lpips_model(gen_tensor[i:i+1], gt_tensor[i:i+1])
            lpips_scores.append(score.item())
    
    return np.mean(lpips_scores)


def compute_psnr(gen_video: np.ndarray, gt_video: np.ndarray, device: str = 'cuda') -> float:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio).
    
    Args:
        gen_video: Generated video [T, H, W, C] in range [0, 255]
        gt_video: Ground truth video [T, H, W, C] in range [0, 255]
        device: Device to use
    
    Returns:
        Average PSNR across all frames in dB
    """
    psnr_metric = PeakSignalNoiseRatio(data_range=255.0).to(device)
    
    # Convert to torch tensors [T, C, H, W]
    gen_tensor = torch.from_numpy(gen_video).permute(0, 3, 1, 2).float().to(device)
    gt_tensor = torch.from_numpy(gt_video).permute(0, 3, 1, 2).float().to(device)
    
    # Compute PSNR for each frame
    psnr_scores = []
    with torch.no_grad():
        for i in range(gen_tensor.shape[0]):
            score = psnr_metric(gen_tensor[i:i+1], gt_tensor[i:i+1])
            psnr_scores.append(score.item())
    
    return np.mean(psnr_scores)


def compute_ssim(gen_video: np.ndarray, gt_video: np.ndarray, device: str = 'cuda') -> float:
    """
    Compute SSIM (Structural Similarity Index Measure).
    
    Args:
        gen_video: Generated video [T, H, W, C] in range [0, 255]
        gt_video: Ground truth video [T, H, W, C] in range [0, 255]
        device: Device to use
    
    Returns:
        Average SSIM across all frames (0-1, higher is better)
    """
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=255.0).to(device)
    
    # Convert to torch tensors [T, C, H, W]
    gen_tensor = torch.from_numpy(gen_video).permute(0, 3, 1, 2).float().to(device)
    gt_tensor = torch.from_numpy(gt_video).permute(0, 3, 1, 2).float().to(device)
    
    # Compute SSIM for each frame
    ssim_scores = []
    with torch.no_grad():
        for i in range(gen_tensor.shape[0]):
            score = ssim_metric(gen_tensor[i:i+1], gt_tensor[i:i+1])
            ssim_scores.append(score.item())
    
    return np.mean(ssim_scores)


def evaluate_video_pair(
    gen_video_path: str,
    gt_video_path: str,
    frame_indices: Optional[List[int]] = None,
    num_frames: int = 16,
    size: Tuple[int, int] = (256, 256),
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate a single generated-ground truth video pair.
    
    Args:
        gen_video_path: Path to generated video
        gt_video_path: Path to ground truth video
        frame_indices: Specific frame indices used during training
        num_frames: Number of frames
        size: Video size (H, W)
        device: Device to use
    
    Returns:
        Dictionary with metrics
    """
    # Load generated video (all frames)
    gen_video = load_generated_video(gen_video_path, num_frames, size)
    
    # Load ground truth video (SAME frames as training)
    gt_video = load_video_frames(gt_video_path, frame_indices, num_frames, size)
    
    # Verify shapes match
    if gen_video.shape != gt_video.shape:
        print(f"  Warning: Shape mismatch - gen {gen_video.shape} vs gt {gt_video.shape}")
        return {'error': 'shape_mismatch'}
    
    # Compute metrics
    metrics = {}
    
    try:
        metrics['lpips'] = compute_lpips(gen_video, gt_video, device)
    except Exception as e:
        print(f"  Warning: LPIPS failed - {e}")
        metrics['lpips'] = None
    
    try:
        metrics['psnr'] = compute_psnr(gen_video, gt_video, device)
    except Exception as e:
        print(f"  Warning: PSNR failed - {e}")
        metrics['psnr'] = None
    
    try:
        metrics['ssim'] = compute_ssim(gen_video, gt_video, device)
    except Exception as e:
        print(f"  Warning: SSIM failed - {e}")
        metrics['ssim'] = None
    
    return metrics


def evaluate_dataset(
    generated_dir: str,
    test_index_path: str,
    output_file: str = "metrics_results.json",
    num_frames: int = 16,
    size: Tuple[int, int] = (256, 256),
    device: str = 'cuda',
):
    """
    Evaluate entire test dataset with correct frame sampling.
    
    Args:
        generated_dir: Directory with generated videos
        test_index_path: Path to test_index.json (REQUIRED - contains frame indices)
        output_file: Where to save results
        num_frames: Number of frames
        size: Video size
        device: Device to use
    """
    generated_dir = Path(generated_dir)
    
    print(f"\n{'='*60}")
    print("VIDEO EVALUATION (CORRECTED FOR FRAME SAMPLING)")
    print(f"{'='*60}")
    print(f"Generated videos: {generated_dir}")
    print(f"Test index: {test_index_path}")
    
    # Load test index
    print(f"\nLoading test index...")
    with open(test_index_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"Test samples: {len(test_data)}")
    
    # Check if frame_indices are provided
    has_frame_indices = 'frame_indices' in test_data[0] if test_data else False
    
    if not has_frame_indices:
        print("\n⚠️  WARNING: test_index.json does not contain 'frame_indices'!")
        print("⚠️  Will use uniform sampling, but this may give incorrect metrics.")
        print("⚠️  See EVALUATION_GUIDE.md for how to add frame_indices.\n")
    
    # Evaluate each sample
    results = []
    all_lpips = []
    all_psnr = []
    all_ssim = []
    
    for sample in tqdm(test_data, desc="Evaluating videos"):
        video_id = sample['video_id']
        
        # Generated video path
        gen_path = generated_dir / f"{video_id}_generated.mp4"
        
        # Ground truth video path
        if 'video_path' in sample:
            gt_path = Path(sample['video_path'])
        else:
            print(f"  Warning: No 'video_path' for {video_id}, skipping")
            continue
        
        # Frame indices (if available)
        frame_indices = sample.get('frame_indices', None)
        
        if not gen_path.exists():
            print(f"  Warning: Missing generated video: {gen_path}")
            continue
        
        if not gt_path.exists():
            print(f"  Warning: Missing ground truth video: {gt_path}")
            continue
        
        try:
            # Evaluate with correct frame sampling
            metrics = evaluate_video_pair(
                str(gen_path),
                str(gt_path),
                frame_indices=frame_indices,
                num_frames=num_frames,
                size=size,
                device=device,
            )
            
            if 'error' in metrics:
                results.append({
                    'video_id': video_id,
                    'error': metrics['error'],
                })
                continue
            
            result = {
                'video_id': video_id,
                'generated_path': str(gen_path),
                'ground_truth_path': str(gt_path),
                'frame_indices': frame_indices,
                'lpips': metrics['lpips'],
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
            }
            
            results.append(result)
            
            # Collect for averaging
            if metrics['lpips'] is not None:
                all_lpips.append(metrics['lpips'])
            if metrics['psnr'] is not None:
                all_psnr.append(metrics['psnr'])
            if metrics['ssim'] is not None:
                all_ssim.append(metrics['ssim'])
                
        except Exception as e:
            print(f"\n❌ Error evaluating {video_id}: {e}")
            results.append({
                'video_id': video_id,
                'error': str(e),
            })
    
    # Compute average metrics
    summary = {
        'num_videos': len(test_data),
        'num_successful': len(all_lpips),
        'frame_indices_provided': has_frame_indices,
        'average_metrics': {
            'lpips': float(np.mean(all_lpips)) if all_lpips else None,
            'psnr': float(np.mean(all_psnr)) if all_psnr else None,
            'ssim': float(np.mean(all_ssim)) if all_ssim else None,
        },
        'std_metrics': {
            'lpips': float(np.std(all_lpips)) if all_lpips else None,
            'psnr': float(np.std(all_psnr)) if all_psnr else None,
            'ssim': float(np.std(all_ssim)) if all_ssim else None,
        }
    }
    
    # Save results
    output = {
        'summary': summary,
        'per_video_results': results,
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Videos evaluated: {summary['num_successful']}/{summary['num_videos']}")
    print(f"Frame indices provided: {'Yes ✓' if has_frame_indices else 'No ✗ (using uniform sampling)'}")
    print(f"\nAverage Metrics:")
    if summary['average_metrics']['lpips']:
        print(f"  LPIPS: {summary['average_metrics']['lpips']:.4f} ± {summary['std_metrics']['lpips']:.4f}")
    if summary['average_metrics']['psnr']:
        print(f"  PSNR:  {summary['average_metrics']['psnr']:.2f} ± {summary['std_metrics']['psnr']:.2f} dB")
    if summary['average_metrics']['ssim']:
        print(f"  SSIM:  {summary['average_metrics']['ssim']:.4f} ± {summary['std_metrics']['ssim']:.4f}")
    print(f"\n✓ Results saved: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated videos (corrected for frame sampling)")
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory with generated videos")
    parser.add_argument("--test_index", type=str, required=True, help="Path to test_index.json (with frame_indices)")
    parser.add_argument("--output_file", type=str, default="metrics_results.json", help="Output JSON file")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--height", type=int, default=256, help="Video height")
    parser.add_argument("--width", type=int, default=256, help="Video width")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    evaluate_dataset(
        generated_dir=args.generated_dir,
        test_index_path=args.test_index,
        output_file=args.output_file,
        num_frames=args.num_frames,
        size=(args.height, args.width),
        device=args.device,
    )