"""Video evaluation script.

This script evaluates videos in the data/video folder using t-LPIPS metric.
It can be run with: python src/evaluate/evaluate.py
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import logging

import torch

# Import from same directory
from metrics import (
    load_video_for_lpips,
    compute_t_lpips,
    TLPIPS,
    calculate_optical_flow_consistency_score,
    VideoPhyEvaluator,
    VideoPhy2Evaluator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_video_files(video_dir: str) -> List[str]:
    """Find all video files in the specified directory.
    
    Args:
        video_dir: Directory containing video files
    
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'}
    video_files = []
    
    if not os.path.exists(video_dir):
        logger.warning(f"Video directory does not exist: {video_dir}")
        return video_files
    
    for file in os.listdir(video_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(video_dir, file))
    
    return sorted(video_files)


def evaluate_video(
    video_path: str,
    lpips_calculator: TLPIPS,
    size: int = 224,
    compute_optical_flow: bool = True,
    max_frames: int = None,
    videophy_evaluator: Optional[VideoPhyEvaluator] = None,
    videophy2_evaluator: Optional[VideoPhy2Evaluator] = None,
    caption: Optional[str] = None
) -> Dict[str, float]:
    """Evaluate a single video using t-LPIPS, optical flow, and VideoPhy metrics.
    
    Args:
        video_path: Path to video file
        lpips_calculator: TLPIPS calculator instance
        size: Target size for video resizing
        compute_optical_flow: Whether to compute optical flow metric
        max_frames: Maximum frames to process for optical flow (None for all)
        videophy_evaluator: Optional VideoPhy evaluator for SA/PC scores
        videophy2_evaluator: Optional VideoPhy-2 evaluator for SA/PC scores
        caption: Optional caption for VideoPhy metrics (required if evaluators provided)
    
    Returns:
        Dictionary with evaluation results
    """
    try:
        logger.info(f"Processing: {video_path}")
        
        # Load video
        frames = load_video_for_lpips(video_path, size=size)
        
        # Compute t-LPIPS
        t_lpips_score = lpips_calculator.compute(frames)
        
        result = {
            'video_path': video_path,
            't_lpips': t_lpips_score,
            'num_frames': frames.shape[0],
            'success': True
        }
        
        # Compute optical flow consistency if requested
        if compute_optical_flow:
            try:
                optical_flow_score = calculate_optical_flow_consistency_score(
                    video_path,
                    max_frames=max_frames,
                    use_decord=True
                )
                result['optical_flow'] = optical_flow_score
            except Exception as e:
                logger.warning(f"  Failed to compute optical flow: {e}")
                result['optical_flow'] = None
                result['optical_flow_error'] = str(e)
        
        # Compute VideoPhy (original) metrics if evaluator provided
        if videophy_evaluator is not None:
            if caption is None:
                logger.warning(f"  Skipping VideoPhy metrics: caption required")
                result['videophy_sa'] = None
                result['videophy_pc'] = None
            else:
                try:
                    videophy_sa = videophy_evaluator.compute_sa(video_path, caption)
                    videophy_pc = videophy_evaluator.compute_pc(video_path)
                    result['videophy_sa'] = videophy_sa
                    result['videophy_pc'] = videophy_pc
                except Exception as e:
                    logger.warning(f"  Failed to compute VideoPhy metrics: {e}")
                    result['videophy_sa'] = None
                    result['videophy_pc'] = None
                    result['videophy_error'] = str(e)
        
        # Compute VideoPhy-2 metrics if evaluator provided
        if videophy2_evaluator is not None:
            if caption is None:
                logger.warning(f"  Skipping VideoPhy-2 metrics: caption required")
                result['videophy2_sa'] = None
                result['videophy2_pc'] = None
            else:
                try:
                    videophy2_sa = videophy2_evaluator.compute_sa(video_path, caption)
                    videophy2_pc = videophy2_evaluator.compute_pc(video_path)
                    result['videophy2_sa'] = videophy2_sa
                    result['videophy2_pc'] = videophy2_pc
                except Exception as e:
                    logger.warning(f"  Failed to compute VideoPhy-2 metrics: {e}")
                    result['videophy2_sa'] = None
                    result['videophy2_pc'] = None
                    result['videophy2_error'] = str(e)
        
        # Log results
        log_parts = [f"t-LPIPS: {t_lpips_score:.4f}"]
        if result.get('optical_flow') is not None:
            log_parts.append(f"Optical Flow: {result['optical_flow']:.4f}")
        if result.get('videophy_sa') is not None:
            log_parts.append(f"VideoPhy SA: {result['videophy_sa']:.4f}, PC: {result['videophy_pc']:.4f}")
        if result.get('videophy2_sa') is not None:
            log_parts.append(f"VideoPhy-2 SA: {result['videophy2_sa']}, PC: {result['videophy2_pc']}")
        log_parts.append(f"Frames: {frames.shape[0]}")
        logger.info("  " + ", ".join(log_parts))
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to evaluate {video_path}: {e}")
        return {
            'video_path': video_path,
            't_lpips': None,
            'optical_flow': None,
            'videophy_sa': None,
            'videophy_pc': None,
            'videophy2_sa': None,
            'videophy2_pc': None,
            'num_frames': None,
            'success': False,
            'error': str(e)
        }


def load_caption_mapping(caption_file: str) -> Dict[str, str]:
    """Load video caption mapping from CSV file.
    
    Args:
        caption_file: Path to CSV file with 'videopath' and 'caption' columns
    
    Returns:
        Dictionary mapping video paths to captions
    """
    import pandas as pd
    try:
        df = pd.read_csv(caption_file)
        if 'videopath' not in df.columns or 'caption' not in df.columns:
            raise ValueError("CSV must contain 'videopath' and 'caption' columns")
        
        # Normalize paths for matching
        mapping = {}
        for _, row in df.iterrows():
            # Use basename for matching if full path doesn't match
            mapping[os.path.basename(row['videopath'])] = row['caption']
            mapping[row['videopath']] = row['caption']
        
        return mapping
    except Exception as e:
        logger.error(f"Failed to load caption file {caption_file}: {e}")
        return {}


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate videos using t-LPIPS, optical flow, and VideoPhy metrics')
    parser.add_argument(
        '--video_dir',
        type=str,
        default='data/video',
        help='Directory containing video files (default: data/video)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=224,
        help='Target size for video resizing (default: 224)'
    )
    parser.add_argument(
        '--net',
        type=str,
        default='vgg',
        choices=['vgg', 'alex'],
        help='LPIPS network type (default: vgg)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run computation on (default: cuda if available, else cpu)'
    )
    parser.add_argument(
        '--no_optical_flow',
        action='store_true',
        help='Skip optical flow computation (faster)'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=None,
        help='Maximum frames to process for optical flow (None for all frames)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file to save results (optional)'
    )
    parser.add_argument(
        '--videophy_checkpoint',
        type=str,
        default=None,
        help='Path to VideoPhy checkpoint (for SA/PC scores, 0-1 scale)'
    )
    parser.add_argument(
        '--videophy2_checkpoint',
        type=str,
        default=None,
        help='Path to VideoPhy-2 checkpoint (for SA/PC scores, 1-5 scale)'
    )
    parser.add_argument(
        '--caption_file',
        type=str,
        default=None,
        help='CSV file with videopath and caption columns (required for VideoPhy metrics)'
    )
    
    args = parser.parse_args()
    
    # Get project root (assuming script is in src/evaluate/)
    project_root = Path(__file__).parent.parent.parent
    video_dir = os.path.join(project_root, args.video_dir)
    
    logger.info(f"Video directory: {video_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"LPIPS network: {args.net}")
    logger.info(f"Resize size: {args.size}")
    logger.info(f"Compute optical flow: {not args.no_optical_flow}")
    if args.max_frames:
        logger.info(f"Max frames for optical flow: {args.max_frames}")
    
    # Find video files
    video_files = find_video_files(video_dir)
    
    if not video_files:
        logger.warning(f"No video files found in {video_dir}")
        return
    
    logger.info(f"Found {len(video_files)} video file(s)")
    
    # Load caption mapping if provided
    caption_mapping = {}
    if args.caption_file:
        caption_file_path = os.path.join(project_root, args.caption_file)
        caption_mapping = load_caption_mapping(caption_file_path)
        logger.info(f"Loaded {len(caption_mapping)} caption(s) from {caption_file_path}")
    
    # Initialize LPIPS calculator
    lpips_calculator = TLPIPS(net=args.net, device=args.device)
    
    # Initialize VideoPhy evaluators if checkpoints provided
    videophy_evaluator = None
    videophy2_evaluator = None
    
    if args.videophy_checkpoint:
        checkpoint_path = os.path.join(project_root, args.videophy_checkpoint)
        try:
            videophy_evaluator = VideoPhyEvaluator(checkpoint_path, device=args.device)
            logger.info(f"Initialized VideoPhy evaluator with checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to initialize VideoPhy evaluator: {e}")
            logger.warning("Continuing without VideoPhy metrics")
    
    if args.videophy2_checkpoint:
        checkpoint_path = os.path.join(project_root, args.videophy2_checkpoint)
        try:
            videophy2_evaluator = VideoPhy2Evaluator(checkpoint_path, device=args.device)
            logger.info(f"Initialized VideoPhy-2 evaluator with checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to initialize VideoPhy-2 evaluator: {e}")
            logger.warning("Continuing without VideoPhy-2 metrics")
    
    # Evaluate each video
    results = []
    for video_path in video_files:
        # Get caption for this video
        caption = None
        if caption_mapping:
            # Try exact path first, then basename
            caption = caption_mapping.get(video_path) or caption_mapping.get(os.path.basename(video_path))
        
        result = evaluate_video(
            video_path,
            lpips_calculator,
            size=args.size,
            compute_optical_flow=not args.no_optical_flow,
            max_frames=args.max_frames,
            videophy_evaluator=videophy_evaluator,
            videophy2_evaluator=videophy2_evaluator,
            caption=caption
        )
        results.append(result)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Evaluation Summary")
    logger.info("="*60)
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if successful_results:
        t_lpips_scores = [r['t_lpips'] for r in successful_results]
        avg_t_lpips = sum(t_lpips_scores) / len(t_lpips_scores)
        
        logger.info(f"Successfully evaluated: {len(successful_results)}/{len(results)} videos")
        logger.info(f"Average t-LPIPS: {avg_t_lpips:.4f}")
        logger.info(f"Min t-LPIPS: {min(t_lpips_scores):.4f}")
        logger.info(f"Max t-LPIPS: {max(t_lpips_scores):.4f}")
        
        # Optical flow statistics if computed
        if not args.no_optical_flow:
            optical_flow_scores = [r.get('optical_flow') for r in successful_results 
                                 if r.get('optical_flow') is not None]
            if optical_flow_scores:
                avg_optical_flow = sum(optical_flow_scores) / len(optical_flow_scores)
                logger.info(f"\nOptical Flow Consistency:")
                logger.info(f"  Average: {avg_optical_flow:.4f} pixels/frame")
                logger.info(f"  Min: {min(optical_flow_scores):.4f} pixels/frame")
                logger.info(f"  Max: {max(optical_flow_scores):.4f} pixels/frame")
                logger.info(f"  (Lower = smoother transitions, Higher = more motion)")
        
        # VideoPhy statistics if computed
        if videophy_evaluator:
            videophy_sa_scores = [r.get('videophy_sa') for r in successful_results 
                                 if r.get('videophy_sa') is not None]
            videophy_pc_scores = [r.get('videophy_pc') for r in successful_results 
                                 if r.get('videophy_pc') is not None]
            if videophy_sa_scores:
                avg_sa = sum(videophy_sa_scores) / len(videophy_sa_scores)
                avg_pc = sum(videophy_pc_scores) / len(videophy_pc_scores) if videophy_pc_scores else 0
                logger.info(f"\nVideoPhy Metrics (0-1 scale, higher is better):")
                logger.info(f"  SA Average: {avg_sa:.4f} (Min: {min(videophy_sa_scores):.4f}, Max: {max(videophy_sa_scores):.4f})")
                if videophy_pc_scores:
                    logger.info(f"  PC Average: {avg_pc:.4f} (Min: {min(videophy_pc_scores):.4f}, Max: {max(videophy_pc_scores):.4f})")
        
        # VideoPhy-2 statistics if computed
        if videophy2_evaluator:
            videophy2_sa_scores = [r.get('videophy2_sa') for r in successful_results 
                                  if r.get('videophy2_sa') is not None]
            videophy2_pc_scores = [r.get('videophy2_pc') for r in successful_results 
                                  if r.get('videophy2_pc') is not None]
            if videophy2_sa_scores:
                avg_sa = sum(videophy2_sa_scores) / len(videophy2_sa_scores)
                avg_pc = sum(videophy2_pc_scores) / len(videophy2_pc_scores) if videophy2_pc_scores else 0
                logger.info(f"\nVideoPhy-2 Metrics (1-5 scale, higher is better):")
                logger.info(f"  SA Average: {avg_sa:.2f} (Min: {min(videophy2_sa_scores)}, Max: {max(videophy2_sa_scores)})")
                if videophy2_pc_scores:
                    logger.info(f"  PC Average: {avg_pc:.2f} (Min: {min(videophy2_pc_scores)}, Max: {max(videophy2_pc_scores)})")
        
        logger.info("\nPer-video results:")
        for result in successful_results:
            info_str = f"  {os.path.basename(result['video_path'])}: " \
                      f"t-LPIPS={result['t_lpips']:.4f}, " \
                      f"Frames={result['num_frames']}"
            if not args.no_optical_flow and result.get('optical_flow') is not None:
                info_str += f", Optical Flow={result['optical_flow']:.4f}"
            if result.get('videophy_sa') is not None:
                info_str += f", VideoPhy SA={result['videophy_sa']:.4f}, PC={result['videophy_pc']:.4f}"
            if result.get('videophy2_sa') is not None:
                info_str += f", VideoPhy-2 SA={result['videophy2_sa']}, PC={result['videophy2_pc']}"
            logger.info(info_str)
    
    if failed_results:
        logger.warning(f"\nFailed to evaluate {len(failed_results)} video(s):")
        for result in failed_results:
            logger.warning(f"  {os.path.basename(result['video_path'])}: {result.get('error', 'Unknown error')}")
    
    # Save results to file if requested
    if args.output:
        import json
        output_path = os.path.join(project_root, args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

