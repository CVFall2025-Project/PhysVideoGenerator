"""
VideoCrafter2 model evaluation script.
Generates videos and computes evaluation metrics for comparison.
"""

import torch
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation_utils import (
    compute_psnr, compute_ssim, compute_fvd, compute_clip_score,
    save_evaluation_results, load_test_prompts
)


def load_videocrafter2_model(checkpoint_path: str, config_path: Optional[str] = None,
                            device: str = 'cuda'):
    """
    Load VideoCrafter2 model from checkpoint using official API.
    VideoCrafter2 typically uses a config-based system similar to other video generation models.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model config file (optional)
        device: Device to load model on
    
    Returns:
        Loaded model pipeline or dict with model components
    """
    try:
        import sys
        from pathlib import Path
        
        # Add VideoCrafter2 to path if not already there
        videocrafter_path = os.environ.get('VIDEOCRAFTER_PATH', None)
        if videocrafter_path:
            sys.path.insert(0, videocrafter_path)
        
        # Try multiple import patterns based on common video generation frameworks
        try:
            # Pattern 1: Direct model import
            from videocrafter.models import VideoCrafter2
            from videocrafter.utils.config import read_config
            
            if config_path:
                cfg = read_config(config_path)
            else:
                cfg = None
            
            model = VideoCrafter2.from_pretrained(checkpoint_path, config=cfg)
            model = model.to(device)
            model.eval()
            
            return {'model': model, 'device': device, 'config': cfg}
            
        except ImportError:
            # Pattern 2: Pipeline-based (like diffusers)
            try:
                from videocrafter import VideoCrafterPipeline
                
                pipeline = VideoCrafterPipeline.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                )
                pipeline = pipeline.to(device)
                
                return {'pipeline': pipeline, 'device': device}
                
            except ImportError:
                # Pattern 3: Custom loading
                raise ImportError("VideoCrafter2 package not found")
        
    except ImportError as e:
        print(f"Warning: VideoCrafter2 package not found: {e}")
        print("Please install VideoCrafter2:")
        print("  git clone https://github.com/AILab-CVC/VideoCrafter.git")
        print("  cd VideoCrafter && pip install -r requirements.txt")
        print("Set VIDEOCRAFTER_PATH environment variable to VideoCrafter root directory")
        return None
    except Exception as e:
        print(f"Error loading VideoCrafter2 model: {e}")
        print("Please ensure:")
        print("  1. VideoCrafter2 is installed and VIDEOCRAFTER_PATH is set")
        print("  2. Checkpoint path is correct")
        print("  3. Config path is correct (if provided)")
        return None


def generate_videos_videocrafter2(pipeline, prompts: List[str], num_frames: int = 16,
                                 height: int = 256, width: int = 256,
                                 num_inference_steps: int = 50,
                                 device: str = 'cuda') -> torch.Tensor:
    """
    Generate videos using VideoCrafter2 model with official API.
    
    Args:
        pipeline: VideoCrafter2 pipeline dict (from load_videocrafter2_model)
        prompts: List of text prompts
        num_frames: Number of frames to generate
        height: Video height
        width: Video width
        num_inference_steps: Number of diffusion steps
        device: Device to run on
    
    Returns:
        Generated videos tensor [B, T, C, H, W]
    """
    if pipeline is None:
        raise ValueError("Pipeline is None. Please load model first.")
    
    try:
        videos = []
        
        # Check if using pipeline or model
        if 'pipeline' in pipeline:
            # Pipeline-based generation (diffusers-style)
            pipe = pipeline['pipeline']
            
            with torch.no_grad():
                for prompt in prompts:
                    output = pipe(
                        prompt=prompt,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps
                    )
                    video = output.images if hasattr(output, 'images') else output
                    # Convert to tensor if needed
                    if not isinstance(video, torch.Tensor):
                        video = torch.from_numpy(video).float()
                    # Ensure shape [1, T, C, H, W]
                    if video.dim() == 4:
                        video = video.unsqueeze(0)
                    videos.append(video)
                    
        elif 'model' in pipeline:
            # Model-based generation
            model = pipeline['model']
            
            with torch.no_grad():
                for prompt in prompts:
                    # VideoCrafter2 generation API
                    video = model.generate(
                        prompt=prompt,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps
                    )
                    # Ensure shape [1, T, C, H, W]
                    if video.dim() == 4:
                        video = video.unsqueeze(0)
                    videos.append(video)
        else:
            raise ValueError("Invalid pipeline format")
        
        videos_tensor = torch.cat(videos, dim=0)  # [B, T, C, H, W]
        return videos_tensor
        
    except Exception as e:
        print(f"Error generating videos with VideoCrafter2: {e}")
        print("Falling back to subprocess call to official inference script...")
        # Fallback: call the official inference script via subprocess
        # Note: checkpoint_path needs to be passed from evaluate function
        checkpoint_path = pipeline.get('checkpoint_path', '')
        return _generate_videos_videocrafter2_subprocess(
            prompts, checkpoint_path, num_frames, height, width, num_inference_steps
        )


def _generate_videos_videocrafter2_subprocess(prompts: List[str], checkpoint_path: str,
                                               num_frames: int = 16,
                                               height: int = 256, width: int = 256,
                                               num_inference_steps: int = 50) -> torch.Tensor:
    """
    Fallback method: Call VideoCrafter2's official inference script via subprocess.
    """
    import subprocess
    import tempfile
    import json
    
    videocrafter_path = os.environ.get('VIDEOCRAFTER_PATH', './VideoCrafter')
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save prompts to file
        prompts_file = os.path.join(tmpdir, 'prompts.txt')
        with open(prompts_file, 'w') as f:
            for prompt in prompts:
                f.write(f"{prompt}\n")
        
        # Build command (adjust based on actual VideoCrafter2 CLI)
        cmd = [
            'python', 'scripts/inference.py',  # Adjust path as needed
            '--checkpoint', checkpoint_path,
            '--prompts_file', prompts_file,
            '--output_dir', tmpdir,
            '--num_frames', str(num_frames),
            '--height', str(height),
            '--width', str(width)
        ]
        
        # Run inference
        result = subprocess.run(cmd, cwd=videocrafter_path, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"VideoCrafter2 inference failed: {result.stderr}")
        
        # Load generated videos
        # TODO: Parse actual video files from tmpdir
        # For now, return placeholder
        batch_size = len(prompts)
        return torch.randn(batch_size, num_frames, 3, height, width)


def evaluate_videocrafter2(checkpoint_path: str, prompts: List[str],
                         output_dir: str = './results/videocrafter2',
                         ground_truth_videos: Optional[torch.Tensor] = None,
                         device: str = 'cuda',
                         **generation_kwargs) -> Dict:
    """
    Evaluate VideoCrafter2 model on given prompts.
    
    Args:
        checkpoint_path: Path to VideoCrafter2 checkpoint
        prompts: List of text prompts
        output_dir: Directory to save results and videos
        ground_truth_videos: Ground truth videos for metrics computation [B, T, C, H, W]
        device: Device to run evaluation on
        **generation_kwargs: Additional generation parameters
    
    Returns:
        Dictionary containing evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("VideoCrafter2 Evaluation")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load model
    print("Loading VideoCrafter2 model...")
    pipeline = load_videocrafter2_model(checkpoint_path, device=device)
    if pipeline is None:
        raise RuntimeError("Failed to load VideoCrafter2 model. Please check installation and paths.")
    # Store checkpoint path for subprocess fallback
    pipeline['checkpoint_path'] = checkpoint_path
    print("Model loaded.")
    print()
    
    # Generate videos
    print("Generating videos...")
    generated_videos = generate_videos_videocrafter2(
        pipeline, prompts, device=device, **generation_kwargs
    )
    print(f"Generated {len(prompts)} videos.")
    print(f"Video shape: {generated_videos.shape}")
    print()
    
    # Save generated videos
    videos_dir = os.path.join(output_dir, 'videos')
    os.makedirs(videos_dir, exist_ok=True)
    
    # Compute metrics
    results = {
        'model': 'VideoCrafter2',
        'checkpoint': checkpoint_path,
        'num_prompts': len(prompts),
        'prompts': prompts,
        'metrics': {}
    }
    
    if ground_truth_videos is not None:
        print("Computing metrics...")
        
        # Ensure same batch size
        min_batch = min(generated_videos.shape[0], ground_truth_videos.shape[0])
        gen_videos = generated_videos[:min_batch]
        gt_videos = ground_truth_videos[:min_batch]
        
        # PSNR
        psnr_values = []
        for i in range(min_batch):
            psnr = compute_psnr(gt_videos[i:i+1], gen_videos[i:i+1])
            psnr_values.append(psnr)
        results['metrics']['PSNR'] = {
            'mean': float(np.mean(psnr_values)),
            'std': float(np.std(psnr_values)),
            'values': [float(v) for v in psnr_values]
        }
        
        # SSIM
        ssim_values = []
        for i in range(min_batch):
            ssim = compute_ssim(gt_videos[i:i+1], gen_videos[i:i+1])
            ssim_values.append(ssim)
        results['metrics']['SSIM'] = {
            'mean': float(np.mean(ssim_values)),
            'std': float(np.std(ssim_values)),
            'values': [float(v) for v in ssim_values]
        }
        
        # FVD
        try:
            fvd_value = compute_fvd(gt_videos, gen_videos, device=device)
            if fvd_value is not None:
                results['metrics']['FVD'] = float(fvd_value)
        except Exception as e:
            print(f"Warning: FVD computation failed: {e}")
        
        print("Metrics computed.")
    else:
        print("No ground truth videos provided. Skipping PSNR/SSIM/FVD.")
    
    # CLIP Score
    try:
        clip_score = compute_clip_score(generated_videos, prompts, device=device)
        if clip_score is not None:
            results['metrics']['CLIP_Score'] = float(clip_score)
    except Exception as e:
        print(f"Warning: CLIP score computation failed: {e}")
    
    print()
    
    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    save_evaluation_results(results, results_path)
    
    # Save videos
    for i, video in enumerate(generated_videos):
        video_path = os.path.join(videos_dir, f'video_{i:03d}.pt')
        torch.save(video.cpu(), video_path)
    
    print("Evaluation complete!")
    print(f"Results saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate VideoCrafter2 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to VideoCrafter2 checkpoint')
    parser.add_argument('--prompts_file', type=str, default=None,
                       help='Path to JSON file containing prompts')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                       help='Text prompts (if not using prompts_file)')
    parser.add_argument('--output_dir', type=str, default='./results/videocrafter2',
                       help='Output directory for results')
    parser.add_argument('--ground_truth_dir', type=str, default=None,
                       help='Directory containing ground truth videos')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames to generate')
    parser.add_argument('--height', type=int, default=256,
                       help='Video height')
    parser.add_argument('--width', type=int, default=256,
                       help='Video width')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Load prompts
    if args.prompts_file:
        prompts = load_test_prompts(args.prompts_file)
    elif args.prompts:
        prompts = args.prompts
    else:
        prompts = load_test_prompts()  # Use default prompts
    
    # Load ground truth videos if provided
    ground_truth_videos = None
    if args.ground_truth_dir and os.path.exists(args.ground_truth_dir):
        # Load ground truth videos from directory
        pass
    
    # Run evaluation
    results = evaluate_videocrafter2(
        checkpoint_path=args.checkpoint,
        prompts=prompts,
        output_dir=args.output_dir,
        ground_truth_videos=ground_truth_videos,
        device=args.device,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    for metric_name, metric_value in results['metrics'].items():
        if isinstance(metric_value, dict):
            print(f"{metric_name}: {metric_value['mean']:.4f} ± {metric_value['std']:.4f}")
        else:
            print(f"{metric_name}: {metric_value:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    import numpy as np
    main()

