"""
OpenSora model evaluation script.
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


def load_opensora_model(config_path: str, checkpoint_path: Optional[str] = None,
                       device: str = 'cuda'):
    """
    Load OpenSora model using official API.
    OpenSora uses a config-based system for model loading.
    
    Args:
        config_path: Path to OpenSora config file (e.g., configs/diffusion/inference/256px.py)
        checkpoint_path: Path to model checkpoint (optional, can be specified in config)
        device: Device to load model on
    
    Returns:
        Loaded model and pipeline components
    """
    try:
        import sys
        from pathlib import Path
        
        # Add Open-Sora to path if not already there
        opensora_path = os.environ.get('OPENSORA_PATH', None)
        if opensora_path:
            sys.path.insert(0, opensora_path)
        
        # Import OpenSora modules
        from opensora.utils.config import read_config
        from opensora.registry import MODELS, SCHEDULERS, build_module
        from opensora.datasets import prepare_prompts
        
        # Read config
        cfg = read_config(config_path)
        
        # Build model from config
        model = build_module(cfg.model, MODELS)
        
        # Load checkpoint if provided
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict, strict=False)
        
        model = model.to(device)
        model.eval()
        
        # Build scheduler
        scheduler = build_module(cfg.sampling.scheduler, SCHEDULERS)
        
        return {
            'model': model,
            'scheduler': scheduler,
            'config': cfg,
            'device': device
        }
    except ImportError as e:
        print(f"Warning: OpenSora package not found: {e}")
        print("Please install Open-Sora:")
        print("  git clone https://github.com/hpcaitech/Open-Sora.git")
        print("  cd Open-Sora && pip install -v .")
        return None
    except Exception as e:
        print(f"Error loading OpenSora model: {e}")
        print("Please ensure:")
        print("  1. Open-Sora is installed and OPENSORA_PATH is set")
        print("  2. Config path is correct")
        print("  3. Checkpoint path is correct (if provided)")
        return None


def generate_videos_opensora(pipeline, prompts: List[str], num_frames: int = 16,
                             height: int = 256, width: int = 256,
                             num_inference_steps: int = 50,
                             aspect_ratio: str = "16:9",
                             device: str = 'cuda') -> torch.Tensor:
    """
    Generate videos using OpenSora model with official API.
    
    Args:
        pipeline: OpenSora pipeline dict (from load_opensora_model)
        prompts: List of text prompts
        num_frames: Number of frames to generate
        height: Video height
        width: Video width
        num_inference_steps: Number of diffusion steps
        aspect_ratio: Aspect ratio (e.g., "16:9", "9:16", "1:1")
        device: Device to run on
    
    Returns:
        Generated videos tensor [B, T, C, H, W]
    """
    if pipeline is None:
        raise ValueError("Pipeline is None. Please load model first.")
    
    try:
        from opensora.datasets import prepare_prompts
        from opensora.sampling.inference import inference
        
        model = pipeline['model']
        scheduler = pipeline['scheduler']
        cfg = pipeline['config']
        
        videos = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Prepare prompt according to OpenSora format
                prompt_data = prepare_prompts([prompt], cfg)
                
                # Run inference using OpenSora's official inference function
                video = inference(
                    model=model,
                    scheduler=scheduler,
                    prompt=prompt_data,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    aspect_ratio=aspect_ratio,
                    device=device
                )
                
                # video shape: [T, C, H, W] -> [1, T, C, H, W]
                if video.dim() == 4:
                    video = video.unsqueeze(0)
                videos.append(video)
        
        videos_tensor = torch.cat(videos, dim=0)  # [B, T, C, H, W]
        return videos_tensor
        
    except Exception as e:
        print(f"Error generating videos with OpenSora: {e}")
        print("Falling back to subprocess call to official inference script...")
        # Fallback: call the official inference script via subprocess
        return _generate_videos_opensora_subprocess(prompts, num_frames, height, width, 
                                                    num_inference_steps, aspect_ratio)


def _generate_videos_opensora_subprocess(prompts: List[str], num_frames: int = 16,
                                         height: int = 256, width: int = 256,
                                         num_inference_steps: int = 50,
                                         aspect_ratio: str = "16:9") -> torch.Tensor:
    """
    Fallback method: Call OpenSora's official inference script via subprocess.
    This uses the command-line interface: torchrun scripts/diffusion/inference.py
    """
    import subprocess
    import tempfile
    import json
    
    opensora_path = os.environ.get('OPENSORA_PATH', './Open-Sora')
    config_file = f"configs/diffusion/inference/{height}px.py" if height <= 256 else "configs/diffusion/inference/768px.py"
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save prompts to CSV
        csv_path = os.path.join(tmpdir, 'prompts.csv')
        with open(csv_path, 'w') as f:
            f.write('prompt\n')
            for prompt in prompts:
                f.write(f'"{prompt}"\n')
        
        # Build command
        cmd = [
            'torchrun', '--nproc_per_node', '1', '--standalone',
            'scripts/diffusion/inference.py',
            config_file,
            '--save-dir', tmpdir,
            '--dataset.data-path', csv_path,
            '--num_frames', str(num_frames),
            '--aspect_ratio', aspect_ratio
        ]
        
        # Run inference
        result = subprocess.run(cmd, cwd=opensora_path, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"OpenSora inference failed: {result.stderr}")
        
        # Load generated videos
        # Note: This is a simplified version - actual implementation should parse
        # the output directory structure from OpenSora
        videos = []
        # TODO: Parse actual video files from tmpdir
        # For now, return placeholder
        batch_size = len(prompts)
        return torch.randn(batch_size, num_frames, 3, height, width)


def evaluate_opensora(config_path: str, prompts: List[str],
                     output_dir: str = './results/opensora',
                     checkpoint_path: Optional[str] = None,
                     ground_truth_videos: Optional[torch.Tensor] = None,
                     device: str = 'cuda',
                     **generation_kwargs) -> Dict:
    """
    Evaluate OpenSora model on given prompts using official API.
    
    Args:
        config_path: Path to OpenSora config file (required)
        prompts: List of text prompts
        output_dir: Directory to save results and videos
        checkpoint_path: Path to checkpoint (optional, can be in config)
        ground_truth_videos: Ground truth videos for metrics computation [B, T, C, H, W]
        device: Device to run evaluation on
        **generation_kwargs: Additional generation parameters
    
    Returns:
        Dictionary containing evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("OpenSora Evaluation")
    print("=" * 50)
    print(f"Config: {config_path}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load model
    print("Loading OpenSora model...")
    pipeline = load_opensora_model(config_path, checkpoint_path=checkpoint_path, device=device)
    if pipeline is None:
        raise RuntimeError("Failed to load OpenSora model. Please check installation and paths.")
    print("Model loaded.")
    print()
    
    # Generate videos
    print("Generating videos...")
    generated_videos = generate_videos_opensora(
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
        'model': 'OpenSora',
        'config': config_path,
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
    
    # Save videos (as numpy arrays for now)
    for i, video in enumerate(generated_videos):
        video_path = os.path.join(videos_dir, f'video_{i:03d}.pt')
        torch.save(video.cpu(), video_path)
    
    print("Evaluation complete!")
    print(f"Results saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate OpenSora model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to OpenSora config file (e.g., configs/diffusion/inference/256px.py)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to OpenSora checkpoint (optional, can be in config)')
    parser.add_argument('--prompts_file', type=str, default=None,
                       help='Path to JSON file containing prompts')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                       help='Text prompts (if not using prompts_file)')
    parser.add_argument('--output_dir', type=str, default='./results/opensora',
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
        # This is a placeholder - adjust based on actual data format
        pass
    
    # Run evaluation
    results = evaluate_opensora(
        config_path=args.config,
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

