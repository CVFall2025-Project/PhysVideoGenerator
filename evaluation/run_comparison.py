"""
Unified comparison runner for all video generation models.
Runs evaluation on all models and generates comparison report.
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.opensora_eval import evaluate_opensora
from evaluation.videocrafter2_eval import evaluate_videocrafter2
from evaluation.hunyuanvideo_eval import evaluate_hunyuanvideo
from evaluation_utils import load_test_prompts


def run_all_evaluations(prompts: List[str],
                       opensora_config: Optional[str] = None,
                       opensora_checkpoint: Optional[str] = None,
                       videocrafter2_checkpoint: Optional[str] = None,
                       hunyuanvideo_checkpoint: Optional[str] = None,
                       ground_truth_videos: Optional = None,
                       output_base_dir: str = './results/comparison',
                       device: str = 'cuda',
                       **generation_kwargs) -> Dict:
    """
    Run evaluation on all available models.
    
    Args:
        prompts: List of text prompts
        opensora_checkpoint: Path to OpenSora checkpoint (None to skip)
        videocrafter2_checkpoint: Path to VideoCrafter2 checkpoint (None to skip)
        hunyuanvideo_checkpoint: Path to HunyuanVideo checkpoint (None to skip)
        ground_truth_videos: Ground truth videos tensor [B, T, C, H, W]
        output_base_dir: Base directory for all results
        device: Device to run on
        **generation_kwargs: Additional generation parameters
    
    Returns:
        Dictionary containing all evaluation results
    """
    os.makedirs(output_base_dir, exist_ok=True)
    
    all_results = {
        'prompts': prompts,
        'models': {}
    }
    
    # Run OpenSora evaluation
    if opensora_config:
        print("\n" + "=" * 70)
        print("Running OpenSora Evaluation")
        print("=" * 70)
        try:
            opensora_results = evaluate_opensora(
                config_path=opensora_config,
                checkpoint_path=opensora_checkpoint,
                prompts=prompts,
                output_dir=os.path.join(output_base_dir, 'opensora'),
                ground_truth_videos=ground_truth_videos,
                device=device,
                **generation_kwargs
            )
            all_results['models']['OpenSora'] = opensora_results
        except Exception as e:
            print(f"Error running OpenSora evaluation: {e}")
            all_results['models']['OpenSora'] = {'error': str(e)}
    else:
        print("Skipping OpenSora (config not provided)")
    
    # Run VideoCrafter2 evaluation
    if videocrafter2_checkpoint:
        print("\n" + "=" * 70)
        print("Running VideoCrafter2 Evaluation")
        print("=" * 70)
        try:
            videocrafter2_results = evaluate_videocrafter2(
                checkpoint_path=videocrafter2_checkpoint,
                prompts=prompts,
                output_dir=os.path.join(output_base_dir, 'videocrafter2'),
                ground_truth_videos=ground_truth_videos,
                device=device,
                **generation_kwargs
            )
            all_results['models']['VideoCrafter2'] = videocrafter2_results
        except Exception as e:
            print(f"Error running VideoCrafter2 evaluation: {e}")
            all_results['models']['VideoCrafter2'] = {'error': str(e)}
    else:
        print("Skipping VideoCrafter2 (checkpoint not provided)")
    
    # Run HunyuanVideo evaluation
    if hunyuanvideo_checkpoint:
        print("\n" + "=" * 70)
        print("Running HunyuanVideo Evaluation")
        print("=" * 70)
        try:
            hunyuanvideo_results = evaluate_hunyuanvideo(
                checkpoint_path=hunyuanvideo_checkpoint,
                prompts=prompts,
                output_dir=os.path.join(output_base_dir, 'hunyuanvideo'),
                ground_truth_videos=ground_truth_videos,
                device=device,
                **generation_kwargs
            )
            all_results['models']['HunyuanVideo'] = hunyuanvideo_results
        except Exception as e:
            print(f"Error running HunyuanVideo evaluation: {e}")
            all_results['models']['HunyuanVideo'] = {'error': str(e)}
    else:
        print("Skipping HunyuanVideo (checkpoint not provided)")
    
    return all_results


def generate_comparison_report(all_results: Dict, output_path: str):
    """
    Generate comparison report from all evaluation results.
    
    Args:
        all_results: Dictionary containing all model results
        output_path: Path to save comparison report
    """
    # Extract metrics for comparison
    comparison_data = []
    
    for model_name, model_results in all_results['models'].items():
        if 'error' in model_results:
            continue
        
        metrics = model_results.get('metrics', {})
        row = {'Model': model_name}
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                row[f'{metric_name}_mean'] = metric_value.get('mean', None)
                row[f'{metric_name}_std'] = metric_value.get('std', None)
            else:
                row[metric_name] = metric_value
        
        comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save as CSV
    csv_path = output_path.replace('.json', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nComparison report (CSV) saved to: {csv_path}")
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Comparison report (JSON) saved to: {output_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Run comparison evaluation on all video generation models'
    )
    parser.add_argument('--prompts_file', type=str, default=None,
                       help='Path to JSON file containing prompts')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                       help='Text prompts (if not using prompts_file)')
    parser.add_argument('--opensora_config', type=str, default=None,
                       help='Path to OpenSora config file (e.g., configs/diffusion/inference/256px.py)')
    parser.add_argument('--opensora_checkpoint', type=str, default=None,
                       help='Path to OpenSora checkpoint (optional)')
    parser.add_argument('--videocrafter2_checkpoint', type=str, default=None,
                       help='Path to VideoCrafter2 checkpoint')
    parser.add_argument('--hunyuanvideo_checkpoint', type=str, default=None,
                       help='Path to HunyuanVideo checkpoint')
    parser.add_argument('--ground_truth_dir', type=str, default=None,
                       help='Directory containing ground truth videos')
    parser.add_argument('--output_dir', type=str, default='./results/comparison',
                       help='Output directory for comparison results')
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
    
    # Check that at least one model is provided
    if not any([args.opensora_config, args.videocrafter2_checkpoint, 
                args.hunyuanvideo_checkpoint]):
        print("Warning: No models provided. Please provide at least one model.")
        print("Usage: python run_comparison.py --opensora_config <path> ...")
        return
    
    # Run all evaluations
    print("\n" + "=" * 70)
    print("Starting Comparison Evaluation")
    print("=" * 70)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    all_results = run_all_evaluations(
        prompts=prompts,
        opensora_config=args.opensora_config,
        opensora_checkpoint=args.opensora_checkpoint,
        videocrafter2_checkpoint=args.videocrafter2_checkpoint,
        hunyuanvideo_checkpoint=args.hunyuanvideo_checkpoint,
        ground_truth_videos=ground_truth_videos,
        output_base_dir=args.output_dir,
        device=args.device,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width
    )
    
    # Generate comparison report
    report_path = os.path.join(args.output_dir, 'comparison_report.json')
    comparison_df = generate_comparison_report(all_results, report_path)
    
    print("\nComparison evaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

