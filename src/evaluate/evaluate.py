"""
VideoPhy and VideoPhy2 Evaluation Script

This script provides a unified interface for evaluating videos using:
- VideoPhy (VideoCon-Physics): Evaluates semantic adherence (SA) and physical commonsense (PC)
- VideoPhy2 (VideoPhy-2-AutoEval): Evaluates SA, PC, and physical rule adherence

Usage:
    python src/evaluate.py \
        --input_csv path/to/videos.csv \
        --checkpoint_videophy path/to/videocon_physics \
        --checkpoint_videophy2 path/to/videophy_2_auto \
        --output_dir results/ \
        --evaluator videophy  # or videophy2 or both

Input CSV format:
    videopath,caption
    path/to/video1.mp4,Description of video 1
    path/to/video2.mp4,Description of video 2

For VideoPhy2 rule evaluation, CSV should also include:
    videopath,caption,rule
    path/to/video1.mp4,Description,Physical rule to check
"""

import os
import sys
import argparse
import pandas as pd
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json

# Add VideoPhy paths to sys.path
VIDEOPHY_BASE = Path(__file__).parent / "evaluate" / "VideoPhy"
VIDEOPHY2_BASE = VIDEOPHY_BASE / "VIDEOPHY2"

def check_model_checkpoint(checkpoint_path: str, evaluator: str) -> bool:
    """Check if model checkpoint exists and is valid."""
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: {evaluator} checkpoint not found at: {checkpoint_path}")
        print(f"\nTo download the checkpoint:")
        if evaluator == "videophy":
            print("  git lfs install")
            print("  git clone https://huggingface.co/videophysics/videocon_physics")
        elif evaluator == "videophy2":
            print("  git lfs install")
            print("  git clone https://huggingface.co/videophysics/videophy_2_auto")
        return False
    
    # Check for required files
    required_files = ["config.json", "tokenizer.model"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(checkpoint_path, f))]
    if missing:
        print(f"WARNING: Missing files in checkpoint: {missing}")
        return False
    
    return True

def prepare_videophy_data(input_csv: str, output_dir: str) -> Tuple[str, str]:
    """
    Prepare data for VideoPhy evaluation.
    Returns paths to sa_testing.csv and physics_testing.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert relative video paths to absolute paths if needed
    df = pd.read_csv(input_csv)
    input_csv_dir = Path(input_csv).parent
    
    # Update videopaths to absolute if they're relative
    updated_paths = []
    for videopath in df['videopath']:
        videopath = Path(videopath)
        if not videopath.is_absolute():
            # Try relative to CSV directory first, then current working directory
            abs_path = input_csv_dir / videopath
            if not abs_path.exists():
                abs_path = Path.cwd() / videopath
            videopath = abs_path
        updated_paths.append(str(videopath.resolve()))
    
    df['videopath'] = updated_paths
    temp_csv = output_dir / "temp_input_absolute.csv"
    df.to_csv(temp_csv, index=False)
    
    # Use VideoPhy's prepare_data.py script
    prepare_script = VIDEOPHY_BASE / "utils" / "prepare_data.py"
    
    if not prepare_script.exists():
        raise FileNotFoundError(f"VideoPhy prepare_data.py not found at {prepare_script}")
    
    # Run the preparation script
    cmd = [
        sys.executable,
        str(prepare_script),
        "--input_csv", str(temp_csv),
        "--output_folder", str(output_dir)
    ]
    
    print(f"Preparing VideoPhy data...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(VIDEOPHY_BASE))
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to prepare VideoPhy data: {result.stderr}")
    
    sa_path = output_dir / "sa_testing.csv"
    physics_path = output_dir / "physics_testing.csv"
    
    if not sa_path.exists() or not physics_path.exists():
        raise RuntimeError("Failed to generate VideoPhy testing CSVs")
    
    return str(sa_path), str(physics_path)

def run_videophy_inference(
    input_csv: str,
    checkpoint: str,
    output_csv: str,
    batch_size: int = 16
) -> str:
    """Run VideoPhy inference for semantic adherence or physical commonsense."""
    inference_script = VIDEOPHY_BASE / "videocon" / "training" / "pipeline_video" / "entailment_inference.py"
    
    if not inference_script.exists():
        raise FileNotFoundError(f"VideoPhy inference script not found at {inference_script}")
    
    # Add VideoPhy to path for imports
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    script_dir = inference_script.parent
    if str(script_dir) not in pythonpath:
        env["PYTHONPATH"] = f"{script_dir}:{pythonpath}" if pythonpath else str(script_dir)
    
    cmd = [
        sys.executable,
        str(inference_script),
        "--input_csv", input_csv,
        "--output_csv", output_csv,
        "--checkpoint", checkpoint,
        "--batch_size", str(batch_size)
    ]
    
    print(f"Running VideoPhy inference...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(script_dir))
    
    if result.returncode != 0:
        print(f"ERROR: VideoPhy inference failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"VideoPhy inference failed: {result.stderr}")
    
    return output_csv

def run_videophy2_inference(
    input_csv: str,
    checkpoint: str,
    output_csv: str,
    task: str,  # 'sa', 'pc', or 'rule'
    batch_size: int = 1,
    num_frames: int = 32
) -> str:
    """Run VideoPhy2 inference for SA, PC, or rule evaluation."""
    inference_script = VIDEOPHY2_BASE / "inference.py"
    
    if not inference_script.exists():
        raise FileNotFoundError(f"VideoPhy2 inference script not found at {inference_script}")
    
    # Add VideoPhy2 to path for imports
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if str(VIDEOPHY2_BASE) not in pythonpath:
        env["PYTHONPATH"] = f"{VIDEOPHY2_BASE}:{pythonpath}" if pythonpath else str(VIDEOPHY2_BASE)
    
    cmd = [
        sys.executable,
        str(inference_script),
        "--input_csv", input_csv,
        "--checkpoint", checkpoint,
        "--output_csv", output_csv,
        "--task", task,
        "--batch_size", str(batch_size),
        "--num_frames", str(num_frames)
    ]
    
    print(f"Running VideoPhy2 inference for task: {task}...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"ERROR: VideoPhy2 inference failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"VideoPhy2 inference failed: {result.stderr}")
    
    return output_csv

def evaluate_videophy(
    input_csv: str,
    checkpoint: str,
    output_dir: str,
    batch_size: int = 16
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate videos using VideoPhy (VideoCon-Physics).
    Returns DataFrames with SA and PC scores.
    """
    print("\n" + "="*60)
    print("Evaluating with VideoPhy (VideoCon-Physics)")
    print("="*60)
    
    if not check_model_checkpoint(checkpoint, "videophy"):
        raise ValueError("Invalid VideoPhy checkpoint")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    sa_csv, physics_csv = prepare_videophy_data(input_csv, output_dir)
    
    # Run SA evaluation
    sa_output = output_dir / "videophy_sa_scores.csv"
    run_videophy_inference(sa_csv, checkpoint, str(sa_output), batch_size)
    
    # Run PC evaluation
    pc_output = output_dir / "videophy_pc_scores.csv"
    run_videophy_inference(physics_csv, checkpoint, str(pc_output), batch_size)
    
    # Load results
    sa_df = pd.read_csv(sa_output, header=None, names=["videopath", "caption", "sa_score"])
    pc_df = pd.read_csv(pc_output, header=None, names=["videopath", "caption", "pc_score"])
    
    # Merge results
    results_df = pd.merge(sa_df, pc_df, on=["videopath", "caption"], how="outer")
    
    # Calculate joint score (SA=1 and PC=1, where scores > threshold)
    # VideoPhy outputs entailment scores (0-1), typically >0.5 means positive
    threshold = 0.5
    results_df["sa_binary"] = (results_df["sa_score"] > threshold).astype(int)
    results_df["pc_binary"] = (results_df["pc_score"] > threshold).astype(int)
    results_df["joint_score"] = ((results_df["sa_binary"] == 1) & (results_df["pc_binary"] == 1)).astype(int)
    
    # Save merged results
    merged_output = output_dir / "videophy_results.csv"
    results_df.to_csv(merged_output, index=False)
    print(f"\nResults saved to: {merged_output}")
    
    # Print summary statistics
    print("\nVideoPhy Evaluation Summary:")
    print(f"  Total videos: {len(results_df)}")
    print(f"  SA > {threshold}: {results_df['sa_binary'].sum()} ({100*results_df['sa_binary'].mean():.1f}%)")
    print(f"  PC > {threshold}: {results_df['pc_binary'].sum()} ({100*results_df['pc_binary'].mean():.1f}%)")
    print(f"  Joint (SA & PC): {results_df['joint_score'].sum()} ({100*results_df['joint_score'].mean():.1f}%)")
    print(f"  Mean SA score: {results_df['sa_score'].mean():.3f}")
    print(f"  Mean PC score: {results_df['pc_score'].mean():.3f}")
    
    return {
        "sa": sa_df,
        "pc": pc_df,
        "merged": results_df
    }

def evaluate_videophy2(
    input_csv: str,
    checkpoint: str,
    output_dir: str,
    batch_size: int = 1,
    num_frames: int = 32,
    evaluate_rules: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate videos using VideoPhy2 (VideoPhy-2-AutoEval).
    Returns DataFrames with SA, PC, and optionally rule scores.
    """
    print("\n" + "="*60)
    print("Evaluating with VideoPhy2 (VideoPhy-2-AutoEval)")
    print("="*60)
    
    if not check_model_checkpoint(checkpoint, "videophy2"):
        raise ValueError("Invalid VideoPhy2 checkpoint")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    input_csv_dir = Path(input_csv).parent
    
    # Check required columns
    if "videopath" not in df.columns or "caption" not in df.columns:
        raise ValueError("Input CSV must contain 'videopath' and 'caption' columns")
    
    # Convert relative video paths to absolute paths if needed
    updated_paths = []
    for videopath in df['videopath']:
        videopath = Path(videopath)
        if not videopath.is_absolute():
            # Try relative to CSV directory first, then current working directory
            abs_path = input_csv_dir / videopath
            if not abs_path.exists():
                abs_path = Path.cwd() / videopath
            videopath = abs_path
        updated_paths.append(str(videopath.resolve()))
    
    df['videopath'] = updated_paths
    
    # Prepare SA/PC CSV (same format for both tasks)
    sa_pc_csv = output_dir / "videophy2_sa_pc_input.csv"
    df[["videopath", "caption"]].to_csv(sa_pc_csv, index=False)
    
    # Run SA evaluation
    sa_output = output_dir / "videophy2_sa_scores.csv"
    run_videophy2_inference(str(sa_pc_csv), checkpoint, str(sa_output), "sa", batch_size, num_frames)
    
    # Run PC evaluation
    pc_output = output_dir / "videophy2_pc_scores.csv"
    run_videophy2_inference(str(sa_pc_csv), checkpoint, str(pc_output), "pc", batch_size, num_frames)
    
    # Load results
    sa_df = pd.read_csv(sa_output)
    pc_df = pd.read_csv(pc_output)
    
    # Merge SA and PC results
    results_df = pd.merge(
        sa_df[["videopath", "score"]].rename(columns={"score": "sa_score"}),
        pc_df[["videopath", "score"]].rename(columns={"score": "pc_score"}),
        on="videopath",
        how="outer"
    )
    
    # Add captions back
    results_df = pd.merge(results_df, df[["videopath", "caption"]], on="videopath", how="left")
    
    # Calculate joint score (SA>=4 and PC>=4)
    results_df["sa_binary"] = (results_df["sa_score"] >= 4).astype(int)
    results_df["pc_binary"] = (results_df["pc_score"] >= 4).astype(int)
    results_df["joint_score"] = ((results_df["sa_binary"] == 1) & (results_df["pc_binary"] == 1)).astype(int)
    
    # Rule evaluation (if requested and rules are available)
    rule_df = None
    if evaluate_rules and "rule" in df.columns:
        rule_csv = output_dir / "videophy2_rule_input.csv"
        df[["videopath", "rule"]].to_csv(rule_csv, index=False)
        
        rule_output = output_dir / "videophy2_rule_scores.csv"
        run_videophy2_inference(str(rule_csv), checkpoint, str(rule_output), "rule", batch_size, num_frames)
        
        rule_df = pd.read_csv(rule_output)
        results_df = pd.merge(
            results_df,
            rule_df[["videopath", "score"]].rename(columns={"score": "rule_score"}),
            on="videopath",
            how="left"
        )
    
    # Save merged results
    merged_output = output_dir / "videophy2_results.csv"
    results_df.to_csv(merged_output, index=False)
    print(f"\nResults saved to: {merged_output}")
    
    # Print summary statistics
    print("\nVideoPhy2 Evaluation Summary:")
    print(f"  Total videos: {len(results_df)}")
    print(f"  SA >= 4: {results_df['sa_binary'].sum()} ({100*results_df['sa_binary'].mean():.1f}%)")
    print(f"  PC >= 4: {results_df['pc_binary'].sum()} ({100*results_df['pc_binary'].mean():.1f}%)")
    print(f"  Joint (SA>=4 & PC>=4): {results_df['joint_score'].sum()} ({100*results_df['joint_score'].mean():.1f}%)")
    print(f"  Mean SA score: {results_df['sa_score'].mean():.2f}")
    print(f"  Mean PC score: {results_df['pc_score'].mean():.2f}")
    
    if rule_df is not None:
        print(f"  Rule adherence: {len(rule_df[rule_df['score'] == 1])} ({100*len(rule_df[rule_df['score'] == 1])/len(rule_df):.1f}%)")
        print(f"  Rule violations: {len(rule_df[rule_df['score'] == 0])} ({100*len(rule_df[rule_df['score'] == 0])/len(rule_df):.1f}%)")
    
    return {
        "sa": sa_df,
        "pc": pc_df,
        "rule": rule_df,
        "merged": results_df
    }

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate videos using VideoPhy and/or VideoPhy2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV file with columns: videopath, caption (and optionally 'rule' for VideoPhy2)"
    )
    
    parser.add_argument(
        "--evaluator",
        type=str,
        choices=["videophy", "videophy2", "both"],
        default="both",
        help="Which evaluator to use: 'videophy', 'videophy2', or 'both'"
    )
    
    parser.add_argument(
        "--checkpoint_videophy",
        type=str,
        default=None,
        help="Path to VideoPhy (VideoCon-Physics) checkpoint directory"
    )
    
    parser.add_argument(
        "--checkpoint_videophy2",
        type=str,
        default=None,
        help="Path to VideoPhy2 (VideoPhy-2-AutoEval) checkpoint directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for VideoPhy (VideoPhy2 uses batch_size=1 by default)"
    )
    
    parser.add_argument(
        "--num_frames",
        type=int,
        default=32,
        help="Number of frames to use for VideoPhy2 evaluation"
    )
    
    parser.add_argument(
        "--evaluate_rules",
        action="store_true",
        help="Evaluate physical rules (VideoPhy2 only, requires 'rule' column in CSV)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    
    # Check evaluator requirements
    if args.evaluator in ["videophy", "both"]:
        if not args.checkpoint_videophy:
            raise ValueError("--checkpoint_videophy is required for VideoPhy evaluation")
    
    if args.evaluator in ["videophy2", "both"]:
        if not args.checkpoint_videophy2:
            raise ValueError("--checkpoint_videophy2 is required for VideoPhy2 evaluation")
    
    # Run evaluations
    results = {}
    
    if args.evaluator in ["videophy", "both"]:
        results["videophy"] = evaluate_videophy(
            args.input_csv,
            args.checkpoint_videophy,
            args.output_dir,
            args.batch_size
        )
    
    if args.evaluator in ["videophy2", "both"]:
        results["videophy2"] = evaluate_videophy2(
            args.input_csv,
            args.checkpoint_videophy2,
            args.output_dir,
            batch_size=1,  # VideoPhy2 typically uses batch_size=1
            num_frames=args.num_frames,
            evaluate_rules=args.evaluate_rules
        )
    
    # Save summary
    summary = {
        "input_csv": args.input_csv,
        "evaluator": args.evaluator,
        "output_dir": args.output_dir
    }
    
    summary_path = Path(args.output_dir) / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

