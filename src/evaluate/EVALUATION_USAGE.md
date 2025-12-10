# VideoPhy and VideoPhy2 Evaluation Usage Guide

## Overview

The `evaluate.py` script provides a unified interface for evaluating videos using:
- **VideoPhy (VideoCon-Physics)**: Evaluates semantic adherence (SA) and physical commonsense (PC)
- **VideoPhy2 (VideoPhy-2-AutoEval)**: Evaluates SA, PC, and physical rule adherence

## Prerequisites

1. **Download Model Checkpoints**:

   For VideoPhy:
   ```bash
   git lfs install
   git clone https://huggingface.co/videophysics/videocon_physics
   ```

   For VideoPhy2:
   ```bash
   git lfs install
   git clone https://huggingface.co/videophysics/videophy_2_auto
   ```

2. **Install Dependencies**:
   ```bash
   pip install pandas  # If not already installed
   ```

## Input CSV Format

### Basic Format (for SA and PC evaluation):
```csv
videopath,caption
path/to/video1.mp4,A wooden spoon stirs the hot soup in the pot.
path/to/video2.mp4,An apple falls and bounces on the hard ground.
```

### With Rules (for VideoPhy2 rule evaluation):
```csv
videopath,caption,rule
path/to/video1.mp4,An apple falls,Objects fall due to gravity
path/to/video2.mp4,Water pours,Conservation of mass
```

**Note**: Video paths can be relative (to the CSV file location) or absolute.

## Usage Examples

### 1. Evaluate with VideoPhy only:
```bash
python src/evaluate.py \
    --input_csv src/evaluate/VideoPhy/examples/example.csv \
    --checkpoint_videophy /path/to/videocon_physics \
    --output_dir results/videophy \
    --evaluator videophy
```

### 2. Evaluate with VideoPhy2 only:
```bash
python src/evaluate.py \
    --input_csv src/evaluate/VideoPhy/examples/example.csv \
    --checkpoint_videophy2 /path/to/videophy_2_auto \
    --output_dir results/videophy2 \
    --evaluator videophy2
```

### 3. Evaluate with both (recommended):
```bash
python src/evaluate.py \
    --input_csv src/evaluate/VideoPhy/examples/example.csv \
    --checkpoint_videophy /path/to/videocon_physics \
    --checkpoint_videophy2 /path/to/videophy_2_auto \
    --output_dir results/both \
    --evaluator both
```

### 4. Evaluate with VideoPhy2 including rule evaluation:
```bash
python src/evaluate.py \
    --input_csv path/to/videos_with_rules.csv \
    --checkpoint_videophy2 /path/to/videophy_2_auto \
    --output_dir results/videophy2_rules \
    --evaluator videophy2 \
    --evaluate_rules
```

## Output

The script generates:

1. **For VideoPhy**:
   - `videophy_sa_scores.csv`: Semantic adherence scores
   - `videophy_pc_scores.csv`: Physical commonsense scores
   - `videophy_results.csv`: Merged results with joint scores

2. **For VideoPhy2**:
   - `videophy2_sa_scores.csv`: Semantic adherence scores (1-5 scale)
   - `videophy2_pc_scores.csv`: Physical commonsense scores (1-5 scale)
   - `videophy2_rule_scores.csv`: Rule adherence scores (if `--evaluate_rules` is used)
   - `videophy2_results.csv`: Merged results with joint scores

3. **Summary**:
   - `evaluation_summary.json`: Summary of evaluation run

## Score Interpretation

### VideoPhy:
- **SA/PC Scores**: Entailment scores (0-1), typically >0.5 means positive
- **Joint Score**: Both SA > 0.5 AND PC > 0.5

### VideoPhy2:
- **SA/PC Scores**: Ratings from 1-5 (higher is better)
- **Joint Score**: Both SA >= 4 AND PC >= 4
- **Rule Scores**: 
  - 0 = violation
  - 1 = adherence
  - 2 = rule cannot be grounded in the video

## Troubleshooting

1. **Checkpoint not found**: Ensure you've downloaded the model checkpoints and provided the correct path
2. **Video paths not found**: Make sure video paths in CSV are correct (relative to CSV location or absolute)
3. **CUDA out of memory**: Reduce `--batch_size` for VideoPhy or ensure GPU has enough memory
4. **Import errors**: Make sure you're running from the project root and all dependencies are installed

## Example with Provided Data

To test with the example videos in the repository:

```bash
python src/evaluate.py \
    --input_csv src/evaluate/VideoPhy/examples/example.csv \
    --checkpoint_videophy /path/to/videocon_physics \
    --checkpoint_videophy2 /path/to/videophy_2_auto \
    --output_dir results/example_eval \
    --evaluator both
```

Note: You'll need to download the model checkpoints first (see Prerequisites).

