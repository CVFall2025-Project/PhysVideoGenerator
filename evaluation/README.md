# Comparison Experiments Evaluation

This directory contains evaluation scripts for comparing PhysVideoGenerator with other video generation models: OpenSora, VideoCrafter2, and HunyuanVideo.

## Structure

- `opensora_eval.py` - Evaluation script for OpenSora model
- `videocrafter2_eval.py` - Evaluation script for VideoCrafter2 model
- `hunyuanvideo_eval.py` - Evaluation script for HunyuanVideo model
- `run_comparison.py` - Unified script to run all model evaluations and generate comparison report

## Usage

### Individual Model Evaluation

#### OpenSora
```bash
# OpenSora uses config-based loading
export OPENSORA_PATH=/path/to/Open-Sora
python evaluation/opensora_eval.py \
    --config configs/diffusion/inference/256px.py \
    --checkpoint /path/to/opensora/checkpoint \
    --prompts_file prompts.json \
    --output_dir ./results/opensora \
    --num_frames 16 \
    --height 256 \
    --width 256
```

#### VideoCrafter2
```bash
# Set VideoCrafter2 repository path
export VIDEOCRAFTER_PATH=/path/to/VideoCrafter
python evaluation/videocrafter2_eval.py \
    --checkpoint /path/to/videocrafter2/checkpoint \
    --prompts_file prompts.json \
    --output_dir ./results/videocrafter2 \
    --num_frames 16 \
    --height 256 \
    --width 256
```

#### HunyuanVideo
```bash
# Set HunyuanVideo repository path
export HUNYUANVIDEO_PATH=/path/to/HunyuanVideo
python evaluation/hunyuanvideo_eval.py \
    --checkpoint /path/to/hunyuanvideo/checkpoint \
    --prompts_file prompts.json \
    --output_dir ./results/hunyuanvideo \
    --num_frames 16 \
    --height 256 \
    --width 256
```

### Run All Comparisons

```bash
# Set environment variables for each model
export OPENSORA_PATH=/path/to/Open-Sora
export VIDEOCRAFTER_PATH=/path/to/VideoCrafter
export HUNYUANVIDEO_PATH=/path/to/HunyuanVideo

python evaluation/run_comparison.py \
    --opensora_config configs/diffusion/inference/256px.py \
    --opensora_checkpoint /path/to/opensora/checkpoint \
    --videocrafter2_checkpoint /path/to/videocrafter2/checkpoint \
    --hunyuanvideo_checkpoint /path/to/hunyuanvideo/checkpoint \
    --prompts_file prompts.json \
    --output_dir ./results/comparison \
    --num_frames 16 \
    --height 256 \
    --width 256
```

## Evaluation Metrics

The evaluation scripts compute the following metrics:

- **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel-level similarity
- **SSIM** (Structural Similarity Index): Measures structural similarity
- **FVD** (Frechet Video Distance): Measures video quality and realism (requires ground truth)
- **CLIP Score**: Measures text-video alignment

## Output Format

Each evaluation generates:
- `evaluation_results.json` - Detailed metrics and results
- `videos/` - Directory containing generated videos
- `comparison_report.json` - Comparison across all models (when using run_comparison.py)
- `comparison_report.csv` - CSV format comparison table

## Prompts Format

Create a JSON file with prompts:
```json
[
    "A beautiful sunset over the ocean",
    "A cat playing with a ball of yarn",
    "A car driving on a highway"
]
```

Or use command-line arguments:
```bash
--prompts "Prompt 1" "Prompt 2" "Prompt 3"
```

## Dependencies

Install comparison-specific dependencies:
```bash
pip install -r requirements_comparison.txt
```

Note: Model-specific packages (OpenSora, VideoCrafter2, HunyuanVideo) need to be installed separately from their respective repositories.

## Environment Setup

Each model requires its repository to be cloned and the path set as an environment variable:

```bash
# OpenSora
export OPENSORA_PATH=/path/to/Open-Sora

# VideoCrafter2  
export VIDEOCRAFTER_PATH=/path/to/VideoCrafter

# HunyuanVideo
export HUNYUANVIDEO_PATH=/path/to/HunyuanVideo
```

## Notes

- All models should be evaluated on the same test prompts for fair comparison
- Ground truth videos are optional but required for PSNR, SSIM, and FVD metrics
- CLIP score can be computed without ground truth videos
- Scripts use official APIs with fallback mechanisms for subprocess calls
- OpenSora requires a config file (--config) in addition to checkpoint
- See `API_UPDATE_NOTES.md` for detailed API implementation information

