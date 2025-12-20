# API Update Notes - Using Official APIs

This document describes the updates made to use official APIs instead of placeholders.

## OpenSora

### Updated Implementation
- **Config-based loading**: OpenSora uses a configuration file system
- **Official inference API**: Uses `opensora.utils.config.read_config` and `opensora.registry.build_module`
- **Inference function**: Uses `opensora.sampling.inference.inference` for video generation
- **Fallback**: Includes subprocess fallback to call official `scripts/diffusion/inference.py`

### Usage
```python
from evaluation.opensora_eval import evaluate_opensora

results = evaluate_opensora(
    config_path="configs/diffusion/inference/256px.py",
    checkpoint_path="path/to/checkpoint.pth",  # Optional
    prompts=["A beautiful sunset"],
    output_dir="./results/opensora"
)
```

### Command Line
```bash
python evaluation/opensora_eval.py \
    --config configs/diffusion/inference/256px.py \
    --checkpoint path/to/checkpoint.pth \
    --prompts "A beautiful sunset" \
    --output_dir ./results/opensora
```

### Requirements
- Open-Sora repository cloned and installed
- Set `OPENSORA_PATH` environment variable to Open-Sora root directory
- Config files available in `configs/diffusion/inference/`

## VideoCrafter2

### Updated Implementation
- **Multiple import patterns**: Tries direct model import, pipeline-based, and custom loading
- **Flexible API support**: Supports both pipeline and model-based generation
- **Subprocess fallback**: Includes fallback to call official inference script
- **Environment variable**: Uses `VIDEOCRAFTER_PATH` to locate repository

### Usage
```python
from evaluation.videocrafter2_eval import evaluate_videocrafter2

results = evaluate_videocrafter2(
    checkpoint_path="path/to/checkpoint.pth",
    prompts=["A beautiful sunset"],
    output_dir="./results/videocrafter2"
)
```

### Command Line
```bash
python evaluation/videocrafter2_eval.py \
    --checkpoint path/to/checkpoint.pth \
    --prompts "A beautiful sunset" \
    --output_dir ./results/videocrafter2
```

### Requirements
- VideoCrafter2 repository cloned
- Set `VIDEOCRAFTER_PATH` environment variable to VideoCrafter2 root directory
- Checkpoint file available

## HunyuanVideo

### Updated Implementation
- **Multiple import patterns**: Tries direct model import, pipeline-based, and config-based loading
- **Flexible API support**: Supports both pipeline and model-based generation
- **Subprocess fallback**: Includes fallback to call official inference script
- **Environment variable**: Uses `HUNYUANVIDEO_PATH` to locate repository

### Usage
```python
from evaluation.hunyuanvideo_eval import evaluate_hunyuanvideo

results = evaluate_hunyuanvideo(
    checkpoint_path="path/to/checkpoint.pth",
    prompts=["A beautiful sunset"],
    output_dir="./results/hunyuanvideo"
)
```

### Command Line
```bash
python evaluation/hunyuanvideo_eval.py \
    --checkpoint path/to/checkpoint.pth \
    --prompts "A beautiful sunset" \
    --output_dir ./results/hunyuanvideo
```

### Requirements
- HunyuanVideo repository cloned
- Set `HUNYUANVIDEO_PATH` environment variable to HunyuanVideo root directory
- Checkpoint file available

## Implementation Status

✅ **OpenSora**: Fully updated with official API
✅ **VideoCrafter2**: Updated with realistic API patterns and fallback mechanisms
✅ **HunyuanVideo**: Updated with realistic API patterns and fallback mechanisms

## Next Steps

1. **Test with Actual Models**:
   - Install each model repository
   - Set environment variables (OPENSORA_PATH, VIDEOCRAFTER_PATH, HUNYUANVIDEO_PATH)
   - Test model loading with actual checkpoints
   - Verify video generation works correctly

2. **Fine-tune API Calls**:
   - Adjust import paths based on actual repository structure
   - Update subprocess commands to match actual CLI interfaces
   - Test fallback mechanisms

3. **Complete Subprocess Parsing**:
   - Implement video file parsing in subprocess fallback functions
   - Handle different output formats from each model
   - Ensure consistent tensor format [B, T, C, H, W]

4. **Update Documentation**:
   - Add troubleshooting section for common issues
   - Document environment setup for each model
   - Add examples for different use cases

## Testing

After updating each model's API:

1. Test model loading with actual checkpoint
2. Test video generation with sample prompts
3. Verify output tensor format: [B, T, C, H, W]
4. Test evaluation metrics computation
5. Test comparison runner with all models

