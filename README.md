# PhysVideoGenerator

A physics-aware video generation framework leveraging Latte Transformer with physics-informed training.

## Overview

PhysVideoGenerator is a research framework for generating physics-aware videos from text descriptions. The system extends the Latte (Latent Diffusion Transformer) architecture with physics-aware modifications to ensure generated videos follow physical laws and dynamics.

### Key Components

- **Latte Transformer**: 3D Diffusion Transformer architecture for video generation
- **Physics-Aware Training**: Custom training objectives that enforce physical constraints
- **VAE Encoder/Decoder**: Efficient video compression and reconstruction
- **T5-XXL Text Encoder**: High-quality text-to-video conditioning
- **Evaluation Metrics**: Comprehensive video quality assessment (FID, SSIM, PSNR, LPIPS)

## Features

- **Physics-Informed Generation**: Videos that respect physical constraints and dynamics
- **Text-to-Video Synthesis**: Generate videos from natural language descriptions
- **Flexible Architecture**: Modular design with separate encoder, decoder, and transformer components
- **Comprehensive Evaluation**: Built-in metrics for assessing video quality and physical accuracy
- **GPU-Accelerated**: Optimized for CUDA-enabled GPUs with mixed precision training
- **Streaming Dataset Support**: Memory-efficient data loading for large video datasets

## Project Structure

```
PhysVideoGenerator/
├── data/                             # Dataset storagemodels 
├── src/                              # Source code
│   ├── encoders/                    # Video and text encoders
│   │   ├── vae_encoder_decoder.py  # VAE for video compression
│   │   └── text_caption_encoder.py # T5 text encoder
│   ├── datasets/                    # Dataset utilities (legacy)
│   │   ├── clean_videos.py
│   │   ├── delete_videos.py
│   │   ├── download_videos.py
│   ├── latte_physics.py            # Latte Transformer with physics
│   ├── train_latte_physics.py      # Training script
│   ├── infer_latte_physics.py      # Inference script
│   ├── evaluate_latte_physics.py   # Evaluation script
│   └── 01_prepare_video_dataset_streaming.py  # Dataset preparation
└── requirements.txt                 # Python dependencies
```

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (16GB+ VRAM recommended)
- PyTorch 2.9.1 with CUDA support

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PhysVideoGenerator.git
cd PhysVideoGenerator
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

Prepare video datasets with streaming support for large collections:

```bash
python src/datasets/download_videos.py
python src/01_prepare_video_dataset_streaming.py
```

These scripts:
- Download OpenVid-1M dataset
- Using pre-trained encoders (VJEPA-2, Latte-1 VAE, T5), prepare embedded latents and tokens

### 2. Train Physics-Aware Model

Train the Latte Transformer with physics-informed objectives:

```bash
python src/train_latte_physics.py
```

**Key Training Features:**
- Physics-aware loss functions
- Mixed precision training (bfloat16)
- Gradient checkpointing for memory efficiency
- Checkpoint saving and resumption

### 3. Generate Videos (Inference)

Generate physics-aware videos from text prompts:

```bash
python src/infer_latte_physics.py
```

**Inference Configuration:**
- Customizable resolution (default: 256x256)
- Adjustable video length (default: 16 frames)
- DDIM sampling with configurable steps
- Guidance scale for text conditioning strength

### 4. Evaluate Generated Videos

Assess video quality using multiple metrics:

```bash
python src/evaluate_latte_physics.py
```

**Evaluation Metrics:**
- **FID (Frechet Inception Distance)**: Overall visual quality
- **SSIM (Structural Similarity Index)**: Frame-level structural similarity
- **PSNR (Peak Signal-to-Noise Ratio)**: Pixel-level reconstruction quality
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual similarity

## Configuration

### Model Architecture

```python
# Latte Transformer Configuration
num_attention_heads = 16
attention_head_dim = 72
in_channels = 4              # VAE latent channels
out_channels = 4
num_layers = 28
sample_size = 32             # 256/8 (VAE downsampling factor)
patch_size = 2
caption_channels = 4096      # T5-XXL embedding dimension
video_length = 16            # Number of frames
```

### Training Parameters

```python
# Training Configuration
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
NUM_DIFFUSION_STEPS = 1000
GUIDANCE_SCALE = 7.5
MIXED_PRECISION = "bf16"     # bfloat16 for stability
GRADIENT_CHECKPOINTING = True
```

### Inference Settings

```python
# Inference Configuration
NUM_FRAMES = 16
HEIGHT = 256
WIDTH = 256
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
FPS = 8                      # Output video frame rate
```

## Model Details

### VAE Encoder/Decoder
- **Base Model**: `maxin-cn/Latte-1`
- **Latent Channels**: 4
- **Spatial Compression**: 8x (256 → 32)
- **Dtype**: bfloat16 for memory efficiency

### Text Encoder
- **Model**: T5-XXL (`google/t5-v1_1-xxl`)
- **Embedding Dimension**: 4096
- **Max Sequence Length**: 128 tokens

### Diffusion Scheduler
- **Type**: DDIM (Denoising Diffusion Implicit Models)
- **Base Config**: From Latte-1 pretrained model
- **Training Steps**: 1000
- **Inference Steps**: 50 (configurable)


## Dependencies

Core dependencies from `requirements.txt`:

- **torch** (2.9.1): Deep learning framework
- **torchvision** (0.24.1): Computer vision utilities
- **diffusers** (0.35.2): Diffusion models library
- **transformers** (4.57.3): NLP and multimodal models
- **accelerate** (1.12.0): Distributed training utilities
- **decord**: Efficient video loading
- **imageio** (2.37.2): Video I/O
- **opencv-python** (4.12.0.88): Image processing
- **lpips**: Perceptual similarity metric
- **torchmetrics**: Video quality metrics


## Known Limitations

- Requires significant GPU memory (16GB+ recommended)
- Training can be time-intensive for large datasets
- Generated videos currently limited to 256x256 resolution
- Physics constraints implementation is domain-specific


## License

This project is for research and educational purposes.

## Acknowledgments

- **Latte Team**: For the foundational Latent Diffusion Transformer architecture
- **Hugging Face**: For the Diffusers and Transformers libraries
- **Stability AI**: For diffusion model research and development
- **Google**: For the T5 text encoder models
