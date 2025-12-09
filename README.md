# PhysVideoGenerator

> A physics-aware video generation framework leveraging CogVideoX with LoRA fine-tuning and VJEPA2 encoders

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Docker Setup](#docker-setup)

## Overview

PhysVideoGenerator is a research framework for generating physics-aware videos from text descriptions. The system combines state-of-the-art video diffusion models with physics understanding through:

- **CogVideoX** with LoRA adapters for efficient fine-tuning
- **VJEPA2-style encoders** for video representation learning
- **DiT (Diffusion Transformer)** architecture for video latent diffusion
- **Physics-aware training** to ensure generated videos follow physical laws

## Features

- **Physics-Aware Video Generation**: Generate videos that respect physical constraints and dynamics
- **Efficient Fine-Tuning**: LoRA (Low-Rank Adaptation) for parameter-efficient model adaptation
- **Multi-Stage Pipeline**: Dataset preparation, encoding, and training workflows
- **Flexible Architecture**: Modular design with separate encoder and decoder components
- **GPU-Accelerated**: Optimized for CUDA-enabled GPUs
- **Docker Support**: Containerized environment for reproducible results

## Project Structure

```
PhysVideoGenerator/
├── configs/              # Configuration files
├── data/                 # Dataset storage
├── dit/                  # Diffusion Transformer models
│   ├── model.py         # DiT model architecture
│   ├── layers.py        # Custom layers
│   └── diffusion.py     # Diffusion process
├── src/                  # Source code
│   ├── encoders/        # Video and text encoders
│   │   ├── vjepa2_encoder.py
│   │   ├── vae_encoder_decoder.py
│   │   └── text_caption_encoder.py
│   ├── datasets/        # Dataset utilities
│   │   ├── download_videos.py
│   │   └── clean_videos.py
│   ├── 01_prepare_video_dataset.py
│   └── 02_cogvideox_physics_with_lora.py
├── vae/                  # Variational Autoencoder components
└── requirements.txt      # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ GPU memory for training

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

Download and prepare video datasets:

```bash
python src/01_prepare_video_dataset.py
```

For streaming datasets:
```bash
python src/01_prepare_video_dataset_streaming.py
```

### 2. Train Physics-Aware Model

Train the CogVideoX model with LoRA adapters:

```bash
python src/02_cogvideox_physics_with_lora.py
```

### 3. Generate Videos

Once trained, use the model to generate physics-aware videos from text prompts.

## Configuration

Key configuration parameters in `src/02_cogvideox_physics_with_lora.py`:

```python
BATCH_SIZE = 2              # Batch size for training
T_STEPS = 1000              # Diffusion timesteps
EPOCHS = 30                 # Training epochs
LORA_RANK = 64              # LoRA rank (64 for new concepts)
LORA_ALPHA = 64             # LoRA alpha scaling
ADAPTER_LR = 1e-3           # Learning rate for adapters
```

### LoRA Configuration

The model uses LoRA for efficient fine-tuning with the following target modules:
- Attention projections: `to_q`, `to_k`, `to_v`, `to_out.0`
- Feed-forward layers: `ff.net.0.proj`, `ff.net.2`

## Training

### Model Architecture

- **VAE Latent Shape**: `[B, 16, 13, 60, 90]` (Channels, Temporal, Height, Width)
- **VFM Sequence Length**: 6144 tokens
- **VFM Dimension**: 1408
- **Text Sequence Length**: 128 tokens
- **Text Dimension**: 1024

### Training Tips

1. Start with a small dataset (50-100 videos) for initial experiments
2. Use gradient checkpointing to reduce memory usage
3. Monitor physics loss alongside reconstruction loss
4. Adjust `LAMBDA_PRED` to balance physics awareness vs. visual quality

### Checkpointing

Resume training from a checkpoint:
```python
RESUME_FROM_CHECKPOINT = 'checkpoints/checkpoint_epoch5_step1000.pt'
```

## Docker Setup

### Build Image

```bash
docker build -t vjepa2-diffusion .
```

### Run Container (GPU)

```bash
docker run --gpus all -it --shm-size=16g vjepa2-diffusion
```

The container includes:
- PyTorch with CUDA support
- All required dependencies
- Pre-configured environment

## Dependencies

- **PyTorch** (2.9.1): Deep learning framework
- **Diffusers** (0.35.2): Diffusion models library
- **Transformers** (4.57.3): NLP and multimodal models
- **PEFT** (0.14.0): Parameter-efficient fine-tuning
- **Decord**: Efficient video loading
- **OpenCV**: Image processing

## License

This project is for research and educational purposes.

## Acknowledgments

- CogVideoX team for the base video generation model
- Meta AI for VJEPA architecture insights
- Diffusers library by Hugging Face
