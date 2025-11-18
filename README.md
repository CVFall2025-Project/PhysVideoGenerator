# PhysVideoGenerator
Preliminary architecture for generating physics aware videos from text

# VJEPA2 → DiT Video Generation

### Setup using Docker + uv

Build the image:
    docker build -t vjepa2-diffusion .

Run (GPU):
    docker run --gpus all -it --shm-size=16g vjepa2-diffusion

Project uses:
- uv (package & environment manager)
- PyTorch
- Diffusers
- Transformers
- VJEPA-style encoders
- DiT for video latent diffusion
