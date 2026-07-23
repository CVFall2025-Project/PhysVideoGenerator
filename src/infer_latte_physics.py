"""
Inference script for Latte + PredictorP physics-aware video generation.

Usage:
    python infer_latte_physics.py \
        --checkpoint ./checkpoints/checkpoint_epoch_50.pt \
        --prompt "A ball rolling down a hill" \
        --output output.mp4
"""

import torch
import numpy as np
from diffusers import DDIMScheduler
from latte_physics import LatteTransformer3DModelWithPhysics
import imageio
from pathlib import Path
import argparse

from encoders.text_caption_enocder import TextEncoder
from encoders.vae_encoder_decoder import VAEEncoder

def save_video(tensor, path, fps=8):
    """
    Saves a [C, T, H, W] tensor as an mp4.
    
    Args:
        tensor: Video tensor in range [-1, 1] with shape [C, T, H, W]
        path: Output path for mp4 file
        fps: Frames per second
    """
    # Reshape to [T, H, W, C] and map to [0, 255]
    video = tensor.permute(1, 2, 3, 0).cpu().numpy()
    video = ((video + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Ensure output directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    imageio.mimsave(path, video, fps=fps, codec='libx264')
    print(f"✓ Saved video: {path}")


@torch.no_grad()
def run_inference(
    prompt: str,
    checkpoint_path: str,
    output_path: str = "output_physics.mp4",
    num_frames: int = 16,
    height: int = 512,
    width: int = 512,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = None,
    device: str = "cuda",
):
    """
    Run physics-aware video generation inference.
    
    Args:
        prompt: Text description of the video to generate
        checkpoint_path: Path to trained model checkpoint
        output_path: Where to save output video
        num_frames: Number of frames to generate (default: 16)
        height: Video height in pixels (default: 256)
        width: Video width in pixels (default: 256)
        num_steps: Number of denoising steps (default: 50)
        guidance_scale: Classifier-free guidance scale (default: 7.5)
        seed: Random seed for reproducibility (default: None)
        device: Device to run on (default: "cuda")
    """
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"🎲 Using seed: {seed}")
    
    print(f"\n{'='*60}")
    print("LOADING MODELS")
    print(f"{'='*60}")
    
    # 1. Load text encoder (T5-XXL)
    print("Loading T5-XXL text encoder...")
    text_encoder = TextEncoder(
        model_name="google/t5-v1_1-xxl",
        device=device
    )
    
    # 2. Load VAE
    print("Loading VAE...")
    vae = VAEEncoder(
        model_name="maxin-cn/Latte-1",
        subfolder="vae",
        device=device,
        torch_dtype=torch.bfloat16
    )
    
    # 3. Load scheduler
    print("Loading DDIM scheduler...")
    scheduler = DDIMScheduler.from_pretrained(
        "maxin-cn/Latte-1",
        subfolder="scheduler"
    )
    
    # 4. Initialize Latte + PredictorP model
    print("Initializing Latte + PredictorP...")
    model = LatteTransformer3DModelWithPhysics(
        num_attention_heads=16,
        attention_head_dim=72,  # Correct value!
        in_channels=4,
        out_channels=4,
        num_layers=28,
        dropout=0.0,
        cross_attention_dim=1152,
        attention_bias=False,
        sample_size=64,  # 512/8 (VAE downscale factor)
        patch_size=2,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=4096,  # T5-XXL embedding dimension
        video_length=num_frames,
        # PredictorP config (IMPORTANT!)
        predictor_hidden_dim=512,
        vjepa_dim=1408,
        vjepa_seq_len=2048,
        use_predictor=True,
    ).to(device, dtype=torch.bfloat16)
    
    # 5. Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle potential key mismatches
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        print(f"Warning: {e}")
        print("Attempting flexible loading...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.eval()
    print(f"✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    print(f"\n{'='*60}")
    print("GENERATING VIDEO")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Frames: {num_frames}")
    print(f"Resolution: {height}×{width}")
    print(f"Steps: {num_steps}")
    print(f"CFG scale: {guidance_scale}")
    
    # 6. Encode text prompt
    print("\nEncoding text...")
    
    # Positive prompt
    text_inputs = text_encoder.tokenizer(
        prompt,
        padding="max_length",
        max_length=226,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        prompt_embeds = text_encoder.model(text_inputs.input_ids)[0]  # [1, 226, 4096]
    
    # Negative prompt (empty for classifier-free guidance)
    uncond_inputs = text_encoder.tokenizer(
        "",
        padding="max_length",
        max_length=226,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        uncond_embeds = text_encoder.model(uncond_inputs.input_ids)[0]  # [1, 226, 4096]
    
    # Concatenate for CFG: [uncond, cond]
    encoder_hidden_states = torch.cat([uncond_embeds, prompt_embeds], dim=0)  # [2, 226, 4096]
    
    # 7. Prepare initial noise
    # IMPORTANT: Model expects [B, T, C, H, W] format!
    latent_h = height // 8  # VAE downscales by 8
    latent_w = width // 8
    
    print(f"Initializing latents: [1, {num_frames}, 4, {latent_h}, {latent_w}]")
    latents = torch.randn(
        1, num_frames, 4, latent_h, latent_w,
        device=device,
        dtype=torch.bfloat16
    )
    
    # 8. Set up scheduler
    scheduler.set_timesteps(num_steps, device=device)
    
    # 9. Denoising loop
    print(f"\nDenoising ({num_steps} steps)...")
    
    for i, t in enumerate(scheduler.timesteps):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2, dim=0)  # [2, T, C, H, W]

        # Model expects [B, C, T, H, W] (matches training layout)
        latent_model_input = latent_model_input.permute(0, 2, 1, 3, 4)

        # Prepare timestep (needs to match batch size for CFG)
        timestep = torch.tensor([t, t], device=device, dtype=torch.long)

        # Predict noise with PredictorP
        # Model automatically runs PredictorP to predict physics!
        noise_pred, predicted_vjepa = model(
            latent_model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            ground_truth_vjepa=None,  # Not provided during inference
            enable_temporal_attentions=True,
        )
        
        # Extract noise prediction
        noise_pred = noise_pred.sample  # [2, C, T, H, W]
        
        # Convert back to [B, T, C, H, W] for scheduler
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)  # [2, T, C, H, W]
        
        # Perform classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Convert back to [B, C, T, H, W] for model compatibility
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
        latents_for_step = latents.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
        
        # Scheduler step
        latents_stepped = scheduler.step(noise_pred, t, latents_for_step).prev_sample
        
        # Convert back to [B, T, C, H, W]
        latents = latents_stepped.permute(0, 2, 1, 3, 4)  # [1, T, C, H, W]
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Step {i+1}/{num_steps}")
    
    print("✓ Denoising complete")
    
    # 10. Decode latents to video
    print("\nDecoding to pixels...")
    
    # Prepare for VAE: [B, C, T, H, W]
    latents = latents.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
    latents = latents / vae.model.config.scaling_factor
    
    # Decode frame by frame (VAE expects 4D input)
    latents_frames = latents.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [T, C, H, W]
    
    video_frames = vae.decode(latents_frames)  # [T, 3, H, W]
    
    # Reshape to [1, T, 3, H, W] then to [3, T, H, W] for saving
    video_tensor = video_frames.reshape(1, num_frames, 3, height, width)[0]  # [T, 3, H, W]
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # [3, T, H, W]
    
    # 11. Save video
    print(f"\nSaving video to {output_path}...")
    save_video(video_tensor, output_path, fps=8)
    
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Video saved: {output_path}")
    print(f"✓ Duration: {num_frames / 8:.1f}s @ 8 fps")
    print(f"✓ Resolution: {height}×{width}")
    
    return video_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics-aware video generation with Latte + PredictorP")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--output", type=str, default="output_physics.mp4", help="Output video path")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    run_inference(
        prompt=args.prompt,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.cfg_scale,
        seed=args.seed,
        device=args.device,
    )