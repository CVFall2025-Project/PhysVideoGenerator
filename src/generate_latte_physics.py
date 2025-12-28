"""
Batch Video Generation for Latte + PredictorP

Generates videos from test_index.json using trained checkpoint.
Decodes latents to actual video files using VAE.

Uses:
- VAE latents (as reference for shape only, starts from noise)
- Text embeddings (loaded from file)
- PredictorP (predicts VJEPA during generation)

Usage:
    python generate_latte_physics.py \
        --checkpoint ./checkpoints/checkpoint_epoch_50.pt \
        --test_index ./data/test_index.json \
        --output_dir ./generated_videos \
        --num_steps 50
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import imageio

from diffusers import DDIMScheduler
from latte_physics import LatteTransformer3DModelWithPhysics
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


@torch.no_grad()
def generate_video_from_prompt(
    model,
    vae,
    scheduler,
    text_embeddings: torch.Tensor,
    num_frames: int = 16,
    height: int = 256,
    width: int = 256,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    device: str = "cuda",
):
    """
    Generate video from text embeddings.
    
    Args:
        model: Latte + PredictorP model
        vae: VAE decoder
        scheduler: DDIM scheduler
        text_embeddings: Pre-computed text embeddings [seq_len, embed_dim]
        num_frames: Number of frames to generate
        height: Video height
        width: Video width
        num_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        device: Device to use
    
    Returns:
        Generated video tensor [C, T, H, W] in range [-1, 1]
    """
    
    # Handle text embedding shape
    if text_embeddings.dim() == 2:
        # [seq_len, embed_dim] -> [1, seq_len, embed_dim]
        text_embeddings = text_embeddings.unsqueeze(0)
    
    # Prepare text embeddings for CFG
    uncond_embeddings = torch.zeros_like(text_embeddings)
    encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings], dim=0)  # [2, seq_len, embed_dim]
    
    # Initialize noise in latent space
    # VAE downscales by 8
    latent_h = height // 8
    latent_w = width // 8
    
    latents = torch.randn(
        1, num_frames, 4, latent_h, latent_w,
        device=device,
        dtype=torch.bfloat16
    )
    
    # Set up scheduler
    scheduler.set_timesteps(num_steps, device=device)
    
    # Denoising loop
    for t in scheduler.timesteps:
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2, dim=0)  # [2, T, C, H, W]
        
        # Prepare timestep
        timestep = torch.tensor([t, t], device=device, dtype=torch.long)
        
        # Predict noise (PredictorP predicts VJEPA automatically!)
        noise_pred, predicted_vjepa = model(
            latent_model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            ground_truth_vjepa=None,  # Predicted by PredictorP
            enable_temporal_attentions=True,
        )
        
        # Extract noise prediction
        noise_pred = noise_pred.sample  # [2, C, T, H, W]
        
        # Convert to [B, T, C, H, W]
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)  # [2, T, C, H, W]
        
        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Convert back for scheduler
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
        latents_for_step = latents.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
        
        # Scheduler step
        latents_stepped = scheduler.step(noise_pred, t, latents_for_step).prev_sample
        
        # Convert back
        latents = latents_stepped.permute(0, 2, 1, 3, 4)  # [1, T, C, H, W]
    
    # Decode latents to video
    latents = latents.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
    latents = latents / vae.model.config.scaling_factor
    
    # Decode frame by frame
    latents_frames = latents.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [T, C, H, W]
    video_frames = vae.model.decode(latents_frames).sample  # [T, 3, H, W]
    
    # Reshape to [C, T, H, W]
    video_tensor = video_frames.reshape(1, num_frames, 3, height, width)[0]  # [T, 3, H, W]
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # [3, T, H, W]
    
    return video_tensor


def batch_generate(
    checkpoint_path: str,
    test_index_path: str,
    output_dir: str = "./generated_videos",
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    num_frames: int = 16,
    height: int = 256,
    width: int = 256,
    device: str = "cuda",
    max_samples: int = None,
):
    """
    Batch generate videos from test_index.json.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        test_index_path: Path to test_index.json
        output_dir: Output directory for generated videos
        num_steps: Number of denoising steps
        guidance_scale: CFG scale
        num_frames: Number of frames
        height: Video height
        width: Video width
        device: Device to use
        max_samples: Maximum number of samples to generate (None = all)
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("BATCH VIDEO GENERATION WITH VAE DECODING")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test index: {test_index_path}")
    print(f"Output dir: {output_dir}")
    print(f"Resolution: {height}×{width}, {num_frames} frames")
    print(f"Steps: {num_steps}, CFG: {guidance_scale}")
    
    # Load test index
    print(f"\nLoading test index...")
    with open(test_index_path, 'r') as f:
        test_data = json.load(f)
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"Samples: {len(test_data)}")
    
    # Load VAE
    print(f"\nLoading VAE decoder...")
    vae = VAEEncoder(
        model_name="maxin-cn/Latte-1",
        subfolder="vae",
        device=device,
        torch_dtype=torch.bfloat16
    )
    
    # Load model
    print(f"Loading Latte + PredictorP...")
    model = LatteTransformer3DModelWithPhysics(
        num_attention_heads=16,
        attention_head_dim=72,
        in_channels=4,
        out_channels=4,
        num_layers=28,
        dropout=0.0,
        cross_attention_dim=1152,
        attention_bias=False,
        sample_size=32,
        patch_size=2,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=4096,
        video_length=num_frames,
        # PredictorP config
        predictor_hidden_dim=512,
        vjepa_dim=1408,
        vjepa_seq_len=2048,
        use_predictor=True,
    ).to(device, dtype=torch.bfloat16)
    
    # Load checkpoint
    print(f"Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load scheduler
    scheduler = DDIMScheduler.from_pretrained("maxin-cn/Latte-1", subfolder="scheduler")
    
    print(f"\n{'='*60}")
    print("GENERATING VIDEOS")
    print(f"{'='*60}\n")
    
    # Generate videos
    results = []
    
    for sample in tqdm(test_data, desc="Generating videos"):
        video_id = sample['video_id']
        
        try:
            # Load text embeddings
            text_latent_path = sample.get('text', sample.get('text_latent'))
            if text_latent_path is None:
                print(f"\n  Warning: No text latent for {video_id}")
                results.append({'video_id': video_id, 'status': 'failed', 'error': 'no_text_latent'})
                continue
            
            text_embeddings = np.load(text_latent_path)
            
            # Handle different formats
            if isinstance(text_embeddings, np.lib.npyio.NpzFile):
                if 'embedding' in text_embeddings:
                    text_embeddings = text_embeddings['embedding']
                elif 'arr_0' in text_embeddings:
                    text_embeddings = text_embeddings['arr_0']
                else:
                    text_embeddings = text_embeddings[list(text_embeddings.keys())[0]]
            
            text_embeddings = torch.from_numpy(text_embeddings).to(device, dtype=torch.bfloat16)
            
            # Handle shape
            if text_embeddings.dim() == 3 and text_embeddings.shape[0] == 1:
                text_embeddings = text_embeddings[0]
            
            # Generate video
            video_tensor = generate_video_from_prompt(
                model=model,
                vae=vae,
                scheduler=scheduler,
                text_embeddings=text_embeddings,
                num_frames=num_frames,
                height=height,
                width=width,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                device=device,
            )
            
            # Save video
            output_path = output_dir / f"{video_id}_generated.mp4"
            save_video(video_tensor, output_path, fps=8)
            
            results.append({
                'video_id': video_id,
                'status': 'success',
                'output_path': str(output_path),
                'prompt': sample.get('text_prompt', ''),
            })
            
        except Exception as e:
            print(f"\n❌ Error generating {video_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'video_id': video_id,
                'status': 'failed',
                'error': str(e),
            })
    
    # Save results
    results_path = output_dir / "generation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'total': len(test_data),
                'successful': sum(1 for r in results if r['status'] == 'success'),
                'failed': sum(1 for r in results if r['status'] == 'failed'),
            },
            'results': results,
        }, f, indent=2)
    
    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Successful: {successful}/{len(results)}")
    if failed > 0:
        print(f"✗ Failed: {failed}/{len(results)}")
    print(f"✓ Videos saved to: {output_dir}")
    print(f"✓ Results saved: {results_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch video generation from test_index.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--test_index", type=str, required=True, help="Path to test_index.json")
    parser.add_argument("--output_dir", type=str, default="./generated_videos", help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--height", type=int, default=256, help="Video height")
    parser.add_argument("--width", type=int, default=256, help="Video width")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples")
    
    args = parser.parse_args()
    
    batch_generate(
        checkpoint_path=args.checkpoint,
        test_index_path=args.test_index,
        output_dir=args.output_dir,
        num_steps=args.steps,
        guidance_scale=args.cfg_scale,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        device=args.device,
        max_samples=args.max_samples,
    )