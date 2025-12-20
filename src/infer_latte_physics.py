import torch
import numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import T5EncoderModel, T5Tokenizer
from latte_physics import LatteTransformer3DModelWithPhysics
import imageio

from encoders.text_caption_enocder import TextEncoder
from encoders.vae_encoder_decoder import VAEEncoder

def save_video(tensor, path, fps=8):
    """Saves a [C, T, H, W] tensor as an mp4."""
    # Reshape to [T, H, W, C] and map to [0, 255]
    video = tensor.permute(1, 2, 3, 0).cpu().numpy()
    video = ((video + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    imageio.mimsave(path, video, fps=fps, codec='libx264')

@torch.no_grad()
def run_inference(
    prompt,
    checkpoint_path,
    output_path="output_physics.mp4",
    num_frames=16,
    height=256,
    width=256,
    num_steps=50,
    guidance_scale=7.5,
    device="cuda"
):
    print(f"Loading components to {device}...")
    # 1. Load the text encoder and VAE from base Latte
    # 4096 channels indicates T5-XXL
    text_encoder = TextEncoder(model_name="google/t5-v1_1-xxl", device=device)
    vae = VAEEncoder(model_name="maxin-cn/Latte-1", subfolder="vae", device=device, torch_dtype=torch.bfloat16)

    scheduler = DDIMScheduler.from_pretrained("maxin-cn/Latte-1", subfolder="scheduler")
    
    # 2. Initialize and load your custom model
    model = LatteTransformer3DModelWithPhysics(
        num_attention_heads=16,
        attention_head_dim=72,
        in_channels=4,
        out_channels=4,
        num_layers=28,
        sample_size=32, # 256/8 (VAE factor)
        patch_size=2,
        caption_channels=4096,
        video_length=num_frames,
    ).to(device)
    
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Encode prompt
    text_inputs = text_encoder.tokenizer(
        prompt, padding="max_length", max_length=226, truncation=True, return_tensors="pt"
    ).to(device)
    prompt_embeds = text_encoder(text_inputs.input_ids)[0]
    
    uncond_inputs = text_encoder.tokenizer(
        "", padding="max_length", max_length=226, truncation=True, return_tensors="pt"
    ).to(device)
    uncond_embeds = text_encoder.model(uncond_inputs.input_ids)[0]
    prompt_embeds = torch.cat([uncond_embeds, prompt_embeds])

    # 4. Prepare initial noise
    # VAE downscales by 8 (256 -> 32)
    latents = torch.randn(1, 4, num_frames, 32, 32).to(device)
    scheduler.set_timesteps(num_steps)

    # 5. Diffusion Loop
    print("Starting denoising...")
    for t in scheduler.timesteps:
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        
        # Predict noise
        # The forward pass automatically runs PredictorP!
        noise_pred, _ = model(
            latent_model_input,
            timestep=t.unsqueeze(0).to(device),
            encoder_hidden_states=prompt_embeds,
        )
        noise_pred = noise_pred.sample # Extract from Transformer2DModelOutput

        # Perform CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Step scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 6. Decode Latents to Video
    print("Decoding to pixels...")
    latents = 1 / vae.config.scaling_factor * latents
    video_tensor = vae.decode(latents.permute(0, 2, 1, 3, 4).flatten(0, 1)).sample
    video_tensor = video_tensor.reshape(1, num_frames, 3, height, width).permute(0, 2, 1, 3, 4)[0]

    # 7. Save
    save_video(video_tensor, output_path)
    print(f"✓ Video saved to {output_path}")

if __name__ == "__main__":
    run_inference(
        prompt="A block sliding down an inclined plane with realistic friction",
        checkpoint_path="./latte_predictor_checkpoints/checkpoint_epoch_6.pt"
    )