import torch
import imageio
from diffusers import AutoencoderKL
import numpy as np
from typing import Optional
from einops import rearrange

class VAEEncoder():
    def __init__(self, model_name, torch_dtype, device: Optional[torch.device] = None):
        self.model = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=torch_dtype).to(device)
        self.model.enable_slicing()
        self.model.enable_tiling()
        self.device = device
        self.torch_dtype = torch_dtype
    
    def encode(self, video_tensor):
        # Encode using the VAE model
        with torch.no_grad():
            b, _, _, _, _ = video_tensor.shape
            video_tensor = rearrange(video_tensor, 'b f c h w -> (b f) c h w').contiguous()
            encoded_frames = self.model.encode(video_tensor).latent_dist.sample().mul_(0.18215)
            encoded_frames = rearrange(encoded_frames, '(b f) c h w -> b f c h w', b=b).contiguous()
        
        return encoded_frames
    
    def decode(self, encoded_frames):
        with torch.no_grad():
            decoded_frames = self.model.decode(encoded_frames).sample
        return decoded_frames

    def save_video(self, tensor, fps, output_path):
        tensor = tensor.to(dtype=torch.float32)
        frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        frames = np.clip(frames, 0, 1) * 255
        frames = frames.astype(np.uint8)
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
