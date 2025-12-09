import torch
import imageio
from diffusers import AutoencoderKLCogVideoX
import numpy as np
from typing import Optional

class VAEEncoder():
    def __init__(self, model_name, torch_dtype, device: Optional[torch.device] = None):
        self.model = AutoencoderKLCogVideoX.from_pretrained(model_name, subfolder="vae", torch_dtype=torch_dtype).to(device)
        self.model.enable_slicing()
        self.model.enable_tiling()
        self.device = device
        self.torch_dtype = torch_dtype
    
    def encode(self, video):
        if 'frames' in video.files:
            frames_array = video['frames']
        else:
            # Use the first available array if 'frames' key doesn't exist
            frames_array = video[video.files[0]]
        
        # Convert frames to tensors
        video_tensor = torch.from_numpy(frames_array).to(self.device).to(self.torch_dtype) # (1, C, T, H, W)
        video.close()

        # Encode using the VAE model
        with torch.no_grad():
            encoded_frames = self.model.encode(video_tensor)[0].sample()
        
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
