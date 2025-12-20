"""
Common evaluation utilities for video generation models.
Provides standardized metrics computation for fair comparison.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from pathlib import Path
from PIL import Image


def compute_psnr(video1: torch.Tensor, video2: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two videos.
    
    Args:
        video1: Ground truth video tensor [B, T, C, H, W] or [T, C, H, W]
        video2: Generated video tensor [B, T, C, H, W] or [T, C, H, W]
    
    Returns:
        PSNR value in dB
    """
    if video1.dim() == 4:
        video1 = video1.unsqueeze(0)
    if video2.dim() == 4:
        video2 = video2.unsqueeze(0)
    
    mse = torch.mean((video1 - video2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0 if video1.max() <= 1.0 else 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(video1: torch.Tensor, video2: torch.Tensor, window_size: int = 11) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two videos.
    Simplified version - computes SSIM frame by frame and averages.
    
    Args:
        video1: Ground truth video tensor [B, T, C, H, W] or [T, C, H, W]
        video2: Generated video tensor [B, T, C, H, W] or [T, C, H, W]
        window_size: Size of the sliding window
    
    Returns:
        SSIM value (0-1)
    """
    try:
        from pytorch_msssim import ssim
    except ImportError:
        # Fallback to simple frame-wise SSIM approximation
        if video1.dim() == 4:
            video1 = video1.unsqueeze(0)
        if video2.dim() == 4:
            video2 = video2.unsqueeze(0)
        
        B, T, C, H, W = video1.shape
        ssim_values = []
        
        for t in range(T):
            for b in range(B):
                frame1 = video1[b, t].unsqueeze(0)  # [1, C, H, W]
                frame2 = video2[b, t].unsqueeze(0)
                
                # Simple SSIM approximation
                mu1 = frame1.mean()
                mu2 = frame2.mean()
                sigma1_sq = ((frame1 - mu1) ** 2).mean()
                sigma2_sq = ((frame2 - mu2) ** 2).mean()
                sigma12 = ((frame1 - mu1) * (frame2 - mu2)).mean()
                
                c1, c2 = 0.01 ** 2, 0.03 ** 2
                ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                          ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
                ssim_values.append(ssim_val.item())
        
        return np.mean(ssim_values)
    
    # Use pytorch_msssim if available
    if video1.dim() == 5:
        B, T, C, H, W = video1.shape
        ssim_values = []
        for t in range(T):
            ssim_val = ssim(video1[:, t], video2[:, t], data_range=1.0)
            ssim_values.append(ssim_val.item())
        return np.mean(ssim_values)
    else:
        return ssim(video1, video2, data_range=1.0).item()


def compute_fvd(videos_real: torch.Tensor, videos_fake: torch.Tensor, 
                device: str = 'cuda') -> float:
    """
    Compute Frechet Video Distance (FVD).
    Requires I3D model for feature extraction.
    
    Args:
        videos_real: Real videos tensor [B, T, C, H, W]
        videos_fake: Generated videos tensor [B, T, C, H, W]
        device: Device to run computation on
    
    Returns:
        FVD value
    """
    try:
        import tensorflow as tf
        from frechet_video_distance import fvd
    except ImportError:
        print("Warning: FVD computation requires tensorflow and frechet_video_distance.")
        print("Skipping FVD metric.")
        return None
    
    # Convert to numpy and ensure proper format
    videos_real_np = videos_real.cpu().numpy()
    videos_fake_np = videos_fake.cpu().numpy()
    
    # FVD expects videos in [B, T, H, W, C] format
    if videos_real_np.shape[2] == 3:  # [B, T, C, H, W]
        videos_real_np = np.transpose(videos_real_np, (0, 1, 3, 4, 2))
        videos_fake_np = np.transpose(videos_fake_np, (0, 1, 3, 4, 2))
    
    # Normalize to [0, 255]
    if videos_real_np.max() <= 1.0:
        videos_real_np = (videos_real_np * 255).astype(np.uint8)
        videos_fake_np = (videos_fake_np * 255).astype(np.uint8)
    
    fvd_value = fvd.calculate_fvd(
        fvd.create_id3_embedding(videos_real_np),
        fvd.create_id3_embedding(videos_fake_np)
    )
    
    return float(fvd_value)


def compute_clip_score(videos: torch.Tensor, text_prompts: List[str], 
                       device: str = 'cuda') -> float:
    """
    Compute CLIP score between generated videos and text prompts.
    
    Args:
        videos: Generated videos tensor [B, T, C, H, W]
        text_prompts: List of text prompts
        device: Device to run computation on
    
    Returns:
        Average CLIP score
    """
    try:
        import clip
    except ImportError:
        print("Warning: CLIP score computation requires clip package.")
        print("Skipping CLIP score metric.")
        return None
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Extract frames from videos (use middle frame)
    B, T, C, H, W = videos.shape
    mid_frame_idx = T // 2
    frames = videos[:, mid_frame_idx]  # [B, C, H, W]
    
    # Preprocess frames
    frames_processed = []
    for i in range(B):
        frame_np = frames[i].permute(1, 2, 0).cpu().numpy()
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame_np)
        frame_processed = preprocess(frame_pil).unsqueeze(0)
        frames_processed.append(frame_processed)
    
    frames_tensor = torch.cat(frames_processed, dim=0).to(device)
    
    # Encode text prompts
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    # Compute similarity
    with torch.no_grad():
        image_features = model.encode_image(frames_tensor)
        text_features = model.encode_text(text_tokens)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = (image_features * text_features).sum(dim=-1)
    
    return similarity.mean().item()


def save_evaluation_results(results: Dict, output_path: str):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary containing evaluation metrics
        output_path: Path to save results JSON file
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {output_path}")


def load_test_prompts(prompts_file: Optional[str] = None) -> List[str]:
    """
    Load test prompts from file or return default prompts.
    
    Args:
        prompts_file: Path to JSON file containing prompts list
    
    Returns:
        List of text prompts
    """
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
        return prompts if isinstance(prompts, list) else prompts.get('prompts', [])
    
    # Default test prompts
    default_prompts = [
        "A beautiful sunset over the ocean",
        "A cat playing with a ball of yarn",
        "A car driving on a highway",
        "A person walking in a park",
        "Birds flying in the sky",
        "Waves crashing on the beach",
        "A dog running in a field",
        "Rain falling on a window",
        "A train moving through a tunnel",
        "A butterfly flying among flowers"
    ]
    
    return default_prompts

