import torch
import numpy as np
import os
import cv2
import glob
from tqdm import tqdm
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# --- CONFIGURATION ---
GENERATED_DIR = "./generated_videos"  
REFERENCE_DIR = "./reference_videos"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_FRAMES = 16 
HEIGHT = 256
WIDTH = 256
# ---------------------

def load_video_tensor(path, num_frames, height, width):
    """Loads a video as a [T, C, H, W] tensor in range [0, 1]."""
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) < num_frames:
        # Pad with last frame if too short
        if len(frames) == 0: return None
        frames += [frames[-1]] * (num_frames - len(frames))
    
    # Numpy [T, H, W, C] -> Torch [T, C, H, W]
    video = np.array(frames).astype(np.float32) / 255.0
    video = torch.from_numpy(video).permute(0, 3, 1, 2)
    return video

def compute_metrics():
    print(f"Initializing metrics on {DEVICE}...")
    
    fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    lpips_loss = lpips.LPIPS(net='alex').to(DEVICE)

    gen_files = sorted(glob.glob(os.path.join(GENERATED_DIR, "*.mp4")))
    ref_files = sorted(glob.glob(os.path.join(REFERENCE_DIR, "*.mp4")))

    if len(gen_files) == 0 or len(ref_files) == 0:
        print("Error: No videos found in one of the directories.")
        return

    min_len = min(len(gen_files), len(ref_files))
    gen_files = gen_files[:min_len]
    ref_files = ref_files[:min_len]

    print(f"Evaluating {min_len} video pairs...")
    
    avg_ssim = 0
    avg_psnr = 0
    avg_lpips = 0

    # Loop through batches
    for i in tqdm(range(min_len)):
        # Load pair
        gen_vid = load_video_tensor(gen_files[i], NUM_FRAMES, HEIGHT, WIDTH) # [T, 3, H, W]
        ref_vid = load_video_tensor(ref_files[i], NUM_FRAMES, HEIGHT, WIDTH)

        if gen_vid is None or ref_vid is None: continue

        gen_vid = gen_vid.to(DEVICE)
        ref_vid = ref_vid.to(DEVICE)

        # Update FID (Flatten time: [T*B, 3, H, W])
        # Note: True FVD requires an I3D model, but aggregating frame-level FID 
        # is a common proxy when I3D weights are tricky to setup.
        # For FID, inputs must be uint8 [0, 255]
        fid_metric.update((ref_vid * 255).byte(), real=True)
        fid_metric.update((gen_vid * 255).byte(), real=False)

        # Compute Per-Sample Metrics (averaged over frames)
        # SSIM
        val_ssim = ssim_metric(gen_vid, ref_vid)
        avg_ssim += val_ssim.item()

        # PSNR
        val_psnr = psnr_metric(gen_vid, ref_vid)
        avg_psnr += val_psnr.item()

        # LPIPS (Expects range [-1, 1])
        gen_norm = gen_vid * 2.0 - 1.0
        ref_norm = ref_vid * 2.0 - 1.0
        val_lpips = lpips_loss(gen_norm, ref_norm).mean()
        avg_lpips += val_lpips.item()


    count = min_len
    final_fid = fid_metric.compute().item()
    final_ssim = avg_ssim / count
    final_psnr = avg_psnr / count
    final_lpips = avg_lpips / count

    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"FID (Frame-level): {final_fid:.4f} (Lower is better)")
    print(f"SSIM:              {final_ssim:.4f} (Higher is better)")
    print(f"PSNR:              {final_psnr:.4f} (Higher is better)")
    print(f"LPIPS:             {final_lpips:.4f} (Lower is better)")
    print("="*30)

if __name__ == "__main__":
    compute_metrics()