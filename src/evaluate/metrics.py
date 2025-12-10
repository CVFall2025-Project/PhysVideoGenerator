"""
metrics_research.py
Research-grade metrics for video evaluation (Option A).
Requires: GPU for FVD (I3D) + RAFT recommended for speed; CPU works but slow.

Usage examples provided at bottom of file.
"""

import os
import sys
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from skimage.metrics import structural_similarity as ssim_sk
from decord import VideoReader, cpu
from torchvision import transforms
from PIL import Image
from einops import rearrange
import lpips
from typing import List, Tuple, Optional

# -------------------------
# GPU / device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Helpers: frame extraction
# -------------------------
def read_video_frames(video_path: str, resize: Optional[Tuple[int,int]] = None, max_frames: Optional[int]=None):
    """
    Returns frames as list of RGB uint8 arrays [H,W,3].
    Uses decord (fast).
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    nframes = len(vr)
    if max_frames is not None and nframes > max_frames:
        # uniform sampling
        idx = np.linspace(0, nframes-1, max_frames).astype(int).tolist()
        batch = vr.get_batch(idx).asnumpy()
    else:
        batch = vr.get_batch(range(nframes)).asnumpy()
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch]
    if resize is not None:
        frames = [cv2.resize(f, resize[::-1], interpolation=cv2.INTER_LINEAR) for f in frames]
    return frames

# -------------------------
# 1) FVD (calls Google official frechet_video_distance)
# -------------------------
def compute_fvd_with_google_repo(real_videos_dir: str, gen_videos_dir: str, fvd_repo_path: str, i3d_checkpoint_path: str, tmp_stats_dir: str="/tmp/fvd_stats"):
    """
    Uses the official repo (google-research/frechet_video_distance).
    Steps:
      1) compute stats for the reference set (if not already)
      2) compute stats for generated set
      3) run compute_fvd.py with both stats -> score
    Requirements:
      - fvd_repo_path: path to cloned frechet_video_distance repo
      - i3d_checkpoint_path: path to I3D checkpoint expected by the repo (see repo README)
    """
    # sanity checks
    if not os.path.isdir(fvd_repo_path):
        raise FileNotFoundError("fvd_repo_path not found: " + fvd_repo_path)
    if not os.path.exists(i3d_checkpoint_path):
        raise FileNotFoundError("i3d_checkpoint_path not found: " + i3d_checkpoint_path)
    os.makedirs(tmp_stats_dir, exist_ok=True)

    # compute stats for reference
    ref_stats = os.path.join(tmp_stats_dir, "ref_stats.npz")
    gen_stats = os.path.join(tmp_stats_dir, "gen_stats.npz")

    # helper to run python inside repo
    def run_cmd(cmd_args):
        print("RUN:", " ".join(cmd_args))
        res = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr, file=sys.stderr)
            raise RuntimeError("Command failed: " + " ".join(cmd_args))
        return res

    # 1) compute reference stats if not present
    if not os.path.exists(ref_stats):
        run_cmd([
            sys.executable,
            os.path.join(fvd_repo_path, "compute_statistics.py"),
            "--kinetics_i3d_ckpt", i3d_checkpoint_path,
            "--videos_dir", real_videos_dir,
            "--output_path", ref_stats,
        ])

    # 2) compute generated stats
    run_cmd([
        sys.executable,
        os.path.join(fvd_repo_path, "compute_statistics.py"),
        "--kinetics_i3d_ckpt", i3d_checkpoint_path,
        "--videos_dir", gen_videos_dir,
        "--output_path", gen_stats,
    ])

    # 3) compute fvd
    res = run_cmd([
        sys.executable,
        os.path.join(fvd_repo_path, "compute_fvd.py"),
        "--statistics_real", ref_stats,
        "--statistics_fake", gen_stats,
    ])
    # parse output
    out = res.stdout
    # The script prints FVD result; parse last numeric occurrence
    import re
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", out)
    if len(nums) == 0:
        raise RuntimeError("Could not parse FVD output.")
    return float(nums[-1])

# -------------------------
# 2) t-LPIPS (temporal LPIPS)
# -------------------------
# Use official LPIPS (alexnet or vgg)
_lpips_model = None
def get_lpips_model(net="alex"):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net=net).to(device).eval()
    return _lpips_model

def compute_t_lpips(frames: List[np.ndarray], resize=(256,256), net="alex"):
    """
    frames: list of RGB numpy arrays uint8
    returns: mean LPIPS between consecutive frames
    """
    model = get_lpips_model(net=net)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize),
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  # to [-1,1]
    ])
    tensors = [preprocess(f).unsqueeze(0).to(device) for f in frames]
    vals = []
    with torch.no_grad():
        for i in range(len(tensors)-1):
            a = tensors[i]
            b = tensors[i+1]
            d = model(a, b)  # returns tensor [1,1,1,1] shape; .item()
            vals.append(d.item())
    return float(np.mean(vals)) if vals else 0.0

# -------------------------
# 3) Optical Flow Consistency (RAFT, forward-backward occlusion-aware)
# -------------------------
# This requires RAFT repo and RAFT checkpoint (e.g., raft-sintel.pth).
# We'll use RAFT to compute flow then warp and compute forward-backward error with occlusion mask.
# Assumes RAFT repo is cloned and RAFT model is loadable as module via sys.path insertion.
def import_raft(raft_repo_path: str):
    # add RAFT to path
    if raft_repo_path not in sys.path:
        sys.path.insert(0, raft_repo_path)
    try:
        import raft
        return raft
    except Exception as e:
        raise ImportError("Could not import RAFT. Ensure RAFT repo is at raft_repo_path and contains __init__.py. Error: " + str(e))

def load_raft_model(raft_repo_path: str, ckpt_path: str):
    raft = import_raft(raft_repo_path)
    # RAFT repo typical API has a create_model function or model class; adapt if necessary
    # We'll follow typical RAFT patterns (the exact import path may differ)
    from raft import RAFT  # may need change depending on repo structure
    model = RAFT().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def flow_warp(flow, coords):
    """
    Warp flow field 'flow' from its native coords to coords using bilinear sampling.
    flow: [H,W,2] numpy
    coords: two arrays X,Y of same shape
    returns: sampled flow at coords shape [H,W,2]
    """
    h,w = flow.shape[:2]
    X = np.clip(np.round(coords[0]).astype(int), 0, w-1)
    Y = np.clip(np.round(coords[1]).astype(int), 0, h-1)
    return flow[Y, X]

def compute_flow_consistency_raft(frames: List[np.ndarray], raft_repo_path: str, raft_ckpt_path: str):
    """
    frames: list of RGB uint8 frames
    Uses RAFT to compute flow F_t (t->t+1). Then warps backward flow and computes:
      err_t = mean(||F_t + warp(B_{t+1})(x)||) masked by occlusion
    Masking: use small-threshold occlusion detection by checking differences magnitude ratio.

    Returns: mean forward-backward error over frames.
    """
    # load RAFT model
    # NOTE: RAFT import may vary by repo. You might need to adapt to repo's naming.
    # For performance, convert frames to floats and normalized tensors
    import importlib
    # try to import RAFT inference util if the repo provides it; else fallback to Farneback for CPU
    try:
        raft = import_raft(raft_repo_path)
        # instantiate RAFT model; the RAFT API differs, so the project-specific script may have an 'inference' helper
        # Many RAFT repos have a function 'demo.py' which loads model and runs inference. For robust use, call that script.
        # For brevity here, if RAFT isn't easily importable, fallback to Farneback (slower but robust).
        from raft import RAFT  # may fail depending on repo layout
        model = RAFT().to(device)
        ckpt = torch.load(raft_ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()
        use_raft = True
    except Exception as e:
        print("RAFT import/init failed:", e)
        print("Falling back to OpenCV Farneback for flow (research-grade RAFT recommended).")
        use_raft = False

    h,w = frames[0].shape[:2]
    fb_errors = []
    for t in range(len(frames)-2):
        im1 = frames[t]
        im2 = frames[t+1]
        im3 = frames[t+2]

        gray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        gray3 = cv2.cvtColor(im3, cv2.COLOR_RGB2GRAY)

        if use_raft:
            # Convert to torch tensor [1,3,H,W] normalized to [0,1]
            def to_tensor_np(img):
                x = torch.from_numpy(img.astype('float32') / 255.0).permute(2,0,1).unsqueeze(0).to(device)
                return x
            with torch.no_grad():
                f12 = model(to_tensor_np(im1), to_tensor_np(im2))[0].cpu().numpy().transpose(1,2,0)
                f23 = model(to_tensor_np(im2), to_tensor_np(im3))[0].cpu().numpy().transpose(1,2,0)
        else:
            f12 = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                              pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                              poly_n=5, poly_sigma=1.2, flags=0)
            f23 = cv2.calcOpticalFlowFarneback(gray2, gray3, None,
                                              pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                              poly_n=5, poly_sigma=1.2, flags=0)

        # warp f23 backwards to coords of f12 using coords after applying f12
        h,w = f12.shape[:2]
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        x_fw = grid_x + f12[...,0]
        y_fw = grid_y + f12[...,1]
        # sample f23 at (y_fw, x_fw)
        xq = np.clip(x_fw, 0, w-1)
        yq = np.clip(y_fw, 0, h-1)
        # bilinear sampling
        def bilinear_sample(flow, xq, yq):
            x0 = np.floor(xq).astype(int); x1 = np.clip(x0+1, 0, w-1)
            y0 = np.floor(yq).astype(int); y1 = np.clip(y0+1, 0, h-1)
            wa = (x1 - xq)*(y1 - yq)
            wb = (xq - x0)*(y1 - yq)
            wc = (x1 - xq)*(yq - y0)
            wd = (xq - x0)*(yq - y0)
            Ia = flow[y0, x0]; Ib = flow[y0, x1]; Ic = flow[y1, x0]; Id = flow[y1, x1]
            return (Ia*wa[...,None] + Ib*wb[...,None] + Ic*wc[...,None] + Id*wd[...,None])

        f23_warp = bilinear_sample(f23, xq, yq)
        # forward-backward residual
        res = f12 + f23_warp  # shape H,W,2
        mag = np.linalg.norm(res, axis=2)
        # occlusion heuristic: if ||f12|| + ||f23|| small or change too big, mark occluded (optionally)
        occl_mask = (np.linalg.norm(f12, axis=2) + np.linalg.norm(f23_warp, axis=2)) > 1e-3
        # compute mean over non-occluded pixels
        if occl_mask.sum() > 0:
            mean_err = mag[occl_mask].mean()
        else:
            mean_err = mag.mean()
        fb_errors.append(mean_err)
    return float(np.mean(fb_errors)) if fb_errors else 0.0

# -------------------------
# 4) VideoCLIP Score (video-text cosine)
# -------------------------
# This assumes you have a pretrained VideoCLIP model (video encoder + CLIP text encoder).
# Many repos provide a checkpoint; load the model and compute:
# video_embedding = video_encoder(frames)  # global pooled
# text_embedding = clip_text_encoder(prompt)
# score = cosine(video_embedding, text_embedding)
def compute_videoclip_score_clip(frames: List[np.ndarray], text: str, clip_model, clip_processor, frame_resize=(224,224)):
    """
    If you don't have VideoCLIP, you can approximate with CLIP image encoder + average pooling across frames (not exact VideoCLIP).
    This function implements the CLIP-approximation which is a practical baseline.
    For research-grade VideoCLIP, replace this with official VideoCLIP model inference.
    """
    # frames: list of HWC RGB uint8
    # clip_model: huggingface CLIPModel
    # clip_processor: huggingface CLIPProcessor
    from transformers import CLIPProcessor, CLIPModel
    # Preprocess each frame with clip_processor
    imgs = [Image.fromarray(f) for f in frames]
    inputs = clip_processor(images=imgs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_feats = clip_model.get_image_features(**inputs)  # [T, dim]
        vid_feat = img_feats.mean(dim=0, keepdim=True)  # [1, dim]
        txt_inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
        txt_feat = clip_model.get_text_features(**txt_inputs)  # [1, dim]
        vid_feat = F.normalize(vid_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        sim = torch.matmul(vid_feat, txt_feat.t()).item()
    return float(sim)

# -------------------------
# 5) Action Accuracy (Kinetics classifier)
# -------------------------
# We'll use PyTorchVideo hub model (e.g., slow_r50) for inference.
def compute_action_label_pytorchvideo(frames: List[np.ndarray], model_name="slow_r50", frames_per_clip=16):
    """
    frames: list of RGB HWC uint8
    model_name: 'slow_r50' or 'x3d_s' etc (available via pytorchvideo.models.hub)
    returns: predicted class index (int) and topk logits if needed
    """
    from pytorchvideo.models.hub import slow_r50, x3d_s
    if model_name == "slow_r50":
        m = slow_r50(pretrained=True).to(device).eval()
    elif model_name == "x3d_s":
        m = x3d_s(pretrained=True).to(device).eval()
    else:
        m = slow_r50(pretrained=True).to(device).eval()

    # sample a clip of frames_per_clip uniformly
    n = len(frames)
    idx = np.linspace(0, n-1, frames_per_clip).astype(int).tolist()
    clip = [frames[i] for i in idx]  # list of HWC
    # preprocess to tensor [B=1, C, T, H, W]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
    ])
    clip_t = torch.stack([transform(f).to(device) for f in clip], dim=1).unsqueeze(0)  # [1, C, T, H, W]
    with torch.no_grad():
        out = m(clip_t)
        probs = F.softmax(out, dim=-1)
        top1 = int(probs.argmax(dim=-1).item())
    return top1, probs.cpu().numpy()

# -------------------------
# 6) SSIM (frame-level averaged)
# -------------------------
def compute_ssim_avg(frames: List[np.ndarray]):
    """
    frames: list of RGB HWC uint8
    returns: mean SSIM across consecutive frames (per-channel averaged)
    """
    vals = []
    for i in range(len(frames)-1):
        a = frames[i]
        b = frames[i+1]
        # skimage expects grayscale or multichannel; specify multichannel=True
        s = ssim_sk(a, b, data_range=255, multichannel=True)
        vals.append(s)
    return float(np.mean(vals)) if vals else 1.0

# -------------------------
# Example runner: single video metrics
# -------------------------
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="path to video")
    parser.add_argument("--prompt", type=str, default="A scene", help="text prompt for VideoCLIP")
    parser.add_argument("--raft_repo", type=str, default=None, help="path to RAFT repo")
    parser.add_argument("--raft_ckpt", type=str, default=None, help="path to RAFT checkpoint")
    parser.add_argument("--fvd_repo", type=str, default=None, help="path to frechet_video_distance repo")
    parser.add_argument("--i3d_ckpt", type=str, default=None, help="path to I3D checkpoint for FVD")
    args = parser.parse_args()

    frames = read_video_frames(args.video, resize=(256,256))
    print("frames:", len(frames))

    print("Computing t-LPIPS...")
    t_lpips = compute_t_lpips(frames, resize=(256,256))

    print("Computing SSIM...")
    ssim_v = compute_ssim_avg(frames)

    print("Computing Optical Flow Consistency (RAFT or Farneback)...")
    flow_cons = compute_flow_consistency_raft(frames, args.raft_repo or "", args.raft_ckpt or "")

    print("Computing VideoCLIP (CLIP-approx)...")
    # lightweight CLIP baseline for VideoCLIP — you should replace with true VideoCLIP for strict paper numbers
    try:
        from transformers import CLIPProcessor, CLIPModel
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        vclip = compute_videoclip_score_clip(frames, args.prompt, clip_model, clip_processor)
    except Exception as e:
        print("CLIP load error:", e)
        vclip = None

    print("Computing Action label (pytorchvideo slow_r50)...")
    action_label, probs = compute_action_label_pytorchvideo(frames, model_name="slow_r50")

    out = {
        "t-LPIPS": t_lpips,
        "SSIM": ssim_v,
        "OpticalFlowConsistency": flow_cons,
        "VideoCLIP_score": vclip,
        "Action_label_top1": int(action_label),
    }
    print(json.dumps(out, indent=2))
