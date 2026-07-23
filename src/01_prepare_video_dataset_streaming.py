"""Prepare video dataset with streaming: download → extract → process → delete

Efficient pipeline that processes each downloaded zip immediately:
 1. Download a single zip part
 2. Extract videos
 3. For each video:
    a. Clean (VAE format + VJEPA format)
    b. VAE encode
    c. VJEPA encode
    d. Delete raw video
 4. Delete extracted videos folder
 5. Repeat for next part
 6. (Optional) Encode text captions for all videos at the end
 7. Build index

Benefits:
 - No intermediate storage of raw videos or cleaned .npz files
 - GPU memory reused per video
 - Only final encodings stored
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from src.datasets import clean_videos
from src.encoders.vae_encoder_decoder import VAEEncoder
from src.encoders.vjepa2_encoder import VJEPA2Encoder
from src.encoders.text_caption_enocder import TextEncoder
import random

logger = logging.getLogger("prepare_video_dataset_streaming")


def ensure_dirs(root: str) -> Dict[str, str]:
    project_root = os.path.abspath(root)
    paths = {
        "project_root": project_root,
        "zip_folder": os.path.join(project_root, "data", "download"),
        "raw_videos": os.path.join(project_root, "data", "raw_videos"),
        "encoded_vae": os.path.join(project_root, "data", "encoded_videos", "vae"),
        "encoded_vjepa": os.path.join(project_root, "data", "encoded_videos", "vjepa"),
        "encoded_text": os.path.join(project_root, "data", "encoded_videos", "text"),
        "csv_data": os.path.join(project_root, "data", "text_csv"),
        "index_file": os.path.join(project_root, "data", "indexed_dataset.json"),
    }
    for p in paths.values():
        if p != paths["index_file"]:
            os.makedirs(p, exist_ok=True)
    return paths


def process_video_full(
    video_path: str,
    vae_encoder: VAEEncoder,
    vjepa_encoder: VJEPA2Encoder,
    processor: clean_videos.VideoProcessor,
    output_paths: Dict[str, str],
) -> Tuple[str, bool]:
    """Process a single video: clean → VAE encode → VJEPA encode. Return (base_name, success)."""
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Load and clean video
        frames = processor.load_video(video_path)
        outputs = processor.process_video(frames)  # {"vae": ..., "vjepa": ...}
        
        # VAE encode
        frames_vae = outputs["vae"]
        vae_tensor = torch.from_numpy(frames_vae).to(vae_encoder.device).to(vae_encoder.torch_dtype)
        vae_encoded = vae_encoder.encode(vae_tensor)
        vae_arr = vae_encoded.detach().cpu().numpy()
        vae_path = os.path.join(output_paths["encoded_vae"], f"{base_name}_vae.npz")
        np.savez_compressed(vae_path, vae_arr)
        
        # VJEPA encode (create minimal NpzFile wrapper for VJEPA encoder)
        frames_vjepa = outputs["vjepa"]
        vjepa_tensor = torch.from_numpy(frames_vjepa).to(vjepa_encoder.device).to(vjepa_encoder.torch_dtype)
        vjepa_features = vjepa_encoder.encode(vjepa_tensor)
        vjepa_arr = vjepa_features.detach().cpu().numpy()
        vjepa_path = os.path.join(output_paths["encoded_vjepa"], f"{base_name}_vjepa.npz")
        np.savez_compressed(vjepa_path, vjepa_arr)
        
        return base_name, True
    
    except Exception as e:
        logger.warning(f"Failed to process {video_path}: {e}")
        return "", False


def run_streaming_pipeline(
    root: str,
    limit: Optional[int] = None,
    do_text: bool = True,
    do_vae: bool = True,
    do_vjepa: bool = True,
) -> Dict[str, str]:
    """
    Stream pipeline: for each zip part, download → extract → process → delete.
    
    Args:
        root: project root
        limit: max videos per part (for testing)
        do_text: encode text captions after all videos
        do_vae: encode VAE
        do_vjepa: encode VJEPA
    
    Returns:
        video_ids: list of processed video IDs
    """
    paths = ensure_dirs(root)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    csv_path = os.path.join(paths["csv_data"], "OpenVid-1M.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"CSV not found: {csv_path}. Skipping text encoding.")
        return {}
    csv_df = pd.read_csv(csv_path)

    # Restrict to the curated subset if the curated CSV is present.
    curated_csv_path = os.path.join(paths["csv_data"], "curated_OpenVid-1M.csv")
    curated_set = None
    if os.path.exists(curated_csv_path):
        curated_set = set(pd.read_csv(curated_csv_path)["video"].astype(str))
        logger.info(f"Curated CSV found: restricting to {len(curated_set)} clips.")

    # Process videos present in the `raw_videos` folder (downloader runs separately)
    if not os.path.exists(paths["raw_videos"]):
        logger.error(f"Raw videos directory not found: {paths['raw_videos']}. Place videos there or run downloader first.")
        return {}

    video_files = [
        f for f in os.listdir(paths["raw_videos"]) if f.lower().endswith((".mp4"))
    ]

    random.shuffle(video_files)

    shortlisted_videos = []
    cnt = 0
    for file_name in video_files:
        if limit is not None and cnt >= limit:
            break
        if curated_set is not None and file_name not in curated_set:
            continue
        row = csv_df[csv_df["video"] == file_name]
        if row.empty:
            continue
        duration = row["seconds"].values[0]
        if duration <= 4.0:
            shortlisted_videos.append(file_name)
            cnt += 1
    
    print(f"Found suitable {len(shortlisted_videos)} videos")

    # Initialize encoders once
    if (do_vae and do_vjepa):

        vae_encoder = VAEEncoder("maxin-cn/Latte-1", torch_dtype=torch.float16, device=device)
        vjepa_encoder = VJEPA2Encoder(model_name="facebook/vjepa2-vitg-fpc64-256", torch_dtype=torch.float16, device=device)
        processor = clean_videos.VideoProcessor(device=device)

        logger.info(f"Processing {len(shortlisted_videos)} videos from {paths['raw_videos']}")

        for video_file in tqdm(shortlisted_videos, desc="Processing videos"):
            video_path = os.path.join(paths["raw_videos"], video_file)
            _, _ = process_video_full(
                video_path,
                vae_encoder,
                vjepa_encoder,
                processor,
                paths,
            )

        # Delete raw video immediately after processing
        delete_command = "rm -rf " + paths["raw_videos"] + "/*.mp4"
        os.system(delete_command)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Streaming encoding complete. Processed {len(shortlisted_videos)} videos total.")
        logger.info(f"{'='*60}")
    
    # Encode text captions if requested (do this after all video processing)
    if do_text:
        run_text_encoding_batch(paths)
    
    # Build index
    build_index(paths)
    
    return {"processed": shortlisted_videos}


def run_text_encoding_batch(paths: Dict[str, str]) -> Dict[str, str]:
    """Encode text captions for processed videos."""
    logger.info("Starting text encoding step")
    csv_path = os.path.join(paths["csv_data"], "OpenVid-1M.csv")
    
    if not os.path.exists(csv_path):
        logger.warning(f"CSV not found: {csv_path}. Skipping text encoding.")
        return {}
    
    csv_df = pd.read_csv(csv_path)

    encoded_video_path_list = os.listdir(paths["encoded_vae"])
    video_ids = set([fname[:-8] for fname in encoded_video_path_list])

    model_name = "google/t5-v1_1-xxl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder = TextEncoder(model_name=model_name, device=device)
    
    saved_map = {}
    for _, row in tqdm(csv_df.iterrows(), total=len(csv_df), desc="Text encoding"):
        filename = row["video"].split(".")[0]
        
        # Only encode if this video was processed
        if filename not in video_ids:
            continue
        
        try:
            text_embedding = text_encoder.encode(row["caption"])
            output_filename = f"{filename}_text.npy"
            output_path = os.path.join(paths["encoded_text"], output_filename)
            np.save(output_path, text_embedding)
            saved_map[filename] = output_path
        except Exception as e:
            logger.warning(f"Failed to encode text for {filename}: {e}")
            continue
    
    logger.info(f"Text encoding complete. Encoded {len(saved_map)} captions.")
    return saved_map


def build_index(paths: Dict[str, str]) -> None:
    """Build index file for processed videos."""
    logger.info("Building indexed dataset")
    index_file = paths["index_file"]
    
    encoded_video_path_list = os.listdir(paths["encoded_vae"])
    video_ids = set([fname[:-8] for fname in encoded_video_path_list])
    
    json_entries = []
    for video_id in sorted(video_ids):
        vae_file = os.path.join(paths["encoded_vae"], f"{video_id}_vae.npz")
        vjepa_file = os.path.join(paths["encoded_vjepa"], f"{video_id}_vjepa.npz")
        text_file = os.path.join(paths["encoded_text"], f"{video_id}_text.npy")
        
        # Only add if files exist
        if not os.path.exists(vae_file):
            continue
        if not os.path.exists(vjepa_file):
            continue
        if not os.path.exists(text_file):
            continue
        
        entry = {
            "video_id": video_id,
            "vae": vae_file,
            "vjepa": vjepa_file,
            "text": text_file,
        }
        json_entries.append(entry)
    
    if not os.path.exists(index_file):
        existing_data = []
    else:
        with open(index_file, "r") as outf:
            try:
                existing_data = json.load(outf)
            except json.JSONDecodeError:
                existing_data = []
    
    existing_data.extend(json_entries)
    
    with open(index_file, "w") as outf:
        json.dump(existing_data, outf, indent=2)
    
    logger.info(f"Wrote index to {index_file} for {len(video_ids)} videos")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Streaming video dataset preparation pipeline")
    parser.add_argument("--root", default=str(Path(__file__).parent.parent.resolve()), help="Project root")
    parser.add_argument("--limit", type=int, default=None, help="Max videos per part (for testing)")
    parser.add_argument("--no-text", dest="text", action="store_false", default=True, help="Skip text encoding")
    parser.add_argument("--no-vae", dest="vae", action="store_false", default=True, help="Skip VAE encoding")
    parser.add_argument("--no-vjepa", dest="vjepa", action="store_false", default=True, help="Skip VJEPA encoding")
    
    args = parser.parse_args()
    
    run_streaming_pipeline(
        root=args.root,
        limit=args.limit,
        do_text=args.text,
        do_vae=args.vae,
        do_vjepa=args.vjepa,
    )
