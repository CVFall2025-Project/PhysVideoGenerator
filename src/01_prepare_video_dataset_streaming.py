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
import tempfile
import shutil

import numpy as np
import torch
import pandas as pd
from glob import glob
from tqdm import tqdm

from src.datasets import download_videos, clean_videos
from src.encoders.vae_encoder_decoder import VAEEncoder
from src.encoders.vjepa2_encoder import VJEPA2Encoder
from src.encoders.text_caption_enocder import TextEncoder

logger = logging.getLogger("prepare_video_dataset_streaming")


def ensure_dirs(root: str) -> Dict[str, str]:
    project_root = os.path.abspath(root)
    paths = {
        "project_root": project_root,
        "zip_folder": os.path.join(project_root, "data", "download"),
        "encoded_vae": os.path.join(project_root, "data", "encoded_videos", "vae"),
        "encoded_vjepa": os.path.join(project_root, "data", "encoded_videos", "vjepa"),
        "encoded_text": os.path.join(project_root, "data", "encoded_videos", "text"),
        "csv_data": os.path.join(project_root, "data", "text_csv"),
        "index_file": os.path.join(project_root, "data", "indexed_dataset.jsonl"),
    }
    for p in paths.values():
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
        with torch.no_grad():
            vae_encoded = vae_encoder.model.encode(vae_tensor)[0].sample()
        vae_arr = vae_encoded.detach().cpu().numpy()
        vae_path = os.path.join(output_paths["encoded_vae"], f"{base_name}_vae.npz")
        np.savez_compressed(vae_path, vae_arr)
        
        # VJEPA encode (create minimal NpzFile wrapper for VJEPA encoder)
        frames_vjepa = outputs["vjepa"]
        vjepa_tensor = torch.from_numpy(frames_vjepa).to(vjepa_encoder.device).to(vjepa_encoder.torch_dtype)
        with torch.inference_mode():
            # Mimic VJEPA2Encoder.encode() logic but inline
            x_hf = vjepa_encoder.transform(vjepa_tensor, return_tensors="pt")["pixel_values_videos"].to(vjepa_encoder.device)
            vjepa_features = vjepa_encoder.model.get_vision_features(x_hf)
        vjepa_arr = vjepa_features.detach().cpu().numpy()
        vjepa_path = os.path.join(output_paths["encoded_vjepa"], f"{base_name}_vjepa.npz")
        np.savez_compressed(vjepa_path, vjepa_arr)
        
        logger.info(f"Encoded {base_name}: VAE→{vae_path}, VJEPA→{vjepa_path}")
        return base_name, True
    
    except Exception as e:
        logger.warning(f"Failed to process {video_path}: {e}")
        return "", False


def run_streaming_pipeline(
    root: str,
    parts_range=range(0, 1),
    limit: Optional[int] = None,
    do_text: bool = True,
    do_vae: bool = True,
    do_vjepa: bool = True,
) -> Dict[str, str]:
    """
    Stream pipeline: for each zip part, download → extract → process → delete.
    
    Args:
        root: project root
        parts_range: parts to download
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
    
    # Initialize encoders once
    vae_encoder = VAEEncoder("THUDM/CogVideoX-2b", torch_dtype=torch.float32, device=device) if do_vae else None
    vjepa_encoder = VJEPA2Encoder(model_name="facebook/vjepa2-vitg-fpc64-256", torch_dtype=torch.float32, device=device) if do_vjepa else None
    processor = clean_videos.VideoProcessor(device=device)
    
    processed_videos = []
    
    # Download and process each part sequentially
    for part_idx in parts_range:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing part {part_idx}")
        logger.info(f"{'='*60}")
        
        # Download and extract this part
        with tempfile.TemporaryDirectory(prefix=f"openvid_part{part_idx}_") as temp_extract_dir:
            try:
                # Download single part
                url = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{part_idx}.zip"
                zip_path = os.path.join(paths["zip_folder"], f"OpenVid_part{part_idx}.zip")
                
                logger.info(f"Downloading {url}...")
                download_videos.download_file(url, zip_path)
                
                # Verify and extract
                if not download_videos.verify_zip(zip_path):
                    logger.error(f"ZIP verification failed for {zip_path}")
                    continue
                
                logger.info(f"Extracting to {temp_extract_dir}...")
                download_videos.extract_zip(zip_path, temp_extract_dir)
                
                # List videos
                video_files = sorted([
                    f for f in os.listdir(temp_extract_dir)
                    if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
                ])
                
                if limit is not None:
                    video_files = video_files[:limit]
                
                logger.info(f"Processing {len(video_files)} videos from part {part_idx}")
                
                # Process each video immediately
                for video_file in tqdm(video_files, desc=f"Part {part_idx} videos"):
                    video_path = os.path.join(temp_extract_dir, video_file)
                    base_name, success = process_video_full(
                        video_path,
                        vae_encoder,
                        vjepa_encoder,
                        processor,
                        paths,
                    )
                    if success:
                        processed_videos.append(base_name)
                    
                    # Delete raw video immediately
                    if os.path.exists(video_path):
                        os.remove(video_path)
                
                logger.info(f"Part {part_idx} complete. Processed {len(video_files)} videos.")
                
                # Delete zip after processing
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            
            except Exception as e:
                logger.error(f"Error processing part {part_idx}: {e}")
                continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Streaming encoding complete. Processed {len(processed_videos)} videos total.")
    logger.info(f"{'='*60}")
    
    # Encode text captions if requested (do this after all video processing)
    if do_text:
        run_text_encoding_batch(paths, processed_videos)
    
    # Build index
    build_index(paths, processed_videos)
    
    return {"processed": processed_videos}


def run_text_encoding_batch(paths: Dict[str, str], video_ids: list) -> Dict[str, str]:
    """Encode text captions for processed videos."""
    logger.info("Starting text encoding step")
    csv_path = os.path.join(paths["csv_data"], "OpenVid-1M.csv")
    
    if not os.path.exists(csv_path):
        logger.warning(f"CSV not found: {csv_path}. Skipping text encoding.")
        return {}
    
    csv_df = pd.read_csv(csv_path)
    model_name = "t5-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder = TextEncoder(model_name=model_name, device=device)
    
    saved_map = {}
    for idx, row in tqdm(csv_df.iterrows(), total=len(csv_df), desc="Text encoding"):
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


def build_index(paths: Dict[str, str], video_ids: list) -> None:
    """Build index file for processed videos."""
    logger.info("Building indexed dataset")
    index_file = paths["index_file"]
    
    with open(index_file, "w") as outf:
        for video_id in sorted(video_ids):
            vae_file = os.path.join(paths["encoded_vae"], f"{video_id}_vae.npz")
            vjepa_file = os.path.join(paths["encoded_vjepa"], f"{video_id}_vjepa.npz")
            text_file = os.path.join(paths["encoded_text"], f"{video_id}_text.npy")
            
            # Only add if files exist
            if not os.path.exists(vae_file):
                continue
            
            entry = {
                "video_id": video_id,
                "vae": os.path.relpath(vae_file, paths["project_root"]),
                "vjepa": os.path.relpath(vjepa_file, paths["project_root"]) if os.path.exists(vjepa_file) else None,
                "text": os.path.relpath(text_file, paths["project_root"]) if os.path.exists(text_file) else None,
                "fps": 12,
            }
            outf.write(json.dumps(entry) + "\n")
    
    logger.info(f"Wrote index to {index_file} for {len(video_ids)} videos")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Streaming video dataset preparation pipeline")
    parser.add_argument("--root", default=str(Path(__file__).parent.parent.resolve()), help="Project root")
    parser.add_argument("--parts", type=int, default=1, help="Number of parts to process (0..N)")
    parser.add_argument("--limit", type=int, default=None, help="Max videos per part (for testing)")
    parser.add_argument("--no-text", dest="text", action="store_false", default=True, help="Skip text encoding")
    parser.add_argument("--no-vae", dest="vae", action="store_false", default=True, help="Skip VAE encoding")
    parser.add_argument("--no-vjepa", dest="vjepa", action="store_false", default=True, help="Skip VJEPA encoding")
    
    args = parser.parse_args()
    
    parts_range = range(0, max(1, args.parts))
    run_streaming_pipeline(
        root=args.root,
        parts_range=parts_range,
        limit=args.limit,
        do_text=args.text,
        do_vae=args.vae,
        do_vjepa=args.vjepa,
    )
