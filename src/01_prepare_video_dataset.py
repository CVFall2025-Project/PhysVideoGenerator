"""Prepare video dataset end-to-end

Pipeline steps (configurable):
 1. download data
 2. clean videos -> .npz
 3. VAE encode (.npy)
 4. VJEPA encode (.npy)  (optional; requires vjepa encoder available)
 5. Text encoding (captions from CSVs) -> .npy
 6. Produce indexed dataset JSON

The script calls available helper modules in `src/datasets` and `src/encoders`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import logging
from typing import Dict, Optional

import numpy as np
import torch
import pandas as pd
from glob import glob

from src.datasets import download_videos
from src.datasets import clean_videos
from src.encoders.vae_encoder_decoder import VAEEncoder
from src.encoders.vjepa2_encoder import VJEPA2Encoder
from src.encoders.text_caption_enocder import TextEncoder

logger = logging.getLogger("prepare_video_dataset")


def ensure_dirs(root: str) -> Dict[str, str]:
    project_root = os.path.abspath(root)
    paths = {
        "project_root": project_root,
        "zip_folder": os.path.join(project_root, "data", "download"),
        "videos_folder": os.path.join(project_root, "data", "raw_videos"),
        "clean_npz": os.path.join(project_root, "data", "clean_video_npz"),
        "encoded_vae": os.path.join(project_root, "data", "encoded_videos", "vae"),
        "encoded_vjepa": os.path.join(project_root, "data", "encoded_videos", "vjepa"),
        "encoded_text": os.path.join(project_root, "data", "encoded_videos", "text"),
        "csv_data": os.path.join(project_root, "data", "text_csv"),
        "index_file": os.path.join(project_root, "data", "indexed_dataset.jsonl"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def run_download(paths: Dict[str, str], parts_range=range(0, 1)) -> None:
    logger.info("Starting download step")
    download_videos.download_openvid(
        parts_range=parts_range,
        zip_folder=paths["zip_folder"],
        videos_folder=paths["videos_folder"],
        csv_data_folder=paths["csv_data"],
    )


def run_clean(paths: Dict[str, str], target_fps=12, target_size=256, max_frames=16, limit: Optional[int]=None) -> None:
    logger.info("Starting cleaning step")
    processor = clean_videos.VideoProcessor(device="cuda" if torch.cuda.is_available() else "cpu")

    processor.process_folder(
        input_dir=paths["videos_folder"],
        output_dir=paths["clean_npz"],
        limit=limit,
    )


def run_vae_encoding(paths, dtype: torch.dtype = torch.float16) -> Dict[str, str]:
    logger.info("Starting VAE encoding step")
    # vae_encoder_decoder.encode_video(npz_folder_path, output_folder_path, dtype, device)
    model = VAEEncoder("THUDM/CogVideoX-2b", torch_dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu")
    
    saved_map = {}
    for path in glob(os.path.join(paths["clean_npz"], "*_vae.npz")):
        filename = os.path.basename(path)

        video_data = np.load(path)
        vae_encoded_frames = model.encode(video_data)
        # video_data is closed inside encode() method

        # Save encoded tensor
        output_filename = os.path.splitext(filename)[0] + ".npz"
        output_path = os.path.join(paths["encoded_vae"], output_filename)
        arr = vae_encoded_frames.detach().cpu().numpy()
        np.savez_compressed(output_path, arr)
        saved_map[os.path.splitext(filename)[0][:-4]] = output_path
    
    logger.info("VAE encoding complete.")
    return saved_map


def run_vjepa_encoding(paths: Dict[str, str]) -> Dict[str, str]:
    logger.info("Starting VJEPA encoding step")
    model = VJEPA2Encoder(model_name="facebook/vjepa2-vitg-fpc64-256", device="cuda" if torch.cuda.is_available() else "cpu")
    
    saved_map = {}
    for path in glob(os.path.join(paths["clean_npz"], "*_vjepa.npz")):
        filename = os.path.basename(path)

        try:
            video_data = np.load(path)
            vjepa_encoded_frames = model.encode(video_data)
            # video_data is closed inside encode() method

            output_filename = os.path.splitext(filename)[0] + ".npz"
            output_path = os.path.join(paths["encoded_vjepa"], output_filename)
            arr = vjepa_encoded_frames.detach().cpu().numpy()
            np.savez_compressed(output_path, arr)
            saved_map[os.path.splitext(filename)[0][:-6]] = output_path
            logger.info(f"Saved VJEPA encoding for {filename} -> {output_path}")
        except Exception as e:
            logger.warning(f"Failed to encode {filename} with VJEPA: {e}")
    
    logger.info("VJEPA encoding complete.")
    return saved_map


def run_text_encoding(paths: Dict[str, str]) -> Dict[str, str]:
    logger.info("Starting text encoding step")
    csv_df = pd.read_csv(os.path.join(paths["csv_data"], "OpenVid-1M.csv"))

    saved_map = {}
    model_name = "google/t5-v1_1-xxl"

    text_encoder = TextEncoder(model_name=model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    for idx, row in csv_df.iterrows():
        filename = row["video"].split(".")[0]
        text_embedding = text_encoder.encode(row["caption"])
        saved_map[filename] = text_embedding
        output_filename = f"{filename}_text.npy"
        output_path = os.path.join(paths["encoded_text"], output_filename)
        np.save(output_path, text_embedding)
    return saved_map


def build_index(paths: Dict[str, str], vae_map: Dict[str, str], vjepa_map: Dict[str, str], text_map: Dict[str, str], fps=12) -> None:
    logger.info("Building indexed dataset")
    index_file = paths["index_file"]
    # Open JSONL file and append entries
    with open(index_file, "w") as outf:
        for fname in sorted(os.listdir(paths["clean_npz"])):
            if not fname.endswith('_vae.npz'):
                continue
            base = os.path.splitext(fname)[0][:-4]  # remove _vae
            entry = {
                "video_id": base,
                "vae": os.path.relpath(vae_map.get(base, "")),
                "vjepa": os.path.relpath(vjepa_map.get(base, "")),
                "text": os.path.relpath(text_map.get(base, "")),
                "fps": fps,
            }
            outf.write(json.dumps(entry) + "\n")
    logger.info(f"Wrote index to {index_file}")


def main(root: str, do_steps: Dict[str, bool], parts_range=range(0, 1), limit: Optional[int] = None):
    paths = ensure_dirs(root)
    if do_steps.get('download', True):
        run_download(paths, parts_range=parts_range)
    if do_steps.get('clean', True):
        run_clean(paths, limit=limit)
    vae_map = {}
    if do_steps.get('vae', True):
        vae_map = run_vae_encoding(paths)
    vjepa_map = {}
    if do_steps.get('vjepa', True):
        vjepa_map = run_vjepa_encoding(paths)
    text_map = {}
    if do_steps.get('text', True):
        text_map = run_text_encoding(paths)

    # Finally, build index if only all steps are done
    if all(do_steps.get(step, True) for step in ['vae', 'vjepa', 'text']):
        build_index(paths, vae_map, vjepa_map, text_map)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="End-to-end prepare video dataset pipeline")
    parser.add_argument("--root", default=str(Path(__file__).parent.parent.resolve()), help="Project root")
    parser.add_argument("--no-download", dest="download", action="store_false", help="Skip download step")
    parser.add_argument("--no-clean", dest="clean", action="store_false", help="Skip cleaning step")
    parser.add_argument("--no-vae", dest="vae", action="store_false", help="Skip VAE encoding step")
    parser.add_argument("--no-vjepa", dest="vjepa", action="store_false", help="Skip VJEPA encoding step")
    parser.add_argument("--no-text", dest="text", action="store_false", help="Skip text encoding step")
    parser.add_argument("--parts", type=int, default=1, help="Number of OpenVid parts to download (0..N) will set range(0, parts)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process during cleaning (for testing)")

    args = parser.parse_args()

    steps = {
        'download': args.download,
        'clean': args.clean,
        'vae': args.vae,
        'vjepa': args.vjepa,
        'text': args.text,
    }

    parts_range = range(0, max(1, args.parts))
    main(root=args.root, do_steps=steps, parts_range=parts_range, limit=args.limit)
