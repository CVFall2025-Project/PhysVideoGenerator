"""iterative_encode_curated.py

Iterate OpenVid-1M parts on HuggingFace, stream-extract only the videos listed
in the curated CSV via HTTP range requests, VAE+VJEPA encode each one, and
delete the mp4 immediately. Text encoding is decoupled — it works straight from
the curated CSV's caption column and can be run before, during, or after video
prep. Idempotent throughout: safe to re-run to pick up where a previous run
left off.

Requires:
    pip install remotezip

Usage:
    # Recommended: text-encode all captions first (cheap, doesn't need videos),
    # then run video encoding without T5 loaded.
    python src/iterative_encode_curated.py --only_text
    python src/iterative_encode_curated.py --skip_text

    # Or everything in one shot
    python src/iterative_encode_curated.py

    # Range of parts / index-only helpers
    python src/iterative_encode_curated.py --start_part 5 --end_part 20
    python src/iterative_encode_curated.py --only_index
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import clean_videos
from src.encoders.vae_encoder_decoder import VAEEncoder
from src.encoders.vjepa2_encoder import VJEPA2Encoder
from src.encoders.text_caption_enocder import TextEncoder

try:
    from remotezip import RemoteZip
except ImportError:
    print("Please install remotezip:  pip install remotezip", file=sys.stderr)
    sys.exit(1)


NUM_PARTS = 187
BASE_URL = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main"
CURATED_CSV = "data/text_csv/curated_OpenVid-1M.csv"
RAW_DIR = "data/raw_videos"
ENCODED_ROOT = "data/encoded_videos"
INDEX_FILE = "data/indexed_dataset.json"


def hf_session() -> requests.Session:
    """requests.Session with an HF auth header if a token is available."""
    session = requests.Session()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            token = None
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def already_encoded(video_id: str) -> bool:
    return (
        os.path.exists(f"{ENCODED_ROOT}/vae/{video_id}_vae.npz")
        and os.path.exists(f"{ENCODED_ROOT}/vjepa/{video_id}_vjepa.npz")
    )


def encode_video(video_path: str, video_id: str, vae, vjepa, processor) -> bool:
    try:
        frames = processor.load_video(video_path)
        outputs = processor.process_video(frames)

        # Store as fp16 — VJEPA runs the model in fp32 by default so its output
        # is fp32 (~10 MB compressed); training casts to bf16 anyway.
        vae_tensor = torch.from_numpy(outputs["vae"]).to(vae.device).to(vae.torch_dtype)
        vae_encoded = vae.encode(vae_tensor).detach().cpu().numpy().astype(np.float16)
        np.savez_compressed(f"{ENCODED_ROOT}/vae/{video_id}_vae.npz", vae_encoded)

        vjepa_tensor = torch.from_numpy(outputs["vjepa"]).to(vjepa.device).to(vjepa.torch_dtype)
        vjepa_encoded = vjepa.encode(vjepa_tensor).detach().cpu().numpy().astype(np.float16)
        np.savez_compressed(f"{ENCODED_ROOT}/vjepa/{video_id}_vjepa.npz", vjepa_encoded)
        return True
    except Exception as e:
        print(f"  ! encode error for {video_id}: {e}", file=sys.stderr)
        return False


def process_part(part_num: int, curated: set, vae, vjepa, processor, session) -> tuple[int, int]:
    """Peek at one OpenVid part remotely, extract curated intersection, encode.
    Returns (kept, encoded) — number range-fetched and number successfully encoded."""
    url = f"{BASE_URL}/OpenVid_part{part_num}.zip"
    print(f"\n=== Part {part_num} ===")
    try:
        with RemoteZip(url, session=session) as z:
            names = z.namelist()
            wanted = [
                n for n in names
                if n.lower().endswith(".mp4")
                and os.path.basename(n) in curated
                and not already_encoded(os.path.splitext(os.path.basename(n))[0])
            ]

            if not wanted:
                total_intersect = sum(1 for n in names if os.path.basename(n) in curated)
                if total_intersect:
                    print(f"  All {total_intersect} curated videos here are already encoded. Skip.")
                else:
                    print(f"  No curated intersection in {len(names)} entries. Skip.")
                return 0, 0

            print(f"  {len(wanted)} new curated videos to fetch from part {part_num}.")

            os.makedirs(RAW_DIR, exist_ok=True)
            encoded_count = 0
            for name in tqdm(wanted, desc=f"  part {part_num}", leave=False):
                bn = os.path.basename(name)
                video_id = os.path.splitext(bn)[0]
                out_path = os.path.join(RAW_DIR, bn)

                try:
                    # Range-fetch just this mp4 from the remote zip
                    try:
                        with z.open(name) as src, open(out_path, "wb") as dst:
                            dst.write(src.read())
                    except Exception as e:
                        print(f"  ! range fetch failed for {bn}: {e}", file=sys.stderr)
                        continue

                    if encode_video(out_path, video_id, vae, vjepa, processor):
                        encoded_count += 1
                finally:
                    # Always clean up the raw mp4 — even on partial fetch, encode
                    # failure, or KeyboardInterrupt. Encoded npz is already on Drive.
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass

            return len(wanted), encoded_count
    except Exception as e:
        print(f"  ! part {part_num} failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 0, 0


def batch_text_encode():
    """Text-encode every caption in the curated CSV. Independent of video
    encoding: works from the CSV alone, so it can be run before, during, or
    after video prep. Idempotent — skips captions already on disk."""
    print("\n=== Batch text encoding ===")
    text_dir = f"{ENCODED_ROOT}/text"
    os.makedirs(text_dir, exist_ok=True)

    df = pd.read_csv(CURATED_CSV)
    df["video_id"] = df["video"].str.replace(".mp4", "", regex=False)

    done_text = {fn[:-9] for fn in os.listdir(text_dir) if fn.endswith("_text.npy")}
    df_todo = df[~df["video_id"].isin(done_text)]

    if len(df_todo) == 0:
        print("  All curated captions already encoded.")
        return

    print(f"  {len(df_todo)} of {len(df)} curated captions still need encoding.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = TextEncoder(model_name="google/t5-v1_1-xxl", device=device)

    for _, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="  text"):
        video_id = row["video_id"]
        try:
            emb = encoder.encode(row["caption"])
            arr = emb.detach().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)
            np.save(f"{text_dir}/{video_id}_text.npy", arr.astype(np.float16))
        except Exception as e:
            print(f"  ! text encode error for {video_id}: {e}", file=sys.stderr)

    del encoder
    gc.collect()
    torch.cuda.empty_cache()


def build_index():
    """Rebuild the index from scratch — only entries with all three modalities present."""
    print("\n=== Building index ===")
    entries = []
    vae_dir = Path(f"{ENCODED_ROOT}/vae")
    vjepa_dir = Path(f"{ENCODED_ROOT}/vjepa")
    text_dir = Path(f"{ENCODED_ROOT}/text")
    for vae_file in sorted(vae_dir.glob("*_vae.npz")):
        video_id = vae_file.name[:-8]
        vjepa_file = vjepa_dir / f"{video_id}_vjepa.npz"
        text_file = text_dir / f"{video_id}_text.npy"
        if vjepa_file.exists() and text_file.exists():
            entries.append({
                "video_id": video_id,
                "vae": str(vae_file.resolve()),
                "vjepa": str(vjepa_file.resolve()),
                "text": str(text_file.resolve()),
            })
    with open(INDEX_FILE, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"  Index: {len(entries)} entries -> {INDEX_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_part", type=int, default=0)
    parser.add_argument("--end_part", type=int, default=NUM_PARTS)
    parser.add_argument("--skip_text", action="store_true", help="Defer text encoding.")
    parser.add_argument("--only_text", action="store_true",
                        help="Skip video encoding; only text-encode the curated captions.")
    parser.add_argument("--only_index", action="store_true", help="Only rebuild the index and exit.")
    args = parser.parse_args()

    if args.only_index:
        build_index()
        return

    if args.only_text:
        batch_text_encode()
        build_index()
        return

    curated = set(pd.read_csv(CURATED_CSV)["video"].astype(str))
    print(f"Curated set: {len(curated)} videos.")

    for sub in ("vae", "vjepa", "text"):
        os.makedirs(f"{ENCODED_ROOT}/{sub}", exist_ok=True)

    os.makedirs(RAW_DIR, exist_ok=True)
    stale = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".mp4")]
    for f in stale:
        os.remove(os.path.join(RAW_DIR, f))
    if stale:
        print(f"Cleared {len(stale)} stale mp4s from {RAW_DIR}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    vae = VAEEncoder("maxin-cn/Latte-1", torch_dtype=torch.float16, device=device)
    vjepa = VJEPA2Encoder(model_name="facebook/vjepa2-vitg-fpc64-256",
                          torch_dtype=torch.float16, device=device)
    processor = clean_videos.VideoProcessor(device=device)
    session = hf_session()

    total_kept = 0
    total_encoded = 0
    for part_num in range(args.start_part, args.end_part):
        kept, encoded = process_part(part_num, curated, vae, vjepa, processor, session)
        total_kept += kept
        total_encoded += encoded

        cumulative = len([f for f in os.listdir(f"{ENCODED_ROOT}/vae") if f.endswith("_vae.npz")])
        print(f"  Cumulative encoded: {cumulative}/{len(curated)}")

        if cumulative >= len(curated):
            print("\nAll curated videos encoded. Stopping early.")
            break

        gc.collect()
        torch.cuda.empty_cache()

    # Free vision encoders before loading T5
    del vae
    del vjepa
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    if not args.skip_text:
        batch_text_encode()

    build_index()

    print(f"\n=== Done ===")
    print(f"Videos kept this run:    {total_kept}")
    print(f"Videos encoded this run: {total_encoded}")


if __name__ == "__main__":
    main()
