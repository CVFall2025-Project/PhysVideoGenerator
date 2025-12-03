"""download_videos.py

Utilities to download and extract OpenVid parts and associated CSVs.

Usage:
    from src.datasets.download_videos import download_openvid

    download_openvid(
        parts_range=range(0, 1),
        zip_folder="/path/to/download",
        videos_folder="/path/to/videos",
        csv_data_folder="/path/to/data",
    )

Or run as a script:
    python -m src.datasets.download_videos

The module uses `curl` and `unzip` (both available on macOS and Linux). It follows redirects, verifies zip integrity with `unzip -t`, and attempts to download split parts if the full archive fails.
"""
from __future__ import annotations

import os
import subprocess
import shutil
import glob
from typing import Iterable, List


def _run(cmd: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=capture_output)


def download_file(url: str, dest: str) -> None:
    """Download a file using curl and save to dest. Follows redirects."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cmd = ["curl", "-L", "-o", dest, url]
    _run(cmd)


def verify_zip(path: str) -> bool:
    """Verify ZIP integrity using `unzip -t`. Returns True if valid."""
    try:
        _run(["unzip", "-t", path], capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def combine_split_parts(zip_folder: str, part_prefix: str, output_zip: str) -> None:
    """Concatenate split parts matching `part_prefix` into output_zip using system cat.
    Example: combine_split_parts(zip_folder, 'OpenVid_part0_part', 'OpenVid_part0.zip')
    """
    pattern = os.path.join(zip_folder, part_prefix + "*")
    parts = sorted(glob.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"No split parts found for pattern: {pattern}")
    # Use binary cat via Python to avoid shell specifics
    with open(output_zip, "wb") as out_f:
        for p in parts:
            print(f"Appending {p} -> {output_zip}")
            with open(p, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f)


def extract_zip(zip_path: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    _run(["unzip", "-j", zip_path, "-d", dest_dir])


def download_openvid(
    parts_range: Iterable[int] = range(0, 1),
    zip_folder: str | None = None,
    videos_folder: str | None = None,
    csv_data_folder: str | None = None,
    error_log_path: str | None = None,
    retry_with_parts: bool = True,
) -> None:
    """Download OpenVid zip parts, extract videos and download CSV metadata.

    - parts_range: iterable of part indices (e.g., range(0, 1))
    - zip_folder: directory to store zip files
    - videos_folder: directory to extract videos to
    - csv_data_folder: directory to save CSV metadata files
    - error_log_path: file to append errors to
    - retry_with_parts: when a full zip download fails, try downloading split parts
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if zip_folder is None:
        zip_folder = os.path.join(project_root, "data", "zip_files")
    if videos_folder is None:
        videos_folder = os.path.join(project_root, "data", "raw_videos")
    if csv_data_folder is None:
        csv_data_folder = os.path.join(project_root, "data", "text_csv")
    if error_log_path is None:
        error_log_path = os.path.join(zip_folder, "download_log.txt")

    os.makedirs(zip_folder, exist_ok=True)
    os.makedirs(videos_folder, exist_ok=True)
    os.makedirs(csv_data_folder, exist_ok=True)

    base_url = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main"

    for i in parts_range:
        url = f"{base_url}/OpenVid_part{i}.zip"
        file_path = os.path.join(zip_folder, f"OpenVid_part{i}.zip")

        # If file exists, verify integrity
        if os.path.exists(file_path):
            print(f"file {file_path} exists. Checking integrity...")
            if verify_zip(file_path):
                print("ZIP file is valid. Skipping download.")
                continue
            else:
                print("ZIP file is corrupted. Removing and re-downloading...")
                os.remove(file_path)

        try:
            print(f"Downloading {url} to {file_path}...")
            download_file(url, file_path)

            # verify and extract
            if verify_zip(file_path):
                print("ZIP verified. Extracting...")
                extract_zip(file_path, videos_folder)
                print("Extraction complete.")
            else:
                raise RuntimeError("ZIP verification failed after download")

        except Exception as e:  # fallback to split-parts strategy
            err_msg = f"file {url} download/extract failed: {e}\n"
            print(err_msg)
            with open(error_log_path, "a") as ef:
                ef.write(err_msg)

            if not retry_with_parts:
                continue

            # try split parts
            part_urls = [
                f"{base_url}/OpenVid_part{i}_partaa",
                f"{base_url}/OpenVid_part{i}_partab",
            ]

            part_files: List[str] = []
            for part_url in part_urls:
                part_file_path = os.path.join(zip_folder, os.path.basename(part_url))
                if os.path.exists(part_file_path):
                    print(f"file {part_file_path} exists. Skipping part download.")
                    part_files.append(part_file_path)
                    continue
                try:
                    download_file(part_url, part_file_path)
                    part_files.append(part_file_path)
                except Exception as part_e:
                    part_err = f"file {part_url} download failed: {part_e}\n"
                    print(part_err)
                    with open(error_log_path, "a") as ef:
                        ef.write(part_err)

            # combine parts if any downloaded
            if part_files:
                try:
                    combined_zip = file_path
                    prefix = f"OpenVid_part{i}_part"
                    combine_split_parts(zip_folder, prefix, combined_zip)
                    if verify_zip(combined_zip):
                        extract_zip(combined_zip, videos_folder)
                        print("Successfully extracted files from combined archive.")
                    else:
                        raise RuntimeError("Combined ZIP failed verification")
                except Exception as comb_e:
                    comb_err = f"Combined ZIP failed: {comb_e}\n"
                    print(comb_err)
                    with open(error_log_path, "a") as ef:
                        ef.write(comb_err)

    # Download CSV metadata files (train splits)
    data_urls = [
        f"{base_url}/data/train/OpenVid-1M.csv",
    ]
    for data_url in data_urls:
        data_path = os.path.join(csv_data_folder, os.path.basename(data_url))
        try:
            print(f"Downloading {data_url} to {data_path}...")
            download_file(data_url, data_path)
        except Exception as e:
            msg = f"Failed to download metadata {data_url}: {e}\n"
            print(msg)
            with open(error_log_path, "a") as ef:
                ef.write(msg)
    
    delete_command = "rm -rf " + zip_folder + "/*.zip"
    os.system(delete_command)

    print("Download/openvid routine finished.")


if __name__ == "__main__":
    # simple CLI wrapper
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    zip_folder_default = os.path.join(project_root, "data", "zip_files")
    videos_folder_default = os.path.join(project_root, "data", "raw_videos")
    data_folder_default = os.path.join(project_root, "data", "text_csv")

    download_openvid(
        parts_range=range(0, 1),
        zip_folder=zip_folder_default,
        videos_folder=videos_folder_default,
        csv_data_folder=data_folder_default,
    )
