import argparse
import os
from glob import glob
import torch
import imageio
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms
import numpy as np
from tqdm import tqdm


def encode_video(npz_folder_path, output_folder_path, dtype, device):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and encodes video frames from .npz files.

    Parameters:
    - npz_folder_path (str): Path to folder containing .npz files.
    - output_folder_path (str): Path to save encoded tensors.
    - dtype (torch.dtype): The data type for computation.
    - device (str): The device to use for computation (e.g., "cuda" or "cpu").

    Returns:
    - dict: Dictionary with npz filenames as keys and encoded tensors as values.
    """
    model = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=dtype).to(device)

    model.enable_slicing()
    model.enable_tiling()

    # Find all .npz files in the folder
    npz_files = sorted(glob(os.path.join(npz_folder_path, "*.npz")))
    
    if not npz_files:
        print(f"No .npz files found in {npz_folder_path}")
        return {}

    encoded_results = {}

    for npz_path in tqdm(npz_files, desc="Encoding videos using VAE:"):
        try:
            filename = os.path.basename(npz_path)
            
            # Load the npz file
            video_data = np.load(npz_path)
            
            # Extract frames from the .npz file (assumes 'frames' key or first array)
            if 'frames' in video_data.files:
                frames_array = video_data['frames']
            else:
                # Use the first available array if 'frames' key doesn't exist
                frames_array = video_data[video_data.files[0]]
            
            # Convert frames to tensors
            frames = [transforms.ToTensor()(frame) for frame in frames_array]
            video_data.close()

            # Stack and permute frames for the VAE model
            frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)

            # Encode using the VAE model
            with torch.no_grad():
                encoded_frames = model.encode(frames_tensor)[0].sample()
            
            # Save encoded tensor
            output_filename = os.path.splitext(filename)[0] + "_encoded.pt"
            output_path = os.path.join(output_folder_path, output_filename)
            torch.save(encoded_frames, output_path)
            
            encoded_results[filename] = encoded_frames
            
        except Exception as e:
            print(f"✗ Error encoding {filename}: {e}")
            continue

    print(f"\nEncoding complete. Encoded {len(encoded_results)}/{len(npz_files)} files.")
    return encoded_results


def decode_video(encoded_tensor_path, dtype, device):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and decodes the encoded video frames.

    Parameters:
    - encoded_tensor_path (str): The path to the encoded tensor file.
    - dtype (torch.dtype): The data type for computation.
    - device (str): The device to use for computation (e.g., "cuda" or "cpu").

    Returns:
    - torch.Tensor: The decoded video frames.
    """
    model = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=dtype).to(device)
    encoded_frames = torch.load(encoded_tensor_path, weights_only=True).to(device).to(dtype)
    with torch.no_grad():
        decoded_frames = model.decode(encoded_frames).sample
    return decoded_frames


def save_video(tensor, output_path):
    """
    Saves the video frames to a video file.

    Parameters:
    - tensor (torch.Tensor): The video frames' tensor.
    - output_path (str): The path to save the output video.
    """
    tensor = tensor.to(dtype=torch.float32)
    frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)
    writer = imageio.get_writer(output_path + "/output.mp4", fps=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVideoX encode/decode demo")
    parser.add_argument(
        "--model_path", type=str, required=True, help="The path to the CogVideoX model"
    )
    parser.add_argument("--video_path", type=str, help="The path to the video file (for encoding)")
    parser.add_argument(
        "--encoded_path", type=str, help="The path to the encoded tensor file (for decoding)"
    )
    parser.add_argument(
        "--output_path", type=str, default=".", help="The path to save the output file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["encode", "decode", "both"],
        required=True,
        help="Mode: encode, decode, or both",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="The data type for computation (e.g., 'float16' or 'bfloat16')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to use for computation (e.g., 'cuda' or 'cpu')",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    if args.mode == "encode":
        assert args.video_path, "Video path must be provided for encoding."
        encoded_output = encode_video(args.model_path, args.video_path, dtype, device)
        torch.save(encoded_output, args.output_path + "/encoded.pt")
        print(
            f"Finished encoding the video to a tensor, save it to a file at {encoded_output}/encoded.pt"
        )
    elif args.mode == "decode":
        assert args.encoded_path, "Encoded tensor path must be provided for decoding."
        decoded_output = decode_video(args.model_path, args.encoded_path, dtype, device)
        save_video(decoded_output, args.output_path)
        print(
            f"Finished decoding the video and saved it to a file at {args.output_path}/output.mp4"
        )
    elif args.mode == "both":
        assert args.video_path, "Video path must be provided for encoding."
        encoded_output = encode_video(args.model_path, args.video_path, dtype, device)
        torch.save(encoded_output, args.output_path + "/encoded.pt")
        decoded_output = decode_video(
            args.model_path, args.output_path + "/encoded.pt", dtype, device
        )
        save_video(decoded_output, args.output_path)
