# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

def forward_vjepa_video(cleaned_video_npz, model_hf, hf_transform):
    # Run a sample inference with VJEPA
    with torch.inference_mode():
        # Read and pre-process the image
        video = torch.from_numpy(cleaned_video_npz).permute(0, 3, 1, 2)  # T x C x H x W
        x_hf = hf_transform(video, return_tensors="pt")["pixel_values_videos"].to("cuda")
        # Extract the patch-wise features from the last layer
        out_patch_features_hf = model_hf.get_vision_features(x_hf)

    return out_patch_features_hf