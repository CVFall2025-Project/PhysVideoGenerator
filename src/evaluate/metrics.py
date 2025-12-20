"""Metrics for video evaluation.

This module provides functions to compute temporal LPIPS (t-LPIPS) metric
and optical flow consistency metric for evaluating video quality and temporal consistency.
"""

from __future__ import annotations

import os
import tempfile
import shutil
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import lpips
import cv2

try:
    import decord
    decord.bridge.set_bridge("native")
except Exception:
    decord = None


def load_video_for_lpips(path: str, size: int = 224) -> torch.Tensor:
    """Load video and prepare it for LPIPS computation.
    
    Args:
        path: Path to video file
        size: Target size for resizing (default: 224)
    
    Returns:
        frames: Tensor of shape [T, 3, H, W] normalized to [-1, 1]
    
    Raises:
        ImportError: If decord is not available
        FileNotFoundError: If video file doesn't exist
    """
    if decord is None:
        raise ImportError("decord is required for video loading. Please install it.")
    
    vr = decord.VideoReader(path)
    frames = vr.get_batch(range(len(vr)))  # T x H x W x 3, uint8

    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # TCHW

    transform = T.Compose([
        T.Resize((size, size)),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])
    ])

    frames = transform(frames)  # apply per-frame transform
    return frames  # [T, 3, H, W]


class TLPIPS:
    """Temporal LPIPS metric calculator.
    
    Computes LPIPS between consecutive frames and returns the average.
    """
    
    def __init__(self, net: str = 'vgg', device: str = 'cuda'):
        """Initialize the LPIPS model.
        
        Args:
            net: Network type for LPIPS ('vgg' or 'alex')
            device: Device to run computation on ('cuda' or 'cpu')
        """
        self.device = device
        self.lpips_model = lpips.LPIPS(net=net).to(device).eval()
    
    def compute(self, frames: torch.Tensor) -> float:
        """Compute temporal LPIPS score.
        
        Args:
            frames: torch tensor [T, 3, H, W], normalized to [-1, 1]
        
        Returns:
            Average LPIPS score across consecutive frame pairs
        """
        frames = frames.to(self.device)
        T_len = frames.shape[0]

        if T_len < 2:
            raise ValueError("Video must have at least 2 frames to compute t-LPIPS")

        scores = []
        with torch.no_grad():
            for t in range(T_len - 1):
                d = self.lpips_model(
                    frames[t].unsqueeze(0),
                    frames[t+1].unsqueeze(0)
                )
                scores.append(d.item())

        return sum(scores) / len(scores)


def compute_t_lpips(frames: torch.Tensor, net: str = 'vgg', device: str = 'cuda') -> float:
    """Compute temporal LPIPS metric for a video.
    
    Convenience function that creates a TLPIPS instance and computes the score.
    
    Args:
        frames: torch tensor [T, 3, H, W], normalized to [-1, 1]
        net: Network type for LPIPS ('vgg' or 'alex')
        device: Device to run computation on ('cuda' or 'cpu')
    
    Returns:
        Average LPIPS score across consecutive frame pairs
    """
    calculator = TLPIPS(net=net, device=device)
    return calculator.compute(frames)


def calculate_dense_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[float, np.ndarray]:
    """Calculate the dense optical flow between two frames using the Farneback method.
    
    Args:
        frame1: First frame as numpy array (H, W, 3) in BGR format
        frame2: Second frame as numpy array (H, W, 3) in BGR format
    
    Returns:
        Tuple of (average_magnitude, flow) where:
        - average_magnitude: Average magnitude of flow vectors
        - flow: 2-channel array (x and y components of motion vectors)
    
    Raises:
        ValueError: If frames are None or have incompatible shapes
    """
    if frame1 is None or frame2 is None:
        raise ValueError("Error: one or both frames are None")
    
    if frame1.shape != frame2.shape:
        raise ValueError(f"Frame shapes don't match: {frame1.shape} vs {frame2.shape}")
    
    # Convert to grayscale (required for Farneback)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow (Farneback parameters are common defaults)
    # The 'flow' result is a 2-channel array (x and y components of the motion vector)
    flow = cv2.calcOpticalFlowFarneback(
        prvs,
        next_frame,
        None,
        0.5,   # pyr_scale
        3,     # levels
        15,    # winsize
        3,     # iterations
        5,     # poly_n
        1.2,   # poly_sigma
        0      # flags
    )
    
    # flow[:,:,0] is the u component (horizontal motion)
    # flow[:,:,1] is the v component (vertical motion)
    
    # Calculate the magnitude of the flow vectors: sqrt(u^2 + v^2)
    # This magnitude is the *speed* of motion at each pixel.
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Calculate the average magnitude across the entire frame
    average_magnitude = np.mean(magnitude)
    
    return average_magnitude, flow


def calculate_optical_flow_consistency_score(
    video_path: str,
    max_frames: Optional[int] = None,
    temp_dir: Optional[str] = None,
    use_decord: bool = True
) -> float:
    """Calculate the average optical flow magnitude over all consecutive frame pairs.
    
    This metric measures temporal consistency by computing the average magnitude
    of optical flow between consecutive frames. Lower scores indicate smoother
    transitions (higher consistency), while higher scores indicate more drastic
    motion (lower consistency).
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process (None for all frames)
        temp_dir: Temporary directory for frame extraction (None for auto-generated)
        use_decord: Whether to use decord for video loading (faster) or cv2
    
    Returns:
        Average optical flow magnitude (pixels/frame)
    """
    flow_magnitudes = []
    cleanup_temp = False
    
    try:
        if use_decord and decord is not None:
            # Use decord for faster loading
            vr = decord.VideoReader(video_path)
            num_frames = len(vr)
            if max_frames:
                num_frames = min(num_frames, max_frames)
            
            frames = vr.get_batch(range(num_frames))  # T x H x W x 3, uint8
            frames_np = frames.asnumpy()  # Convert to numpy
            
            # Calculate optical flow between consecutive pairs
            for i in range(num_frames - 1):
                frame1 = frames_np[i]  # H x W x 3, RGB
                frame2 = frames_np[i + 1]
                
                # Convert RGB to BGR for OpenCV
                frame1_bgr = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
                frame2_bgr = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
                
                try:
                    avg_mag, _ = calculate_dense_optical_flow(frame1_bgr, frame2_bgr)
                    flow_magnitudes.append(avg_mag)
                except Exception as e:
                    continue
        
        else:
            # Fallback to cv2 with temporary frame extraction
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp(prefix='optical_flow_')
                cleanup_temp = True
            else:
                os.makedirs(temp_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file: {video_path}")
            
            frame_paths = []
            frame_count = 0
            
            # Extract and save frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_path = os.path.join(temp_dir, f"frame_{frame_count:04d}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
            
            cap.release()
            
            # Calculate optical flow between consecutive pairs
            for i in range(len(frame_paths) - 1):
                frame1 = cv2.imread(frame_paths[i])
                frame2 = cv2.imread(frame_paths[i + 1])
                
                if frame1 is None or frame2 is None:
                    continue
                
                try:
                    avg_mag, _ = calculate_dense_optical_flow(frame1, frame2)
                    flow_magnitudes.append(avg_mag)
                except Exception as e:
                    continue
            
            # Clean up temporary directory
            if cleanup_temp and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    except Exception as e:
        if cleanup_temp and temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise RuntimeError(f"Error processing video {video_path}: {e}")
    
    if not flow_magnitudes:
        return 0.0
    
    # Calculate the final score
    average_flow_consistency_score = np.mean(flow_magnitudes)
    
    return average_flow_consistency_score


# VideoPhy metrics - require model checkpoints
class VideoPhyEvaluator:
    """VideoPhy (original) evaluator for SA and PC scores.
    
    Uses entailment-based inference that outputs scores 0-1.
    Requires checkpoint from: https://huggingface.co/videophysics/videocon_physics
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', batch_size: int = 16):
        """Initialize VideoPhy evaluator.
        
        Args:
            checkpoint_path: Path to VideoPhy checkpoint directory
            device: Device to run computation on
            batch_size: Batch size for inference
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of model components."""
        if self._initialized:
            return
        
        try:
            import sys
            import os
            # Add VideoPhy paths to sys.path
            videophy_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'VideoPhy', 'videocon', 'training', 'pipeline_video'
            )
            if videophy_path not in sys.path:
                sys.path.insert(0, videophy_path)
            
            from transformers.models.llama.tokenization_llama import LlamaTokenizer
            from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
            from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
            from data_utils.xgpt3_dataset import MultiModalDataset
            from utils import batchify
            from torch.utils.data import DataLoader
            import torch.nn as nn
            
            self.tokenizer = LlamaTokenizer.from_pretrained(self.checkpoint_path)
            image_processor = MplugOwlImageProcessor.from_pretrained(self.checkpoint_path)
            self.processor = MplugOwlProcessor(image_processor, self.tokenizer)
            
            self.model = MplugOwlForConditionalGeneration.from_pretrained(
                self.checkpoint_path,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            self.model.eval()
            
            self.softmax = nn.Softmax(dim=2)
            self._initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VideoPhy model: {e}")
    
    def _get_entail_score(self, logits, input_ids):
        """Extract entailment score from model logits."""
        logits = self.softmax(logits)
        token_id_yes = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        token_id_no = self.tokenizer.encode('No', add_special_tokens=False)[0]
        
        entailment = []
        for j in range(len(logits)):
            for i in range(len(input_ids[j])):
                if input_ids[j][i] == self.tokenizer.pad_token_id:
                    i = i - 1
                    break
                elif i == len(input_ids[j]) - 1:
                    break
            score = logits[j][i][token_id_yes] / (logits[j][i][token_id_yes] + logits[j][i][token_id_no])
            entailment.append(score)
        return torch.stack(entailment)
    
    def compute_sa(self, video_path: str, caption: str) -> float:
        """Compute Semantic Adherence (SA) score.
        
        Args:
            video_path: Path to video file
            caption: Text caption describing the video
        
        Returns:
            SA score between 0 and 1 (higher is better)
        """
        self._initialize()
        
        # Prepare prompt
        prompt = f'''
        The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        Human: <|video|>
        Human: Does this video entail the description: "{caption}"?
        AI: 
        '''
        
        # Create temporary CSV for dataset
        import tempfile
        import pandas as pd
        import csv
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['videopath', 'caption'])
            writer.writerow([video_path, prompt])
            temp_csv = f.name
        
        try:
            from data_utils.xgpt3_dataset import MultiModalDataset
            from utils import batchify
            from torch.utils.data import DataLoader
            
            dataset = MultiModalDataset(temp_csv, self.tokenizer, self.processor, max_length=256, loss_objective='sequential')
            dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, collate_fn=batchify)
            
            with torch.no_grad():
                for inputs in dataloader:
                    for k, v in inputs.items():
                        if torch.is_tensor(v):
                            if v.dtype == torch.float:
                                inputs[k] = v.bfloat16()
                            inputs[k] = inputs[k].to(self.device)
                    
                    outputs = self.model(
                        pixel_values=inputs['pixel_values'],
                        video_pixel_values=inputs['video_pixel_values'],
                        labels=None,
                        num_images=inputs['num_images'],
                        num_videos=inputs['num_videos'],
                        input_ids=inputs['input_ids'],
                        non_padding_mask=inputs['non_padding_mask'],
                        non_media_mask=inputs['non_media_mask'],
                        prompt_mask=inputs['prompt_mask']
                    )
                    logits = outputs['logits']
                    entail_scores = self._get_entail_score(logits, inputs['input_ids'])
                    return entail_scores[0].item()
        finally:
            os.unlink(temp_csv)
    
    def compute_pc(self, video_path: str) -> float:
        """Compute Physical Commonsense (PC) score.
        
        Args:
            video_path: Path to video file
        
        Returns:
            PC score between 0 and 1 (higher is better)
        """
        self._initialize()
        
        # Prepare prompt
        prompt = '''
            The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
            Human: <|video|>
            Human: Does this video follow the physical laws?
            AI: 
        '''
        
        # Create temporary CSV for dataset
        import tempfile
        import csv
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['videopath', 'caption'])
            writer.writerow([video_path, prompt])
            temp_csv = f.name
        
        try:
            from data_utils.xgpt3_dataset import MultiModalDataset
            from utils import batchify
            from torch.utils.data import DataLoader
            
            dataset = MultiModalDataset(temp_csv, self.tokenizer, self.processor, max_length=256, loss_objective='sequential')
            dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, collate_fn=batchify)
            
            with torch.no_grad():
                for inputs in dataloader:
                    for k, v in inputs.items():
                        if torch.is_tensor(v):
                            if v.dtype == torch.float:
                                inputs[k] = v.bfloat16()
                            inputs[k] = inputs[k].to(self.device)
                    
                    outputs = self.model(
                        pixel_values=inputs['pixel_values'],
                        video_pixel_values=inputs['video_pixel_values'],
                        labels=None,
                        num_images=inputs['num_images'],
                        num_videos=inputs['num_videos'],
                        input_ids=inputs['input_ids'],
                        non_padding_mask=inputs['non_padding_mask'],
                        non_media_mask=inputs['non_media_mask'],
                        prompt_mask=inputs['prompt_mask']
                    )
                    logits = outputs['logits']
                    entail_scores = self._get_entail_score(logits, inputs['input_ids'])
                    return entail_scores[0].item()
        finally:
            os.unlink(temp_csv)


class VideoPhy2Evaluator:
    """VideoPhy-2 evaluator for SA and PC scores.
    
    Uses generation-based inference that outputs scores 1-5.
    Requires checkpoint from: https://huggingface.co/videophysics/videophy_2_auto
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', num_frames: int = 32):
        """Initialize VideoPhy-2 evaluator.
        
        Args:
            checkpoint_path: Path to VideoPhy-2 checkpoint directory
            device: Device to run computation on
            num_frames: Number of frames to extract from video
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_frames = num_frames
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._initialized = False
        
        self.generate_kwargs = {
            'do_sample': False,
            'top_k': 1,
            'temperature': 0.001,
            'max_length': 256,
        }
        
        self.num_map = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5
        }
    
    def _initialize(self):
        """Lazy initialization of model components."""
        if self._initialized:
            return
        
        try:
            import sys
            import os
            # Add VideoPhy2 paths to sys.path
            videophy2_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'VideoPhy', 'VIDEOPHY2'
            )
            if videophy2_path not in sys.path:
                sys.path.insert(0, videophy2_path)
            
            from transformers.models.llama.tokenization_llama import LlamaTokenizer
            from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
            from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
            from template import PROMPT_SA, PROMPT_PHYSICS
            
            self.PROMPT_SA = PROMPT_SA
            self.PROMPT_PHYSICS = PROMPT_PHYSICS
            
            self.tokenizer = LlamaTokenizer.from_pretrained(self.checkpoint_path)
            image_processor = MplugOwlImageProcessor.from_pretrained(self.checkpoint_path)
            self.processor = MplugOwlProcessor(image_processor, self.tokenizer)
            
            self.model = MplugOwlForConditionalGeneration.from_pretrained(
                self.checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map={'': 'cpu'}
            )
            self.model.eval()
            self.model = self.model.to(self.device).to(torch.bfloat16)
            
            self._initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VideoPhy-2 model: {e}")
    
    def _parse_score(self, output: str) -> int:
        """Parse score from model output."""
        output_lower = output.lower().strip()
        
        for key, val in self.num_map.items():
            if key in output_lower:
                return val
        
        # Try to extract digit
        digits = ''.join([c for c in output_lower if c.isdigit()])
        if digits and int(digits) in self.num_map.values():
            return int(digits)
        
        return 0  # Default to 0 if parsing fails
    
    def compute_sa(self, video_path: str, caption: str) -> int:
        """Compute Semantic Adherence (SA) score.
        
        Args:
            video_path: Path to video file
            caption: Text caption describing the video
        
        Returns:
            SA score between 1 and 5 (higher is better)
        """
        self._initialize()
        
        prompt = self.PROMPT_SA.format(caption=caption)
        
        with torch.no_grad():
            inputs = self.processor(
                text=[prompt],
                videos=[video_path],
                num_frames=self.num_frames,
                return_tensors='pt'
            )
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            res = self.model.generate(**inputs, **self.generate_kwargs)
            output = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
            
            return self._parse_score(output)
    
    def compute_pc(self, video_path: str) -> int:
        """Compute Physical Commonsense (PC) score.
        
        Args:
            video_path: Path to video file
        
        Returns:
            PC score between 1 and 5 (higher is better)
        """
        self._initialize()
        
        prompt = self.PROMPT_PHYSICS
        
        with torch.no_grad():
            inputs = self.processor(
                text=[prompt],
                videos=[video_path],
                num_frames=self.num_frames,
                return_tensors='pt'
            )
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            res = self.model.generate(**inputs, **self.generate_kwargs)
            output = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
            
            return self._parse_score(output)

