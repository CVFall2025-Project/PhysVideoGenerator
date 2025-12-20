"""
Evaluation module for video generation models.
Provides evaluation scripts for OpenSora, VideoCrafter2, and HunyuanVideo.
"""

from .opensora_eval import evaluate_opensora
from .videocrafter2_eval import evaluate_videocrafter2
from .hunyuanvideo_eval import evaluate_hunyuanvideo
from .run_comparison import run_all_evaluations, generate_comparison_report

__all__ = [
    'evaluate_opensora',
    'evaluate_videocrafter2',
    'evaluate_hunyuanvideo',
    'run_all_evaluations',
    'generate_comparison_report'
]


