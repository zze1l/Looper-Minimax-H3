"""
Looper-Minimax-H3 for ComfyUI
Enables seamless, perpetual video loops using the MiniMax H3 video diffusion model.
Supports both Pixel/Video workflows and Native Latent workflows.
"""

from .comfyui_looperRaw_MinimaxH3 import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

WEB_DIRECTORY = "./comfyui_looperRaw_MinimaxH3/web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
