"""ComfyUI custom node pack wrapping the dancekit toolkit.

Beat-locked OpenPose skeleton sequences for pose-conditioned video generation
(ControlNet / SteadyDancer / MiniMax H3). See README.md; if imports fail here,
see install.md for how to install dancekit alongside this pack.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
