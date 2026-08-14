"""Node registry for the dancekit ComfyUI pack."""

from __future__ import annotations

from . import audio_nodes, io_nodes, library_nodes, pose_nodes, server_routes

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _mod in (audio_nodes, pose_nodes, library_nodes, io_nodes):
    NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)
