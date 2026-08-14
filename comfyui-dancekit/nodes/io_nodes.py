"""Pose JSON interop with ComfyUI's POSE_KEYPOINT format, plus skeleton rendering."""

from __future__ import annotations

from pathlib import Path

from . import _dk
from ._dk import cache_pose, make_pose, pose_to_image_batch, poseio, resolve_out_path


class DKLoadPoseJSON:
    """Load a ComfyUI POSE_KEYPOINT JSON (DWPreprocessor -> SavePoseKpsAsJsonFile)."""

    CATEGORY = "dancekit"
    RETURN_TYPES = ("DK_POSE", "INT", "STRING")
    RETURN_NAMES = ("pose", "frames", "info")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the pose JSON. Coordinates are sniffed: some "
                               "node versions write pixels, some write normalised "
                               "0..1; both load correctly."}),
                "fps": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Frame rate of the clip the poses came from. The JSON "
                               "format does not record it, and Retime needs it to "
                               "place keyposes in time."}),
                "person": ("INT", {
                    "default": -1, "min": -1, "max": 16,
                    "tooltip": "Which detected person to take per frame. -1 = the "
                               "most confidently detected body, which in a dance clip "
                               "is almost always the dancer rather than someone in "
                               "the background."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def run(self, path, fps, person, unique_id=None):
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"Pose JSON not found: {path!r}")
        pose, meta = poseio.load_pose_json(path, person=(person if person >= 0 else None))
        cache_pose(unique_id, pose, fps)
        info = (f"{pose.shape[0]} frames at {fps} fps "
                f"({pose.shape[0] / fps:.2f}s), canvas "
                f"{int(meta['canvas_width'])}x{int(meta['canvas_height'])}")
        return (make_pose(pose, fps, meta=meta), int(pose.shape[0]), info)


class DKSavePoseJSON:
    """Write a DK_POSE as ComfyUI POSE_KEYPOINT JSON."""

    CATEGORY = "dancekit"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("DK_POSE", {}),
                "path": ("STRING", {
                    "default": "dancekit/pose.json",
                    "tooltip": "Where to write the JSON. Relative paths land in "
                               "ComfyUI's output directory."}),
                "canvas_width": ("INT", {
                    "default": 832, "min": 16, "max": 8192,
                    "tooltip": "Canvas size recorded in the JSON. Only metadata "
                               "unless normalised is off, but downstream nodes that "
                               "re-render from the JSON use it for aspect ratio."}),
                "canvas_height": ("INT", {
                    "default": 1472, "min": 16, "max": 8192,
                    "tooltip": "Canvas height recorded in the JSON."}),
                "normalised": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write coordinates as 0..1 (what dancekit and most "
                               "current nodes emit) instead of pixels. Loaders sniff "
                               "either, so leave on unless a downstream tool insists "
                               "on pixels."}),
            },
        }

    def run(self, pose, path, canvas_width, canvas_height, normalised):
        out = resolve_out_path(path, "pose.json")
        meta = dict(pose.get("meta") or {})
        meta["canvas_width"] = canvas_width
        meta["canvas_height"] = canvas_height
        poseio.save_pose_json(out, pose["pose"], meta, normalised=normalised)
        return (str(out),)


class DKRenderSkeleton:
    """Render a DK_POSE to an IMAGE batch for ControlNet / pose conditioning."""

    CATEGORY = "dancekit"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("DK_POSE", {}),
                "width": ("INT", {
                    "default": 832, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Render width. Match your generation resolution "
                               "(multiples of 32 for most video models)."}),
                "height": ("INT", {
                    "default": 1472, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Render height. 832x1472 suits 9:16 dance content."}),
                "thickness": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05,
                    "tooltip": "Limb/joint thickness multiplier. The canonical "
                               "OpenPose colours and filled-ellipse limbs are kept "
                               "exactly -- ControlNet OpenPose models were trained on "
                               "precisely that rendering, so matching it is what makes "
                               "the conditioning bite. Only scale thickness "
                               "moderately."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def run(self, pose, width, height, thickness, unique_id=None):
        cache_pose(unique_id, pose["pose"], pose["fps"], grid=pose.get("grid"))
        images = pose_to_image_batch(pose["pose"], width, height, thickness)
        return (images,)


NODE_CLASS_MAPPINGS = {
    "DKLoadPoseJSON": DKLoadPoseJSON,
    "DKSavePoseJSON": DKSavePoseJSON,
    "DKRenderSkeleton": DKRenderSkeleton,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DKLoadPoseJSON": "DanceKit Load Pose JSON",
    "DKSavePoseJSON": "DanceKit Save Pose JSON",
    "DKRenderSkeleton": "DanceKit Render Skeleton",
}
