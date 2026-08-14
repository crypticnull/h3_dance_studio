"""Vocabulary nodes: Harvest, Save Library, Load Library."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import _dk
from ._dk import (ProgressBar, apply_dropped, cache_library, dkrender, harvest,
                  make_library, resolve_out_path)


def _contact_sheet_tensor(lib: dict, cell: int = 220, cols: int = 6):
    """Contact sheet as a ComfyUI IMAGE tensor (1,H,W,3) float32 0..1."""
    torch = _dk.get_torch()
    names = list(lib.keys())
    if not names:
        return torch.zeros((1, cell, cell, 3), dtype=torch.float32)
    cols = min(cols, max(1, len(names)))
    rows = int(np.ceil(len(names) / cols))
    sheet = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for i, n in enumerate(names):
        r, c = divmod(i, cols)
        img = dkrender.draw_pose(np.asarray(lib[n]), cell, cell, thickness=0.8)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = img
    return torch.from_numpy(sheet.astype(np.float32) / 255.0)[None]


class DKHarvest:
    """Turn a folder of clips / pose JSONs into a deduped pose vocabulary."""

    CATEGORY = "dancekit"
    RETURN_TYPES = ("DK_LIBRARY", "IMAGE", "STRING")
    RETURN_NAMES = ("library", "contact_sheet", "manifest")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {
                    "default": "",
                    "tooltip": "Folder (walked recursively) or single file. Accepts "
                               "ComfyUI pose JSON (DWPreprocessor -> "
                               "SavePoseKpsAsJsonFile) and video files side by side. "
                               "Video needs rtmlib installed; JSON does not. Leave "
                               "empty to harvest only the optional DK_POSE input."}),
                "prominence": ("FLOAT", {
                    "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Keypose threshold, relative to each clip's own speed "
                               "spread. LOWER finds more candidate shapes per clip. "
                               "Only held shapes (speed minima) are harvested -- "
                               "transitions are not vocabulary."}),
                "min_gap": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "Minimum seconds between harvested keyposes within a "
                               "clip."}),
                "conf": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Confidence a core joint needs for the frame to count "
                               "as a usable shape. A pose missing an ankle is a "
                               "cropped frame, not a shape; rejections are counted "
                               "per reason in the manifest."}),
                "foreshorten": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How much observed limb-length ratio is blended back "
                               "after rebuilding each shape on the standard body. "
                               "It's the only depth cue a 2D skeleton has (an arm "
                               "pointing at the camera really is short): 0 = fully "
                               "canonical, flattest and most stable; 1 = keep observed "
                               "proportions, most depth, least stable."}),
                "max_poses": ("INT", {
                    "default": 32, "min": 1, "max": 256,
                    "tooltip": "Vocabulary size cap. Shapes that recurred across the "
                               "footage rank above one-off outliers."}),
                "min_distance": ("FLOAT", {
                    "default": 0.30, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "Dedupe threshold in RADIANS of mean weighted bone "
                               "angle -- roughly 'how different two shapes must be to "
                               "earn separate slots'. Angles, not joint positions: "
                               "two dancers of different builds hitting the same "
                               "shape differ a lot in xy and barely at all in angle. "
                               "Lower = richer, noisier vocabulary."}),
                "drop_neutral": ("FLOAT", {
                    "default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Discard shapes closer than this (same radians metric) "
                               "to plain standing. The composer already has neutral; "
                               "harvested near-neutrals just eat slots."}),
                "min_count": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Require a shape to recur in this many frames before "
                               "it earns a slot. The best filter for noisy footage: "
                               "detector glitches don't repeat, real vocabulary "
                               "does."}),
                "mirror_invariant": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Treat a pose and its mirror as the same shape. "
                               "Usually what you want -- Compose generates mirrors "
                               "itself, so keeping both sides wastes slots."}),
                "skip_slowmo": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Drop clips flagged as conformed slow motion "
                               "(duplicated frames / abnormally low peak speed). "
                               "Slow-mo footage teaches weightless, dreamy movement "
                               "-- poison as vocabulary and as LoRA data alike."}),
                "src_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Override the frame rate assumed for pose JSONs that "
                               "don't record one. 0 = use the file's fps, falling "
                               "back to 30."}),
                "dropped": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "JSON list of pose names to exclude, e.g. "
                               "[\"h03\",\"h07\"]. The library browser below fills "
                               "this in when you click poses off; it is applied on "
                               "the next run. Names are positional (h00, h01, ...) so "
                               "they only stay meaningful while the inputs above are "
                               "unchanged."}),
            },
            "optional": {
                "pose": ("DK_POSE", {
                    "tooltip": "Harvest an in-graph pose sequence too (e.g. straight "
                               "from a DWPreprocessor batch), in addition to whatever "
                               "source_path yields."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def run(self, source_path, prominence, min_gap, conf, foreshorten, max_poses,
            min_distance, drop_neutral, min_count, mirror_invariant, skip_slowmo,
            src_fps, dropped, pose=None, unique_id=None):
        sources = list(harvest.iter_sources(source_path)) if source_path else []
        if not sources and pose is None:
            raise ValueError(
                f"Nothing to harvest: source_path {source_path!r} yields no "
                ".json/video files and no DK_POSE input is connected.")

        extractor = None
        extractor_err = None
        if any(s.suffix.lower() in harvest.VIDEO_EXT for s in sources):
            try:
                extractor = harvest.make_rtmlib_extractor()
            except Exception as e:  # rtmlib not installed
                extractor_err = str(e)

        candidates, reports = [], []
        pbar = ProgressBar(len(sources) + (1 if pose is not None else 0))

        for src in sources:
            try:
                if src.suffix.lower() in harvest.VIDEO_EXT and extractor is None:
                    reports.append({"source": src.name,
                                    "skipped": f"no pose extractor ({extractor_err or 'rtmlib missing'})"})
                    pbar.update(1)
                    continue
                seq, fps = harvest.poses_from_file(src, extractor=extractor)
                if src_fps > 0 and src.suffix.lower() == ".json":
                    fps = src_fps
                cands, rep = harvest.harvest_sequence(
                    seq, fps=fps, prominence=prominence, min_gap_s=min_gap,
                    conf_thresh=conf, foreshorten=foreshorten, source=src.name)
                if skip_slowmo and rep["slowmo"]["likely_slowmo"]:
                    rep["skipped"] = "likely slow motion"
                    reports.append(rep)
                    pbar.update(1)
                    continue
                candidates.extend(cands)
                reports.append(rep)
            except Exception as e:
                reports.append({"source": src.name, "error": str(e)})
            pbar.update(1)

        if pose is not None:
            fps = src_fps if src_fps > 0 else pose["fps"]
            cands, rep = harvest.harvest_sequence(
                pose["pose"], fps=fps, prominence=prominence, min_gap_s=min_gap,
                conf_thresh=conf, foreshorten=foreshorten, source="<DK_POSE input>")
            candidates.extend(cands)
            reports.append(rep)
            pbar.update(1)

        lib, meta = harvest.build_vocabulary(
            candidates, max_poses=max_poses, min_distance=min_distance,
            drop_near_neutral=drop_neutral, mirror_invariant=mirror_invariant,
            min_cluster_size=min_count)

        dklib = apply_dropped(make_library(lib, meta), dropped)
        cache_library(unique_id, dklib["lib"], dklib["meta"])

        manifest = {
            "poses": len(dklib["lib"]),
            "candidates": len(candidates),
            "dropped_by_user": sorted(_dk.parse_dropped(dropped)),
            "vocabulary": dklib["meta"],
            "sources": reports,
        }
        sheet = _contact_sheet_tensor(dklib["lib"])
        return (dklib, sheet, json.dumps(manifest, indent=2))


class DKSaveLibrary:
    """Write a DK_LIBRARY to a .npz on disk."""

    CATEGORY = "dancekit"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library": ("DK_LIBRARY", {}),
                "path": ("STRING", {
                    "default": "dancekit/vocabulary.npz",
                    "tooltip": "Where to write the .npz. Relative paths land in "
                               "ComfyUI's output directory."}),
            },
        }

    def run(self, library, path):
        out = resolve_out_path(path, "vocabulary.npz")
        harvest.save_library(out, library["lib"], library["meta"])
        return (str(out),)


class DKLoadLibrary:
    """Load a vocabulary .npz from disk."""

    CATEGORY = "dancekit"
    RETURN_TYPES = ("DK_LIBRARY", "STRING")
    RETURN_NAMES = ("library", "summary")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a vocabulary.npz written by Harvest / Save "
                               "Library (or by `python -m dancekit harvest`)."}),
                "dropped": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "JSON list of pose names to exclude, e.g. "
                               "[\"h03\",\"h07\"]. The library browser fills this in "
                               "when you click poses off."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def run(self, path, dropped, unique_id=None):
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"Library file not found: {path!r}")
        lib, meta, emap = harvest.load_library(path)
        base, base_emap = _dk.strip_mirrors(lib, emap)
        dklib = apply_dropped(make_library(base, meta, base_emap), dropped)
        cache_library(unique_id, dklib["lib"], dklib["meta"])
        summary = (f"{len(dklib['lib'])} poses"
                   + (f" ({len(base) - len(dklib['lib'])} dropped by user)"
                      if len(dklib["lib"]) != len(base) else "")
                   + f" from {path}")
        return (dklib, summary)


NODE_CLASS_MAPPINGS = {
    "DKHarvest": DKHarvest,
    "DKSaveLibrary": DKSaveLibrary,
    "DKLoadLibrary": DKLoadLibrary,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DKHarvest": "DanceKit Harvest",
    "DKSaveLibrary": "DanceKit Save Library",
    "DKLoadLibrary": "DanceKit Load Library",
}
