"""Choreography nodes: Compose, Retime, SMPL projection, Trim, Pose Info."""

from __future__ import annotations

import json

import numpy as np

from . import _dk
from ._dk import (cache_pose, compose, dkretime, keypose, library_for_compose,
                  make_pose, smpl2d)

# MiniMax H3 accepts frame counts with frames % 17 == 5.
H3_FRAME_COUNTS = [22, 39, 56, 73, 90, 107, 124]


class DKCompose:
    """Generate original beat-locked choreography from a pose vocabulary."""

    CATEGORY = "dancekit"
    RETURN_TYPES = ("DK_POSE", "STRING")
    RETURN_NAMES = ("pose", "info")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "beat_grid": ("DK_BEATGRID", {
                    "tooltip": "Timing grid from DanceKit Beat Grid. The song's own "
                               "structure decides what happens when: sections are "
                               "detected from phrase energy and each section gets one "
                               "motif, replayed on every return."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Output frame rate. Match your video model (24 for most "
                               "I2V pipelines)."}),
                "subdivision": ("INT", {
                    "default": 1, "min": 1, "max": 4,
                    "tooltip": "Pose changes per beat. 1 = quarter notes (calmer), "
                               "2 = eighths -- busier and closer to actual TikTok "
                               "choreo density. 4 = sixteenths, usually too frantic."}),
                "phrase_beats": ("INT", {
                    "default": 8, "min": 2, "max": 32,
                    "tooltip": "Beats per phrase (motif length). 8 matches the "
                               "8-count dancers actually think in."}),
                "sections": ("INT", {
                    "default": 3, "min": 1, "max": 8,
                    "tooltip": "How many distinct musical sections to write material "
                               "for. Phrases are clustered by energy; a returning "
                               "chorus gets the SAME motif back, which is what makes "
                               "the result read as choreography, not flailing."}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31 - 1,
                    "tooltip": "Change to get a different dance for the same song. "
                               "Same seed + same song = same dance."}),
                "snap": ("FLOAT", {
                    "default": 0.65, "min": 0.05, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction of each beat interval spent moving. LOWER = "
                               "the shape arrives early and holds until the next beat, "
                               "which reads as a sharp 'hit'. 0.45-0.6 for popping, "
                               "0.8-1.0 for sustained/flowy movement."}),
                "overshoot": ("FLOAT", {
                    "default": 0.12, "min": 0.0, "max": 0.6, "step": 0.01,
                    "tooltip": "Slight past-the-pose travel that settles back -- the "
                               "snap you see in popping/hip-hop. 0.15-0.25 reads as "
                               "snap; above 0.4 limbs visibly break."}),
                "variation": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Probability of flipping a repeated motif to the other "
                               "side / varying its tail. 0 = every repeat identical, "
                               "1 = maximum 'same thing, other side'."}),
                "bounce": ("FLOAT", {
                    "default": 0.012, "min": 0.0, "max": 0.05, "step": 0.001,
                    "tooltip": "Per-beat vertical pulse amplitude. Real dancers never "
                               "fully stop; a dead-still hold between hits is the "
                               "single biggest tell of synthetic motion. 0 disables."}),
                "max_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                    "tooltip": "Only compose the first N seconds of the song. "
                               "0 = full track."}),
            },
            "optional": {
                "library": ("DK_LIBRARY", {
                    "tooltip": "Custom pose vocabulary from Harvest / Load Library. "
                               "Without it, the generic 19-pose built-in library is "
                               "used -- clearly beat-locked but not stylish. Style "
                               "comes from the vocabulary."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def run(self, beat_grid, fps, subdivision, phrase_beats, sections, seed, snap,
            overshoot, variation, bounce, max_seconds, library=None, unique_id=None):
        lib, emap = library_for_compose(library)
        pose, info = compose.compose(
            beat_grid, fps=fps, subdivision=subdivision, phrase_beats=phrase_beats,
            sections=sections, seed=seed, snap=snap, overshoot=overshoot,
            library=lib, energy_map=emap, variation=variation,
            max_seconds=(max_seconds if max_seconds > 0 else None), bounce=bounce)
        cache_pose(unique_id, pose, fps, grid=beat_grid)
        return (make_pose(pose, fps, grid=beat_grid), json.dumps(info, indent=2))


class DKRetime:
    """Warp an existing dance's timing onto a new song's beat grid."""

    CATEGORY = "dancekit"
    RETURN_TYPES = ("DK_POSE", "STRING")
    RETURN_NAMES = ("pose", "info")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("DK_POSE", {
                    "tooltip": "Source dance (from Load Pose JSON, or any DK_POSE). "
                               "Supplies the choreography and all its inner detail; "
                               "the beat grid replaces only its clock."}),
                "beat_grid": ("DK_BEATGRID", {
                    "tooltip": "Target timing. Detected keyposes (held shapes) get "
                               "pinned to these grid points."}),
                "src_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Frame rate of the SOURCE pose sequence. 0 = use the "
                               "fps carried on the DK_POSE. Getting this wrong scales "
                               "every keypose time, so timing lands consistently "
                               "off-beat."}),
                "prominence": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Keypose detection threshold, relative to the clip's "
                               "own speed spread (so it transfers between energetic "
                               "and languid clips). LOWER finds more held shapes. If "
                               "retime says it needs more anchors, lower this."}),
                "min_gap": ("FLOAT", {
                    "default": 0.12, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "Minimum seconds between detected keyposes. Stops one "
                               "long hold being counted as several hits."}),
                "max_keyposes": ("INT", {
                    "default": 0, "min": 0, "max": 512,
                    "tooltip": "Cap the number of keyposes, keeping the deepest holds "
                               "(the most emphatic hits). 0 = no cap."}),
                "subdivision": ("INT", {
                    "default": 1, "min": 1, "max": 4,
                    "tooltip": "Grid density to pin keyposes to. 1 = quarter notes, "
                               "2 = eighths. Most TikTok choreo hits on eighths."}),
                "stride": ("INT", {
                    "default": 1, "min": 1, "max": 8,
                    "tooltip": "Map keyposes to every Nth grid point. Use 2 when the "
                               "source dances half-time relative to your song."}),
                "start_index": ("INT", {
                    "default": 0, "min": 0, "max": 256,
                    "tooltip": "First grid point to use. Shift so the phrase starts "
                               "on a downbeat rather than mid-bar."}),
                "mode": (["sequential", "nearest"], {
                    "default": "sequential",
                    "tooltip": "sequential: keypose k -> grid point start+k*stride; "
                               "preserves the phrase exactly, right when the source "
                               "really is on-beat. nearest: each keypose snaps to the "
                               "closest grid point -- better for loose/freestyle "
                               "sources."}),
                "snap": ("FLOAT", {
                    "default": 0.7, "min": 0.05, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction of each interval spent moving. Lower = arrive "
                               "early and hold = sharper hits. 1.0 = continuous "
                               "motion, preserves the source's flow."}),
                "overshoot": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.6, "step": 0.01,
                    "tooltip": "Past-the-pose travel that settles back. 0.1-0.25 adds "
                               "hip-hop snap; above 0.4 looks broken. The source "
                               "usually has its own follow-through, hence default 0."}),
                "root_damping": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Pull global travel back toward the clip's mean "
                               "position so the dancer stays in frame instead of "
                               "walking out of a generated shot. 1 = locked in "
                               "place."}),
                "loop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Repeat the source phrase until it covers the whole "
                               "song. Feed it a clean 8-count or you will see the "
                               "seam -- the loop point jumps."}),
                "out_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Output frame rate. Match your video model."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def run(self, pose, beat_grid, src_fps, prominence, min_gap, max_keyposes,
            subdivision, stride, start_index, mode, snap, overshoot, root_damping,
            loop, out_fps, unique_id=None):
        seq = pose["pose"]
        fps_src = src_fps if src_fps > 0 else pose["fps"]
        keys = keypose.detect_keyposes(
            seq, fps=fps_src, min_gap_s=min_gap, prominence=prominence,
            max_count=(max_keyposes if max_keyposes > 0 else None))
        grid_times = beat_grid.subdivide(subdivision)
        retimed, info = dkretime.retime_to_grid(
            seq, fps_src, keys, grid_times, out_fps=out_fps, snap=snap,
            overshoot=overshoot, stride=stride, start_index=start_index,
            mode=mode, root_damping=root_damping, loop=loop)
        info["keyposes_detected"] = int(len(keys))
        cache_pose(unique_id, retimed, out_fps, grid=beat_grid)
        return (make_pose(retimed, out_fps, meta=pose.get("meta"), grid=beat_grid),
                json.dumps(info, indent=2))


class DKSMPLToOpenPose:
    """Project 3D SMPL motion (EDGE / AtomicDance / OpenDance) to OpenPose 2D."""

    CATEGORY = "dancekit"
    RETURN_TYPES = ("DK_POSE", "STRING")
    RETURN_NAMES = ("pose", "info")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to .npy / .npz / .pkl of (T, 24+, 3) SMPL joint "
                               "positions -- what EDGE, AtomicDance and OpenDance "
                               "emit."}),
                "fps": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Frame rate the motion was generated at (EDGE outputs "
                               "30). Carried on the DK_POSE for downstream retiming."}),
                "azimuth": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Camera yaw in degrees. 0 = front-on. Projection is "
                               "orthographic on purpose: these models emit motion in "
                               "a canonical space with no camera, and inventing a "
                               "focal length adds distortion the video model has to "
                               "fight."}),
                "elevation": ("FLOAT", {
                    "default": 0.0, "min": -89.0, "max": 89.0, "step": 1.0,
                    "tooltip": "Camera pitch in degrees. Positive looks down at the "
                               "dancer. Keep small; pose conditioning gets confused "
                               "by strong top-down views."}),
                "up_axis": (["y", "z"], {
                    "default": "y",
                    "tooltip": "Which world axis is 'up' in the motion file. SMPL is "
                               "y-up; some pipelines re-export z-up. If the skeleton "
                               "lies on its side, switch this."}),
                "headroom": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 0.4, "step": 0.01,
                    "tooltip": "Margin above the figure as a fraction of canvas "
                               "height. Scaling is fixed over the WHOLE clip -- "
                               "per-frame fitting would make the dancer zoom in and "
                               "out every time they raise an arm."}),
                "floor": ("FLOAT", {
                    "default": 0.04, "min": 0.0, "max": 0.4, "step": 0.01,
                    "tooltip": "Margin below the figure as a fraction of canvas "
                               "height."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def run(self, motion_path, fps, azimuth, elevation, up_axis, headroom, floor,
            unique_id=None):
        joints = smpl2d.load_joints(motion_path)
        pose = smpl2d.smpl_to_openpose(joints, azimuth=azimuth, elevation=elevation,
                                       up_axis=up_axis, headroom=headroom, floor=floor)
        cache_pose(unique_id, pose, fps)
        info = (f"frames: {pose.shape[0]}  fps: {fps}  "
                f"duration: {pose.shape[0] / fps:.2f}s  "
                f"source joints: {np.asarray(joints).shape}")
        return (make_pose(pose, fps), info)


class DKTrim:
    """Trim or pad a pose sequence to a target frame count."""

    CATEGORY = "dancekit"
    RETURN_TYPES = ("DK_POSE", "INT", "STRING")
    RETURN_NAMES = ("pose", "frames", "info")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("DK_POSE", {}),
                "target_frames": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "Desired frame count. 0 = keep current length (useful "
                               "with h3_frame_grid to just snap to a valid count)."}),
                "start_frame": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "Drop this many frames from the front first. Use to "
                               "start the clip on a downbeat."}),
                "pad_mode": (["hold", "loop"], {
                    "default": "hold",
                    "tooltip": "When the target is longer than the sequence: 'hold' "
                               "freezes the last frame (safe but static -- remember a "
                               "dead hold is the biggest tell of synthetic motion), "
                               "'loop' repeats from the start (seam jumps unless the "
                               "clip is a clean cycle)."}),
                "h3_frame_grid": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Snap the target to the nearest valid MiniMax H3 frame "
                               "count. H3 requires frames % 17 == 5: 22, 39, 56, 73, "
                               "90, 107, 124. Counts above 124 clamp to 124."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def run(self, pose, target_frames, start_frame, pad_mode, h3_frame_grid,
            unique_id=None):
        seq = pose["pose"]
        if start_frame > 0:
            seq = seq[min(start_frame, seq.shape[0] - 1):]
        T = seq.shape[0]
        target = target_frames if target_frames > 0 else T
        if h3_frame_grid:
            target = min(H3_FRAME_COUNTS, key=lambda n: abs(n - target))

        if target <= T:
            out = seq[:target]
            action = f"trimmed {T} -> {target}"
        elif pad_mode == "loop":
            reps = int(np.ceil(target / T))
            out = np.concatenate([seq] * reps, axis=0)[:target]
            action = f"looped {T} -> {target}"
        else:
            pad = np.repeat(seq[-1:], target - T, axis=0)
            out = np.concatenate([seq, pad], axis=0)
            action = f"held last frame {T} -> {target}"

        info = (f"{action}  ({out.shape[0] / pose['fps']:.2f}s at {pose['fps']} fps)"
                + ("  [H3 grid: frames % 17 == 5]" if h3_frame_grid else ""))
        cache_pose(unique_id, out, pose["fps"], grid=pose.get("grid"))
        return (make_pose(out, pose["fps"], meta=pose.get("meta"),
                          grid=pose.get("grid")), int(out.shape[0]), info)


class DKPoseInfo:
    """Diagnostics: keyposes, speed, implied tempo, slow-motion check."""

    CATEGORY = "dancekit"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("DK_POSE", {}),
                "prominence": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Keypose detection threshold (relative to the clip's "
                               "own speed spread). Lower finds more held shapes."}),
                "min_gap": ("FLOAT", {
                    "default": 0.12, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "Minimum seconds between keyposes."}),
            },
        }

    def run(self, pose, prominence, min_gap):
        seq, fps = pose["pose"], pose["fps"]
        keys = keypose.detect_keyposes(seq, fps=fps, min_gap_s=min_gap,
                                       prominence=prominence)
        rep = keypose.phrase_report(seq, keys, fps)
        slow = keypose.detect_slowmo(seq, fps)
        lines = [f"frames: {rep['frames']}  fps: {fps}  duration: {rep['duration_s']}s",
                 f"keyposes: {rep['keyposes']}  mean gap: {rep['mean_gap_s']}s  "
                 f"gap CV: {rep['gap_cv']}"]
        if rep.get("implied_bpm"):
            lines.append(f"implied tempo: {rep['implied_bpm']} BPM "
                         "(unstable gap CV > ~0.5 usually means freestyle, "
                         "speed-ramped, or a broken pose track)")
        lines.append(f"mean speed: {rep['mean_speed']}  peak: {rep['peak_speed']}")
        lines.append(f"slow-mo check: dup ratio {slow['duplicate_frame_ratio']}, "
                     f"likely_slowmo={slow['likely_slowmo']}"
                     + (" -- conformed slow motion produces weightless movement; "
                        "avoid as a retime source" if slow["likely_slowmo"] else ""))
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "DKCompose": DKCompose,
    "DKRetime": DKRetime,
    "DKSMPLToOpenPose": DKSMPLToOpenPose,
    "DKTrim": DKTrim,
    "DKPoseInfo": DKPoseInfo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DKCompose": "DanceKit Compose",
    "DKRetime": "DanceKit Retime",
    "DKSMPLToOpenPose": "DanceKit SMPL to OpenPose",
    "DKTrim": "DanceKit Trim / Frame Count",
    "DKPoseInfo": "DanceKit Pose Info",
}
