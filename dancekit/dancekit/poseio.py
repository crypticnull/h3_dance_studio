"""Read/write pose sequences in ComfyUI's POSE_KEYPOINT JSON format.

ComfyUI (DWPreprocessor -> SavePoseKpsAsJsonFile) emits a list of per-frame dicts:

    [{"people": [{"pose_keypoints_2d": [x,y,c, x,y,c, ...54 floats...], ...}],
      "canvas_width": 832, "canvas_height": 1472}, ...]

Coordinates are sometimes normalised 0..1 and sometimes in pixels depending on which
node version wrote them, so we sniff and normalise on load.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .skeleton import NUM_JOINTS

EXTRA_KEYS = ("face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d")


def _pick_person(frame: dict, index: int | None) -> dict | None:
    people = frame.get("people") or []
    if not people:
        return None
    if index is not None:
        return people[index] if index < len(people) else None
    # Default: the most confidently detected body, which in a TikTok dance clip is
    # almost always the dancer rather than someone in the background.
    def score(p):
        kp = np.asarray(p.get("pose_keypoints_2d") or [], dtype=float)
        return float(kp[2::3].sum()) if kp.size else 0.0
    return max(people, key=score)


def load_pose_json(path: str | Path, person: int | None = None):
    """Returns (pose (T,18,3) normalised 0..1, meta dict)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = [data]

    W = float(data[0].get("canvas_width") or 1)
    H = float(data[0].get("canvas_height") or 1)

    frames, extras = [], []
    for fr in data:
        p = _pick_person(fr, person)
        if p is None:
            frames.append(np.zeros((NUM_JOINTS, 3)))
            extras.append({})
            continue
        kp = np.asarray(p.get("pose_keypoints_2d") or [], dtype=float)
        if kp.size < NUM_JOINTS * 3:
            kp = np.pad(kp, (0, NUM_JOINTS * 3 - kp.size))
        frames.append(kp[: NUM_JOINTS * 3].reshape(NUM_JOINTS, 3))
        extras.append({k: p[k] for k in EXTRA_KEYS if k in p})

    pose = np.stack(frames)

    # Sniff coordinate space. Normalised data never exceeds ~1.
    if np.nanmax(np.abs(pose[:, :, :2])) > 1.5:
        pose[:, :, 0] /= max(W, 1.0)
        pose[:, :, 1] /= max(H, 1.0)

    return pose, {"canvas_width": W, "canvas_height": H, "extras": extras}


def save_pose_json(path: str | Path, pose: np.ndarray, meta: dict | None = None,
                   normalised: bool = True) -> None:
    meta = meta or {}
    W = float(meta.get("canvas_width") or 832)
    H = float(meta.get("canvas_height") or 1472)
    extras = meta.get("extras") or []

    out = []
    for t in range(pose.shape[0]):
        kp = pose[t].copy()
        if not normalised:
            kp[:, 0] *= W
            kp[:, 1] *= H
        person = {"pose_keypoints_2d": [round(float(v), 6) for v in kp.reshape(-1)]}
        if t < len(extras):
            person.update(extras[t])
        out.append({"people": [person], "canvas_width": W, "canvas_height": H})

    Path(path).write_text(json.dumps(out))


def save_npz(path: str | Path, pose: np.ndarray, **kw) -> None:
    np.savez_compressed(path, pose=pose, **kw)


def load_npz(path: str | Path):
    d = np.load(path, allow_pickle=True)
    return d["pose"], {k: d[k] for k in d.files if k != "pose"}
