"""Project 3D motion (SMPL / joint positions) down to OpenPose BODY_18.

This is the bridge from a music-to-motion model -- EDGE, AtomicDance, OpenDance -- to a
2D pose-conditioned video model. Those models emit 3D SMPL joints; ControlNet wants an
OpenPose skeleton image. Nothing off the shelf joins the two.
"""

from __future__ import annotations

import numpy as np

from .skeleton import NUM_JOINTS

# SMPL 24-joint order -> OpenPose 18. SMPL has no facial keypoints, so eyes and ears
# are synthesised from the head/neck axis; ControlNet only needs them to be plausible.
SMPL_TO_OP = {
    1: 12,   # neck        <- SMPL neck
    2: 17,   # r_shoulder
    3: 19,   # r_elbow
    4: 21,   # r_wrist
    5: 16,   # l_shoulder
    6: 18,   # l_elbow
    7: 20,   # l_wrist
    8: 2,    # r_hip
    9: 5,    # r_knee
    10: 8,   # r_ankle
    11: 1,   # l_hip
    12: 4,   # l_knee
    13: 7,   # l_ankle
    0: 15,   # nose        <- SMPL head
}


def rotation(azimuth: float = 0.0, elevation: float = 0.0) -> np.ndarray:
    az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
    ry = np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]])
    rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    return rx @ ry


def project(joints3d: np.ndarray, azimuth=0.0, elevation=0.0,
            up_axis: str = "y", flip_y: bool = True) -> np.ndarray:
    """(T,J,3) world -> (T,J,2) image-space, orthographic.

    Orthographic rather than perspective on purpose: these models output motion in a
    canonical space with no camera, and a made-up focal length introduces distortion
    the video model then has to fight.
    """
    j = np.asarray(joints3d, dtype=float)
    if up_axis == "z":
        j = j[..., [0, 2, 1]]
    R = rotation(azimuth, elevation)
    p = j @ R.T
    xy = p[..., :2].copy()
    if flip_y:
        xy[..., 1] *= -1.0       # world Y up -> image Y down
    return xy


def frame_to_canvas(xy: np.ndarray, headroom: float = 0.10, floor: float = 0.04,
                    fixed_scale: bool = True) -> np.ndarray:
    """Fit the whole sequence into 0..1 with margins.

    Scaling once over the WHOLE clip (fixed_scale) rather than per frame is important:
    per-frame fitting makes the dancer appear to zoom in and out every time they raise
    an arm, which the video model faithfully reproduces as a breathing camera.
    """
    out = xy.copy()
    if fixed_scale:
        lo = np.nanmin(out.reshape(-1, 2), axis=0)
        hi = np.nanmax(out.reshape(-1, 2), axis=0)
    else:
        lo = np.nanmin(out, axis=1, keepdims=True)
        hi = np.nanmax(out, axis=1, keepdims=True)

    span_y = np.maximum(hi[..., 1] - lo[..., 1], 1e-6)
    usable = 1.0 - headroom - floor
    s = usable / span_y
    if np.ndim(s) == 0:
        s = float(s)

    out[..., 0] = (out[..., 0] - (lo[..., 0] + hi[..., 0]) / 2) * s + 0.5
    out[..., 1] = (out[..., 1] - lo[..., 1]) * s + headroom
    return out


def smpl_to_openpose(joints3d: np.ndarray, azimuth: float = 0.0, elevation: float = 0.0,
                     up_axis: str = "y", headroom: float = 0.10,
                     floor: float = 0.04) -> np.ndarray:
    """(T,24+,3) SMPL joints -> (T,18,3) OpenPose pose, normalised, conf 1.0."""
    j = np.asarray(joints3d, dtype=float)
    if j.ndim != 3 or j.shape[-1] != 3:
        raise ValueError(f"expected (T,J,3), got {j.shape}")

    xy = project(j, azimuth, elevation, up_axis=up_axis)
    xy = frame_to_canvas(xy, headroom=headroom, floor=floor)

    T = xy.shape[0]
    out = np.zeros((T, NUM_JOINTS, 3))
    for op_i, smpl_i in SMPL_TO_OP.items():
        if smpl_i < xy.shape[1]:
            out[:, op_i, :2] = xy[:, smpl_i]
            out[:, op_i, 2] = 1.0

    # Synthesise the face points OpenPose expects from the neck->head axis.
    neck, nose = out[:, 1, :2], out[:, 0, :2]
    axis = nose - neck
    n = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = axis / np.maximum(n, 1e-6)
    perp = np.stack([-axis[:, 1], axis[:, 0]], axis=-1)
    head = np.maximum(n, 1e-6)

    for idx, (along, across) in {14: (0.30, -0.28), 15: (0.30, 0.28),
                                 16: (0.12, -0.52), 17: (0.12, 0.52)}.items():
        out[:, idx, :2] = nose + axis * head * along + perp * head * across
        out[:, idx, 2] = 1.0

    return out


def orbit(n_frames: int, start: float = 0.0, degrees: float = 0.0) -> np.ndarray:
    """Per-frame azimuth for a slow camera orbit. Small values (10-25 degrees over a
    clip) add life; large ones confuse pose conditioning."""
    return start + np.linspace(0.0, degrees, n_frames)


def load_joints(path: str) -> np.ndarray:
    """Load 3D joints from .npy, .npz (key 'joints'/'poses'/'motion') or a pickle."""
    import pickle
    from pathlib import Path

    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=True)
        for k in ("joints", "joints3d", "poses", "motion", "pred_joints"):
            if k in d:
                return d[k]
        return d[d.files[0]]
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for k in ("full_pose", "joints", "joints3d", "smpl_poses", "pred_joints"):
            if k in obj:
                return np.asarray(obj[k])
    return np.asarray(obj)
