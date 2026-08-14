"""OpenPose BODY_18 (COCO) skeleton definition, bone-space conversion, and FK.

Everything downstream works on arrays shaped (T, 18, 3) -> (frames, joints, [x, y, conf])
with x/y normalised to 0..1 in canvas space. That matches what ComfyUI's
DWPreprocessor -> SavePoseKpsAsJsonFile emits, so the round trip is lossless.
"""

from __future__ import annotations

import numpy as np

# --- Joint layout (OpenPose COCO-18) ------------------------------------------------

JOINT_NAMES = [
    "nose",       # 0
    "neck",       # 1   <- root
    "r_shoulder", # 2
    "r_elbow",    # 3
    "r_wrist",    # 4
    "l_shoulder", # 5
    "l_elbow",    # 6
    "l_wrist",    # 7
    "r_hip",      # 8
    "r_knee",     # 9
    "r_ankle",    # 10
    "l_hip",      # 11
    "l_knee",     # 12
    "l_ankle",    # 13
    "r_eye",      # 14
    "l_eye",      # 15
    "r_ear",      # 16
    "l_ear",      # 17
]

NUM_JOINTS = 18
ROOT = 1  # neck

# parent[j] = index of j's parent, or -1 for the root
PARENTS = np.array([1, -1, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 0, 0, 14, 15])

# Traversal order guaranteeing parents are resolved before children.
FK_ORDER = [1, 0, 2, 5, 8, 11, 3, 6, 9, 12, 14, 15, 4, 7, 10, 13, 16, 17]

# Joints whose motion actually carries "dance". Used to weight motion energy so a
# bobbing head doesn't read as choreography.
LIMB_WEIGHTS = np.array(
    [0.3, 0.5, 1.0, 1.4, 1.6, 1.0, 1.4, 1.6, 1.0, 1.4, 1.6, 1.0, 1.4, 1.6, 0.1, 0.1, 0.1, 0.1]
)

# --- Drawing (canonical OpenPose colours; ControlNet models were trained on these) ---

LIMB_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
    (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
    (0, 15), (15, 17),
]

COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]


# --- Normalisation -------------------------------------------------------------------

def torso_scale(pose: np.ndarray) -> np.ndarray:
    """Per-frame body scale = neck->mid-hip distance. Shape (T,).

    Falls back to shoulder width, then to 1.0, when hips are missing/occluded.
    """
    neck = pose[:, ROOT, :2]
    r_hip, l_hip = pose[:, 8, :2], pose[:, 11, :2]
    hip_ok = (pose[:, 8, 2] > 0) & (pose[:, 11, 2] > 0)
    mid_hip = 0.5 * (r_hip + l_hip)
    scale = np.linalg.norm(mid_hip - neck, axis=-1)

    sh_ok = (pose[:, 2, 2] > 0) & (pose[:, 5, 2] > 0)
    sh = np.linalg.norm(pose[:, 2, :2] - pose[:, 5, :2], axis=-1) * 1.6
    scale = np.where(hip_ok & (scale > 1e-4), scale, np.where(sh_ok, sh, 0.0))
    med = np.median(scale[scale > 1e-4]) if np.any(scale > 1e-4) else 1.0
    scale = np.where(scale > 1e-4, scale, med)
    return np.maximum(scale, 1e-4)


def normalise(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strip out global translation and body scale.

    This is what stops camera pans, zooms and the dancer walking across frame from
    being read as choreography. Returns (body_local, root_xy, scale).
    """
    root = pose[:, ROOT, :2].copy()
    scale = torso_scale(pose)
    local = pose.copy()
    local[:, :, :2] = (pose[:, :, :2] - root[:, None, :]) / scale[:, None, None]
    return local, root, scale


def denormalise(local: np.ndarray, root: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = local.copy()
    out[:, :, :2] = local[:, :, :2] * scale[:, None, None] + root[:, None, :]
    return out


# --- Bone space ----------------------------------------------------------------------
# Linearly interpolating raw 2D joint positions between two distant poses makes limbs
# shrink and pop (the midpoint of an arc is inside the circle). Interpolating bone
# ANGLE and LENGTH separately, then rebuilding by forward kinematics, keeps limbs solid.

def to_bones(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T,18,3) -> (angles (T,18), lengths (T,18), root_xy (T,2)).

    Angle/length at the root index are unused placeholders.
    """
    xy = pose[:, :, :2]
    par = xy[:, PARENTS, :]
    vec = xy - par
    ang = np.arctan2(vec[:, :, 1], vec[:, :, 0])
    ln = np.linalg.norm(vec, axis=-1)
    ang[:, ROOT] = 0.0
    ln[:, ROOT] = 0.0
    return ang, ln, xy[:, ROOT, :].copy()


def from_bones(ang: np.ndarray, ln: np.ndarray, root: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Forward kinematics back to (T,18,3)."""
    T = ang.shape[0]
    xy = np.zeros((T, NUM_JOINTS, 2), dtype=float)
    xy[:, ROOT, :] = root
    for j in FK_ORDER:
        if j == ROOT:
            continue
        p = PARENTS[j]
        xy[:, j, 0] = xy[:, p, 0] + ln[:, j] * np.cos(ang[:, j])
        xy[:, j, 1] = xy[:, p, 1] + ln[:, j] * np.sin(ang[:, j])
    return np.concatenate([xy, conf[:, :, None]], axis=-1)


def shortest_arc(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    """Signed angular delta from a0 to a1, wrapped to (-pi, pi].

    Without this an elbow crossing the +/-pi boundary spins the long way round.
    """
    return (a1 - a0 + np.pi) % (2 * np.pi) - np.pi


def unwrap_sequence(ang: np.ndarray) -> np.ndarray:
    """Make an angle track continuous over time so interpolation behaves."""
    return np.unwrap(ang, axis=0)


# --- Gap filling ---------------------------------------------------------------------

def fill_missing(pose: np.ndarray, conf_thresh: float = 0.15) -> np.ndarray:
    """Linearly interpolate joints the detector dropped, and hold at the ends.

    Occlusion dropouts otherwise show up as a joint snapping to (0,0), which the
    keypose detector would happily mistake for a very energetic dance move.
    """
    out = pose.copy()
    T = out.shape[0]
    for j in range(NUM_JOINTS):
        ok = out[:, j, 2] > conf_thresh
        if ok.all():
            continue
        if not ok.any():
            out[:, j, :2] = 0.0
            out[:, j, 2] = 0.0
            continue
        idx = np.arange(T)
        for c in (0, 1):
            out[:, j, c] = np.interp(idx, idx[ok], out[ok, j, c])
        # Keep a low-but-nonzero confidence on filled frames so renderers still draw them.
        out[~ok, j, 2] = 0.1
    return out
