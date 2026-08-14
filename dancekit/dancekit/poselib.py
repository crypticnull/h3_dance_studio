"""A built-in library of dance keyposes, authored in bone-angle space.

Authoring poses as joint xy is miserable; authoring them as "upper arm at -60 degrees,
forearm at -100" is tractable. Everything here is defined as angle overrides on a
neutral standing figure and resolved through forward kinematics.

Angles are in degrees, in image space: 0 = pointing right, +90 = pointing DOWN
(y grows downward in image coordinates), -90 = pointing up.

The figure faces camera, so the character's right side is on the image left.
"""

from __future__ import annotations

import numpy as np

from .skeleton import FK_ORDER, NUM_JOINTS, PARENTS, ROOT

# --- neutral figure -------------------------------------------------------------------
# (angle_deg, length) per joint, indexed by joint id. Root entry is ignored.

NEUTRAL: dict[int, tuple[float, float]] = {
    0:  (-90.0, 0.085),   # nose      <- neck
    2:  (180.0, 0.075),   # r_shoulder<- neck
    5:  (0.0,   0.075),   # l_shoulder<- neck
    3:  (90.0,  0.115),   # r_elbow   <- r_shoulder
    4:  (90.0,  0.105),   # r_wrist   <- r_elbow
    6:  (90.0,  0.115),   # l_elbow   <- l_shoulder
    7:  (90.0,  0.105),   # l_wrist   <- l_elbow
    8:  (106.0, 0.215),   # r_hip     <- neck
    11: (74.0,  0.215),   # l_hip     <- neck
    9:  (90.0,  0.180),   # r_knee    <- r_hip
    10: (90.0,  0.175),   # r_ankle   <- r_knee
    12: (90.0,  0.180),   # l_knee    <- l_hip
    13: (90.0,  0.175),   # l_ankle   <- l_knee
    14: (-155.0, 0.028),  # r_eye     <- nose
    15: (-25.0,  0.028),  # l_eye     <- nose
    16: (155.0,  0.030),  # r_ear     <- r_eye
    17: (25.0,   0.030),  # l_ear     <- l_eye
}

# Left/right joint pairs, for mirroring a pose.
MIRROR_PAIRS = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (14, 15), (16, 17)]


def build_pose(overrides: dict[int, float] | None = None,
               length_scale: dict[int, float] | None = None,
               root: tuple[float, float] = (0.5, 0.34)) -> np.ndarray:
    """Resolve a pose from angle overrides. Returns (18,3) normalised 0..1."""
    overrides = overrides or {}
    length_scale = length_scale or {}

    ang = np.zeros(NUM_JOINTS)
    ln = np.zeros(NUM_JOINTS)
    for j, (a, l) in NEUTRAL.items():
        ang[j] = np.deg2rad(overrides.get(j, a))
        ln[j] = l * length_scale.get(j, 1.0)

    xy = np.zeros((NUM_JOINTS, 2))
    xy[ROOT] = root
    for j in FK_ORDER:
        if j == ROOT:
            continue
        p = PARENTS[j]
        xy[j, 0] = xy[p, 0] + ln[j] * np.cos(ang[j])
        xy[j, 1] = xy[p, 1] + ln[j] * np.sin(ang[j])

    return np.concatenate([xy, np.ones((NUM_JOINTS, 1))], axis=-1)


def mirror(pose: np.ndarray, axis: float = 0.5) -> np.ndarray:
    """Flip a pose left-to-right. The cheapest source of real-looking variation --
    choreography is full of 'same thing, other side'."""
    out = pose.copy()
    out[:, 0] = 2 * axis - out[:, 0]
    for a, b in MIRROR_PAIRS:
        out[[a, b]] = out[[b, a]]
    return out


def shift(pose: np.ndarray, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    out = pose.copy()
    out[:, 0] += dx
    out[:, 1] += dy
    return out


# --- the library ----------------------------------------------------------------------
# energy: 0 = low/contained, 1 = full extension. Used by the composer to follow the
# track's dynamics instead of picking uniformly at random.

LIBRARY: dict[str, dict] = {
    "neutral":      dict(energy=0.15, ov={}),

    # Weight shifts. Small shapes; the workhorses between bigger hits.
    "hip_pop_r":    dict(energy=0.30, ov={8: 114, 11: 80, 9: 82, 10: 94,
                                          12: 96, 3: 80, 6: 100}),
    "hip_pop_l":    dict(energy=0.30, ov={8: 100, 11: 66, 9: 84, 12: 98,
                                          13: 86, 3: 80, 6: 100}),

    # Arm shapes over a stable base.
    "arms_up_v":    dict(energy=0.95, ov={3: -125, 4: -118, 6: -55, 7: -62}),
    "arms_out_t":   dict(energy=0.70, ov={3: 178, 4: 175, 6: 2, 7: 5}),
    "cross_arms":   dict(energy=0.45, ov={3: 62, 4: 18, 6: 118, 7: 162},
                          ls={4: 0.9, 7: 0.9}),
    "point_r_high": dict(energy=0.80, ov={3: -152, 4: -146, 6: 78, 7: 84}),
    "point_l_high": dict(energy=0.80, ov={3: 102, 4: 96, 6: -28, 7: -34}),
    "hands_head":   dict(energy=0.55, ov={3: 168, 4: -62, 6: 12, 7: -118},
                          ls={4: 0.95, 7: 0.95}),

    # Torso.
    "body_roll_in": dict(energy=0.35, ov={0: -82, 3: 96, 4: 74, 6: 84, 7: 106,
                                          8: 110, 11: 78, 9: 86, 12: 94}),
    "lean_back":    dict(energy=0.55, ov={0: -68, 3: -158, 4: -172, 6: -22, 7: -8,
                                          8: 116, 11: 84}),
    "reach_down":   dict(energy=0.50, ov={0: -102, 3: 112, 4: 100, 6: 68, 7: 80,
                                          9: 76, 10: 96, 12: 104, 13: 84},
                          ls={3: 0.95, 6: 0.95}),

    # Legs. Side lunges read clearly from a front-on camera; forward lunges do not.
    "lunge_r":      dict(energy=0.75, ov={8: 112, 9: 132, 10: 104, 11: 78, 12: 74,
                                          13: 98, 3: 168, 4: 176, 6: 26, 7: 12},
                          ls={12: 0.82, 13: 0.82}),
    "lunge_l":      dict(energy=0.75, ov={11: 68, 12: 48, 13: 76, 8: 102, 9: 106,
                                          10: 82, 6: 12, 7: 4, 3: 154, 4: 168},
                          ls={9: 0.82, 10: 0.82}),
    "squat":        dict(energy=0.60, dy=0.075,
                          ov={9: 118, 10: 62, 12: 62, 13: 118,
                              3: 146, 4: 128, 6: 34, 7: 52, 8: 114, 11: 66},
                          ls={9: 0.78, 10: 0.78, 12: 0.78, 13: 0.78}),
    "knee_up_r":    dict(energy=0.85, ov={9: 152, 10: 88, 3: -138, 4: -150,
                                          6: 34, 7: 18},
                          ls={9: 0.55, 10: 0.9}),
    "knee_up_l":    dict(energy=0.85, ov={12: 28, 13: 92, 6: -42, 7: -30,
                                          3: 146, 4: 162},
                          ls={12: 0.55, 13: 0.9}),
    "kick_r":       dict(energy=0.90, dy=-0.01,
                          ov={9: 158, 10: 162, 3: -160, 4: -170,
                              6: 40, 7: 30, 11: 78}),
    "step_wide":    dict(energy=0.65, ov={8: 118, 9: 108, 11: 62, 12: 72,
                                          3: 156, 4: 172, 6: 24, 7: 8}),
}


def get(name: str) -> np.ndarray:
    spec = LIBRARY[name]
    # dy drops or lifts the whole figure -- forward kinematics is rooted at the neck,
    # so without this a squat reads as "bent knees" rather than "lower".
    root = (0.5, 0.34 + float(spec.get("dy", 0.0)))
    return build_pose(spec.get("ov"), spec.get("ls"), root=root)


def all_poses() -> dict[str, np.ndarray]:
    """Every library pose plus its mirror, so the composer has both sides available."""
    out = {}
    for name in LIBRARY:
        out[name] = get(name)
        out[name + "_m"] = mirror(get(name))
    return out


def energies() -> dict[str, float]:
    out = {}
    for name, spec in LIBRARY.items():
        out[name] = spec["energy"]
        out[name + "_m"] = spec["energy"]
    return out


def library_from_poses(pose_seq: np.ndarray, indices) -> dict[str, np.ndarray]:
    """Build a custom library by pulling frames out of a pose sequence.

    Use this to harvest keyposes from your own generated clips -- you get an
    original vocabulary without copying anyone's choreography wholesale.
    """
    return {f"p{int(i):04d}": pose_seq[int(i)].copy() for i in indices}
