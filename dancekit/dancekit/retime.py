"""Warp a real dance's timing onto a new song's beat grid.

The source clip supplies the choreography (including all the inner detail between
hits). The beat grid supplies the timing. We build a monotonic map from output time
back to source time that pins every detected keypose to a grid point, then resample
the source in bone space so limbs stay solid through the warp.
"""

from __future__ import annotations

import numpy as np

from .skeleton import (NUM_JOINTS, fill_missing, from_bones, to_bones,
                       unwrap_sequence)


# --- easing ---------------------------------------------------------------------------

def ease(u: np.ndarray, snap: float = 0.7, overshoot: float = 0.0) -> np.ndarray:
    """Map normalised segment position u in [0,1] to normalised progress.

    snap      - fraction of the interval spent moving. 1.0 is sustained/continuous;
                0.5 completes the move in the first half and holds the shape until the
                next beat, which is what reads as a sharp "hit".
    overshoot - slight past-the-pose travel that settles back, the snap you see in
                popping/hip-hop styles. 0.1-0.25 is plenty; above ~0.4 it looks broken.
    """
    snap = float(np.clip(snap, 0.05, 1.0))
    v = np.clip(u / snap, 0.0, 1.0)
    if overshoot > 0:
        c = 1.70158 * float(overshoot)
        w = v - 1.0
        return 1.0 + (c + 1.0) * w ** 3 + c * w ** 2
    return v * v * (3.0 - 2.0 * v)          # smoothstep


# --- resampling ------------------------------------------------------------------------

def resample_bones(pose: np.ndarray, src_pos: np.ndarray,
                   root_damping: float = 0.0) -> np.ndarray:
    """Sample `pose` at fractional frame positions `src_pos`, in bone space.

    Interpolating bone angle + length rather than raw joint xy is what stops limbs
    collapsing toward the body on big fast transitions.
    """
    pose = fill_missing(pose)
    ang, ln, root = to_bones(pose)
    ang = unwrap_sequence(ang)               # continuous tracks -> no +/-pi spins

    T = pose.shape[0]
    src_pos = np.clip(src_pos, 0, T - 1)
    idx = np.arange(T)

    out_ang = np.empty((len(src_pos), NUM_JOINTS))
    out_ln = np.empty((len(src_pos), NUM_JOINTS))
    for j in range(NUM_JOINTS):
        out_ang[:, j] = np.interp(src_pos, idx, ang[:, j])
        out_ln[:, j] = np.interp(src_pos, idx, ln[:, j])

    out_root = np.stack([np.interp(src_pos, idx, root[:, c]) for c in (0, 1)], axis=-1)
    if root_damping > 0:
        # Pull global travel back toward the clip's mean position so the dancer stays
        # in frame instead of walking out of a generated shot.
        out_root = out_root * (1 - root_damping) + out_root.mean(axis=0) * root_damping

    conf = np.stack([np.interp(src_pos, idx, pose[:, j, 2]) for j in range(NUM_JOINTS)], axis=-1)
    return from_bones(out_ang, out_ln, out_root, conf)


# --- anchor matching -------------------------------------------------------------------

def build_anchors(key_times: np.ndarray, grid_times: np.ndarray,
                  stride: int = 1, start_index: int = 0,
                  mode: str = "sequential") -> tuple[np.ndarray, np.ndarray]:
    """Pair source keypose times with target grid times.

    sequential - keypose k -> grid point start_index + k*stride. Preserves the phrase
                 exactly and is what you want when the source really is on-beat.
    nearest    - each keypose snaps to the closest grid point, deduped and forced
                 monotonic. Better for loose/freestyle sources.
    """
    key_times = np.asarray(key_times, dtype=float)
    grid_times = np.asarray(grid_times, dtype=float)

    if mode == "nearest":
        span = key_times[-1] - key_times[0] if len(key_times) > 1 else 1.0
        gspan = grid_times[-1] - grid_times[0] if len(grid_times) > 1 else 1.0
        scaled = grid_times[0] + (key_times - key_times[0]) * (gspan / max(span, 1e-9))
        chosen, used = [], set()
        for t in scaled:
            order = np.argsort(np.abs(grid_times - t))
            pick = next((int(i) for i in order if int(i) not in used), None)
            if pick is None:
                break
            used.add(pick)
            chosen.append(pick)
        chosen = np.sort(np.array(chosen, dtype=int))
        n = min(len(chosen), len(key_times))
        return key_times[:n], grid_times[chosen[:n]]

    picks = start_index + np.arange(len(key_times)) * stride
    valid = picks < len(grid_times)
    return key_times[valid], grid_times[picks[valid]]


# --- main entry ------------------------------------------------------------------------

def retime_to_grid(pose: np.ndarray, src_fps: float, key_frames: np.ndarray,
                   grid_times: np.ndarray, out_fps: float = 24.0,
                   out_frames: int | None = None, snap: float = 0.7,
                   overshoot: float = 0.0, stride: int = 1, start_index: int = 0,
                   mode: str = "sequential", root_damping: float = 0.0,
                   loop: bool = False):
    """Retime `pose` so its keyposes land on `grid_times`.

    Returns (retimed_pose, info_dict).
    """
    key_times = np.asarray(key_frames, dtype=float) / src_fps
    src_anchor, tgt_anchor = build_anchors(key_times, grid_times, stride, start_index, mode)

    if len(src_anchor) < 2:
        raise ValueError(
            "Need at least 2 anchors. Lower --prominence to detect more keyposes, "
            "or check that the pose track isn't mostly empty."
        )

    # Loop the source phrase until it covers the requested output span. Source anchor
    # times are extended monotonically past the end of the clip and wrapped back into
    # range at sampling time, so the phrase repeats. Feed it a clean 8-count or the
    # loop point will visibly jump.
    phrase_len = float(src_anchor[-1] - src_anchor[0])
    if loop and len(tgt_anchor) < len(grid_times) and phrase_len > 0:
        n = len(src_anchor)
        rel = src_anchor - src_anchor[0]
        s_ext, t_ext = [src_anchor[0]], [tgt_anchor[0]]
        gi, rep = start_index, 0
        while gi + stride < len(grid_times):
            for k in range(1, n):
                gi += stride
                if gi >= len(grid_times):
                    break
                s_ext.append(src_anchor[0] + rep * phrase_len + rel[k])
                t_ext.append(grid_times[gi])
            rep += 1
        seg_src, seg_tgt = np.array(s_ext), np.array(t_ext)
    else:
        seg_src, seg_tgt = src_anchor, tgt_anchor

    duration = float(seg_tgt[-1])
    if out_frames is None:
        out_frames = int(np.floor(duration * out_fps)) + 1
    t_out = np.arange(out_frames) / out_fps

    # Per-output-frame source time, eased inside each anchor interval.
    src_time = np.empty(out_frames, dtype=float)
    for i in range(len(seg_tgt) - 1):
        t0, t1 = seg_tgt[i], seg_tgt[i + 1]
        s0, s1 = seg_src[i], seg_src[i + 1]
        m = (t_out >= t0) & (t_out < t1)
        if not m.any():
            continue
        u = (t_out[m] - t0) / max(t1 - t0, 1e-9)
        src_time[m] = s0 + (s1 - s0) * ease(u, snap=snap, overshoot=overshoot)

    src_time[t_out < seg_tgt[0]] = seg_src[0]
    src_time[t_out >= seg_tgt[-1]] = seg_src[-1]

    retimed = resample_bones(pose, src_time * src_fps, root_damping=root_damping)

    info = {
        "anchors": int(len(seg_src)),
        "out_frames": int(out_frames),
        "out_fps": float(out_fps),
        "duration_s": round(duration, 3),
        "snap": snap,
        "overshoot": overshoot,
        "anchor_pairs_s": [[round(float(a), 3), round(float(b), 3)]
                           for a, b in zip(seg_src, seg_tgt)],
    }
    return retimed, info
