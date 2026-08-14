"""Find the frames that actually matter in a dance clip.

Choreography is a series of held shapes connected by fast transitions. The held
shapes -- the "hits" -- are local minima of body speed. Those are the frames worth
pinning to a beat; everything between them is travel and can be re-timed freely.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .skeleton import LIMB_WEIGHTS, fill_missing, normalise


def body_speed(pose: np.ndarray, smooth: float = 1.5) -> np.ndarray:
    """Per-frame limb-weighted speed in body-local space. Shape (T,).

    Normalising first means a camera push-in or the dancer stepping sideways does not
    register as movement -- only the body changing shape does.
    """
    local, _, _ = normalise(fill_missing(pose))
    xy = gaussian_filter1d(local[:, :, :2], smooth, axis=0, mode="nearest")
    d = np.linalg.norm(np.diff(xy, axis=0), axis=-1)          # (T-1, 18)
    w = LIMB_WEIGHTS / LIMB_WEIGHTS.sum()
    spd = (d * w[None, :]).sum(axis=1)
    return np.concatenate([spd[:1], spd])                      # pad back to T


def detect_keyposes(pose: np.ndarray, fps: float = 30.0, min_gap_s: float = 0.12,
                    prominence: float = 0.15, smooth: float = 1.5,
                    max_count: int | None = None) -> np.ndarray:
    """Frame indices of held shapes (speed minima).

    prominence is relative to the clip's own speed spread, so it transfers between
    energetic and languid clips without retuning.
    """
    spd = body_speed(pose, smooth=smooth)
    if spd.size < 3:
        return np.array([0], dtype=int)

    spread = float(np.percentile(spd, 90) - np.percentile(spd, 10))
    prom = max(prominence * spread, 1e-6)
    distance = max(int(round(min_gap_s * fps)), 1)

    idx, props = find_peaks(-spd, distance=distance, prominence=prom)

    # A clip usually starts and ends on a held shape; the peak finder can't see those.
    idx = np.unique(np.concatenate([[0], idx, [len(spd) - 1]])).astype(int)

    if max_count is not None and len(idx) > max_count:
        # Keep the deepest holds -- the most emphatic hits in the phrase.
        keep = np.argsort(spd[idx])[:max_count]
        idx = np.sort(idx[keep])

    return idx


def phrase_report(pose: np.ndarray, keyposes: np.ndarray, fps: float) -> dict:
    """Diagnostics: is this clip actually danced to a steady tempo?

    If implied_bpm is wildly unstable the clip is probably freestyle, speed-ramped, or
    the pose track is broken -- all of which make it a bad retiming source.
    """
    spd = body_speed(pose)
    t = keyposes / fps
    gaps = np.diff(t)
    out = {
        "frames": int(pose.shape[0]),
        "duration_s": round(float(pose.shape[0] / fps), 2),
        "keyposes": int(len(keyposes)),
        "mean_gap_s": round(float(gaps.mean()), 3) if len(gaps) else None,
        "gap_cv": round(float(gaps.std() / gaps.mean()), 3) if len(gaps) and gaps.mean() else None,
        "mean_speed": round(float(spd.mean()), 5),
        "peak_speed": round(float(spd.max()), 5),
    }
    if out["mean_gap_s"]:
        out["implied_bpm"] = round(60.0 / out["mean_gap_s"], 1)
    return out


def detect_slowmo(pose: np.ndarray, fps: float, dup_thresh: float = 1e-4) -> dict:
    """Flag conformed slow motion, which teaches a LoRA weightless, dreamy movement.

    Two signals: near-identical consecutive frames (frame-duplicated conform) and an
    unusually low peak speed for the clip's length.
    """
    local, _, _ = normalise(fill_missing(pose))
    d = np.linalg.norm(np.diff(local[:, :, :2], axis=0), axis=-1).mean(axis=1)
    dup_ratio = float((d < dup_thresh).mean())
    spd = body_speed(pose)
    return {
        "duplicate_frame_ratio": round(dup_ratio, 3),
        "peak_speed": round(float(spd.max()), 5),
        "likely_slowmo": bool(dup_ratio > 0.2 or float(spd.max()) < 0.004),
    }
