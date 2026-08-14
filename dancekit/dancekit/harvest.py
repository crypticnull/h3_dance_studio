"""Build a pose vocabulary from real footage or your own generated clips.

Point it at a folder. It extracts poses, keeps the held shapes, throws away the junk,
normalises everything onto one body, dedupes what's left, and hands back a vocabulary
the composer can write new choreography from.

What this deliberately does NOT keep is sequence or timing. It harvests SHAPES. The
ordering, phrasing and beat placement are generated fresh by `compose`, so a vocabulary
built from real footage doesn't reproduce anyone's routine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .keypose import detect_keyposes, detect_slowmo
from .poselib import NEUTRAL, mirror
from .skeleton import (FK_ORDER, LIMB_WEIGHTS, NUM_JOINTS, PARENTS, ROOT,
                       fill_missing, from_bones, shortest_arc, to_bones)

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Joints that must be present for a pose to be usable. A dance pose missing an ankle is
# a cropped frame, not a shape.
CORE_JOINTS = np.array([1, 2, 5, 8, 11, 3, 6, 9, 12, 10, 13])

# Bones that carry choreographic meaning; the face is excluded from every comparison.
BODY_BONES = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

CANON_LEN = np.zeros(NUM_JOINTS)
for _j, (_a, _l) in NEUTRAL.items():
    CANON_LEN[_j] = _l


# --- quality -----------------------------------------------------------------------------

def pose_quality(pose: np.ndarray, conf_thresh: float = 0.3,
                 max_len_ratio: float = 2.2) -> tuple[bool, str]:
    """Is this single frame a usable shape? Returns (ok, reason)."""
    if np.any(pose[CORE_JOINTS, 2] < conf_thresh):
        return False, "missing core joints"

    _, ln, _ = to_bones(pose[None])
    ln = ln[0]
    torso = ln[8] + ln[11]
    if torso < 1e-4:
        return False, "degenerate torso"

    # Detector failures show up as limbs several times their plausible length.
    ratio = np.divide(ln, np.maximum(CANON_LEN, 1e-6) * (torso / (CANON_LEN[8] + CANON_LEN[11])))
    if np.nanmax(ratio[BODY_BONES]) > max_len_ratio:
        return False, "implausible limb length"

    span_y = pose[:, 1].max() - pose[:, 1].min()
    if span_y < 0.15:
        return False, "figure too small in frame"
    return True, "ok"


# --- canonicalisation ---------------------------------------------------------------------

def canonicalize(pose: np.ndarray, foreshorten: float = 0.6,
                 root: tuple[float, float] = (0.5, 0.34)) -> np.ndarray:
    """Rebuild a pose on the standard body, keeping its angles.

    Poses harvested from different clips arrive at different framings, distances and body
    proportions. Interpolating between them raw makes the figure grow and shrink between
    every hit. Substituting canonical limb lengths fixes that.

    foreshorten blends back some of the observed length ratio, which is the only depth cue
    a 2D skeleton has -- an arm pointing at the camera really is short. 0 = fully canonical
    (flattest, most consistent), 1 = keep observed proportions (most depth, least stable).
    """
    ang, ln, _ = to_bones(pose[None])
    ang, ln = ang[0], ln[0]

    torso = ln[8] + ln[11]
    scale = torso / max(CANON_LEN[8] + CANON_LEN[11], 1e-6)
    observed = np.divide(ln, np.maximum(scale, 1e-6))
    ratio = np.clip(np.divide(observed, np.maximum(CANON_LEN, 1e-6)), 0.35, 1.0)

    out_len = CANON_LEN * (1.0 - foreshorten + foreshorten * ratio)
    out_len[ROOT] = 0.0

    xy = np.zeros((NUM_JOINTS, 2))
    xy[ROOT] = root
    for j in FK_ORDER:
        if j == ROOT:
            continue
        p = PARENTS[j]
        xy[j, 0] = xy[p, 0] + out_len[j] * np.cos(ang[j])
        xy[j, 1] = xy[p, 1] + out_len[j] * np.sin(ang[j])

    return np.concatenate([xy, np.ones((NUM_JOINTS, 1))], axis=-1)


def pose_energy(pose: np.ndarray) -> float:
    """How big a shape reads, as a raw score. Rank-normalised later.

    Distance from the root to the extremities alone is a poor measure: legs dominate it
    and legs are roughly the same length whatever you're doing, so every pose scores
    about the same. Reach above the shoulders and lateral spread are what actually read
    as effort, so score those.
    """
    neck = pose[ROOT, :2]
    wrists = pose[[4, 7], :2]
    ankles = pose[[10, 13], :2]

    lift = float(np.mean(np.clip(neck[1] - wrists[:, 1], 0.0, None))) / 0.30
    spread = float(np.mean(np.abs(np.concatenate([wrists[:, 0], ankles[:, 0]]) - neck[0]))) / 0.22
    xs, ys = pose[:, 0], pose[:, 1]
    area = float((xs.max() - xs.min()) * (ys.max() - ys.min())) / 0.35
    return float(0.45 * lift + 0.35 * spread + 0.20 * area)


def rank_normalise(values: list[float]) -> list[float]:
    """Spread scores evenly over 0..1 by rank.

    The composer only needs a relative ordering, and absolute extension scores bunch up
    in a narrow band for any real vocabulary -- which would leave it unable to tell a big
    shape from a small one. Ranking guarantees usable dynamic range whatever came in.
    """
    n = len(values)
    if n == 1:
        return [0.5]
    order = np.argsort(np.argsort(np.asarray(values, dtype=float)))
    return [float(r) / (n - 1) for r in order]


# --- distance ------------------------------------------------------------------------------

def bone_angles(pose: np.ndarray) -> np.ndarray:
    ang, _, _ = to_bones(pose[None])
    return ang[0]


def pose_distance(a: np.ndarray, b: np.ndarray, mirror_invariant: bool = True) -> float:
    """Weighted mean angular difference between two poses, in radians.

    Angles rather than joint positions: two dancers of different builds hitting the same
    shape differ a lot in xy and barely at all in angle. Mirror-invariant by default,
    because the composer generates mirrors itself -- keeping both sides of a pose in the
    vocabulary just wastes slots.
    """
    def d(x, y):
        diff = np.abs(shortest_arc(bone_angles(x), bone_angles(y)))
        w = LIMB_WEIGHTS[BODY_BONES]
        return float((diff[BODY_BONES] * w).sum() / w.sum())

    out = d(a, b)
    if mirror_invariant:
        out = min(out, d(a, mirror(b)))
    return out


def distance_matrix(poses: list[np.ndarray], mirror_invariant: bool = True) -> np.ndarray:
    n = len(poses)
    angs = np.stack([bone_angles(p) for p in poses])
    mangs = np.stack([bone_angles(mirror(p)) for p in poses]) if mirror_invariant else None
    w = LIMB_WEIGHTS[BODY_BONES]
    w = w / w.sum()

    D = np.zeros((n, n))
    for i in range(n):
        diff = np.abs(shortest_arc(angs[i][None, BODY_BONES], angs[:, BODY_BONES]))
        row = (diff * w).sum(axis=1)
        if mirror_invariant:
            mdiff = np.abs(shortest_arc(angs[i][None, BODY_BONES], mangs[:, BODY_BONES]))
            row = np.minimum(row, (mdiff * w).sum(axis=1))
        D[i] = row
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


# --- harvesting ------------------------------------------------------------------------------

def harvest_sequence(pose_seq: np.ndarray, fps: float = 30.0, prominence: float = 0.12,
                     min_gap_s: float = 0.10, conf_thresh: float = 0.3,
                     foreshorten: float = 0.6, source: str = "") -> tuple[list[dict], dict]:
    """Pull candidate shapes out of one clip. Returns (candidates, report)."""
    pose_seq = fill_missing(pose_seq)
    slow = detect_slowmo(pose_seq, fps)
    keys = detect_keyposes(pose_seq, fps=fps, prominence=prominence, min_gap_s=min_gap_s)

    cands, rejects = [], {}
    for k in keys:
        raw = pose_seq[int(k)]
        ok, why = pose_quality(raw, conf_thresh=conf_thresh)
        if not ok:
            rejects[why] = rejects.get(why, 0) + 1
            continue
        p = canonicalize(raw, foreshorten=foreshorten)
        cands.append({"pose": p, "energy": pose_energy(p),
                      "source": source, "frame": int(k)})

    report = {"source": source, "frames": int(pose_seq.shape[0]),
              "keyposes": int(len(keys)), "kept": len(cands),
              "rejected": rejects, "slowmo": slow}
    return cands, report


def build_vocabulary(candidates: list[dict], max_poses: int = 32,
                     min_distance: float = 0.30, drop_near_neutral: float = 0.18,
                     mirror_invariant: bool = True,
                     min_cluster_size: int = 1) -> tuple[dict, list[dict]]:
    """Cluster candidate shapes into a deduped vocabulary.

    min_distance is in radians of mean weighted bone angle -- roughly "how different two
    shapes must be to earn separate slots". 0.30 is a good starting point; lower it for a
    richer, noisier vocabulary.
    """
    from .poselib import get as builtin_get

    if not candidates:
        return {}, []

    neutral = builtin_get("neutral")
    kept = [c for c in candidates
            if pose_distance(c["pose"], neutral, mirror_invariant) > drop_near_neutral]
    if not kept:
        kept = list(candidates)

    poses = [c["pose"] for c in kept]
    if len(poses) == 1:
        entries = [{**kept[0], "count": 1}]
    else:
        D = distance_matrix(poses, mirror_invariant)
        Z = linkage(squareform(D, checks=False), method="average")
        labels = fcluster(Z, t=min_distance, criterion="distance")

        entries = []
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            if len(idx) < min_cluster_size:
                continue
            # Medoid: the member closest to all the others. More representative than a
            # mean pose, which can land in an anatomically impossible average.
            sub = D[np.ix_(idx, idx)]
            medoid = idx[int(np.argmin(sub.sum(axis=1)))]
            entries.append({**kept[medoid], "count": int(len(idx))})

        # Prefer shapes that recurred (real vocabulary) over one-off outliers, then
        # break ties toward bigger shapes.
        entries.sort(key=lambda e: (-e["count"], -e["energy"]))
        entries = entries[:max_poses]

    # Rank-normalise across the final set so the composer gets a full 0..1 dynamic range.
    ranked = rank_normalise([e["energy"] for e in entries])

    lib, meta = {}, []
    for i, (e, en) in enumerate(zip(entries, ranked)):
        name = f"h{i:02d}"
        lib[name] = e["pose"]
        meta.append({"name": name, "energy": round(float(en), 3),
                     "raw_energy": round(float(e["energy"]), 3),
                     "count": int(e.get("count", 1)), "source": e["source"],
                     "frame": int(e["frame"])})
    return lib, meta


# --- library IO ---------------------------------------------------------------------------

def save_library(path: str | Path, lib: dict[str, np.ndarray], meta: list[dict]) -> Path:
    import json
    path = Path(path)
    names = list(lib.keys())
    np.savez_compressed(path, names=np.array(names),
                        poses=np.stack([lib[n] for n in names]),
                        meta=json.dumps(meta))
    return path


def load_library(path: str | Path) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    """Returns (library, meta, energy_map). Mirrors are added so the composer can
    alternate sides."""
    import json
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["names"]]
    poses = d["poses"]
    meta = json.loads(str(d["meta"])) if "meta" in d else []
    e_by_name = {m["name"]: m["energy"] for m in meta}

    lib, emap = {}, {}
    for n, p in zip(names, poses):
        e = float(e_by_name.get(n, pose_energy(p)))
        lib[n] = p
        emap[n] = e
        lib[n + "_m"] = mirror(p)
        emap[n + "_m"] = e
    return lib, meta, emap


# --- folder walk -----------------------------------------------------------------------------

def iter_sources(root: str | Path):
    root = Path(root)
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in VIDEO_EXT or p.suffix.lower() == ".json":
            yield p


def poses_from_file(path: Path, extractor=None, max_frames: int = 0):
    """Load a pose sequence from a ComfyUI JSON, or run a detector over a video.

    Returns (pose_seq, fps).
    """
    from . import poseio

    if path.suffix.lower() == ".json":
        pose, meta = poseio.load_pose_json(path)
        return pose, float(meta.get("fps") or 30.0)

    if extractor is None:
        raise RuntimeError(
            f"{path.name} is a video and no pose extractor is available.\n"
            "Install rtmlib (pip install rtmlib onnxruntime-gpu), or export poses from "
            "ComfyUI (DWPreprocessor -> SavePoseKpsAsJsonFile) and harvest the JSON.")
    return extractor(path, max_frames=max_frames)


def make_rtmlib_extractor(mode: str = "balanced", device: str = "cuda"):
    """Video -> (pose_seq, fps) using rtmlib's DWPose. Imported lazily so the rest of the
    toolkit stays dependency-light."""
    from rtmlib import Body
    import cv2

    model = Body(mode=mode, backend="onnxruntime", device=device)

    def extract(path: Path, max_frames: int = 0):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames, n = [], 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            kps, scores = model(img)
            if len(kps) == 0:
                frames.append(np.zeros((NUM_JOINTS, 3)))
            else:
                best = int(np.argmax(scores.sum(axis=1)))
                k = np.concatenate([kps[best], scores[best][:, None]], axis=-1)[:NUM_JOINTS]
                k[:, 0] /= max(W, 1)
                k[:, 1] /= max(H, 1)
                frames.append(k)
            n += 1
            if max_frames and n >= max_frames:
                break
        cap.release()
        return np.stack(frames), float(fps)

    return extract
