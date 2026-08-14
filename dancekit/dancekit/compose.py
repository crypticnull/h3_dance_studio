"""Generate original choreography for a song, from a pose vocabulary and a beat grid.

No source dance clip. The song's own structure decides what happens when.

The thing that separates choreography from flailing is not timing accuracy -- it is
MOTIF REPETITION. A dancer states a phrase, repeats it on the other side, states it
again on the chorus. Random poses land perfectly on the beat and still read as noise.
So the composer builds a small bank of motifs, binds each to a musical section, and
brings the same motif back whenever that section returns.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2

from .beatgrid import BeatGrid
from .poselib import all_poses, energies
from .retime import ease
from .skeleton import NUM_JOINTS, from_bones, to_bones, unwrap_sequence


# --- musical structure -----------------------------------------------------------------

def phrase_energy(grid: BeatGrid, anchors: np.ndarray, slots: int) -> np.ndarray:
    """Mean onset strength per phrase, normalised 0..1."""
    n_phrases = int(np.ceil(len(anchors) / slots))
    if grid.onset_env is None or grid.onset_env.size < 2:
        return np.full(n_phrases, 0.5)
    strength = np.interp(anchors, grid.onset_times, grid.onset_env)
    out = np.array([strength[i * slots:(i + 1) * slots].mean() for i in range(n_phrases)])
    rng = out.max() - out.min()
    return (out - out.min()) / rng if rng > 1e-9 else np.full(n_phrases, 0.5)


def label_sections(energy: np.ndarray, k: int = 3, seed: int = 0) -> np.ndarray:
    """Cluster phrases into k sections by energy so verses and choruses get different
    material -- and, more importantly, so a returning chorus gets the SAME material."""
    k = int(max(1, min(k, len(energy))))
    if k == 1 or len(energy) < 2:
        return np.zeros(len(energy), dtype=int)
    # A track with no dynamics -- or a `beatgrid.synthetic` grid, which carries no onset
    # envelope at all -- gives every phrase the same energy. kmeans on identical points
    # divides by a zero distance sum, so answer directly: it is all one section.
    if float(np.ptp(energy)) < 1e-9:
        return np.zeros(len(energy), dtype=int)
    feats = np.stack([energy, np.gradient(energy)], axis=-1)
    np.random.seed(seed)
    _, labels = kmeans2(feats, k, minit="++", seed=seed)
    return labels.astype(int)


# --- motif construction -----------------------------------------------------------------

def side_bias(pose: np.ndarray) -> float:
    """Which way a pose leans, from the extremities relative to the spine.

    Alternating sides is what stops a phrase looking like a twitch. Deciding "side" by
    whether a name ends in `_m` fails badly: the mirror of a symmetric pose is the same
    pose, so the composer ends up alternating neutral / neutral and calling it dancing.
    Measuring the actual lateral offset is what we want.
    """
    root_x = pose[1, 0]
    ext = pose[[4, 7, 10, 13], 0]
    return float(np.mean(ext - root_x))


def _base(name: str) -> str:
    return name[:-2] if name.endswith("_m") else name


def _pick(names, e_target, e_map, rng, exclude=(), want_side: float = 0.0,
          bias_map=None, neutral_penalty: float = 0.45):
    cand = [n for n in names if _base(n) not in {_base(x) for x in exclude if x}]
    if not cand:
        cand = list(names)

    # Match the target energy...
    w = np.array([np.exp(-((e_map[n] - e_target) ** 2) / 0.06) for n in cand])
    # ...prefer the requested lateral direction...
    if bias_map is not None and abs(want_side) > 1e-9:
        b = np.array([bias_map[n] for n in cand])
        w = w * (1.0 + 1.8 * np.clip(np.sign(want_side) * b / 0.09, -1, 1))
    # ...and keep the resting pose from eating the phrase.
    w = w * np.array([neutral_penalty if _base(n) == "neutral" else 1.0 for n in cand])

    w = np.clip(w, 1e-9, None)
    return str(rng.choice(cand, p=w / w.sum()))


def make_motif(names, e_map, slots: int, e_target: float, rng, bias_map=None) -> list[str]:
    """One phrase of material: an accent on the downbeat, alternating lateral direction,
    no repeat of the last two shapes, and a softer moment mid-phrase so the eye gets a
    rest before the next statement."""
    motif, prev, prev2 = [], None, None
    for s in range(slots):
        if s == 0:
            e = min(1.0, e_target + 0.25)          # hit the downbeat hardest
        elif s == slots // 2:
            e = max(0.0, e_target - 0.15)          # breathe at the half
        else:
            e = e_target
        want = 1.0 if (s % 2 == 0) else -1.0
        n = _pick(names, e, e_map, rng, exclude=(prev, prev2),
                  want_side=want, bias_map=bias_map)
        motif.append(n)
        prev, prev2 = n, prev
    return motif


# --- interpolation -----------------------------------------------------------------------

def _interp_anchor_poses(poses: np.ndarray, anchor_t: np.ndarray, t_out: np.ndarray,
                         snap: float, overshoot: float) -> np.ndarray:
    """Bone-space interpolation through a sequence of anchor poses."""
    ang, ln, root = to_bones(poses)
    ang = unwrap_sequence(ang)

    T = len(t_out)
    o_ang = np.zeros((T, NUM_JOINTS))
    o_ln = np.zeros((T, NUM_JOINTS))
    o_root = np.zeros((T, 2))

    for i in range(len(anchor_t) - 1):
        t0, t1 = anchor_t[i], anchor_t[i + 1]
        m = (t_out >= t0) & (t_out < t1)
        if not m.any():
            continue
        u = (t_out[m] - t0) / max(t1 - t0, 1e-9)
        f = ease(u, snap=snap, overshoot=overshoot)[:, None]
        o_ang[m] = ang[i] + (ang[i + 1] - ang[i]) * f
        o_ln[m] = ln[i] + (ln[i + 1] - ln[i]) * f
        o_root[m] = root[i] + (root[i + 1] - root[i]) * f

    tail = t_out >= anchor_t[-1]
    o_ang[tail], o_ln[tail], o_root[tail] = ang[-1], ln[-1], root[-1]
    head = t_out < anchor_t[0]
    o_ang[head], o_ln[head], o_root[head] = ang[0], ln[0], root[0]

    conf = np.ones((T, NUM_JOINTS))
    return from_bones(o_ang, o_ln, o_root, conf)


# --- main entry ----------------------------------------------------------------------------

def compose(grid: BeatGrid, fps: float = 24.0, subdivision: int = 1,
            phrase_beats: int = 8, sections: int = 3, seed: int = 0,
            snap: float = 0.65, overshoot: float = 0.12,
            library: dict[str, np.ndarray] | None = None,
            energy_map: dict[str, float] | None = None,
            variation: float = 0.5, max_seconds: float | None = None,
            bounce: float = 0.012):
    """Compose a dance for `grid`. Returns (pose (T,18,3), info dict)."""
    rng = np.random.default_rng(seed)
    lib = library or all_poses()
    e_map = energy_map or energies()
    if energy_map is None and library is not None:
        # Custom library with no energies supplied: rank poses by how extended they are.
        e_map = {}
        for n, p in lib.items():
            spread = float(np.linalg.norm(p[:, :2] - p[1, :2], axis=-1).mean())
            e_map[n] = spread
        lo, hi = min(e_map.values()), max(e_map.values())
        e_map = {n: (v - lo) / (hi - lo) if hi > lo else 0.5 for n, v in e_map.items()}

    names = list(lib.keys())
    bias_map = {n: side_bias(p) for n, p in lib.items()}

    anchors = grid.subdivide(subdivision)
    if max_seconds:
        anchors = anchors[anchors <= max_seconds]
    if len(anchors) < 2:
        raise ValueError("Beat grid too short to compose against.")

    slots = int(phrase_beats * subdivision)
    energy = phrase_energy(grid, anchors, slots)
    labels = label_sections(energy, k=sections, seed=seed)

    # One motif per section, reused on every return of that section.
    motifs = {}
    for lab in np.unique(labels):
        e_t = float(energy[labels == lab].mean())
        motifs[int(lab)] = make_motif(names, e_map, slots, e_t, rng, bias_map)

    # Lay motifs down phrase by phrase. Every other repeat of a section flips sides,
    # which is how choreography stays recognisable without being monotonous.
    seen: dict[int, int] = {}
    seq_names: list[str] = []
    for pi, lab in enumerate(labels):
        lab = int(lab)
        rep = seen.get(lab, 0)
        seen[lab] = rep + 1
        motif = list(motifs[lab])
        if rep % 2 == 1 and rng.random() < variation:
            motif = [n[:-2] if n.endswith("_m") else n + "_m" for n in motif]
            motif = [n if n in lib else motifs[lab][i] for i, n in enumerate(motif)]
        if rep >= 2 and rng.random() < variation * 0.5:
            motif[-1] = _pick(names, min(1.0, float(energy[pi]) + 0.3), e_map, rng,
                              bias_map=bias_map)
        seq_names.extend(motif)

    seq_names = seq_names[:len(anchors)]
    anchor_poses = np.stack([lib[n] for n in seq_names])

    duration = float(anchors[len(seq_names) - 1])
    n_out = int(np.floor(duration * fps)) + 1
    t_out = np.arange(n_out) / fps

    pose = _interp_anchor_poses(anchor_poses, anchors[:len(seq_names)], t_out,
                                snap=snap, overshoot=overshoot)

    # A small vertical pulse on every beat. Real dancers never fully stop, and a
    # dead-still hold between hits is the single biggest tell of synthetic motion.
    if bounce > 0:
        spb = 60.0 / max(grid.tempo, 1e-6)
        ph = 2 * np.pi * t_out / spb
        pose[:, :, 1] += bounce * (np.sin(ph) ** 2)[:, None]

    info = {
        "tempo": round(float(grid.tempo), 2),
        "frames": int(n_out),
        "fps": float(fps),
        "duration_s": round(duration, 2),
        "phrases": int(len(labels)),
        "slots_per_phrase": slots,
        "sections": {int(k): motifs[int(k)] for k in motifs},
        "section_sequence": [int(x) for x in labels],
        "seed": seed,
    }
    return pose, info
