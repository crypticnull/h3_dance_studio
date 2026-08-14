"""Shared fixtures.

The important one is `click_track`: a synthetic drum pattern with exactly known beat
times, a kick on the downbeat and a *louder* snare on 2 and 4. That last detail is the
point -- it is the specific trap `_estimate_downbeats` is written to avoid, so a test
signal without it would pass no matter how the downbeat scoring worked.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dancekit import poselib
from dancekit.skeleton import NUM_JOINTS

SR = 22050


# --- audio ---------------------------------------------------------------------------

def _kick(sr: int, dur: float = 0.18, f0: float = 110.0, f1: float = 45.0) -> np.ndarray:
    """Pitch-swept low sine with a fast decay. Energy sits under 400 Hz, which is the
    band `_estimate_downbeats` scores."""
    t = np.arange(int(sr * dur)) / sr
    f = f1 + (f0 - f1) * np.exp(-t * 45.0)
    phase = 2 * np.pi * np.cumsum(f) / sr
    return np.sin(phase) * np.exp(-t * 22.0)


def _hp_noise(sr: int, dur: float, decay: float, k: int, seed: int) -> np.ndarray:
    """Noise burst with a crude high-pass (subtract a moving average) and a decay."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    n = rng.standard_normal(t.size)
    hp = n - np.convolve(n, np.ones(k) / k, mode="same")
    return hp * np.exp(-t * decay)


def _snare(sr: int, dur: float = 0.16) -> np.ndarray:
    """Noise crack over a 190 Hz body. Broadband and deliberately the loudest transient
    in the mix, so anything scoring broadband onset strength picks beats 2 and 4."""
    t = np.arange(int(sr * dur)) / sr
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30.0)
    return _hp_noise(sr, dur, decay=30.0, k=12, seed=0) + 0.7 * body


def _hat(sr: int, dur: float = 0.05) -> np.ndarray:
    return _hp_noise(sr, dur, decay=90.0, k=4, seed=1)


def make_click_track(bpm: float = 120.0, bars: int = 8, sr: int = SR,
                     beats_per_bar: int = 4, lead_in: float = 0.0,
                     hats: bool = True):
    """Render a 4/4 drum pattern. Returns (audio, sr, beat_times, downbeat_times).

    Kick on 1 and 3 (louder on 1), snare on 2 and 4 at a higher peak amplitude than
    either kick, and hi-hats on eighths.

    The hats matter: without an eighth-note layer the tracker has only four transients
    per bar to work from and settles on half-time, which is a property of the stimulus
    rather than of the analyser. Produced music always carries that layer.
    """
    spb = 60.0 / bpm
    n_beats = bars * beats_per_bar
    beat_times = lead_in + np.arange(n_beats) * spb
    duration = float(beat_times[-1] + spb * 2)
    y = np.zeros(int(sr * duration) + sr)

    kick, snare, hat = _kick(sr), _snare(sr), _hat(sr)
    for i, bt in enumerate(beat_times):
        pos = int(round(bt * sr))
        phase = i % beats_per_bar
        if phase == 0:
            y[pos:pos + kick.size] += kick * 1.35      # downbeat: hardest kick
        elif phase == 2:
            y[pos:pos + kick.size] += kick * 0.80      # beat 3: softer kick
        else:
            y[pos:pos + snare.size] += snare * 1.60    # 2 and 4: loudest transient

    if hats:
        for e in np.arange(lead_in, duration, spb / 2):
            pos = int(round(e * sr))
            y[pos:pos + hat.size] += hat * 0.35

    y = y / np.max(np.abs(y))
    downbeats = beat_times[::beats_per_bar]
    return y.astype(np.float32), sr, beat_times, downbeats


@pytest.fixture(scope="session")
def click_track(tmp_path_factory):
    """A 120 BPM / 8 bar click track on disk, with its ground truth.

    Session-scoped because writing and analysing audio is the slowest thing in the
    suite and the file is never mutated.
    """
    import soundfile as sf

    y, sr, beats, downbeats = make_click_track(bpm=120.0, bars=8)
    path = tmp_path_factory.mktemp("audio") / "click_120.wav"
    sf.write(path, y, sr)
    return {"path": str(path), "bpm": 120.0, "beats": beats,
            "downbeats": downbeats, "sr": sr, "duration": len(y) / sr}


# --- poses ---------------------------------------------------------------------------

@pytest.fixture
def sample_poses() -> dict[str, np.ndarray]:
    """A handful of distinct built-in shapes, keyed by name."""
    return {n: poselib.get(n) for n in
            ("neutral", "arms_up_v", "lunge_r", "squat", "arms_out_t")}


def _blend_in_bone_space(a: np.ndarray, b: np.ndarray, f: float) -> np.ndarray:
    """Interpolate two poses through bone angle and length.

    The fixture has to do this rather than lerp raw xy, or the synthetic "footage" would
    itself contain the collapsing limbs that bone-space interpolation exists to prevent
    -- and a retiming test against it would be measuring the fixture's artefact.
    """
    from dancekit.skeleton import from_bones, shortest_arc, to_bones

    ang, ln, root = to_bones(np.stack([a, b]))
    out_ang = (ang[0] + shortest_arc(ang[0], ang[1]) * f)[None]
    out_ln = (ln[0] + (ln[1] - ln[0]) * f)[None]
    out_root = (root[0] + (root[1] - root[0]) * f)[None]
    return from_bones(out_ang, out_ln, out_root, np.ones((1, NUM_JOINTS)))[0]


def make_hold_move_hold(shapes: list[np.ndarray], hold_frames: int = 10,
                        move_frames: int = 5, jitter: float = 0.0,
                        seed: int = 0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build a pose sequence that dwells on each shape then travels to the next.

    Returns (pose (T,18,3), holds) where `holds` gives the inclusive (start, end) frame
    range of each dwell. Keypose detection should put a keypose somewhere inside every
    one of those ranges -- which frame it picks does not matter, since the pose is the
    same throughout.

    `jitter` adds per-frame gaussian noise to the joint positions. Real pose tracks
    always carry some detector noise, and its absence is itself a signal: byte-identical
    consecutive frames are what `detect_slowmo` looks for.
    """
    frames, holds = [], []
    for i, shape in enumerate(shapes):
        start = len(frames)
        frames.extend([shape] * hold_frames)
        holds.append((start, len(frames) - 1))
        if i + 1 < len(shapes):
            nxt = shapes[i + 1]
            for k in range(1, move_frames + 1):
                u = k / (move_frames + 1)
                # Smoothstep so speed peaks mid-transition and the dwells are true minima.
                f = u * u * (3 - 2 * u)
                frames.append(_blend_in_bone_space(shape, nxt, f))

    pose = np.stack(frames)
    if jitter > 0:
        rng = np.random.default_rng(seed)
        pose[:, :, :2] += rng.normal(0.0, jitter, pose[:, :, :2].shape)
    return pose, holds


@pytest.fixture
def hold_sequence(sample_poses):
    """A 5-shape dance with known hold ranges, at 30 fps.

    Carries a little jitter so it behaves like a real pose track: without it the held
    frames are byte-identical, which `detect_slowmo` correctly reads as a frame-
    duplicated conform.
    """
    shapes = [sample_poses[n] for n in
              ("neutral", "arms_up_v", "lunge_r", "squat", "arms_out_t")]
    pose, holds = make_hold_move_hold(shapes, hold_frames=12, move_frames=6,
                                      jitter=3e-4)
    return pose, holds, 30.0


@pytest.fixture
def pose_json(tmp_path, hold_sequence):
    """The hold sequence written out as ComfyUI POSE_KEYPOINT JSON."""
    pose, _, _ = hold_sequence
    frames = []
    for t in range(pose.shape[0]):
        frames.append({
            "people": [{"pose_keypoints_2d":
                        [float(v) for v in pose[t].reshape(-1)]}],
            "canvas_width": 832, "canvas_height": 1472,
        })
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(frames))
    return path


@pytest.fixture
def smpl_motion() -> np.ndarray:
    """(T,24,3) synthetic SMPL-ish joints: a standing figure with one arm swinging.

    Only the joints `SMPL_TO_OP` actually reads are given sensible positions; the rest
    are filled so the array shape is honest.
    """
    T = 40
    joints = np.zeros((T, 24, 3))
    # (smpl_index, x, y, z) for a figure standing with Y up.
    base = {
        15: (0.0, 1.65, 0.0),    # head
        12: (0.0, 1.45, 0.0),    # neck
        17: (-0.18, 1.40, 0.0),  # r_shoulder
        16: (0.18, 1.40, 0.0),   # l_shoulder
        19: (-0.20, 1.12, 0.0),  # r_elbow
        18: (0.20, 1.12, 0.0),   # l_elbow
        21: (-0.22, 0.85, 0.0),  # r_wrist
        20: (0.22, 0.85, 0.0),   # l_wrist
        2:  (-0.10, 0.95, 0.0),  # r_hip
        1:  (0.10, 0.95, 0.0),   # l_hip
        5:  (-0.11, 0.52, 0.0),  # r_knee
        4:  (0.11, 0.52, 0.0),   # l_knee
        8:  (-0.11, 0.08, 0.0),  # r_ankle
        7:  (0.11, 0.08, 0.0),   # l_ankle
    }
    for idx, xyz in base.items():
        joints[:, idx] = xyz

    # Swing the left arm up and back down over the clip.
    swing = np.sin(np.linspace(0, np.pi, T))
    joints[:, 18, 1] += 0.25 * swing
    joints[:, 20, 1] += 0.55 * swing
    joints[:, 20, 0] += 0.10 * swing
    return joints


def assert_no_limb_collapse(seq: np.ndarray, sources: np.ndarray,
                            tol: float = 0.02) -> None:
    """Interpolated limb lengths must stay within the range the source poses span.

    This is the property bone-space interpolation exists to protect. Raw xy lerping
    between two distant poses pulls the midpoint inside the arc, so a limb shrinks well
    below either endpoint and pops back -- an interpolated length outside the source
    range is exactly that failure.

    Note it is not "all limbs are one fixed length": library poses deliberately scale
    limbs (`squat` shortens the legs, `knee_up_r` the raised thigh), so a sequence
    crossing them genuinely varies. What must not happen is going outside that span.
    """
    from dancekit.skeleton import ROOT, to_bones

    _, ln_seq, _ = to_bones(seq)
    _, ln_src, _ = to_bones(np.asarray(sources))

    for j in range(NUM_JOINTS):
        if j == ROOT:
            continue
        lo, hi = ln_src[:, j].min(), ln_src[:, j].max()
        if hi < 1e-6:
            continue
        slack = tol * hi
        assert ln_seq[:, j].min() >= lo - slack, (
            f"bone {j} shrank to {ln_seq[:, j].min():.4f}, below its source range "
            f"[{lo:.4f}, {hi:.4f}]")
        assert ln_seq[:, j].max() <= hi + slack, (
            f"bone {j} stretched to {ln_seq[:, j].max():.4f}, above its source range "
            f"[{lo:.4f}, {hi:.4f}]")
