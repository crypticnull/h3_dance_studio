"""Finding the held shapes in a clip, and spotting conformed slow motion."""

from __future__ import annotations

import numpy as np
import pytest

from dancekit import poselib
from dancekit.keypose import (body_speed, detect_keyposes, detect_slowmo, phrase_report)
from dancekit.skeleton import NUM_JOINTS

from .conftest import make_hold_move_hold


# --- body speed -------------------------------------------------------------------------

def test_body_speed_has_one_value_per_frame(hold_sequence):
    pose, _, _ = hold_sequence
    assert body_speed(pose).shape == (pose.shape[0],)


def test_body_speed_is_non_negative(hold_sequence):
    pose, _, _ = hold_sequence
    assert np.all(body_speed(pose) >= 0)


def test_a_still_figure_has_no_speed():
    pose = np.stack([poselib.get("neutral")] * 30)
    assert body_speed(pose).max() < 1e-9


def test_body_speed_ignores_the_dancer_walking_across_frame():
    """Global translation is not choreography. If it registered, every clip with a
    camera pan would be full of phantom keyposes."""
    still = np.stack([poselib.get("arms_up_v")] * 30)
    travelling = still.copy()
    travelling[:, :, 0] += np.linspace(0, 0.4, 30)[:, None]
    assert body_speed(travelling).max() < 1e-6


def test_body_speed_ignores_a_camera_push_in():
    still = np.stack([poselib.get("arms_up_v")] * 30)
    zooming = still.copy()
    for t in range(30):
        centre = zooming[t, 1, :2].copy()
        zooming[t, :, :2] = centre + (zooming[t, :, :2] - centre) * (1 + 0.02 * t)
    assert body_speed(zooming).max() < 1e-6


def test_body_speed_rises_when_the_body_changes_shape(hold_sequence):
    pose, holds, _ = hold_sequence
    spd = body_speed(pose)
    mid_transition = (holds[0][1] + holds[1][0]) // 2
    hold_centre = (holds[0][0] + holds[0][1]) // 2
    assert spd[mid_transition] > spd[hold_centre] * 5


def test_body_speed_weights_limbs_over_the_face():
    """A bobbing head must not read as dancing."""
    base = poselib.get("neutral")

    head = np.stack([base] * 20)
    head[:, [0, 14, 15, 16, 17], 1] += 0.02 * np.sin(np.linspace(0, 6, 20))[:, None]

    arm = np.stack([base] * 20)
    arm[:, [4], 1] += 0.02 * np.sin(np.linspace(0, 6, 20))[:, None]

    assert body_speed(arm).max() > body_speed(head).max()


# --- keypose detection --------------------------------------------------------------------

def test_keyposes_land_on_the_held_shapes(hold_sequence):
    """The whole retiming path rests on this: the frames it finds must be the frames
    the dancer was actually holding a shape."""
    pose, holds, fps = hold_sequence
    idx = detect_keyposes(pose, fps=fps)

    for start, end in holds:
        inside = idx[(idx >= start) & (idx <= end)]
        assert len(inside) >= 1, f"no keypose anywhere in the hold at frames {start}..{end}"


def test_every_keypose_sits_inside_a_hold(hold_sequence):
    """The converse of the above: a keypose in the middle of a transition would pin a
    half-finished shape to a beat."""
    pose, holds, fps = hold_sequence
    idx = detect_keyposes(pose, fps=fps)

    for i in idx:
        assert any(start <= i <= end for start, end in holds), (
            f"keypose at frame {i} falls in a transition, not on a held shape")


def test_keyposes_avoid_the_transitions(hold_sequence):
    pose, _, fps = hold_sequence
    idx = detect_keyposes(pose, fps=fps)
    spd = body_speed(pose)
    assert spd[idx].mean() < spd.mean(), "keyposes should sit at speed minima"


def test_keyposes_include_the_first_and_last_frame(hold_sequence):
    """A clip usually starts and ends on a held shape and the peak finder cannot see
    either, so they are added explicitly."""
    pose, _, fps = hold_sequence
    idx = detect_keyposes(pose, fps=fps)
    assert idx[0] == 0
    assert idx[-1] == pose.shape[0] - 1


def test_keyposes_are_sorted_and_unique(hold_sequence):
    pose, _, fps = hold_sequence
    idx = detect_keyposes(pose, fps=fps)
    assert np.all(np.diff(idx) > 0)


def test_keyposes_are_valid_frame_indices(hold_sequence):
    pose, _, fps = hold_sequence
    idx = detect_keyposes(pose, fps=fps)
    assert idx.dtype.kind == "i"
    assert idx.min() >= 0 and idx.max() < pose.shape[0]


def test_lower_prominence_finds_more_shapes(hold_sequence):
    pose, _, fps = hold_sequence
    loose = detect_keyposes(pose, fps=fps, prominence=0.02)
    strict = detect_keyposes(pose, fps=fps, prominence=0.6)
    assert len(loose) >= len(strict)


def test_max_count_keeps_the_deepest_holds(hold_sequence):
    pose, _, fps = hold_sequence
    idx = detect_keyposes(pose, fps=fps, prominence=0.02, max_count=3)
    assert len(idx) <= 3
    assert np.all(np.diff(idx) > 0), "capping must not disturb the ordering"


def test_min_gap_enforces_spacing():
    shapes = [poselib.get(n) for n in ("neutral", "arms_up_v", "neutral", "squat")]
    pose, _ = make_hold_move_hold(shapes, hold_frames=4, move_frames=2)
    idx = detect_keyposes(pose, fps=30.0, min_gap_s=0.5, prominence=0.01)
    interior = idx[1:-1]
    if len(interior) > 1:
        assert np.all(np.diff(interior) >= 15 - 1)


def test_detect_keyposes_on_a_degenerate_clip():
    assert detect_keyposes(np.zeros((1, NUM_JOINTS, 3)), fps=30.0).tolist() == [0]
    assert len(detect_keyposes(np.zeros((2, NUM_JOINTS, 3)), fps=30.0)) >= 1


def test_detect_keyposes_on_a_motionless_clip():
    pose = np.stack([poselib.get("neutral")] * 40)
    idx = detect_keyposes(pose, fps=30.0)
    assert len(idx) >= 2 and idx[0] == 0 and idx[-1] == 39


# --- diagnostics ---------------------------------------------------------------------------

def test_phrase_report_describes_the_clip(hold_sequence):
    pose, _, fps = hold_sequence
    idx = detect_keyposes(pose, fps=fps)
    rep = phrase_report(pose, idx, fps)

    assert rep["frames"] == pose.shape[0]
    assert rep["keyposes"] == len(idx)
    assert rep["duration_s"] == pytest.approx(pose.shape[0] / fps, abs=0.01)
    assert rep["peak_speed"] >= rep["mean_speed"]
    assert rep["implied_bpm"] == pytest.approx(60.0 / rep["mean_gap_s"], abs=0.1)


def test_phrase_report_reports_even_timing_as_low_variation():
    """gap_cv is how the CLI tells a danced clip from freestyle; evenly spaced hits
    must score low."""
    shapes = [poselib.get(n) for n in
              ("neutral", "arms_up_v", "neutral", "squat", "neutral", "lunge_r")]
    pose, _ = make_hold_move_hold(shapes, hold_frames=10, move_frames=5)
    idx = detect_keyposes(pose, fps=30.0)
    rep = phrase_report(pose, idx, 30.0)
    assert rep["gap_cv"] < 0.5


def test_phrase_report_handles_a_single_keypose():
    pose = np.stack([poselib.get("neutral")] * 10)
    rep = phrase_report(pose, np.array([0]), 30.0)
    assert rep["mean_gap_s"] is None
    assert "implied_bpm" not in rep


# --- slow motion -----------------------------------------------------------------------------

def test_duplicated_frames_are_flagged_as_slow_motion():
    """Conformed slow motion repeats frames; that teaches weightless movement and is
    poison both as a retiming source and as training data."""
    shapes = [poselib.get(n) for n in ("neutral", "arms_up_v", "squat")]
    pose, _ = make_hold_move_hold(shapes, hold_frames=6, move_frames=6)
    conformed = np.repeat(pose, 3, axis=0)          # 3x frame-duplicated conform

    out = detect_slowmo(conformed, fps=30.0)
    assert out["likely_slowmo"] is True
    assert out["duplicate_frame_ratio"] > 0.2


def test_normal_motion_is_not_flagged():
    """Jitter matters here: a real pose track never repeats a frame exactly, and it is
    that exact repetition the duplicate-frame test looks for."""
    shapes = [poselib.get(n) for n in
              ("neutral", "arms_up_v", "squat", "lunge_r", "arms_out_t")]
    pose, _ = make_hold_move_hold(shapes, hold_frames=3, move_frames=3, jitter=1e-3)
    out = detect_slowmo(pose, fps=30.0)
    assert out["likely_slowmo"] is False


def test_long_static_holds_read_as_duplicate_frames():
    """A documented consequence worth pinning: a synthetic clip whose holds are exactly
    repeated frames trips the duplicate-frame test even at normal speed. Detector jitter
    is what keeps real footage clear of it."""
    shapes = [poselib.get(n) for n in ("neutral", "arms_up_v", "squat")]
    pose, _ = make_hold_move_hold(shapes, hold_frames=8, move_frames=2, jitter=0.0)
    assert detect_slowmo(pose, fps=30.0)["duplicate_frame_ratio"] > 0.2


def test_very_slow_movement_is_flagged_on_peak_speed():
    """The other tell: real footage retimed rather than frame-duplicated never reaches
    a plausible peak speed."""
    a, b = poselib.get("neutral"), poselib.get("arms_up_v")
    u = np.linspace(0, 1, 400)[:, None, None]
    pose = a * (1 - u) + b * u

    out = detect_slowmo(pose, fps=30.0)
    assert out["likely_slowmo"] is True


def test_detect_slowmo_reports_numbers_not_numpy_scalars():
    """These go straight into a JSON report."""
    import json
    pose = np.stack([poselib.get("neutral")] * 10)
    json.dumps(detect_slowmo(pose, fps=30.0))
