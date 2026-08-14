"""Warping a source dance onto a new beat grid.

The load-bearing test here is `test_keyposes_land_on_the_grid`: the entire premise of
the project is that a detected held shape ends up at the beat time it was assigned.
"""

from __future__ import annotations

import numpy as np
import pytest

from dancekit import poselib
from dancekit.beatgrid import synthetic
from dancekit.keypose import detect_keyposes
from dancekit.retime import build_anchors, ease, resample_bones, retime_to_grid
from dancekit.skeleton import NUM_JOINTS, ROOT, to_bones

from .conftest import assert_no_limb_collapse, make_hold_move_hold


# --- easing -------------------------------------------------------------------------------

def test_ease_starts_at_zero_and_reaches_one():
    u = np.linspace(0, 1, 101)
    f = ease(u, snap=1.0)
    assert f[0] == pytest.approx(0.0, abs=1e-12)
    assert f[-1] == pytest.approx(1.0, abs=1e-12)


def test_ease_is_monotonic():
    u = np.linspace(0, 1, 501)
    assert np.all(np.diff(ease(u, snap=1.0)) >= -1e-12)
    assert np.all(np.diff(ease(u, snap=0.5)) >= -1e-12)


def test_snap_finishes_the_move_early_and_holds():
    """snap=0.5 means the shape arrives halfway through the interval and is held until
    the next beat. That hold is what reads as a 'hit' rather than a drift."""
    u = np.linspace(0, 1, 101)
    f = ease(u, snap=0.5)
    assert f[u <= 0.5][-1] == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(f[u >= 0.5], 1.0, atol=1e-9)


def test_lower_snap_arrives_sooner():
    u = np.linspace(0, 1, 201)
    early = ease(u, snap=0.4)
    late = ease(u, snap=1.0)
    assert np.all(early >= late - 1e-12)


def test_overshoot_travels_past_the_pose_and_settles_back():
    u = np.linspace(0, 1, 401)
    f = ease(u, snap=1.0, overshoot=0.2)
    assert f.max() > 1.0, "overshoot should carry the shape past its target"
    assert f[-1] == pytest.approx(1.0, abs=1e-9), "and then settle exactly onto it"


def test_no_overshoot_never_exceeds_the_target():
    f = ease(np.linspace(0, 1, 401), snap=1.0, overshoot=0.0)
    assert f.max() <= 1.0 + 1e-12


def test_bigger_overshoot_travels_further():
    u = np.linspace(0, 1, 401)
    assert ease(u, overshoot=0.3).max() > ease(u, overshoot=0.1).max()


def test_snap_is_clamped_to_a_usable_range():
    """A snap of 0 would divide by zero; the clamp keeps a silly value survivable."""
    f = ease(np.linspace(0, 1, 51), snap=0.0)
    assert np.all(np.isfinite(f))
    assert f[-1] == pytest.approx(1.0, abs=1e-9)


# --- resampling ----------------------------------------------------------------------------

def test_resample_at_integer_positions_returns_the_source(hold_sequence):
    pose, _, _ = hold_sequence
    out = resample_bones(pose, np.arange(pose.shape[0], dtype=float))
    np.testing.assert_allclose(out[:, :, :2], pose[:, :, :2], atol=1e-9)


def test_resample_keeps_limbs_solid_through_a_big_transition():
    """This is why interpolation happens in bone space. Lerping raw joint xy between two
    distant poses pulls the midpoint inside the arc, so limbs visibly shrink."""
    a, b = poselib.get("arms_up_v"), poselib.get("reach_down")
    pose = np.stack([a, b])
    out = resample_bones(pose, np.linspace(0, 1, 60))
    assert_no_limb_collapse(out, pose, tol=0.001)


def test_bone_space_beats_naive_xy_lerp_on_limb_length():
    a, b = poselib.get("arms_up_v"), poselib.get("reach_down")
    u = np.linspace(0, 1, 60)

    bone = resample_bones(np.stack([a, b]), u)
    naive = a[None] * (1 - u[:, None, None]) + b[None] * u[:, None, None]

    def worst_forearm_wobble(seq):
        _, ln, _ = to_bones(seq)
        track = ln[:, 4]                       # r_wrist <- r_elbow
        return (track.max() - track.min()) / track.max()

    assert worst_forearm_wobble(bone) < worst_forearm_wobble(naive) / 4


def test_resample_clamps_outside_the_clip(hold_sequence):
    pose, _, _ = hold_sequence
    T = pose.shape[0]
    out = resample_bones(pose, np.array([-5.0, 0.0, T - 1.0, T + 20.0]))
    np.testing.assert_allclose(out[0, :, :2], out[1, :, :2], atol=1e-9)
    np.testing.assert_allclose(out[3, :, :2], out[2, :, :2], atol=1e-9)


def test_resample_output_length_follows_the_request(hold_sequence):
    pose, _, _ = hold_sequence
    assert resample_bones(pose, np.linspace(0, 10, 37)).shape == (37, NUM_JOINTS, 3)


def test_root_damping_pulls_travel_back_toward_centre():
    """Otherwise a source clip where the dancer crosses frame walks straight out of a
    generated shot."""
    base = poselib.get("neutral")
    pose = np.stack([base] * 40)
    pose[:, :, 0] += np.linspace(0, 0.5, 40)[:, None]

    loose = resample_bones(pose, np.arange(40.0), root_damping=0.0)
    tight = resample_bones(pose, np.arange(40.0), root_damping=0.9)

    assert np.ptp(tight[:, ROOT, 0]) < np.ptp(loose[:, ROOT, 0]) * 0.2


def test_root_damping_does_not_deform_the_body():
    pose = np.stack([poselib.get("lunge_r")] * 20)
    pose[:, :, 0] += np.linspace(0, 0.4, 20)[:, None]
    out = resample_bones(pose, np.arange(20.0), root_damping=0.8)
    assert_no_limb_collapse(out, pose, tol=1e-9)


def test_resample_fills_detector_dropouts(hold_sequence):
    pose, _, _ = hold_sequence
    holed = pose.copy()
    holed[5:9, 4, 2] = 0.0
    out = resample_bones(holed, np.arange(pose.shape[0], dtype=float))
    assert np.all(np.isfinite(out))
    assert np.all(out[5:9, 4, 2] > 0)


# --- anchor matching --------------------------------------------------------------------------

def test_sequential_anchors_map_keyposes_in_order():
    keys = np.array([0.0, 0.4, 0.9, 1.5])
    grid = np.arange(8) * 0.5
    src, tgt = build_anchors(keys, grid, mode="sequential")
    np.testing.assert_allclose(src, keys)
    np.testing.assert_allclose(tgt, [0.0, 0.5, 1.0, 1.5])


def test_stride_maps_to_every_nth_grid_point():
    """For a half-time source, one keypose per two grid points."""
    keys = np.array([0.0, 0.4, 0.9])
    grid = np.arange(8) * 0.5
    _, tgt = build_anchors(keys, grid, stride=2)
    np.testing.assert_allclose(tgt, [0.0, 1.0, 2.0])


def test_start_index_offsets_the_entry_point():
    keys = np.array([0.0, 0.4])
    grid = np.arange(8) * 0.5
    _, tgt = build_anchors(keys, grid, start_index=3)
    np.testing.assert_allclose(tgt, [1.5, 2.0])


def test_sequential_drops_keyposes_past_the_end_of_the_grid():
    keys = np.arange(10) * 0.3
    grid = np.arange(4) * 0.5
    src, tgt = build_anchors(keys, grid)
    assert len(src) == len(tgt) == 4


def test_nearest_mode_snaps_each_keypose_to_its_closest_grid_point():
    """Nearest mode first stretches the keypose span across the whole grid span, then
    snaps. With a source that already spans the grid the stretch is a no-op, which is
    what isolates the snapping here."""
    grid = np.arange(8) * 0.5                       # 0 .. 3.5 s
    keys = np.array([0.0, 0.52, 1.03, 1.48, 3.5])   # same span as the grid
    _, tgt = build_anchors(keys, grid, mode="nearest")
    np.testing.assert_allclose(tgt, [0.0, 0.5, 1.0, 1.5, 3.5])


def test_nearest_mode_stretches_the_source_across_the_grid():
    """A short source is spread over the whole grid rather than left bunched at the
    start -- so the last keypose lands near the end of the song, not a second in."""
    grid = np.arange(8) * 0.5                       # 0 .. 3.5 s
    keys = np.array([0.0, 0.52, 1.03, 1.48])        # only 1.48 s of source
    _, tgt = build_anchors(keys, grid, mode="nearest")
    np.testing.assert_allclose(tgt, [0.0, 1.0, 2.5, 3.5])


def test_nearest_mode_output_is_monotonic_and_deduped():
    """Two keyposes snapping to the same beat would make time run backwards."""
    keys = np.array([0.0, 0.05, 0.09, 0.52, 0.55])
    grid = np.arange(10) * 0.5
    _, tgt = build_anchors(keys, grid, mode="nearest")
    assert np.all(np.diff(tgt) > 0)
    assert len(set(tgt.tolist())) == len(tgt)


def test_nearest_mode_rescales_a_source_of_a_different_length():
    """A 2 s source onto an 8 s grid should spread across it, not bunch at the start."""
    keys = np.linspace(0, 2.0, 5)
    grid = np.arange(17) * 0.5              # 0 .. 8 s
    _, tgt = build_anchors(keys, grid, mode="nearest")
    assert tgt[-1] > 6.0


# --- the full retime ---------------------------------------------------------------------------

@pytest.fixture
def danced_clip():
    """A source clip with clean, detectable holds at 30 fps."""
    shapes = [poselib.get(n) for n in
              ("neutral", "arms_up_v", "lunge_r", "squat", "arms_out_t", "knee_up_r")]
    pose, holds = make_hold_move_hold(shapes, hold_frames=10, move_frames=6)
    return pose, holds, 30.0


def test_keyposes_land_on_the_grid(danced_clip):
    """The core promise of the tool. At each target beat time, the output pose must be
    the source keypose assigned to it."""
    pose, _, src_fps = danced_clip
    keys = detect_keyposes(pose, fps=src_fps)
    grid = synthetic(120.0, duration=20.0)
    out_fps = 24.0

    retimed, info = retime_to_grid(pose, src_fps, keys, grid.beats, out_fps=out_fps)

    for src_t, tgt_t in info["anchor_pairs_s"]:
        frame = int(round(tgt_t * out_fps))
        if frame >= retimed.shape[0]:
            continue
        expected = pose[int(round(src_t * src_fps))]
        np.testing.assert_allclose(
            retimed[frame, :, :2], expected[:, :2], atol=2e-3,
            err_msg=f"output at beat {tgt_t:.3f}s is not the source shape at {src_t:.3f}s")


def test_anchor_times_are_actual_grid_times(danced_clip):
    pose, _, src_fps = danced_clip
    keys = detect_keyposes(pose, fps=src_fps)
    grid = synthetic(120.0, duration=20.0)

    _, info = retime_to_grid(pose, src_fps, keys, grid.beats)
    beats = set(np.round(grid.beats, 9))
    for _, tgt_t in info["anchor_pairs_s"]:
        assert round(tgt_t, 3) in {round(b, 3) for b in beats}


def test_retimed_output_has_the_requested_length(danced_clip):
    pose, _, src_fps = danced_clip
    keys = detect_keyposes(pose, fps=src_fps)
    grid = synthetic(120.0, duration=20.0)

    retimed, info = retime_to_grid(pose, src_fps, keys, grid.beats, out_fps=24.0,
                                   out_frames=97)
    assert retimed.shape == (97, NUM_JOINTS, 3)
    assert info["out_frames"] == 97


def test_retime_keeps_limbs_solid(danced_clip):
    pose, _, src_fps = danced_clip
    keys = detect_keyposes(pose, fps=src_fps)
    grid = synthetic(120.0, duration=20.0)
    retimed, _ = retime_to_grid(pose, src_fps, keys, grid.beats, snap=0.6, overshoot=0.15)
    assert_no_limb_collapse(retimed, pose, tol=0.01)


def test_retime_output_is_finite(danced_clip):
    pose, _, src_fps = danced_clip
    keys = detect_keyposes(pose, fps=src_fps)
    grid = synthetic(120.0, duration=20.0)
    retimed, _ = retime_to_grid(pose, src_fps, keys, grid.beats)
    assert np.all(np.isfinite(retimed))


def test_source_time_never_runs_backwards(danced_clip):
    """The map from output time to source time has to stay monotonic or the dance plays
    in reverse in places."""
    pose, _, src_fps = danced_clip
    keys = detect_keyposes(pose, fps=src_fps)
    grid = synthetic(120.0, duration=20.0)
    _, info = retime_to_grid(pose, src_fps, keys, grid.beats)

    pairs = np.array(info["anchor_pairs_s"])
    assert np.all(np.diff(pairs[:, 0]) > 0), "source anchor times went backwards"
    assert np.all(np.diff(pairs[:, 1]) > 0), "target anchor times went backwards"


def test_retime_needs_at_least_two_anchors(danced_clip):
    pose, _, src_fps = danced_clip
    with pytest.raises(ValueError, match="at least 2 anchors"):
        retime_to_grid(pose, src_fps, np.array([0]), np.array([0.0, 0.5, 1.0]))


def test_loop_extends_a_short_phrase_over_a_long_grid(danced_clip):
    """A 2 s source against a 20 s song: without looping the dance stops after the
    source runs out."""
    pose, _, src_fps = danced_clip
    keys = detect_keyposes(pose, fps=src_fps)
    grid = synthetic(120.0, duration=30.0)

    once, info_once = retime_to_grid(pose, src_fps, keys, grid.beats, loop=False)
    looped, info_loop = retime_to_grid(pose, src_fps, keys, grid.beats, loop=True)

    assert info_loop["anchors"] > info_once["anchors"]
    assert info_loop["duration_s"] > info_once["duration_s"]
    assert looped.shape[0] > once.shape[0]


def test_looping_replays_the_phrase(danced_clip):
    """The point of looping: the same shapes come back round, so the second pass should
    revisit poses from the first."""
    pose, _, src_fps = danced_clip
    keys = detect_keyposes(pose, fps=src_fps)
    grid = synthetic(120.0, duration=30.0)

    _, info = retime_to_grid(pose, src_fps, keys, grid.beats, loop=True)
    src_times = np.array(info["anchor_pairs_s"])[:, 0]
    phrase_len = keys[-1] / src_fps - keys[0] / src_fps
    assert src_times[-1] > phrase_len, "source time should run past the end of the clip"


def test_loop_keeps_target_times_on_the_grid(danced_clip):
    pose, _, src_fps = danced_clip
    keys = detect_keyposes(pose, fps=src_fps)
    grid = synthetic(120.0, duration=30.0)

    _, info = retime_to_grid(pose, src_fps, keys, grid.beats, loop=True)
    beats = {round(b, 3) for b in grid.beats}
    for _, tgt_t in info["anchor_pairs_s"]:
        assert round(tgt_t, 3) in beats


def test_nearest_mode_survives_a_loose_source(danced_clip):
    pose, _, src_fps = danced_clip
    keys = detect_keyposes(pose, fps=src_fps)
    grid = synthetic(120.0, duration=20.0)
    retimed, info = retime_to_grid(pose, src_fps, keys, grid.beats, mode="nearest")
    assert np.all(np.isfinite(retimed))
    assert info["anchors"] >= 2


def test_info_is_json_serialisable(danced_clip):
    import json
    pose, _, src_fps = danced_clip
    keys = detect_keyposes(pose, fps=src_fps)
    grid = synthetic(120.0, duration=20.0)
    _, info = retime_to_grid(pose, src_fps, keys, grid.beats)
    json.loads(json.dumps(info))


def test_nearest_mode_stops_when_the_grid_runs_out():
    """More keyposes than grid points: pairing has to stop rather than reuse a beat."""
    keys = np.linspace(0, 3.0, 20)
    grid = np.arange(4) * 0.5
    src, tgt = build_anchors(keys, grid, mode="nearest")
    assert len(src) == len(tgt) <= len(grid)
    assert np.all(np.diff(tgt) > 0)


def test_grid_points_before_the_first_anchor_hold_the_opening_shape(danced_clip):
    """Output frames earlier than the first anchor must show the first shape, not
    extrapolate backwards out of the clip."""
    pose, _, src_fps = danced_clip
    keys = detect_keyposes(pose, fps=src_fps)
    grid = synthetic(120.0, duration=20.0, offset=1.0)   # first beat a second in

    retimed, info = retime_to_grid(pose, src_fps, keys, grid.beats, out_fps=24.0)
    first_anchor_frame = int(round(info["anchor_pairs_s"][0][1] * 24.0))
    assert first_anchor_frame > 0
    for f in range(first_anchor_frame):
        np.testing.assert_allclose(retimed[f, :, :2], retimed[first_anchor_frame, :, :2],
                                   atol=1e-9)
