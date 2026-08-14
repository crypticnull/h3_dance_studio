"""Skeleton layout, bone-space conversion, normalisation and gap filling."""

from __future__ import annotations

import numpy as np
import pytest

from dancekit import poselib
from dancekit.skeleton import (COLORS, FK_ORDER, JOINT_NAMES, LIMB_PAIRS, LIMB_WEIGHTS,
                               NUM_JOINTS, PARENTS, ROOT, denormalise, fill_missing,
                               from_bones, normalise, shortest_arc, to_bones,
                               torso_scale, unwrap_sequence)


# --- structural invariants -------------------------------------------------------------

def test_layout_tables_agree_on_joint_count():
    assert len(JOINT_NAMES) == NUM_JOINTS
    assert len(PARENTS) == NUM_JOINTS
    assert len(LIMB_WEIGHTS) == NUM_JOINTS
    assert len(FK_ORDER) == NUM_JOINTS


def test_root_is_the_only_parentless_joint():
    assert PARENTS[ROOT] == -1
    assert list(PARENTS).count(-1) == 1


def test_fk_order_resolves_parents_before_children():
    """FK walks this order once with no second pass, so a child appearing before its
    parent would silently build from an uninitialised (0,0)."""
    seen = set()
    for j in FK_ORDER:
        if j != ROOT:
            assert PARENTS[j] in seen, f"joint {j} precedes its parent {PARENTS[j]}"
        seen.add(j)
    assert seen == set(range(NUM_JOINTS))


def test_limb_pairs_and_colours_are_drawable():
    for a, b in LIMB_PAIRS:
        assert 0 <= a < NUM_JOINTS and 0 <= b < NUM_JOINTS
        assert a != b
    assert len(COLORS) >= len(LIMB_PAIRS)
    assert len(COLORS) >= NUM_JOINTS
    for c in COLORS:
        assert len(c) == 3 and all(0 <= v <= 255 for v in c)


# --- bone space ------------------------------------------------------------------------

def test_bone_round_trip_is_lossless(sample_poses):
    pose = np.stack(list(sample_poses.values()))
    ang, ln, root = to_bones(pose)
    rebuilt = from_bones(ang, ln, root, pose[:, :, 2])
    np.testing.assert_allclose(rebuilt, pose, atol=1e-9)


def test_bone_lengths_match_euclidean_distance_to_parent(sample_poses):
    pose = sample_poses["lunge_r"][None]
    _, ln, _ = to_bones(pose)
    for j in range(NUM_JOINTS):
        if j == ROOT:
            continue
        expected = np.linalg.norm(pose[0, j, :2] - pose[0, PARENTS[j], :2])
        assert ln[0, j] == pytest.approx(expected, abs=1e-12)


def test_root_bone_entries_are_zeroed_placeholders(sample_poses):
    ang, ln, _ = to_bones(sample_poses["squat"][None])
    assert ang[0, ROOT] == 0.0
    assert ln[0, ROOT] == 0.0


def test_from_bones_places_root_where_told():
    ang = np.zeros((3, NUM_JOINTS))
    ln = np.full((3, NUM_JOINTS), 0.1)
    root = np.array([[0.2, 0.3], [0.5, 0.5], [0.9, 0.1]])
    out = from_bones(ang, ln, root, np.ones((3, NUM_JOINTS)))
    np.testing.assert_allclose(out[:, ROOT, :2], root)


# --- angle handling ---------------------------------------------------------------------

def test_shortest_arc_crosses_the_pi_boundary_the_short_way():
    """An elbow going from 179 to -179 degrees has moved 2 degrees, not -358."""
    a0 = np.deg2rad(np.array([179.0, -179.0, 0.0, 170.0]))
    a1 = np.deg2rad(np.array([-179.0, 179.0, 10.0, -160.0]))
    got = np.rad2deg(shortest_arc(a0, a1))
    np.testing.assert_allclose(got, [2.0, -2.0, 10.0, 30.0], atol=1e-9)


def test_shortest_arc_picks_one_direction_for_an_exact_half_turn():
    """A 180 degree flip is genuinely ambiguous -- both directions are equally short.
    Pinning which one comes back only so a change to it is a visible decision."""
    got = shortest_arc(np.array([np.pi / 2]), np.array([-np.pi / 2]))
    assert abs(got[0]) == pytest.approx(np.pi)


def test_shortest_arc_stays_within_half_a_turn():
    rng = np.random.default_rng(4)
    a0 = rng.uniform(-20, 20, 500)
    a1 = rng.uniform(-20, 20, 500)
    d = shortest_arc(a0, a1)
    assert np.all(d > -np.pi - 1e-12) and np.all(d <= np.pi + 1e-12)
    # The delta must still land on the same angle, modulo a full turn.
    np.testing.assert_allclose(np.cos(a0 + d), np.cos(a1), atol=1e-9)
    np.testing.assert_allclose(np.sin(a0 + d), np.sin(a1), atol=1e-9)


def test_unwrap_sequence_removes_wrap_discontinuities():
    """A joint rotating steadily past pi must read as continuous, or interpolating it
    spins the limb the long way round."""
    true_track = np.linspace(0, 4 * np.pi, 200)
    wrapped = (true_track + np.pi) % (2 * np.pi) - np.pi
    ang = np.tile(wrapped[:, None], (1, NUM_JOINTS))
    out = unwrap_sequence(ang)
    steps = np.abs(np.diff(out[:, 3]))
    assert steps.max() < 0.5, "unwrapped track still contains a jump"
    np.testing.assert_allclose(out[:, 3] - out[0, 3], true_track - true_track[0], atol=1e-6)


# --- normalisation ------------------------------------------------------------------------

def test_normalise_denormalise_round_trip(hold_sequence):
    pose, _, _ = hold_sequence
    local, root, scale = normalise(pose)
    np.testing.assert_allclose(denormalise(local, root, scale), pose, atol=1e-9)


def test_normalise_is_invariant_to_translation(hold_sequence):
    """A dancer walking across frame, or a camera pan, must not read as choreography."""
    pose, _, _ = hold_sequence
    moved = pose.copy()
    drift = np.linspace(0, 0.3, pose.shape[0])
    moved[:, :, 0] += drift[:, None]
    moved[:, :, 1] += 0.1

    np.testing.assert_allclose(normalise(pose)[0][:, :, :2],
                               normalise(moved)[0][:, :, :2], atol=1e-9)


def test_normalise_is_invariant_to_scale(hold_sequence):
    """A camera push-in must not read as movement either."""
    pose, _, _ = hold_sequence
    zoomed = pose.copy()
    zoomed[:, :, :2] *= 1.7

    np.testing.assert_allclose(normalise(pose)[0][:, :, :2],
                               normalise(zoomed)[0][:, :, :2], atol=1e-9)


def test_torso_scale_uses_neck_to_mid_hip():
    pose = poselib.get("neutral")[None]
    mid_hip = 0.5 * (pose[0, 8, :2] + pose[0, 11, :2])
    expected = np.linalg.norm(mid_hip - pose[0, ROOT, :2])
    assert torso_scale(pose)[0] == pytest.approx(expected, rel=1e-9)


def test_torso_scale_falls_back_to_shoulder_width_when_hips_drop_out():
    pose = np.stack([poselib.get("neutral")] * 2)
    pose[1, [8, 11], 2] = 0.0                      # hips lost to occlusion
    sh = np.linalg.norm(pose[1, 2, :2] - pose[1, 5, :2]) * 1.6
    assert torso_scale(pose)[1] == pytest.approx(sh, rel=1e-9)


def test_torso_scale_falls_back_to_the_median_when_nothing_is_visible():
    pose = np.stack([poselib.get("neutral")] * 3)
    pose[1, :, 2] = 0.0                            # frame 1 has no usable body at all
    scale = torso_scale(pose)
    assert scale[1] == pytest.approx(np.median([scale[0], scale[2]]), rel=1e-9)


def test_torso_scale_never_returns_zero():
    """Downstream divides by this, so a degenerate frame must not produce a divide by
    zero and a canvas full of infinities."""
    pose = np.zeros((2, NUM_JOINTS, 3))
    assert np.all(torso_scale(pose) > 0)
    assert np.all(np.isfinite(normalise(pose)[0]))


# --- gap filling ---------------------------------------------------------------------------

def test_fill_missing_interpolates_a_dropout():
    pose = np.zeros((5, NUM_JOINTS, 3))
    pose[:, :, 2] = 1.0
    pose[:, 4, 0] = [0.0, 0.0, 0.0, 0.0, 0.4]
    pose[1:4, 4, 2] = 0.0                          # wrist lost for three frames

    out = fill_missing(pose)
    np.testing.assert_allclose(out[:, 4, 0], [0.0, 0.1, 0.2, 0.3, 0.4], atol=1e-9)
    assert np.all(out[1:4, 4, 2] > 0), "filled joints need drawable confidence"


def test_fill_missing_holds_at_the_ends():
    pose = np.zeros((4, NUM_JOINTS, 3))
    pose[:, :, 2] = 1.0
    pose[:, 7, 1] = [0.9, 0.5, 0.6, 0.9]
    pose[0, 7, 2] = 0.0
    pose[3, 7, 2] = 0.0

    out = fill_missing(pose)
    assert out[0, 7, 1] == pytest.approx(0.5)      # held from the first good frame
    assert out[3, 7, 1] == pytest.approx(0.6)      # held from the last good frame


def test_fill_missing_zeroes_a_joint_never_seen():
    pose = np.zeros((4, NUM_JOINTS, 3))
    pose[:, :, 2] = 1.0
    pose[:, 10, :2] = 0.77
    pose[:, 10, 2] = 0.0

    out = fill_missing(pose)
    assert np.all(out[:, 10, :2] == 0.0)
    assert np.all(out[:, 10, 2] == 0.0)


def test_fill_missing_leaves_good_frames_untouched(hold_sequence):
    pose, _, _ = hold_sequence
    np.testing.assert_allclose(fill_missing(pose), pose, atol=1e-12)


def test_fill_missing_does_not_mutate_its_input():
    pose = np.zeros((3, NUM_JOINTS, 3))
    pose[:, :, 2] = 1.0
    pose[:, 4, 0] = [0.0, 5.0, 1.0]
    pose[1, 4, 2] = 0.0
    before = pose.copy()

    fill_missing(pose)
    np.testing.assert_array_equal(pose, before)
