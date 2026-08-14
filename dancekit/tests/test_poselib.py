"""The built-in pose vocabulary and its bone-angle authoring."""

from __future__ import annotations

import numpy as np
import pytest

from dancekit import poselib
from dancekit.poselib import (LIBRARY, MIRROR_PAIRS, NEUTRAL, all_poses, build_pose,
                              energies, library_from_poses, mirror, shift)
from dancekit.skeleton import NUM_JOINTS, PARENTS, ROOT, to_bones


# --- forward kinematics ------------------------------------------------------------------

def test_build_pose_returns_a_well_formed_pose():
    pose = build_pose()
    assert pose.shape == (NUM_JOINTS, 3)
    assert np.all(pose[:, 2] == 1.0)
    assert np.all(np.isfinite(pose))


def test_build_pose_honours_the_root_argument():
    pose = build_pose(root=(0.25, 0.6))
    np.testing.assert_allclose(pose[ROOT, :2], [0.25, 0.6])


def test_neutral_angles_come_back_out_of_the_pose():
    """Authoring is in angle space, so the resolved pose must actually carry the angles
    that were written down -- otherwise every override in LIBRARY means something else."""
    pose = build_pose()
    ang, _, _ = to_bones(pose[None])
    for j, (deg, _) in NEUTRAL.items():
        assert np.rad2deg(ang[0, j]) == pytest.approx(deg, abs=1e-6)


def test_overrides_change_only_the_named_bone_angle():
    ang_before, _, _ = to_bones(build_pose()[None])
    ang_after, _, _ = to_bones(build_pose({3: -125.0})[None])

    assert np.rad2deg(ang_after[0, 3]) == pytest.approx(-125.0, abs=1e-6)
    for j in NEUTRAL:
        if j != 3:
            assert ang_after[0, j] == pytest.approx(ang_before[0, j], abs=1e-9)


def test_moving_a_parent_carries_its_children():
    """The elbow angle is unchanged but the wrist must travel with the shoulder, or the
    limb comes apart."""
    base = build_pose()
    swung = build_pose({3: -125.0})
    assert not np.allclose(base[4, :2], swung[4, :2])

    _, ln_base, _ = to_bones(base[None])
    _, ln_swung, _ = to_bones(swung[None])
    assert ln_swung[0, 4] == pytest.approx(ln_base[0, 4], abs=1e-9)


def test_length_scale_shortens_only_the_named_bone():
    _, ln, _ = to_bones(build_pose(length_scale={4: 0.5})[None])
    assert ln[0, 4] == pytest.approx(NEUTRAL[4][1] * 0.5, abs=1e-9)
    assert ln[0, 7] == pytest.approx(NEUTRAL[7][1], abs=1e-9)


def test_angle_convention_is_image_space():
    """0 points right and +90 points DOWN, because y grows downward in image space.
    Getting this backwards would silently render every pose upside down."""
    pose = build_pose({3: 0.0})       # r_elbow straight out to image right
    assert pose[3, 0] > pose[2, 0]
    assert pose[3, 1] == pytest.approx(pose[2, 1], abs=1e-9)

    pose = build_pose({3: 90.0})      # straight down
    assert pose[3, 1] > pose[2, 1]


# --- mirroring ------------------------------------------------------------------------------

def test_mirror_is_its_own_inverse(sample_poses):
    for name, pose in sample_poses.items():
        np.testing.assert_allclose(mirror(mirror(pose)), pose, atol=1e-12,
                                   err_msg=f"{name} did not survive a double mirror")


def test_mirror_swaps_left_and_right_joints():
    pose = poselib.get("lunge_r")
    flipped = mirror(pose)
    for a, b in MIRROR_PAIRS:
        assert flipped[a, 0] == pytest.approx(1.0 - pose[b, 0], abs=1e-12)
        assert flipped[a, 1] == pytest.approx(pose[b, 1], abs=1e-12)


def test_mirror_preserves_bone_lengths():
    """A mirrored pose is the same body seen from the other side; if limbs change length
    the mirror is not a rigid flip."""
    pose = poselib.get("knee_up_r")
    _, ln, _ = to_bones(pose[None])
    _, ln_m, _ = to_bones(mirror(pose)[None])
    for a, b in MIRROR_PAIRS:
        assert ln_m[0, a] == pytest.approx(ln[0, b], abs=1e-12)


def test_mirroring_a_symmetric_pose_changes_nothing():
    np.testing.assert_allclose(mirror(poselib.get("neutral")),
                               poselib.get("neutral"), atol=1e-12)


def test_mirror_about_a_custom_axis():
    pose = poselib.get("lunge_r")
    flipped = mirror(pose, axis=0.3)
    assert flipped[ROOT, 0] == pytest.approx(0.6 - pose[ROOT, 0], abs=1e-12)


def test_shift_translates_without_deforming():
    pose = poselib.get("squat")
    moved = shift(pose, dx=0.1, dy=-0.05)
    np.testing.assert_allclose(moved[:, 0], pose[:, 0] + 0.1, atol=1e-12)
    np.testing.assert_allclose(moved[:, 1], pose[:, 1] - 0.05, atol=1e-12)

    _, ln, _ = to_bones(pose[None])
    _, ln_moved, _ = to_bones(moved[None])
    np.testing.assert_allclose(ln_moved, ln, atol=1e-12)


# --- the library itself -----------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(LIBRARY))
def test_every_library_pose_is_finite_and_on_canvas(name):
    """A pose whose joints leave 0..1 renders partly off the edge of the frame, which
    ControlNet reads as a cropped body."""
    pose = poselib.get(name)
    assert pose.shape == (NUM_JOINTS, 3)
    assert np.all(np.isfinite(pose))
    assert pose[:, :2].min() >= 0.0, f"{name} runs off the canvas"
    assert pose[:, :2].max() <= 1.0, f"{name} runs off the canvas"


@pytest.mark.parametrize("name", sorted(LIBRARY))
def test_every_library_pose_has_a_sane_energy(name):
    assert 0.0 <= LIBRARY[name]["energy"] <= 1.0


@pytest.mark.parametrize("name", sorted(LIBRARY))
def test_library_overrides_name_real_joints(name):
    spec = LIBRARY[name]
    for key in ("ov", "ls"):
        for j in (spec.get(key) or {}):
            assert j in NEUTRAL, f"{name} {key} targets joint {j}, which has no bone"


def test_library_poses_are_distinct():
    """Two identical entries waste a vocabulary slot and bias the composer's sampling."""
    poses = {n: poselib.get(n) for n in LIBRARY}
    names = list(poses)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            spread = np.abs(poses[a][:, :2] - poses[b][:, :2]).max()
            assert spread > 1e-3, f"{a} and {b} are the same pose"


def test_all_poses_pairs_every_shape_with_its_mirror():
    poses = all_poses()
    assert len(poses) == 2 * len(LIBRARY)
    for name in LIBRARY:
        assert name in poses and name + "_m" in poses
        np.testing.assert_allclose(poses[name + "_m"], mirror(poses[name]), atol=1e-12)


def test_energies_cover_all_poses_and_match_across_mirrors():
    """The composer indexes the energy map by pose name; a missing key is a KeyError
    mid-compose, and a mirror with a different energy would make sides asymmetric."""
    poses, e = all_poses(), energies()
    assert set(e) == set(poses)
    for name in LIBRARY:
        assert e[name] == e[name + "_m"]


def test_library_spans_a_useful_energy_range():
    vals = list(energies().values())
    assert min(vals) < 0.25 and max(vals) > 0.85, (
        "the composer matches poses to a target energy, so a library bunched in the "
        "middle leaves it nothing to pick for quiet or peak moments")


def test_library_from_poses_pulls_named_frames(hold_sequence):
    pose, holds, _ = hold_sequence
    centres = [(a + b) // 2 for a, b in holds]
    lib = library_from_poses(pose, centres[:3])
    assert len(lib) == 3
    for name, p in lib.items():
        assert p.shape == (NUM_JOINTS, 3)
    np.testing.assert_allclose(lib[f"p{centres[0]:04d}"], pose[centres[0]])


def test_library_from_poses_copies_rather_than_views(hold_sequence):
    """A view would let a later edit of the vocabulary corrupt the source sequence."""
    pose, holds, _ = hold_sequence
    centres = [(a + b) // 2 for a, b in holds]
    lib = library_from_poses(pose, centres[:1])
    key = next(iter(lib))
    lib[key][0, 0] = 999.0
    assert pose[centres[0], 0, 0] != 999.0


def test_parents_and_neutral_describe_the_same_skeleton():
    """Every non-root joint needs a bone definition or FK silently leaves it at (0,0)."""
    for j in range(NUM_JOINTS):
        if j == ROOT:
            continue
        assert j in NEUTRAL, f"joint {j} (parent {PARENTS[j]}) has no NEUTRAL entry"
