"""3D SMPL joints -> 2D OpenPose projection."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from dancekit.skeleton import NUM_JOINTS, ROOT
from dancekit.smpl2d import (SMPL_TO_OP, frame_to_canvas, load_joints, orbit, project,
                             rotation, smpl_to_openpose)


# --- rotation and projection ----------------------------------------------------------

def test_rotation_is_orthonormal():
    for az, el in [(0, 0), (37, 0), (0, 21), (145, -33)]:
        R = rotation(az, el)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_identity_rotation_at_zero():
    np.testing.assert_allclose(rotation(0, 0), np.eye(3), atol=1e-15)


def test_project_flips_y_into_image_space():
    """World Y is up, image Y grows downward. Without the flip every dance renders
    upside down."""
    j = np.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]])
    xy = project(j, flip_y=True)
    assert xy[0, 0, 1] < xy[0, 1, 1], "the higher joint must get the smaller image y"

    xy_noflip = project(j, flip_y=False)
    assert xy_noflip[0, 0, 1] > xy_noflip[0, 1, 1]


def test_project_is_orthographic_and_ignores_depth():
    """No focal length is invented, so moving a joint along the view axis must not
    change its image position."""
    near = np.array([[[0.3, 0.5, 0.0]]])
    far = np.array([[[0.3, 0.5, 9.0]]])
    np.testing.assert_allclose(project(near), project(far), atol=1e-12)


def test_azimuth_turns_the_figure():
    j = np.array([[[1.0, 0.0, 0.0]]])
    np.testing.assert_allclose(project(j, azimuth=90.0)[0, 0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(project(j, azimuth=180.0)[0, 0, 0], -1.0, atol=1e-12)


def test_z_up_input_is_swizzled():
    """Some exporters emit Z-up. Read as Y-up it renders the dancer lying down."""
    j = np.array([[[0.0, 0.0, 1.0]]])
    y_up = project(j, up_axis="y")
    z_up = project(j, up_axis="z")
    assert abs(y_up[0, 0, 1]) < 1e-12       # the height went into the ignored axis
    assert z_up[0, 0, 1] == pytest.approx(-1.0, abs=1e-12)


def test_project_preserves_shape():
    j = np.zeros((7, 24, 3))
    assert project(j).shape == (7, 24, 2)


# --- canvas fitting ------------------------------------------------------------------------

def test_frame_to_canvas_respects_margins():
    xy = np.array([[[0.0, 0.0], [0.0, 2.0], [-1.0, 1.0], [1.0, 1.0]]])
    out = frame_to_canvas(xy, headroom=0.10, floor=0.04)
    assert out[..., 1].min() == pytest.approx(0.10, abs=1e-9)
    assert out[..., 1].max() == pytest.approx(1.0 - 0.04, abs=1e-9)


def test_frame_to_canvas_centres_horizontally():
    xy = np.array([[[2.0, 0.0], [4.0, 1.0]]])
    out = frame_to_canvas(xy)
    assert out[..., 0].mean() == pytest.approx(0.5, abs=1e-9)


def test_fixed_scale_keeps_the_figure_from_breathing():
    """Per-frame fitting rescales the dancer every time they raise an arm, and the video
    model faithfully reproduces that as a zooming camera."""
    xy = np.zeros((2, 3, 2))
    xy[0] = [[0, 0], [0, 1.0], [0.2, 0.5]]      # compact frame
    xy[1] = [[0, 0], [0, 2.0], [0.2, 1.0]]      # same figure, arm up

    fixed = frame_to_canvas(xy, fixed_scale=True)
    per_frame = frame_to_canvas(xy, fixed_scale=False)

    fixed_h = [np.ptp(fixed[t, :, 1]) for t in range(2)]
    frame_h = [np.ptp(per_frame[t, :, 1]) for t in range(2)]
    assert fixed_h[1] > fixed_h[0] * 1.5, "fixed scaling must keep the size difference"
    assert frame_h[0] == pytest.approx(frame_h[1], abs=1e-9), (
        "per-frame scaling normalises both frames to the same height")


def test_frame_to_canvas_survives_a_perfectly_flat_sequence():
    """A figure with no vertical extent would divide by zero without the epsilon."""
    xy = np.zeros((3, 4, 2))
    assert np.all(np.isfinite(frame_to_canvas(xy)))


# --- the full conversion ----------------------------------------------------------------------

def test_smpl_to_openpose_shape_and_confidence(smpl_motion):
    out = smpl_to_openpose(smpl_motion)
    assert out.shape == (smpl_motion.shape[0], NUM_JOINTS, 3)
    assert np.all(out[:, :, 2] == 1.0)
    assert np.all(np.isfinite(out))


def test_smpl_to_openpose_fills_every_joint(smpl_motion):
    """A joint left at (0,0) draws a limb shooting into the corner of the frame."""
    out = smpl_to_openpose(smpl_motion)
    for j in range(NUM_JOINTS):
        assert np.any(out[:, j, :2] != 0.0), f"joint {j} was never populated"


def test_smpl_to_openpose_stays_on_canvas(smpl_motion):
    out = smpl_to_openpose(smpl_motion)
    assert out[:, :, :2].min() >= -1e-9
    assert out[:, :, :2].max() <= 1.0 + 1e-9


def test_smpl_to_openpose_puts_the_head_above_the_feet(smpl_motion):
    out = smpl_to_openpose(smpl_motion)
    assert np.all(out[:, 0, 1] < out[:, 10, 1]), "nose should be above the right ankle"
    assert np.all(out[:, ROOT, 1] < out[:, 8, 1]), "neck should be above the hip"


def test_smpl_to_openpose_keeps_left_and_right_the_right_way_round(smpl_motion):
    """The figure faces camera, so the character's right side sits on the image left.
    Swapping these mirrors every generated dance."""
    out = smpl_to_openpose(smpl_motion)
    assert np.all(out[:, 2, 0] < out[:, 5, 0]), "r_shoulder should be left of l_shoulder"
    assert np.all(out[:, 8, 0] < out[:, 11, 0]), "r_hip should be left of l_hip"


def test_smpl_to_openpose_tracks_the_moving_arm(smpl_motion):
    """The fixture swings the left arm up through the middle of the clip; that has to
    survive the projection or the motion is being dropped."""
    out = smpl_to_openpose(smpl_motion)
    wrist_y = out[:, 7, 1]
    mid = len(wrist_y) // 2
    assert wrist_y[mid] < wrist_y[0] - 0.05, "left wrist should be higher mid-clip"


def test_synthesised_face_points_sit_around_the_head(smpl_motion):
    """SMPL has no facial keypoints. The synthesised ones only need to be plausible,
    but an eye behind the neck is not."""
    out = smpl_to_openpose(smpl_motion)
    nose, neck = out[:, 0, :2], out[:, ROOT, :2]
    head_len = np.linalg.norm(nose - neck, axis=-1)
    for idx in (14, 15, 16, 17):
        d = np.linalg.norm(out[:, idx, :2] - nose, axis=-1)
        assert np.all(d < head_len), f"joint {idx} is further from the nose than the head is long"

    assert np.all(out[:, 14, 0] < out[:, 15, 0]), "r_eye should be left of l_eye in image space"
    assert np.all(out[:, 16, 0] < out[:, 17, 0]), "r_ear should be left of l_ear"


def test_smpl_to_openpose_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        smpl_to_openpose(np.zeros((10, 24)))
    with pytest.raises(ValueError):
        smpl_to_openpose(np.zeros((10, 24, 2)))


def test_smpl_to_openpose_handles_extra_joints(smpl_motion):
    """SMPL-X and friends emit more than 24 joints; the extras must just be ignored."""
    padded = np.concatenate([smpl_motion, np.zeros((smpl_motion.shape[0], 20, 3))], axis=1)
    np.testing.assert_allclose(smpl_to_openpose(padded), smpl_to_openpose(smpl_motion))


def test_smpl_mapping_is_a_bijection_over_the_joints_it_covers():
    assert len(set(SMPL_TO_OP.values())) == len(SMPL_TO_OP)
    assert set(SMPL_TO_OP) == set(range(14)), (
        "OpenPose joints 0..13 come from SMPL; 14..17 are synthesised")


# --- camera and loading -----------------------------------------------------------------------

def test_orbit_spans_the_requested_arc():
    a = orbit(50, start=10.0, degrees=20.0)
    assert a.shape == (50,)
    assert a[0] == pytest.approx(10.0)
    assert a[-1] == pytest.approx(30.0)


def test_orbit_of_zero_degrees_is_a_locked_camera():
    a = orbit(30, start=5.0, degrees=0.0)
    assert np.all(a == 5.0)


def test_load_joints_from_npy(tmp_path, smpl_motion):
    p = tmp_path / "m.npy"
    np.save(p, smpl_motion)
    np.testing.assert_allclose(load_joints(str(p)), smpl_motion)


@pytest.mark.parametrize("key", ["joints", "joints3d", "poses", "motion", "pred_joints"])
def test_load_joints_finds_the_known_npz_keys(tmp_path, smpl_motion, key):
    p = tmp_path / f"m_{key}.npz"
    np.savez(p, **{key: smpl_motion})
    np.testing.assert_allclose(load_joints(str(p)), smpl_motion)


def test_load_joints_falls_back_to_the_only_array_in_an_npz(tmp_path, smpl_motion):
    p = tmp_path / "odd.npz"
    np.savez(p, something_else=smpl_motion)
    np.testing.assert_allclose(load_joints(str(p)), smpl_motion)


@pytest.mark.parametrize("key", ["full_pose", "joints", "joints3d", "smpl_poses",
                                 "pred_joints"])
def test_load_joints_from_a_pickled_dict(tmp_path, smpl_motion, key):
    p = tmp_path / f"m_{key}.pkl"
    p.write_bytes(pickle.dumps({key: smpl_motion, "unrelated": 1}))
    np.testing.assert_allclose(load_joints(str(p)), smpl_motion)


def test_load_joints_from_a_bare_pickled_array(tmp_path, smpl_motion):
    p = tmp_path / "bare.pkl"
    p.write_bytes(pickle.dumps(smpl_motion))
    np.testing.assert_allclose(load_joints(str(p)), smpl_motion)
