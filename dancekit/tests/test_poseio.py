"""Reading and writing ComfyUI POSE_KEYPOINT JSON."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dancekit import poselib
from dancekit.poseio import load_npz, load_pose_json, save_npz, save_pose_json
from dancekit.skeleton import NUM_JOINTS


def _frame(pose: np.ndarray, W: int = 832, H: int = 1472, **extra) -> dict:
    person = {"pose_keypoints_2d": [float(v) for v in pose.reshape(-1)]}
    person.update(extra)
    return {"people": [person], "canvas_width": W, "canvas_height": H}


# --- loading ------------------------------------------------------------------------------

def test_load_returns_normalised_poses(pose_json, hold_sequence):
    pose, _, _ = hold_sequence
    loaded, meta = load_pose_json(pose_json)
    assert loaded.shape == pose.shape
    np.testing.assert_allclose(loaded, pose, atol=1e-6)
    assert meta["canvas_width"] == 832
    assert meta["canvas_height"] == 1472


def test_pixel_coordinates_are_sniffed_and_normalised(tmp_path):
    """Some node versions write pixels and some write 0..1. Reading pixels as normalised
    would put every joint thousands of canvases off-screen."""
    pose = poselib.get("arms_up_v").copy()
    px = pose.copy()
    px[:, 0] *= 832
    px[:, 1] *= 1472

    path = tmp_path / "px.json"
    path.write_text(json.dumps([_frame(px)]))

    loaded, _ = load_pose_json(path)
    np.testing.assert_allclose(loaded[0, :, :2], pose[:, :2], atol=1e-6)
    assert loaded[:, :, :2].max() <= 1.5


def test_normalised_coordinates_are_left_alone(tmp_path):
    pose = poselib.get("arms_up_v")
    path = tmp_path / "norm.json"
    path.write_text(json.dumps([_frame(pose)]))

    loaded, _ = load_pose_json(path)
    np.testing.assert_allclose(loaded[0, :, :2], pose[:, :2], atol=1e-9)


def test_a_bare_dict_is_read_as_a_single_frame(tmp_path):
    pose = poselib.get("squat")
    path = tmp_path / "one.json"
    path.write_text(json.dumps(_frame(pose)))

    loaded, _ = load_pose_json(path)
    assert loaded.shape == (1, NUM_JOINTS, 3)


def test_the_most_confident_person_is_picked(tmp_path):
    """In a dance clip the dancer is the confidently detected body; someone in the
    background is not."""
    dancer = poselib.get("arms_up_v").copy()
    bystander = poselib.get("neutral").copy()
    bystander[:, 2] = 0.1

    frame = {"people": [
        {"pose_keypoints_2d": [float(v) for v in bystander.reshape(-1)]},
        {"pose_keypoints_2d": [float(v) for v in dancer.reshape(-1)]},
    ], "canvas_width": 832, "canvas_height": 1472}

    path = tmp_path / "two.json"
    path.write_text(json.dumps([frame]))

    loaded, _ = load_pose_json(path)
    np.testing.assert_allclose(loaded[0, :, :2], dancer[:, :2], atol=1e-9)


def test_an_explicit_person_index_overrides_the_choice(tmp_path):
    a, b = poselib.get("arms_up_v"), poselib.get("squat")
    frame = {"people": [
        {"pose_keypoints_2d": [float(v) for v in a.reshape(-1)]},
        {"pose_keypoints_2d": [float(v) for v in b.reshape(-1)]},
    ], "canvas_width": 832, "canvas_height": 1472}

    path = tmp_path / "pick.json"
    path.write_text(json.dumps([frame]))

    np.testing.assert_allclose(load_pose_json(path, person=1)[0][0, :, :2],
                               b[:, :2], atol=1e-9)


def test_an_out_of_range_person_index_gives_an_empty_frame(tmp_path):
    path = tmp_path / "oor.json"
    path.write_text(json.dumps([_frame(poselib.get("neutral"))]))
    loaded, _ = load_pose_json(path, person=5)
    assert np.all(loaded[0] == 0.0)


def test_frames_with_nobody_in_them_become_zeros(tmp_path):
    """A detector dropping the figure entirely must not shift every later frame by one."""
    pose = poselib.get("neutral")
    data = [_frame(pose), {"people": [], "canvas_width": 832, "canvas_height": 1472},
            _frame(pose)]
    path = tmp_path / "gap.json"
    path.write_text(json.dumps(data))

    loaded, _ = load_pose_json(path)
    assert loaded.shape == (3, NUM_JOINTS, 3)
    assert np.all(loaded[1] == 0.0)
    assert np.all(loaded[1, :, 2] == 0.0), "an empty frame must have zero confidence"


def test_short_keypoint_arrays_are_padded(tmp_path):
    """Some writers emit fewer than 18 joints; the array shape downstream has to hold."""
    frame = {"people": [{"pose_keypoints_2d": [0.5, 0.5, 1.0] * 12}],
             "canvas_width": 832, "canvas_height": 1472}
    path = tmp_path / "short.json"
    path.write_text(json.dumps([frame]))

    loaded, _ = load_pose_json(path)
    assert loaded.shape == (1, NUM_JOINTS, 3)
    assert np.all(loaded[0, 12:] == 0.0)


def test_extra_keypoints_beyond_body_18_are_dropped(tmp_path):
    frame = {"people": [{"pose_keypoints_2d": [0.5, 0.5, 1.0] * 25}],
             "canvas_width": 832, "canvas_height": 1472}
    path = tmp_path / "long.json"
    path.write_text(json.dumps([frame]))
    assert load_pose_json(path)[0].shape == (1, NUM_JOINTS, 3)


def test_face_and_hand_keypoints_are_carried_through(tmp_path):
    """They are not used, but dropping them silently would lose data on a round trip."""
    pose = poselib.get("neutral")
    path = tmp_path / "extras.json"
    path.write_text(json.dumps([_frame(pose, face_keypoints_2d=[1.0, 2.0, 3.0],
                                       hand_left_keypoints_2d=[4.0])]))

    _, meta = load_pose_json(path)
    assert meta["extras"][0]["face_keypoints_2d"] == [1.0, 2.0, 3.0]
    assert meta["extras"][0]["hand_left_keypoints_2d"] == [4.0]


def test_missing_canvas_dimensions_do_not_crash(tmp_path):
    path = tmp_path / "nocanvas.json"
    path.write_text(json.dumps([{"people": [
        {"pose_keypoints_2d": [float(v) for v in poselib.get("neutral").reshape(-1)]}]}]))
    loaded, meta = load_pose_json(path)
    assert loaded.shape == (1, NUM_JOINTS, 3)
    assert meta["canvas_width"] == 1


# --- saving --------------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path, hold_sequence):
    pose, _, _ = hold_sequence
    path = tmp_path / "rt.json"
    save_pose_json(path, pose)
    loaded, _ = load_pose_json(path)
    np.testing.assert_allclose(loaded, pose, atol=1e-6)


def test_round_trip_through_pixel_space(tmp_path, hold_sequence):
    """Writing pixels and reading them back has to land on the same normalised pose, or
    the ComfyUI hand-off is lossy."""
    pose, _, _ = hold_sequence
    path = tmp_path / "rt_px.json"
    save_pose_json(path, pose, meta={"canvas_width": 832, "canvas_height": 1472},
                   normalised=False)

    loaded, _ = load_pose_json(path)
    np.testing.assert_allclose(loaded[:, :, :2], pose[:, :, :2], atol=1e-5)


def test_saved_json_matches_the_comfyui_shape(tmp_path, hold_sequence):
    pose, _, _ = hold_sequence
    path = tmp_path / "shape.json"
    save_pose_json(path, pose)

    data = json.loads(path.read_text())
    assert isinstance(data, list) and len(data) == pose.shape[0]
    for fr in data[:3]:
        assert set(fr) >= {"people", "canvas_width", "canvas_height"}
        kp = fr["people"][0]["pose_keypoints_2d"]
        assert len(kp) == NUM_JOINTS * 3
        assert all(isinstance(v, float) for v in kp)


def test_save_defaults_to_a_portrait_canvas(tmp_path):
    path = tmp_path / "default.json"
    save_pose_json(path, poselib.get("neutral")[None])
    data = json.loads(path.read_text())
    assert data[0]["canvas_width"] == 832
    assert data[0]["canvas_height"] == 1472


def test_save_writes_back_the_extras_it_was_given(tmp_path):
    pose = poselib.get("neutral")[None]
    path = tmp_path / "extras_out.json"
    save_pose_json(path, pose, meta={"extras": [{"face_keypoints_2d": [9.0]}]})

    data = json.loads(path.read_text())
    assert data[0]["people"][0]["face_keypoints_2d"] == [9.0]


def test_save_survives_fewer_extras_than_frames(tmp_path, hold_sequence):
    pose, _, _ = hold_sequence
    path = tmp_path / "few_extras.json"
    save_pose_json(path, pose, meta={"extras": [{"face_keypoints_2d": [1.0]}]})
    assert len(json.loads(path.read_text())) == pose.shape[0]


def test_meta_from_load_feeds_straight_back_into_save(tmp_path, pose_json):
    """load -> save with the returned meta is the obvious usage; it has to work."""
    pose, meta = load_pose_json(pose_json)
    out = tmp_path / "again.json"
    save_pose_json(out, pose, meta=meta)

    again, _ = load_pose_json(out)
    np.testing.assert_allclose(again, pose, atol=1e-6)


# --- npz ------------------------------------------------------------------------------------

def test_npz_round_trip(tmp_path, hold_sequence):
    pose, _, _ = hold_sequence
    path = tmp_path / "p.npz"
    save_npz(path, pose, fps=np.float64(24.0))

    loaded, extra = load_npz(path)
    np.testing.assert_allclose(loaded, pose)
    assert float(extra["fps"]) == 24.0


def test_npz_without_extras(tmp_path, hold_sequence):
    pose, _, _ = hold_sequence
    path = tmp_path / "bare.npz"
    save_npz(path, pose)
    loaded, extra = load_npz(path)
    np.testing.assert_allclose(loaded, pose)
    assert extra == {}
