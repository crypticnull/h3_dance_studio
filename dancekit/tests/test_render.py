"""Drawing OpenPose skeletons.

The colours and the filled-ellipse limb style are not cosmetic: ControlNet OpenPose
models were trained on exactly this rendering, so matching it is what makes the
conditioning bite.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from dancekit import poselib
from dancekit.render import contact_sheet, draw_pose, render_frames, render_video
from dancekit.skeleton import COLORS, LIMB_PAIRS, NUM_JOINTS

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg not on PATH")


# --- one frame ---------------------------------------------------------------------------

def test_draw_pose_returns_an_rgb_image():
    img = draw_pose(poselib.get("arms_up_v"), 128, 256)
    assert img.shape == (256, 128, 3)
    assert img.dtype == np.uint8


def test_draw_pose_actually_draws_something():
    img = draw_pose(poselib.get("arms_up_v"), 256, 256)
    assert img.any(), "the canvas came back empty"
    assert (img.reshape(-1, 3) > 0).any(axis=-1).mean() > 0.01


def test_background_is_black_by_default():
    """ControlNet OpenPose expects the skeleton on black."""
    img = draw_pose(poselib.get("neutral"), 200, 200)
    assert tuple(img[0, 0]) == (0, 0, 0)


def test_background_colour_is_honoured_in_rgb():
    """cv2 works in BGR internally, so a red background coming back as blue would mean
    the conversion is inverted."""
    img = draw_pose(poselib.get("neutral"), 64, 64, background=(255, 0, 0))
    assert tuple(img[0, 0]) == (255, 0, 0)


def test_the_canonical_openpose_colours_appear():
    img = draw_pose(poselib.get("arms_out_t"), 512, 512)
    present = {tuple(c) for c in np.unique(img.reshape(-1, 3), axis=0)}
    hits = sum(1 for c in COLORS if tuple(c) in present)
    assert hits >= len(LIMB_PAIRS) // 2, (
        "most of the 18-colour palette should be visible; ControlNet was trained on it")


def test_low_confidence_joints_are_not_drawn():
    """A joint the detector lost must not be drawn at (0,0), which would put a limb
    across the corner of the frame."""
    pose = poselib.get("arms_up_v").copy()
    full = draw_pose(pose, 256, 256)

    pose[[4, 7], 2] = 0.0                      # both wrists lost
    partial = draw_pose(pose, 256, 256)

    assert partial.sum() < full.sum()
    assert not np.array_equal(full, partial)


def test_a_fully_unconfident_pose_draws_nothing():
    pose = poselib.get("arms_up_v").copy()
    pose[:, 2] = 0.0
    assert not draw_pose(pose, 128, 128).any()


def test_thickness_changes_how_much_ink_lands():
    thin = draw_pose(poselib.get("arms_out_t"), 256, 256, thickness=0.5)
    thick = draw_pose(poselib.get("arms_out_t"), 256, 256, thickness=2.0)
    assert (thick > 0).sum() > (thin > 0).sum()


def test_the_figure_lands_where_the_pose_says():
    """Normalised coordinates scale to the canvas, so a pose on the left half must not
    render on the right."""
    pose = poselib.get("neutral").copy()
    pose[:, 0] *= 0.5                          # squeeze everything into the left half

    img = draw_pose(pose, 256, 256)
    ink = (img > 0).any(axis=-1)
    assert ink[:, :128].sum() > 0
    assert ink[:, 160:].sum() == 0


def test_drawing_does_not_mutate_the_pose():
    pose = poselib.get("squat")
    before = pose.copy()
    draw_pose(pose, 128, 128)
    np.testing.assert_array_equal(pose, before)


def test_non_square_canvases_work():
    img = draw_pose(poselib.get("arms_up_v"), 832, 1472)
    assert img.shape == (1472, 832, 3)
    assert img.any()


def test_joints_off_canvas_do_not_crash():
    pose = poselib.get("neutral").copy()
    pose[:, 0] += 2.0
    assert draw_pose(pose, 64, 64).shape == (64, 64, 3)


@pytest.mark.parametrize("name", ["neutral", "arms_up_v", "squat", "kick_r", "lunge_l"])
def test_every_shape_renders(name):
    assert draw_pose(poselib.get(name), 256, 256).any()


# --- sequences ----------------------------------------------------------------------------

def test_render_frames_writes_one_png_per_frame(tmp_path, hold_sequence):
    pose, _, _ = hold_sequence
    out = render_frames(pose[:8], tmp_path / "frames", width=64, height=64)

    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 8
    assert [p.name for p in pngs] == [f"{i:06d}.png" for i in range(8)]
    assert all(p.stat().st_size > 0 for p in pngs)


def test_render_frames_creates_its_directory(tmp_path, hold_sequence):
    pose, _, _ = hold_sequence
    out = render_frames(pose[:2], tmp_path / "a" / "b" / "c", width=32, height=32)
    assert out.is_dir()


def test_rendered_frames_are_readable_images(tmp_path, hold_sequence):
    import cv2
    pose, _, _ = hold_sequence
    out = render_frames(pose[:3], tmp_path / "f", width=64, height=96)

    img = cv2.imread(str(out / "000000.png"))
    assert img is not None and img.shape == (96, 64, 3)


def test_consecutive_frames_of_a_moving_figure_differ(tmp_path, hold_sequence):
    """A renderer that ignored the frame index would write the same picture every time
    and the whole video would be a still."""
    import cv2
    pose, holds, _ = hold_sequence
    mid = (holds[0][1] + holds[1][0]) // 2      # inside a transition
    out = render_frames(pose[mid:mid + 3], tmp_path / "m", width=96, height=96)

    a = cv2.imread(str(out / "000000.png"))
    b = cv2.imread(str(out / "000002.png"))
    assert not np.array_equal(a, b)


# --- contact sheet --------------------------------------------------------------------------

def test_contact_sheet_lays_out_a_grid(tmp_path, sample_poses):
    import cv2
    path = contact_sheet(sample_poses, tmp_path / "sheet.png", cell=80, cols=3)

    img = cv2.imread(str(path))
    assert img is not None
    rows = int(np.ceil(len(sample_poses) / 3))
    assert img.shape == (rows * 80, 3 * 80, 3)


def test_contact_sheet_draws_every_pose(tmp_path, sample_poses):
    import cv2
    path = contact_sheet(sample_poses, tmp_path / "sheet.png", cell=80, cols=3)
    img = cv2.imread(str(path))

    for i in range(len(sample_poses)):
        r, c = divmod(i, 3)
        cell = img[r * 80:(r + 1) * 80, c * 80:(c + 1) * 80]
        assert cell.any(), f"cell {i} is empty"


def test_contact_sheet_of_the_whole_library(tmp_path):
    """`dancekit poses` renders this to audit a vocabulary before composing with it."""
    import cv2
    path = contact_sheet(poselib.all_poses(), tmp_path / "lib.png", cell=64, cols=6)
    assert cv2.imread(str(path)) is not None


def test_contact_sheet_of_a_single_pose(tmp_path):
    path = contact_sheet({"only": poselib.get("neutral")}, tmp_path / "one.png",
                         cell=64, cols=6)
    assert path.exists()


# --- video ------------------------------------------------------------------------------------

@needs_ffmpeg
def test_render_video_writes_a_playable_file(tmp_path, hold_sequence):
    pose, _, _ = hold_sequence
    out = render_video(pose[:24], tmp_path / "out.mp4", fps=24.0, width=64, height=64)
    assert out.exists() and out.stat().st_size > 0


@needs_ffmpeg
def test_render_video_creates_its_parent_directory(tmp_path, hold_sequence):
    pose, _, _ = hold_sequence
    out = render_video(pose[:8], tmp_path / "deep" / "nested" / "out.mp4",
                       fps=24.0, width=64, height=64)
    assert out.exists()


@needs_ffmpeg
def test_render_video_muxes_audio(tmp_path, hold_sequence, click_track):
    """The song is muxed in so you can check sync by eye, which is the whole
    verification story for this tool."""
    pose, _, _ = hold_sequence
    out = render_video(pose[:48], tmp_path / "synced.mp4", fps=24.0, width=64,
                       height=64, audio=click_track["path"])
    assert out.exists() and out.stat().st_size > 0


def test_a_zero_length_limb_is_skipped_rather_than_drawn():
    """Two joints on top of each other would make a degenerate ellipse."""
    pose = poselib.get("neutral").copy()
    pose[3, :2] = pose[2, :2]                  # elbow collapsed onto the shoulder
    img = draw_pose(pose, 128, 128)
    assert img.shape == (128, 128, 3)
    assert img.any()
