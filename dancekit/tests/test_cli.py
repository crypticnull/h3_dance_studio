"""End-to-end runs through the command line.

These are the paths a user actually touches, and they are the only tests that check the
commands write the files the README says they write.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from dancekit import poselib
from dancekit.cli import main
from dancekit.poseio import load_pose_json
from dancekit.skeleton import NUM_JOINTS

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg not on PATH")


# --- beats -------------------------------------------------------------------------------

def test_beats_prints_a_grid(click_track, capsys):
    main(["beats", click_track["path"]])
    out = capsys.readouterr().out
    assert "tempo" in out
    assert "beats:" in out


def test_beats_writes_json(click_track, tmp_path, capsys):
    dest = tmp_path / "grid.json"
    main(["beats", click_track["path"], "-o", str(dest)])

    data = json.loads(dest.read_text())
    assert data["tempo"] == pytest.approx(click_track["bpm"], abs=0.01)
    assert len(data["beats"]) > 20
    assert data["downbeats"][0] in data["beats"]


def test_beats_honours_a_forced_tempo(click_track, tmp_path):
    dest = tmp_path / "g.json"
    main(["beats", click_track["path"], "-o", str(dest), "--bpm", "60",
          "--no-rigid"])
    assert json.loads(dest.read_text())["tempo"] == pytest.approx(60.0, abs=1.0)


# --- poses -------------------------------------------------------------------------------

def test_poses_writes_a_contact_sheet(tmp_path, capsys):
    import cv2
    dest = tmp_path / "library.png"
    main(["poses", "-o", str(dest)])

    assert cv2.imread(str(dest)) is not None
    assert f"{len(poselib.LIBRARY)} poses" in capsys.readouterr().out


def test_poses_can_include_mirrors(tmp_path, capsys):
    main(["poses", "-o", str(tmp_path / "m.png"), "--mirrors"])
    assert f"{2 * len(poselib.LIBRARY)} poses" in capsys.readouterr().out


# --- compose -------------------------------------------------------------------------------

@needs_ffmpeg
def test_compose_writes_everything_the_readme_promises(click_track, tmp_path, capsys):
    out = tmp_path / "out"
    main(["compose", click_track["path"], "-o", str(out), "--seed", "3",
          "--width", "128", "--height", "224", "--fps", "24"])

    assert (out / "pose.json").exists()
    assert (out / "skeleton.mp4").exists()
    assert (out / "compose.json").exists()

    info = json.loads((out / "compose.json").read_text())
    assert info["seed"] == 3
    assert info["frames"] > 0

    pose, meta = load_pose_json(out / "pose.json")
    assert pose.shape == (info["frames"], NUM_JOINTS, 3)
    assert meta["canvas_width"] == 128 and meta["canvas_height"] == 224
    assert np.all(np.isfinite(pose))

    printed = capsys.readouterr().out
    assert "tempo" in printed and "sections:" in printed


@needs_ffmpeg
def test_compose_frames_flag_writes_a_png_sequence(click_track, tmp_path):
    out = tmp_path / "out"
    main(["compose", click_track["path"], "-o", str(out), "--frames",
          "--width", "64", "--height", "64", "--seconds", "4"])

    pngs = sorted((out / "frames").glob("*.png"))
    assert len(pngs) > 10


@needs_ffmpeg
def test_compose_is_reproducible_from_the_command_line(click_track, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for dest in (a, b):
        main(["compose", click_track["path"], "-o", str(dest), "--seed", "7",
              "--width", "64", "--height", "64", "--seconds", "6"])

    assert (a / "pose.json").read_text() == (b / "pose.json").read_text()


@needs_ffmpeg
def test_compose_with_a_harvested_library(click_track, tmp_path, capsys):
    """The two halves of the generated path joined up: harvest a vocabulary, compose
    with it."""
    from dancekit.harvest import build_vocabulary, canonicalize, pose_energy, save_library

    cands = [{"pose": canonicalize(poselib.get(n)), "energy": pose_energy(poselib.get(n)),
              "source": "s.mp4", "frame": i}
             for i, n in enumerate(["arms_up_v", "squat", "lunge_r", "kick_r"])]
    lib, meta = build_vocabulary(cands, min_distance=0.20)
    vocab = tmp_path / "vocabulary.npz"
    save_library(vocab, lib, meta)

    out = tmp_path / "out"
    main(["compose", click_track["path"], "-o", str(out), "--library", str(vocab),
          "--width", "64", "--height", "64", "--seconds", "6"])

    assert (out / "pose.json").exists()
    assert "vocabulary:" in capsys.readouterr().out


# --- retime -----------------------------------------------------------------------------------

@needs_ffmpeg
def test_retime_maps_a_clip_onto_a_song(pose_json, click_track, tmp_path, capsys):
    out = tmp_path / "rt"
    main(["retime", str(pose_json), click_track["path"], "-o", str(out),
          "--src-fps", "30", "--width", "64", "--height", "64"])

    assert (out / "pose.json").exists()
    assert (out / "skeleton.mp4").exists()

    info = json.loads((out / "retime.json").read_text())
    assert info["anchors"] >= 2
    assert info["out_frames"] > 0

    pose, _ = load_pose_json(out / "pose.json")
    assert pose.shape[0] == info["out_frames"]

    printed = capsys.readouterr().out
    assert "keyposes" in printed and "anchors" in printed


@needs_ffmpeg
def test_retime_loop_covers_the_whole_song(pose_json, click_track, tmp_path):
    once, looped = tmp_path / "once", tmp_path / "looped"
    common = ["--src-fps", "30", "--width", "64", "--height", "64"]
    main(["retime", str(pose_json), click_track["path"], "-o", str(once)] + common)
    main(["retime", str(pose_json), click_track["path"], "-o", str(looped), "--loop"]
         + common)

    a = json.loads((once / "retime.json").read_text())
    b = json.loads((looped / "retime.json").read_text())
    assert b["anchors"] > a["anchors"]


# --- smpl ---------------------------------------------------------------------------------------

@needs_ffmpeg
def test_smpl_converts_3d_motion(smpl_motion, tmp_path, capsys):
    motion = tmp_path / "motion.npy"
    np.save(motion, smpl_motion)

    out = tmp_path / "smpl_out"
    main(["smpl", str(motion), "-o", str(out), "--width", "64", "--height", "64"])

    assert (out / "pose.json").exists()
    assert (out / "skeleton.mp4").exists()

    pose, _ = load_pose_json(out / "pose.json")
    assert pose.shape == (smpl_motion.shape[0], NUM_JOINTS, 3)
    assert "->" in capsys.readouterr().out


@needs_ffmpeg
def test_smpl_accepts_a_flattened_array(smpl_motion, tmp_path):
    """EDGE and friends sometimes emit (T, J*3)."""
    motion = tmp_path / "flat.npy"
    np.save(motion, smpl_motion.reshape(smpl_motion.shape[0], -1))

    out = tmp_path / "flat_out"
    main(["smpl", str(motion), "-o", str(out), "--width", "64", "--height", "64"])
    assert (out / "pose.json").exists()


# --- harvest -------------------------------------------------------------------------------------

def test_harvest_builds_a_vocabulary_from_json(pose_json, tmp_path, capsys):
    out = tmp_path / "vocab"
    main(["harvest", str(pose_json), "-o", str(out), "--src-fps", "30",
          "--cell", "64", "--cols", "4"])

    assert (out / "vocabulary.npz").exists()
    assert (out / "vocabulary.png").exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["clips"] and manifest["vocabulary"]
    for entry in manifest["vocabulary"]:
        assert entry["source"] == pose_json.name
        assert 0.0 <= entry["energy"] <= 1.0

    assert "vocabulary entries" in capsys.readouterr().out


def test_harvest_output_loads_and_composes(pose_json, tmp_path):
    from dancekit.beatgrid import synthetic
    from dancekit.compose import compose
    from dancekit.harvest import load_library

    out = tmp_path / "vocab"
    main(["harvest", str(pose_json), "-o", str(out), "--src-fps", "30",
          "--cell", "64", "--cols", "4"])

    lib, meta, emap = load_library(out / "vocabulary.npz")
    pose, _ = compose(synthetic(120.0, duration=16.0), library=lib, energy_map=emap,
                      seed=1)
    assert np.all(np.isfinite(pose))


def test_harvest_walks_a_folder(pose_json, tmp_path):
    folder = tmp_path / "clips"
    folder.mkdir()
    shutil.copy(pose_json, folder / "one.json")
    shutil.copy(pose_json, folder / "two.json")

    out = tmp_path / "vocab"
    main(["harvest", str(folder), "-o", str(out), "--src-fps", "30",
          "--cell", "64", "--cols", "4"])

    manifest = json.loads((out / "manifest.json").read_text())
    assert {c["source"] for c in manifest["clips"]} == {"one.json", "two.json"}


def test_harvest_on_an_empty_folder_exits_with_a_message(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc:
        main(["harvest", str(empty), "-o", str(tmp_path / "v")])
    assert "no videos or pose json" in str(exc.value)


def test_harvest_explains_itself_when_the_filters_reject_everything(pose_json, tmp_path):
    """--min-count is the best filter for noisy footage, since detector glitches do not
    repeat -- but set too high it leaves nothing, and the user needs to be told which
    knob did it rather than getting a numpy traceback."""
    with pytest.raises(SystemExit) as exc:
        main(["harvest", str(pose_json), "-o", str(tmp_path / "v"), "--src-fps", "30",
              "--min-count", "999", "--cell", "64", "--cols", "4"])

    msg = str(exc.value)
    assert "--min-count" in msg
    assert "999" in msg


# --- render ----------------------------------------------------------------------------------------

@needs_ffmpeg
def test_render_turns_pose_json_into_video(pose_json, tmp_path, capsys):
    dest = tmp_path / "skeleton.mp4"
    main(["render", str(pose_json), "-o", str(dest), "--width", "64", "--height", "64"])

    assert dest.exists() and dest.stat().st_size > 0
    assert "frames ->" in capsys.readouterr().out


@needs_ffmpeg
def test_render_muxes_audio(pose_json, click_track, tmp_path):
    dest = tmp_path / "synced.mp4"
    main(["render", str(pose_json), "-o", str(dest), "--audio", click_track["path"],
          "--width", "64", "--height", "64"])
    assert dest.exists() and dest.stat().st_size > 0


# --- parser ------------------------------------------------------------------------------------------

def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        main([])


def test_an_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit):
        main(["waltz"])


@pytest.mark.parametrize("cmd", ["beats", "compose", "poses", "extract", "retime",
                                 "smpl", "harvest", "render", "prep"])
def test_every_documented_subcommand_exists(cmd):
    """The module docstring advertises these; a missing one is a broken README."""
    with pytest.raises(SystemExit) as exc:
        main([cmd, "--help"])
    assert exc.value.code == 0


# --- error paths ----------------------------------------------------------------------------------

def test_extract_without_rtmlib_says_how_to_fix_it(tmp_path):
    """The most likely first failure for anyone following the video path."""
    try:
        import rtmlib  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("rtmlib is installed, so this branch cannot be reached")

    with pytest.raises(SystemExit) as exc:
        main(["extract", str(tmp_path / "clip.mp4"), "-o", str(tmp_path / "p.json")])
    assert "rtmlib" in str(exc.value)


def test_harvest_skips_files_it_cannot_read_and_keeps_going(pose_json, tmp_path, capsys):
    folder = tmp_path / "clips"
    folder.mkdir()
    shutil.copy(pose_json, folder / "good.json")
    (folder / "broken.json").write_text("{not json at all")

    out = tmp_path / "vocab"
    main(["harvest", str(folder), "-o", str(out), "--src-fps", "30",
          "--cell", "64", "--cols", "4"])

    manifest = json.loads((out / "manifest.json").read_text())
    assert {c["source"] for c in manifest["clips"]} == {"good.json"}
    assert "skip broken.json" in capsys.readouterr().err


def test_harvest_skip_slowmo_drops_conformed_clips(tmp_path, capsys):
    """Conformed slow motion teaches weightless movement, so --skip-slowmo drops it.
    Here every clip is slow motion, so nothing survives."""
    pose = np.repeat(np.stack([poselib.get(n) for n in
                               ("neutral", "arms_up_v", "squat")] * 3), 4, axis=0)
    frames = [{"people": [{"pose_keypoints_2d": [float(v) for v in p.reshape(-1)]}],
               "canvas_width": 832, "canvas_height": 1472} for p in pose]
    clip = tmp_path / "slow.json"
    clip.write_text(json.dumps(frames))

    with pytest.raises(SystemExit) as exc:
        main(["harvest", str(clip), "-o", str(tmp_path / "v"), "--src-fps", "30",
              "--skip-slowmo", "--cell", "64", "--cols", "4"])

    assert "no usable poses" in str(exc.value)
    assert "looks like slow motion" in capsys.readouterr().out
