"""Turning clips into a pose vocabulary."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dancekit import poselib
from dancekit.harvest import (BODY_BONES, CANON_LEN, CORE_JOINTS, VIDEO_EXT, bone_angles,
                              build_vocabulary, canonicalize, distance_matrix,
                              harvest_sequence, iter_sources, load_library, pose_distance,
                              pose_energy, pose_quality, poses_from_file, rank_normalise,
                              save_library)
from dancekit.poselib import mirror
from dancekit.skeleton import NUM_JOINTS, ROOT, to_bones

from .conftest import make_hold_move_hold


# --- quality gates -----------------------------------------------------------------------

def test_a_clean_pose_passes():
    ok, why = pose_quality(poselib.get("arms_up_v"))
    assert ok, why


def test_a_cropped_frame_is_rejected():
    """A dance pose missing an ankle is a cropped frame, not a shape."""
    pose = poselib.get("lunge_r").copy()
    pose[10, 2] = 0.0
    ok, why = pose_quality(pose)
    assert not ok and why == "missing core joints"


@pytest.mark.parametrize("joint", CORE_JOINTS.tolist())
def test_every_core_joint_is_actually_required(joint):
    pose = poselib.get("neutral").copy()
    pose[joint, 2] = 0.0
    assert not pose_quality(pose)[0]


def test_an_implausible_limb_is_rejected():
    """Detector failures show up as limbs several times their plausible length."""
    pose = poselib.get("neutral").copy()
    pose[4, :2] = pose[3, :2] + np.array([0.9, 0.0])
    ok, why = pose_quality(pose)
    assert not ok and why == "implausible limb length"


def test_a_figure_too_small_in_frame_is_rejected():
    pose = poselib.get("neutral").copy()
    centre = pose[ROOT, :2].copy()
    pose[:, :2] = centre + (pose[:, :2] - centre) * 0.15
    ok, why = pose_quality(pose)
    assert not ok and why == "figure too small in frame"


def test_a_degenerate_torso_is_rejected():
    pose = poselib.get("neutral").copy()
    pose[[8, 11], :2] = pose[ROOT, :2]
    ok, why = pose_quality(pose)
    assert not ok and why == "degenerate torso"


def test_quality_scales_with_the_figure_not_the_frame():
    """A dancer filling more of frame is not a different shape; the gate must judge
    proportion, not absolute pixel size."""
    pose = poselib.get("lunge_r").copy()
    centre = pose[ROOT, :2].copy()
    big = pose.copy()
    big[:, :2] = centre + (big[:, :2] - centre) * 1.6
    assert pose_quality(big)[0]


# --- canonicalisation -----------------------------------------------------------------------

def test_canonicalize_puts_the_shape_on_the_standard_body():
    """Shapes from clips at different distances make the figure grow and shrink between
    every hit unless they are rebuilt on one body."""
    pose = poselib.get("lunge_r").copy()
    centre = pose[ROOT, :2].copy()
    pose[:, :2] = centre + (pose[:, :2] - centre) * 1.8      # filmed closer

    out = canonicalize(pose, foreshorten=0.0)
    _, ln, _ = to_bones(out[None])
    for j in BODY_BONES:
        if CANON_LEN[j] > 0:
            assert ln[0, j] == pytest.approx(CANON_LEN[j], rel=1e-6)


def test_canonicalize_keeps_the_angles():
    pose = poselib.get("knee_up_r")
    out = canonicalize(pose, foreshorten=0.0)
    np.testing.assert_allclose(bone_angles(out)[BODY_BONES],
                               bone_angles(pose)[BODY_BONES], atol=1e-9)


def test_canonicalize_is_scale_invariant():
    """Two frames of the same shape at different distances must canonicalise to the
    same pose -- that is the entire point."""
    pose = poselib.get("arms_out_t")
    near, far = pose.copy(), pose.copy()
    c = pose[ROOT, :2].copy()
    near[:, :2] = c + (near[:, :2] - c) * 0.7
    far[:, :2] = c + (far[:, :2] - c) * 1.5

    np.testing.assert_allclose(canonicalize(near, foreshorten=0.0),
                               canonicalize(far, foreshorten=0.0), atol=1e-9)


def test_foreshorten_blends_observed_proportions_back_in():
    """A foreshortened arm really is short -- it is the only depth cue a 2D skeleton
    has -- so foreshorten=1 should keep it short and 0 should not."""
    pose = poselib.get("neutral").copy()
    pose[4, :2] = pose[3, :2] + (pose[4, :2] - pose[3, :2]) * 0.5   # arm toward camera

    flat = canonicalize(pose, foreshorten=0.0)
    deep = canonicalize(pose, foreshorten=1.0)

    _, ln_flat, _ = to_bones(flat[None])
    _, ln_deep, _ = to_bones(deep[None])
    assert ln_deep[0, 4] < ln_flat[0, 4]
    assert ln_flat[0, 4] == pytest.approx(CANON_LEN[4], rel=1e-6)


def test_canonicalize_always_lands_on_the_same_root():
    for name in ("neutral", "squat", "kick_r"):
        out = canonicalize(poselib.get(name))
        np.testing.assert_allclose(out[ROOT, :2], [0.5, 0.34], atol=1e-9)


def test_canonicalize_returns_full_confidence():
    out = canonicalize(poselib.get("neutral"))
    assert np.all(out[:, 2] == 1.0)


# --- energy and ranking ------------------------------------------------------------------------

def test_bigger_shapes_score_higher_energy():
    """Distance from root to extremities alone is a poor measure because legs dominate
    it; reach above the shoulders and lateral spread are what read as effort."""
    assert pose_energy(poselib.get("arms_up_v")) > pose_energy(poselib.get("neutral"))
    assert pose_energy(poselib.get("arms_out_t")) > pose_energy(poselib.get("neutral"))


def test_energy_is_mirror_symmetric():
    for name in ("lunge_r", "point_r_high", "knee_up_r"):
        p = poselib.get(name)
        assert pose_energy(mirror(p)) == pytest.approx(pose_energy(p), abs=1e-9)


def test_rank_normalise_spreads_scores_over_the_full_range():
    """Absolute extension scores bunch into a narrow band for any real vocabulary, which
    would leave the composer unable to tell a big shape from a small one."""
    out = rank_normalise([0.50, 0.51, 0.52, 0.53, 0.54])
    assert out == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_rank_normalise_preserves_the_ordering():
    vals = [3.0, 1.0, 2.0, 5.0]
    out = rank_normalise(vals)
    assert np.argsort(out).tolist() == np.argsort(vals).tolist()


def test_rank_normalise_of_a_single_value():
    assert rank_normalise([7.0]) == [0.5]


# --- distance ------------------------------------------------------------------------------------

def test_a_pose_is_zero_distance_from_itself():
    p = poselib.get("lunge_r")
    assert pose_distance(p, p) == pytest.approx(0.0, abs=1e-12)


def test_distance_is_mirror_invariant_by_default():
    """The composer generates mirrors itself, so keeping both sides in a vocabulary just
    wastes slots."""
    p = poselib.get("lunge_r")
    assert pose_distance(p, mirror(p), mirror_invariant=True) == pytest.approx(0.0, abs=1e-9)
    assert pose_distance(p, mirror(p), mirror_invariant=False) > 0.1


def test_distinct_shapes_are_far_apart():
    a, b = poselib.get("arms_up_v"), poselib.get("squat")
    assert pose_distance(a, b) > 0.3


def test_distance_ignores_the_eyes_and_ears():
    """Facial keypoints carry no choreography, so detector noise on them must not make
    a shape look new."""
    a = poselib.get("arms_out_t")
    b = a.copy()
    b[[14, 15, 16, 17], :2] += 0.05
    assert pose_distance(a, b) == pytest.approx(0.0, abs=1e-9)


def test_distance_still_counts_where_the_head_is_pointing():
    """The neck-to-nose bone is in BODY_BONES: looking up and looking down are
    genuinely different shapes, unlike a jittering ear."""
    a = poselib.get("arms_out_t")
    b = a.copy()
    b[0, :2] += 0.05
    assert pose_distance(a, b) > 1e-3


def test_distance_is_scale_invariant():
    """Angles rather than joint positions, because two dancers of different builds
    hitting the same shape differ a lot in xy and barely at all in angle."""
    a = poselib.get("lunge_r")
    b = a.copy()
    c = a[ROOT, :2].copy()
    b[:, :2] = c + (b[:, :2] - c) * 1.5
    assert pose_distance(a, b) == pytest.approx(0.0, abs=1e-9)


def test_distance_matrix_matches_pairwise_distance():
    poses = [poselib.get(n) for n in ("neutral", "arms_up_v", "squat", "lunge_r")]
    D = distance_matrix(poses)
    for i in range(len(poses)):
        for j in range(len(poses)):
            assert D[i, j] == pytest.approx(pose_distance(poses[i], poses[j]), abs=1e-9)


def test_distance_matrix_is_symmetric_with_a_zero_diagonal():
    poses = [poselib.get(n) for n in ("neutral", "arms_up_v", "squat", "lunge_r")]
    D = distance_matrix(poses)
    np.testing.assert_allclose(D, D.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)


# --- harvesting a clip --------------------------------------------------------------------------

@pytest.fixture
def clip():
    shapes = [poselib.get(n) for n in
              ("neutral", "arms_up_v", "lunge_r", "squat", "arms_out_t", "kick_r")]
    pose, holds = make_hold_move_hold(shapes, hold_frames=10, move_frames=6, jitter=5e-4)
    return pose, holds


def test_harvest_keeps_the_held_shapes(clip):
    pose, holds = clip
    cands, report = harvest_sequence(pose, fps=30.0, source="clip.mp4")

    assert len(cands) >= 3
    assert report["source"] == "clip.mp4"
    assert report["frames"] == pose.shape[0]
    assert report["kept"] == len(cands)
    for c in cands:
        assert c["pose"].shape == (NUM_JOINTS, 3)
        assert any(s <= c["frame"] <= e for s, e in holds), (
            f"candidate at frame {c['frame']} is not a held shape")


def test_harvested_candidates_are_canonicalised(clip):
    pose, _ = clip
    cands, _ = harvest_sequence(pose, fps=30.0)
    for c in cands:
        np.testing.assert_allclose(c["pose"][ROOT, :2], [0.5, 0.34], atol=1e-9)


def test_harvest_records_why_frames_were_rejected(clip):
    """Rejections are counted per reason so you can see what your footage is costing."""
    pose, _ = clip
    broken = pose.copy()
    broken[:, 10, 2] = 0.0                       # every frame loses an ankle

    _, report = harvest_sequence(broken, fps=30.0)
    assert report["kept"] == 0
    assert report["rejected"].get("missing core joints", 0) > 0


def test_harvest_reports_the_slow_motion_check(clip):
    pose, _ = clip
    _, report = harvest_sequence(pose, fps=30.0)
    assert "likely_slowmo" in report["slowmo"]


def test_harvest_report_is_json_serialisable(clip):
    pose, _ = clip
    _, report = harvest_sequence(pose, fps=30.0)
    json.dumps(report)


# --- building a vocabulary --------------------------------------------------------------------

def _cands(names, energy=None):
    return [{"pose": canonicalize(poselib.get(n)), "energy": pose_energy(poselib.get(n)),
             "source": "s.mp4", "frame": i} for i, n in enumerate(names)]


def test_build_vocabulary_dedupes_repeats():
    """A shape hit four times in the footage is one vocabulary entry, not four."""
    cands = _cands(["arms_up_v"] * 4 + ["squat"] * 3 + ["lunge_r"])
    lib, meta = build_vocabulary(cands, min_distance=0.30)
    assert len(lib) == 3
    assert len(meta) == 3


def test_recurring_shapes_are_counted():
    cands = _cands(["arms_up_v"] * 4 + ["squat"] * 2)
    _, meta = build_vocabulary(cands, min_distance=0.30)
    counts = sorted(m["count"] for m in meta)
    assert counts == [2, 4]


def test_recurring_shapes_rank_above_one_offs():
    """Shapes that recurred are real vocabulary; a one-off is usually a detector glitch."""
    cands = _cands(["squat"] * 5 + ["arms_up_v"])
    _, meta = build_vocabulary(cands, min_distance=0.30)
    assert meta[0]["count"] > meta[-1]["count"]


def test_min_cluster_size_drops_one_off_outliers():
    cands = _cands(["squat"] * 4 + ["arms_up_v"])
    _, meta = build_vocabulary(cands, min_distance=0.30, min_cluster_size=2)
    assert all(m["count"] >= 2 for m in meta)


def test_max_poses_caps_the_vocabulary():
    names = ["arms_up_v", "squat", "lunge_r", "kick_r", "arms_out_t", "knee_up_r",
             "step_wide", "lean_back"]
    _, meta = build_vocabulary(_cands(names), min_distance=0.20, max_poses=3)
    assert len(meta) == 3


def test_the_two_lunges_really_are_mirrors_of_each_other():
    """Guards the two tests below: they are authored from different override numbers,
    so if that drifts, the mirror tests would silently stop testing mirroring."""
    r, l = canonicalize(poselib.get("lunge_r")), canonicalize(poselib.get("lunge_l"))
    assert pose_distance(l, mirror(r), mirror_invariant=False) == pytest.approx(0.0, abs=1e-9)


def test_mirrors_collapse_into_one_slot():
    """Whatever threshold is used, a shape and its mirror are the same slot -- the
    composer generates mirrors itself."""
    cands = _cands(["lunge_r", "lunge_l"])
    lib, _ = build_vocabulary(cands, min_distance=0.05, mirror_invariant=True)
    assert len(lib) == 1


def test_keep_mirrors_gives_both_sides_a_slot():
    """The two lunges sit 0.24 rad apart, so the threshold has to be below that for
    them to be distinguishable at all."""
    cands = _cands(["lunge_r", "lunge_l"])
    lib, _ = build_vocabulary(cands, min_distance=0.05, mirror_invariant=False)
    assert len(lib) == 2


def test_near_neutral_shapes_are_dropped():
    cands = _cands(["neutral", "neutral", "arms_up_v", "squat"])
    _, meta = build_vocabulary(cands, drop_near_neutral=0.18)
    assert len(meta) == 2, "plain standing is not vocabulary"


def test_dropping_everything_falls_back_rather_than_returning_nothing():
    """If every candidate is near neutral, an empty vocabulary would be useless; the
    fallback keeps the set."""
    lib, meta = build_vocabulary(_cands(["neutral"] * 3), drop_near_neutral=0.18)
    assert len(lib) >= 1


def test_a_lower_min_distance_gives_a_richer_vocabulary():
    names = ["arms_up_v", "arms_out_t", "squat", "lunge_r", "kick_r", "step_wide"]
    loose, _ = build_vocabulary(_cands(names), min_distance=0.10)
    tight, _ = build_vocabulary(_cands(names), min_distance=1.20)
    assert len(loose) >= len(tight)


def test_vocabulary_energies_span_the_full_range():
    names = ["arms_up_v", "squat", "lunge_r", "kick_r", "arms_out_t"]
    _, meta = build_vocabulary(_cands(names), min_distance=0.20)
    e = [m["energy"] for m in meta]
    assert min(e) == pytest.approx(0.0)
    assert max(e) == pytest.approx(1.0)


def test_vocabulary_meta_records_provenance():
    """The manifest is how you find which clip and frame a shape came from."""
    _, meta = build_vocabulary(_cands(["arms_up_v", "squat"]), min_distance=0.20)
    for m in meta:
        assert m["source"] == "s.mp4"
        assert isinstance(m["frame"], int)
        assert m["name"].startswith("h")


def test_empty_candidates_give_an_empty_vocabulary():
    lib, meta = build_vocabulary([])
    assert lib == {} and meta == []


def test_a_single_candidate_works():
    lib, meta = build_vocabulary(_cands(["arms_up_v"]))
    assert len(lib) == 1
    assert meta[0]["energy"] == pytest.approx(0.5)


# --- library IO ------------------------------------------------------------------------------------

def test_library_round_trip(tmp_path):
    names = ["arms_up_v", "squat", "lunge_r"]
    lib, meta = build_vocabulary(_cands(names), min_distance=0.20)

    path = tmp_path / "vocab.npz"
    save_library(path, lib, meta)
    loaded, loaded_meta, emap = load_library(path)

    assert loaded_meta == meta
    for name, pose in lib.items():
        np.testing.assert_allclose(loaded[name], pose, atol=1e-9)


def test_load_library_adds_mirrors(tmp_path):
    """So the composer can alternate sides."""
    lib, meta = build_vocabulary(_cands(["arms_up_v", "squat"]), min_distance=0.20)
    path = tmp_path / "v.npz"
    save_library(path, lib, meta)

    loaded, _, emap = load_library(path)
    assert len(loaded) == 2 * len(lib)
    for name, pose in lib.items():
        np.testing.assert_allclose(loaded[name + "_m"], mirror(pose), atol=1e-9)
        assert emap[name + "_m"] == emap[name]


def test_loaded_energy_map_covers_every_pose(tmp_path):
    lib, meta = build_vocabulary(_cands(["arms_up_v", "squat", "kick_r"]),
                                 min_distance=0.20)
    path = tmp_path / "v.npz"
    save_library(path, lib, meta)

    loaded, _, emap = load_library(path)
    assert set(emap) == set(loaded)
    assert all(0.0 <= v <= 1.0 for v in emap.values())


def test_a_harvested_library_composes(tmp_path):
    """The end of the harvest path: a saved vocabulary drives the composer."""
    from dancekit.beatgrid import synthetic
    from dancekit.compose import compose

    lib, meta = build_vocabulary(_cands(["arms_up_v", "squat", "lunge_r", "kick_r"]),
                                 min_distance=0.20)
    path = tmp_path / "v.npz"
    save_library(path, lib, meta)
    loaded, _, emap = load_library(path)

    pose, info = compose(synthetic(120.0, duration=16.0), library=loaded,
                         energy_map=emap, seed=1)
    assert np.all(np.isfinite(pose))
    used = {n for motif in info["sections"].values() for n in motif}
    assert used.issubset(set(loaded))


# --- source discovery ---------------------------------------------------------------------------------

def test_iter_sources_finds_videos_and_json(tmp_path):
    for name in ("a.mp4", "b.MOV", "c.json", "d.webm"):
        (tmp_path / name).write_text("")
    (tmp_path / "notes.txt").write_text("")
    (tmp_path / "cover.png").write_text("")

    found = {p.name for p in iter_sources(tmp_path)}
    assert found == {"a.mp4", "b.MOV", "c.json", "d.webm"}


def test_iter_sources_walks_subdirectories(tmp_path):
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    (tmp_path / "sub" / "deep" / "x.mp4").write_text("")
    assert [p.name for p in iter_sources(tmp_path)] == ["x.mp4"]


def test_iter_sources_accepts_a_single_file(tmp_path):
    f = tmp_path / "one.mp4"
    f.write_text("")
    assert [p for p in iter_sources(f)] == [f]


def test_video_extensions_are_lowercase():
    assert all(e == e.lower() and e.startswith(".") for e in VIDEO_EXT)


def test_poses_from_file_reads_json(pose_json, hold_sequence):
    pose, _, _ = hold_sequence
    loaded, fps = poses_from_file(pose_json)
    np.testing.assert_allclose(loaded, pose, atol=1e-6)
    assert fps == 30.0


def test_poses_from_a_video_without_a_detector_explains_itself(tmp_path):
    """The error has to name the fix, because this is the most likely first failure for
    anyone pointing harvest at a folder of clips."""
    clip = tmp_path / "x.mp4"
    clip.write_text("")
    with pytest.raises(RuntimeError, match="rtmlib"):
        poses_from_file(clip, extractor=None)


def test_poses_from_a_video_uses_the_extractor_it_is_given(tmp_path):
    clip = tmp_path / "x.mp4"
    clip.write_text("")
    sentinel = np.zeros((3, NUM_JOINTS, 3))

    def fake(path, max_frames=0):
        assert path == clip
        return sentinel, 24.0

    pose, fps = poses_from_file(clip, extractor=fake)
    assert pose is sentinel and fps == 24.0
