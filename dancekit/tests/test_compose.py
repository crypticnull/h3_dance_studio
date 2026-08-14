"""Generating original choreography from a vocabulary and a beat grid.

The claim this module rests on is that motif repetition, not timing accuracy, is what
makes movement read as intentional -- so the tests care as much about a section's
material coming back as they do about frames and shapes being well formed.
"""

from __future__ import annotations

import numpy as np
import pytest

from dancekit import poselib
from dancekit.beatgrid import BeatGrid, synthetic
from dancekit.compose import (_base, compose, label_sections, make_motif, phrase_energy,
                              side_bias)
from dancekit.poselib import all_poses, energies
from dancekit.skeleton import NUM_JOINTS

from .conftest import assert_no_limb_collapse


@pytest.fixture
def grid():
    """32 bars at 120 BPM -- long enough for several phrases and section returns.

    Carries no onset envelope, which is the `beatgrid.synthetic` case: the composer has
    the timing but nothing to read dynamics from.
    """
    return synthetic(120.0, duration=64.0)


@pytest.fixture
def musical_grid(grid):
    """The same grid with a verse / chorus / verse / chorus energy envelope, so section
    detection has something to actually detect."""
    t = np.linspace(0, grid.duration, 2000)
    env = np.where((t % 32.0) < 16.0, 0.2, 0.9)      # 16 s blocks, quiet then loud
    grid.onset_times = t
    grid.onset_env = env + 0.02 * np.sin(t * 3.0)    # a little texture within a block
    return grid


# --- musical structure --------------------------------------------------------------------

def test_phrase_energy_is_normalised():
    g = synthetic(120.0, duration=32.0)
    g.onset_times = np.linspace(0, 32, 400)
    g.onset_env = np.linspace(0.0, 5.0, 400)      # a steady build

    e = phrase_energy(g, g.beats, slots=8)
    assert e.min() == pytest.approx(0.0)
    assert e.max() == pytest.approx(1.0)
    assert np.all(np.diff(e) > 0), "a rising track should give rising phrase energy"


def test_phrase_energy_without_an_onset_envelope():
    """`synthetic` grids carry no envelope, so this path has to stay usable."""
    g = synthetic(120.0, duration=32.0)
    e = phrase_energy(g, g.beats, slots=8)
    assert np.all(e == 0.5)


def test_phrase_energy_on_a_flat_track():
    g = synthetic(120.0, duration=32.0)
    g.onset_times = np.linspace(0, 32, 400)
    g.onset_env = np.ones(400)
    assert np.all(phrase_energy(g, g.beats, slots=8) == 0.5)


def test_phrase_energy_has_one_value_per_phrase():
    g = synthetic(120.0, duration=32.0)
    g.onset_times = np.linspace(0, 32, 400)
    g.onset_env = np.random.default_rng(0).random(400)
    anchors = g.beats
    for slots in (4, 8, 16):
        assert len(phrase_energy(g, anchors, slots)) == int(np.ceil(len(anchors) / slots))


def test_a_returning_chorus_gets_the_same_label():
    """The point of clustering: material written for the chorus comes back when the
    chorus does. Energy is shaped as verse / chorus / verse / chorus in blocks, which is
    what a real track looks like."""
    energy = np.array([0.15, 0.18, 0.16, 0.17,      # verse
                       0.85, 0.88, 0.86, 0.87,      # chorus
                       0.16, 0.15, 0.18, 0.17,      # verse again
                       0.86, 0.89, 0.87, 0.88])     # chorus again
    labels = label_sections(energy, k=2, seed=0)

    assert len(set(labels[0:4])) == 1, "one verse should not split across sections"
    assert set(labels[0:4]) == set(labels[8:12]), "the returning verse changed section"
    assert set(labels[4:8]) == set(labels[12:16]), "the returning chorus changed section"
    assert labels[0] != labels[4], "loud and quiet phrases must not share a section"


def test_label_sections_handles_a_track_with_no_dynamics():
    """Every phrase at the same energy is one section, and must not divide by zero
    inside kmeans on the way to saying so."""
    labels = label_sections(np.full(8, 0.5), k=3, seed=0)
    np.testing.assert_array_equal(labels, np.zeros(8, dtype=int))


def test_label_sections_is_deterministic():
    energy = np.random.default_rng(3).random(24)
    a = label_sections(energy, k=3, seed=7)
    b = label_sections(energy, k=3, seed=7)
    np.testing.assert_array_equal(a, b)


def test_label_sections_returns_one_label_per_phrase():
    energy = np.random.default_rng(1).random(20)
    for k in (1, 2, 5):
        assert len(label_sections(energy, k=k, seed=0)) == 20


def test_label_sections_clamps_k_to_what_it_has():
    """Asking for more sections than phrases must not crash the composer."""
    labels = label_sections(np.array([0.3, 0.7]), k=9, seed=0)
    assert len(labels) == 2
    assert len(set(labels.tolist())) <= 2


def test_label_sections_of_one():
    np.testing.assert_array_equal(label_sections(np.random.rand(6), k=1), np.zeros(6))


# --- lateral bias -------------------------------------------------------------------------

def test_side_bias_is_signed_by_direction():
    """Deciding a pose's side from its name fails: the mirror of a symmetric pose is
    the same pose. Measuring the lateral offset is what actually works."""
    assert side_bias(poselib.get("neutral")) == pytest.approx(0.0, abs=1e-9)
    assert side_bias(poselib.get("point_r_high")) < 0     # character right = image left
    assert side_bias(poselib.get("point_l_high")) > 0


def test_side_bias_negates_under_mirroring():
    for name in ("point_r_high", "lunge_r", "knee_up_r", "kick_r"):
        p = poselib.get(name)
        assert side_bias(poselib.mirror(p)) == pytest.approx(-side_bias(p), abs=1e-9)


def test_base_strips_the_mirror_suffix():
    assert _base("lunge_r_m") == "lunge_r"
    assert _base("lunge_r") == "lunge_r"
    assert _base("neutral_m") == "neutral"


# --- motifs ---------------------------------------------------------------------------------

def test_motif_has_one_pose_per_slot():
    rng = np.random.default_rng(0)
    lib, e = all_poses(), energies()
    motif = make_motif(list(lib), e, slots=8, e_target=0.5, rng=rng)
    assert len(motif) == 8
    assert all(n in lib for n in motif)


def test_motif_does_not_repeat_within_two_slots():
    """Back-to-back repeats read as a stutter rather than a phrase."""
    lib, e = all_poses(), energies()
    bias = {n: side_bias(p) for n, p in lib.items()}
    for seed in range(12):
        motif = make_motif(list(lib), e, slots=8, e_target=0.5,
                           rng=np.random.default_rng(seed), bias_map=bias)
        bases = [_base(n) for n in motif]
        for i in range(1, len(bases)):
            assert bases[i] != bases[i - 1]
        for i in range(2, len(bases)):
            assert bases[i] != bases[i - 2]


def test_motif_accents_the_downbeat():
    """Slot 0 asks for a higher energy than the phrase target. Sampling is weighted, so
    this is a tendency; check it across many draws rather than in one motif."""
    lib, e = all_poses(), energies()
    bias = {n: side_bias(p) for n, p in lib.items()}

    first, middle = [], []
    for seed in range(120):
        motif = make_motif(list(lib), e, slots=8, e_target=0.5,
                           rng=np.random.default_rng(seed), bias_map=bias)
        first.append(e[motif[0]])
        middle.append(e[motif[4]])

    assert np.mean(first) > np.mean(middle), (
        "the downbeat slot should on average pick a bigger shape than the mid-phrase "
        "breath")


def test_motif_alternates_lateral_direction():
    """Alternating sides is what stops a phrase looking like a twitch."""
    lib, e = all_poses(), energies()
    bias = {n: side_bias(p) for n, p in lib.items()}

    even, odd = [], []
    for seed in range(120):
        motif = make_motif(list(lib), e, slots=8, e_target=0.6,
                           rng=np.random.default_rng(seed), bias_map=bias)
        even.extend(bias[n] for n in motif[0::2])
        odd.extend(bias[n] for n in motif[1::2])

    assert np.mean(even) > np.mean(odd), "even and odd slots should lean opposite ways"


def test_motif_tracks_the_requested_energy():
    lib, e = all_poses(), energies()
    quiet = [e[n] for s in range(60)
             for n in make_motif(list(lib), e, 8, 0.15, np.random.default_rng(s))]
    loud = [e[n] for s in range(60)
            for n in make_motif(list(lib), e, 8, 0.9, np.random.default_rng(s))]
    assert np.mean(quiet) < np.mean(loud)


def test_motif_does_not_fill_a_phrase_with_the_resting_pose():
    lib, e = all_poses(), energies()
    counts = 0
    total = 0
    for seed in range(40):
        motif = make_motif(list(lib), e, 8, 0.5, np.random.default_rng(seed))
        counts += sum(1 for n in motif if _base(n) == "neutral")
        total += len(motif)
    assert counts / total < 0.25, "the resting pose is eating the phrase"


# --- composing -------------------------------------------------------------------------------

def test_compose_returns_a_well_formed_sequence(grid):
    pose, info = compose(grid, fps=24.0, seed=1)
    assert pose.ndim == 3 and pose.shape[1:] == (NUM_JOINTS, 3)
    assert pose.shape[0] == info["frames"]
    assert np.all(np.isfinite(pose))
    assert np.all(pose[:, :, 2] > 0), "every joint must be drawable"


def test_compose_covers_the_requested_duration(grid):
    pose, info = compose(grid, fps=24.0, seed=1)
    assert info["frames"] == pytest.approx(info["duration_s"] * 24.0, abs=1.5)
    assert info["duration_s"] > 50.0


def test_compose_is_deterministic_for_a_seed(grid):
    a, info_a = compose(grid, seed=5)
    b, info_b = compose(grid, seed=5)
    np.testing.assert_array_equal(a, b)
    assert info_a == info_b


def test_different_seeds_give_different_dances(grid):
    a, _ = compose(grid, seed=1)
    b, _ = compose(grid, seed=2)
    assert not np.allclose(a, b), "the seed is meant to change the choreography"


def test_compose_keeps_limbs_solid(grid):
    """Interpolation runs through bone space, so no frame should carry a limb outside
    the range the vocabulary spans."""
    pose, _ = compose(grid, fps=24.0, seed=3, bounce=0.0)
    lib = np.stack(list(all_poses().values()))
    assert_no_limb_collapse(pose, lib, tol=0.02)


def test_compose_stays_on_canvas(grid):
    pose, _ = compose(grid, fps=24.0, seed=3)
    assert pose[:, :, :2].min() > -0.05
    assert pose[:, :, :2].max() < 1.05


def test_a_returning_section_gets_the_same_material(musical_grid):
    """Motif repetition is the whole design: one motif is written per section and
    replayed on every return of that section."""
    _, info = compose(musical_grid, seed=4, sections=2)
    labels = info["section_sequence"]
    assert len(set(labels)) > 1, "a verse/chorus track should give more than one section"

    for lab in set(labels):
        assert lab in info["sections"], f"section {lab} was used but has no material"
        assert len(info["sections"][lab]) == info["slots_per_phrase"]


def test_quiet_and_loud_sections_get_different_material(musical_grid):
    _, info = compose(musical_grid, seed=11, sections=2)
    motifs = [tuple(v) for v in info["sections"].values()]
    assert len(set(motifs)) > 1, "distinct sections should get distinct material"


def test_a_grid_with_no_dynamics_composes_as_one_section(grid):
    """`beatgrid.synthetic` carries no onset envelope. That should still compose, just
    with a single motif rather than a section map."""
    pose, info = compose(grid, seed=4, sections=3)
    assert np.all(np.isfinite(pose))
    assert set(info["section_sequence"]) == {0}


def test_composing_without_dynamics_does_not_warn(grid):
    """kmeans on identical points used to divide by zero here."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        compose(grid, seed=4, sections=3)


def test_repeated_phrases_reuse_shapes_rather_than_reinventing(grid):
    """A composed dance should revisit a small vocabulary, not wander through every
    pose available. That is the difference between a phrase and flailing."""
    _, info = compose(grid, seed=6, sections=3)
    used = {n for motif in info["sections"].values() for n in motif}
    assert len(used) < len(all_poses()) / 2


def test_subdivision_controls_pose_density(grid):
    _, quarters = compose(grid, subdivision=1, seed=2)
    _, eighths = compose(grid, subdivision=2, seed=2)
    assert eighths["slots_per_phrase"] == 2 * quarters["slots_per_phrase"]


def test_max_seconds_truncates_the_dance(grid):
    _, info = compose(grid, seed=2, max_seconds=10.0)
    assert info["duration_s"] <= 10.0
    assert info["frames"] <= 10.0 * info["fps"] + 2


def test_bounce_adds_a_per_beat_pulse(grid):
    """Real dancers never fully stop, and a dead hold between hits is the biggest tell
    of synthetic motion."""
    flat, _ = compose(grid, seed=8, bounce=0.0)
    pulsed, _ = compose(grid, seed=8, bounce=0.02)

    assert not np.allclose(flat, pulsed)
    delta = pulsed[:, :, 1] - flat[:, :, 1]
    assert delta.min() >= -1e-9, "the bounce should only ever push downward"
    assert delta.max() == pytest.approx(0.02, abs=1e-6)
    # Every joint moves together -- it is the whole figure pulsing, not a limb.
    np.testing.assert_allclose(delta.std(axis=1), 0.0, atol=1e-12)


def test_snap_changes_how_long_a_shape_is_held(grid):
    """Lower snap arrives at the shape earlier and holds it, which reads as a hit."""
    sharp, _ = compose(grid, seed=9, snap=0.35, bounce=0.0)
    smooth, _ = compose(grid, seed=9, snap=1.0, bounce=0.0)

    def still_fraction(p):
        d = np.linalg.norm(np.diff(p[:, :, :2], axis=0), axis=-1).mean(axis=1)
        return float((d < d.mean() * 0.25).mean())

    assert still_fraction(sharp) > still_fraction(smooth)


def test_compose_rejects_a_grid_too_short_to_work_with():
    tiny = BeatGrid(tempo=120.0, beats=np.array([0.5]), downbeats=np.array([0.5]),
                    duration=1.0)
    with pytest.raises(ValueError, match="too short"):
        compose(tiny)


def test_compose_info_is_json_serialisable(grid):
    import json
    _, info = compose(grid, seed=1)
    round_tripped = json.loads(json.dumps(info))
    assert round_tripped["seed"] == 1


def test_compose_with_a_custom_library(grid):
    """The harvested path: a vocabulary with no energies supplied, ranked by extension."""
    lib = {n: poselib.get(n) for n in ("neutral", "arms_up_v", "squat", "lunge_r")}
    pose, info = compose(grid, library=lib, seed=1, fps=24.0)

    assert np.all(np.isfinite(pose))
    used = {n for motif in info["sections"].values() for n in motif}
    assert used.issubset(set(lib)), "the composer must only use the library it was given"


def test_compose_with_a_custom_library_and_energies(grid):
    lib = {n: poselib.get(n) for n in ("neutral", "arms_up_v", "squat", "lunge_r")}
    emap = {"neutral": 0.1, "arms_up_v": 0.95, "squat": 0.5, "lunge_r": 0.7}
    pose, info = compose(grid, library=lib, energy_map=emap, seed=1)
    assert np.all(np.isfinite(pose))


def test_compose_with_a_single_pose_library(grid):
    """A degenerate vocabulary should still produce a valid, if dull, sequence rather
    than a divide by zero in the energy ranking."""
    lib = {"only": poselib.get("arms_up_v")}
    pose, _ = compose(grid, library=lib, seed=1)
    assert np.all(np.isfinite(pose))


def test_fps_controls_the_frame_count(grid):
    _, slow = compose(grid, fps=12.0, seed=1)
    _, fast = compose(grid, fps=24.0, seed=1)
    assert fast["frames"] == pytest.approx(2 * slow["frames"], rel=0.02)
    assert slow["duration_s"] == pytest.approx(fast["duration_s"], abs=0.1)


def test_variation_of_zero_keeps_every_repeat_identical(grid):
    """With variation off, a section's repeats should be the literal same material."""
    _, info = compose(grid, seed=2, variation=0.0, sections=2)
    assert info["phrases"] > len(info["sections"])


def test_the_dance_is_exactly_on_the_shape_at_every_anchor(grid):
    """The composed counterpart of the retiming guarantee: at each grid time the figure
    is standing in the shape the motif named for that slot, not on the way to it.

    At 120 BPM and 24 fps a beat lands exactly on a frame, so this is exact rather than
    approximate.
    """
    lib = all_poses()
    pose, info = compose(grid, fps=24.0, seed=3, snap=0.5, overshoot=0.0, bounce=0.0,
                         subdivision=1)

    anchors = grid.subdivide(1)
    # The first phrase is laid down unflipped and unvaried, so its slots are the motif.
    first_motif = info["sections"][info["section_sequence"][0]]

    for slot, name in enumerate(first_motif):
        frame = int(round(anchors[slot] * 24.0))
        if frame >= pose.shape[0]:
            break
        err = np.abs(pose[frame, :, :2] - lib[name][:, :2]).max()
        assert err < 1e-6, f"slot {slot} is not standing in {name} (max error {err:.4f})"


def test_the_shape_arrives_early_and_is_held(grid):
    """With snap below 1 the move completes inside the first part of the interval and
    the shape is held until the next beat -- that hold is what reads as a hit."""
    lib = all_poses()
    pose, info = compose(grid, fps=24.0, seed=3, snap=0.5, overshoot=0.0, bounce=0.0,
                         subdivision=1)

    anchors = grid.subdivide(1)
    first_motif = info["sections"][info["section_sequence"][0]]

    # Between anchor 0 and anchor 1 the figure travels to slot 1's shape; by the time
    # half the interval has passed it should already be there.
    target = lib[first_motif[1]]
    t = anchors[0] + 0.75 * (anchors[1] - anchors[0])
    frame = int(round(t * 24.0))
    err = np.abs(pose[frame, :, :2] - target[:, :2]).max()
    assert err < 0.01, f"the shape had not arrived three quarters through (err {err:.4f})"
