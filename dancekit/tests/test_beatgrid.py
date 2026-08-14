"""Beat, tempo and downbeat analysis.

The accuracy tests measure against synthetic tracks whose beat times are known exactly,
which is the only way to check the claims the README makes about this module.
"""

from __future__ import annotations

import numpy as np
import pytest

from dancekit import beatgrid
from dancekit.beatgrid import BeatGrid, _estimate_downbeats, _fit_rigid, analyse, synthetic

from .conftest import make_click_track


# --- BeatGrid views -------------------------------------------------------------------

def test_synthetic_grid_is_exactly_on_tempo():
    g = synthetic(120.0, duration=10.0)
    np.testing.assert_allclose(np.diff(g.beats), 0.5, atol=1e-12)
    assert g.beats[0] == 0.0
    assert g.beats[-1] <= 10.0


def test_synthetic_grid_marks_every_bar():
    g = synthetic(120.0, duration=8.0, beats_per_bar=4)
    np.testing.assert_allclose(np.diff(g.downbeats), 2.0, atol=1e-12)
    assert set(g.downbeats).issubset(set(g.beats))


def test_synthetic_honours_an_offset():
    g = synthetic(100.0, duration=5.0, offset=0.25)
    assert g.beats[0] == pytest.approx(0.25)


def test_subdivide_inserts_evenly_spaced_points():
    g = synthetic(120.0, duration=4.0)          # beats every 0.5 s
    eighths = g.subdivide(2)
    np.testing.assert_allclose(np.diff(eighths), 0.25, atol=1e-12)
    # Every original beat must survive subdivision -- they are the anchors that matter.
    for b in g.beats:
        assert np.min(np.abs(eighths - b)) < 1e-12


def test_subdivide_counts_are_right():
    g = synthetic(120.0, duration=4.0)
    n = len(g.beats)
    assert len(g.subdivide(1)) == n
    assert len(g.subdivide(2)) == 2 * (n - 1) + 1
    assert len(g.subdivide(4)) == 4 * (n - 1) + 1


def test_subdivide_of_one_or_less_returns_the_beats():
    g = synthetic(120.0, duration=4.0)
    for n in (0, 1, -3):
        np.testing.assert_allclose(g.subdivide(n), g.beats)


def test_subdivide_does_not_alias_the_beat_array():
    g = synthetic(120.0, duration=4.0)
    out = g.subdivide(1)
    out[0] = 99.0
    assert g.beats[0] != 99.0


def test_subdivide_handles_a_degenerate_grid():
    g = BeatGrid(tempo=120.0, beats=np.array([1.5]), downbeats=np.array([1.5]),
                 duration=3.0)
    np.testing.assert_allclose(g.subdivide(4), [1.5])


def test_to_frames_rounds_to_the_nearest_frame():
    g = BeatGrid(tempo=120.0, beats=np.array([0.0, 0.5, 1.0]),
                 downbeats=np.array([0.0]), duration=2.0)
    np.testing.assert_array_equal(g.to_frames(24.0), [0, 12, 24])
    np.testing.assert_array_equal(g.to_frames(24.0, np.array([0.02, 0.03])), [0, 1])


def test_phase_at_walks_round_the_bar():
    g = synthetic(120.0, duration=8.0)          # 0.5 s per beat, downbeat at 0.0
    assert g.phase_at(0.0) == pytest.approx(0.0)
    assert g.phase_at(0.5) == pytest.approx(1.0)
    assert g.phase_at(1.5) == pytest.approx(3.0)
    assert g.phase_at(2.0) == pytest.approx(0.0, abs=1e-9)   # next bar


def test_phase_at_is_safe_without_downbeats():
    g = BeatGrid(tempo=0.0, beats=np.array([]), downbeats=np.array([]), duration=1.0)
    assert g.phase_at(3.0) == 0.0


def test_as_dict_is_json_serialisable():
    import json
    g = synthetic(128.0, duration=4.0)
    d = g.as_dict()
    json.loads(json.dumps(d))                   # must not raise on numpy scalars
    assert isinstance(d["tempo"], float)
    assert d["beats"] and isinstance(d["beats"][0], float)
    assert len(d["beats"]) == len(g.beats)


# --- rigid tempo fitting -----------------------------------------------------------------

def test_fit_rigid_recovers_a_clean_tempo():
    beats = np.arange(32) * 0.5 + 0.13
    bpm, phase = _fit_rigid(beats)
    assert bpm == pytest.approx(120.0, abs=1e-9)
    assert phase == pytest.approx(0.13, abs=1e-9)


def test_fit_rigid_removes_tracker_jitter():
    """The tracker's beat times are quantised to its hop size. Fitting one line through
    them is what turns that into a grid that stays locked over a whole track."""
    rng = np.random.default_rng(1)
    true = np.arange(64) * 0.5 + 0.2
    jittered = true + rng.uniform(-0.006, 0.006, true.size)

    bpm, phase = _fit_rigid(jittered)
    fitted = phase + np.arange(64) * (60.0 / bpm)
    assert np.abs(fitted - true).max() < np.abs(jittered - true).max()
    assert bpm == pytest.approx(120.0, abs=0.05)


def test_fit_rigid_survives_a_few_outliers():
    """A dropped or doubled beat should not drag the whole grid off tempo."""
    beats = np.arange(40) * 0.5
    beats[7] += 0.20
    beats[23] -= 0.18
    bpm, _ = _fit_rigid(beats)
    assert bpm == pytest.approx(120.0, abs=0.5)


def test_fit_rigid_declines_to_guess_from_too_few_beats():
    bpm, phase = _fit_rigid(np.array([1.0, 2.0]))
    assert bpm == 0.0
    assert phase == 1.0
    assert _fit_rigid(np.array([])) == (0.0, 0.0)


# --- downbeat estimation --------------------------------------------------------------------

def _low_band(bpm=120.0, bars=8, pattern=(1.35, 0.2, 0.8, 0.2), sr=200):
    """A synthetic low-band power envelope with a chosen per-beat amplitude pattern."""
    spb = 60.0 / bpm
    times = np.arange(0, bars * 4 * spb + spb, 1.0 / sr)
    power = np.zeros_like(times)
    beats = np.arange(bars * 4) * spb
    for i, b in enumerate(beats):
        amp = pattern[i % 4]
        power += amp * np.exp(-np.clip(times - b, 0, None) * 25.0) * (times >= b)
    return beats, power, times


def test_downbeat_lands_on_the_loudest_kick():
    beats, power, times = _low_band(pattern=(1.35, 0.2, 0.8, 0.2))
    db = _estimate_downbeats(beats, power, times, beats_per_bar=4)
    np.testing.assert_allclose(db, beats[::4], atol=1e-12)


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
def test_downbeat_finds_the_bar_phase_wherever_it_sits(shift):
    pattern = tuple(np.roll([1.35, 0.2, 0.8, 0.2], shift))
    beats, power, times = _low_band(pattern=pattern)
    db = _estimate_downbeats(beats, power, times, beats_per_bar=4)
    np.testing.assert_allclose(db, beats[shift::4], atol=1e-12)


def test_downbeat_estimation_needs_a_full_bar():
    beats = np.array([0.0, 0.5])
    db = _estimate_downbeats(beats, np.ones(10), np.linspace(0, 1, 10), beats_per_bar=4)
    np.testing.assert_allclose(db, [0.0])


def test_downbeats_are_a_subset_of_beats():
    beats, power, times = _low_band()
    db = _estimate_downbeats(beats, power, times)
    assert set(np.round(db, 9)).issubset(set(np.round(beats, 9)))


# --- full analysis against ground truth --------------------------------------------------------

def test_analyse_recovers_the_tempo(click_track):
    g = analyse(click_track["path"])
    assert g.tempo == pytest.approx(click_track["bpm"], abs=0.01), (
        "the README claims tempo exact to +/-0.01 BPM on produced material")


def test_analyse_recovers_every_beat(click_track):
    g = analyse(click_track["path"])
    truth = click_track["beats"]
    for t in truth:
        assert np.min(np.abs(g.beats - t)) < 0.05, f"no beat found near {t:.3f}s"
    assert len(g.beats) >= len(truth)


def test_analyse_beat_error_stays_under_a_quarter_frame(click_track):
    """~11 ms is the figure in the README, about a quarter of a frame at 24 fps."""
    g = analyse(click_track["path"])
    err = [np.min(np.abs(g.beats - t)) for t in click_track["beats"]]
    assert np.mean(err) < 0.015
    assert np.max(err) < 0.030


def test_analyse_finds_the_downbeat_not_the_snare(click_track):
    """The fixture deliberately puts the loudest transient on 2 and 4. Scoring broadband
    onset strength would land the downbeat there, which is exactly wrong."""
    g = analyse(click_track["path"])
    truth = click_track["downbeats"]
    for t in truth[:6]:
        assert np.min(np.abs(g.downbeats - t)) < 0.05, (
            f"downbeat near {t:.3f}s missed -- probably locked onto the snare")


def test_analyse_reports_the_real_duration(click_track):
    g = analyse(click_track["path"])
    assert g.duration == pytest.approx(click_track["duration"], rel=0.01)


def test_rigid_grid_is_perfectly_even(click_track):
    g = analyse(click_track["path"], rigid=True)
    gaps = np.diff(g.beats)
    assert gaps.std() < 1e-9, "a rigid grid must not drift at all"
    assert gaps.mean() == pytest.approx(60.0 / g.tempo, rel=1e-9)


def test_rigid_grid_extends_across_the_whole_track(click_track):
    """The fitted line is extended over intro and outro the tracker skipped, so the
    grid covers material the beat tracker never labelled."""
    g = analyse(click_track["path"], rigid=True)
    assert g.beats[0] >= -1e-9
    assert g.beats[-1] <= g.duration + 1e-9
    expected = int(np.floor(g.duration / (60.0 / g.tempo)))
    assert len(g.beats) >= expected - 1


def test_non_rigid_analysis_still_tracks_the_beat(click_track):
    g = analyse(click_track["path"], rigid=False)
    assert g.tempo == pytest.approx(click_track["bpm"], abs=1.0)
    for t in click_track["beats"][2:-2]:
        assert np.min(np.abs(g.beats - t)) < 0.05


def test_forcing_the_bpm_overrides_estimation(click_track):
    g = analyse(click_track["path"], bpm=60.0, rigid=False)
    assert g.tempo == pytest.approx(60.0, abs=1.0)


def test_offset_shifts_the_whole_grid(click_track):
    base = analyse(click_track["path"])
    moved = analyse(click_track["path"], offset=0.25)
    n = min(len(base.beats), len(moved.beats))
    np.testing.assert_allclose(moved.beats[:n], base.beats[:n] + 0.25, atol=1e-9)


def test_downbeat_index_forces_the_bar_phase(click_track):
    g = analyse(click_track["path"], downbeat_index=1)
    beats = list(np.round(g.beats, 9))
    assert np.round(g.downbeats[0], 9) == beats[1]
    np.testing.assert_allclose(np.diff(g.downbeats), 4 * 60.0 / g.tempo, atol=1e-6)


def test_beats_per_bar_changes_the_downbeat_spacing(click_track):
    g = analyse(click_track["path"], beats_per_bar=3, downbeat_index=0)
    np.testing.assert_allclose(np.diff(g.downbeats), 3 * 60.0 / g.tempo, atol=1e-6)
    assert g.beats_per_bar == 3


def test_auto_align_moves_the_grid_earlier(click_track):
    """Spectral-flux envelopes peak after the transient, so an unaligned grid sits late.
    The correction should pull it back toward the true beat times."""
    aligned = analyse(click_track["path"], auto_align=True)
    raw = analyse(click_track["path"], auto_align=False)

    def mean_err(g):
        return np.mean([np.min(np.abs(g.beats - t)) for t in click_track["beats"]])

    assert mean_err(aligned) <= mean_err(raw) + 1e-9


def test_onset_envelope_is_populated(click_track):
    g = analyse(click_track["path"])
    assert g.onset_env is not None and g.onset_env.size > 100
    assert g.onset_times.shape == g.onset_env.shape
    assert g.onset_env.max() > 0


@pytest.mark.parametrize("bpm", [90.0, 128.0, 140.0])
def test_analyse_across_tempos(tmp_path, bpm):
    import soundfile as sf

    y, sr, beats, downbeats = make_click_track(bpm=bpm, bars=8)
    path = tmp_path / f"click_{int(bpm)}.wav"
    sf.write(path, y, sr)

    g = analyse(str(path))
    # Half/double-time confusion is a known and documented failure; accept an octave.
    ratio = g.tempo / bpm
    assert min(abs(ratio - r) for r in (0.5, 1.0, 2.0)) < 0.01, (
        f"tempo {g.tempo:.2f} is not {bpm} or an octave of it")


def test_sparse_material_tracks_half_time_and_bpm_rescues_it(tmp_path):
    """A documented limit, pinned rather than hidden: with only four transients a bar
    and no eighth-note layer, the tracker settles on half-time. The README's advice is
    to pass --bpm, so check that actually recovers the grid."""
    import soundfile as sf

    y, sr, beats, _ = make_click_track(bpm=120.0, bars=8, hats=False)
    path = tmp_path / "sparse.wav"
    sf.write(path, y, sr)

    loose = analyse(str(path))
    assert loose.tempo == pytest.approx(60.0, abs=1.0), (
        "if this starts passing at 120, the tracker improved and the note in "
        "dancekit/README.md about half-time can be softened")

    forced = analyse(str(path), bpm=120.0)
    # Note that `bpm` seeds the tracker rather than pinning the output: the rigid refit
    # re-derives tempo from the tracked beats afterwards, so the result lands near the
    # requested tempo rather than exactly on it.
    assert forced.tempo == pytest.approx(120.0, abs=0.1)
    err = [np.min(np.abs(forced.beats - t)) for t in beats]
    assert np.mean(err) < 0.030


def test_forced_bpm_is_honoured_exactly_without_the_rigid_refit(tmp_path):
    """With `rigid=False` nothing re-estimates the tempo, so a forced BPM survives."""
    import soundfile as sf

    y, sr, _, _ = make_click_track(bpm=120.0, bars=8, hats=False)
    path = tmp_path / "sparse_norigid.wav"
    sf.write(path, y, sr)

    assert analyse(str(path), bpm=120.0, rigid=False).tempo == pytest.approx(120.0, abs=0.01)


def test_analyse_with_a_lead_in(tmp_path):
    """A track that does not start on beat 1 still needs a grid aligned to its beats."""
    import soundfile as sf

    y, sr, beats, downbeats = make_click_track(bpm=120.0, bars=8, lead_in=0.7)
    path = tmp_path / "leadin.wav"
    sf.write(path, y, sr)

    g = analyse(str(path))
    for t in beats[4:-4]:
        assert np.min(np.abs(g.beats - t)) < 0.05


def test_trim_silence_drops_a_silent_lead_in(tmp_path):
    """Trimming changes where t=0 sits, so the reported duration must shrink with it."""
    import soundfile as sf

    y, sr, _, _ = make_click_track(bpm=120.0, bars=4)
    padded = np.concatenate([np.zeros(sr * 2, dtype=np.float32), y])
    path = tmp_path / "padded.wav"
    sf.write(path, padded, sr)

    kept = analyse(str(path), trim_silence=False)
    trimmed = analyse(str(path), trim_silence=True)
    assert trimmed.duration < kept.duration - 1.0
