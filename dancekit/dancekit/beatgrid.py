"""Turn a song into a hard timing grid: tempo, beats, downbeats, subdivisions.

This is the piece that makes the difference between a model that might notice the
music and a pipeline where the beat is a constraint the motion cannot drift off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BeatGrid:
    tempo: float
    beats: np.ndarray          # beat times in seconds
    downbeats: np.ndarray      # subset of `beats` landing on beat 1 of each bar
    duration: float
    sr: int = 22050
    onset_env: np.ndarray = field(default=None, repr=False)
    onset_times: np.ndarray = field(default=None, repr=False)
    beats_per_bar: int = 4

    # -- derived views ---------------------------------------------------------------

    def subdivide(self, n: int = 2) -> np.ndarray:
        """Insert n-1 evenly spaced points between consecutive beats.

        n=2 -> eighth notes, n=4 -> sixteenths. Most TikTok choreo hits on eighths,
        so n=2 is usually the right anchor density.
        """
        if n <= 1 or len(self.beats) < 2:
            return self.beats.copy()
        segs = [np.linspace(self.beats[i], self.beats[i + 1], n, endpoint=False)
                for i in range(len(self.beats) - 1)]
        return np.concatenate(segs + [self.beats[-1:]])

    def to_frames(self, fps: float, times: np.ndarray | None = None) -> np.ndarray:
        t = self.beats if times is None else times
        return np.round(np.asarray(t) * fps).astype(int)

    def phase_at(self, t: float) -> float:
        """Position within the bar at time t, in beats (0.0 == downbeat)."""
        if len(self.downbeats) == 0 or self.tempo <= 0:
            return 0.0
        spb = 60.0 / self.tempo
        return ((t - self.downbeats[0]) / spb) % self.beats_per_bar

    def as_dict(self) -> dict:
        return {
            "tempo": float(self.tempo),
            "duration": float(self.duration),
            "beats_per_bar": int(self.beats_per_bar),
            "beats": [round(float(x), 4) for x in self.beats],
            "downbeats": [round(float(x), 4) for x in self.downbeats],
        }


def _estimate_downbeats(beats: np.ndarray, low_power: np.ndarray,
                        low_times: np.ndarray, beats_per_bar: int = 4,
                        window: float = 0.08) -> np.ndarray:
    """Pick the bar phase whose beats carry the most LOW-BAND energy.

    librosa has no downbeat tracker. Two traps here, both learned the hard way:

    1. Scoring broadband onset strength picks the snare (beats 2 and 4 in pop), which
       is exactly wrong -- the snare is usually the loudest transient in the mix. The
       kick marks beat 1, so score the sub-bass band only.
    2. Score LINEAR power, not dB. Decibels compress exactly the loudness difference
       that distinguishes the downbeat kick from the beat-3 kick; in testing a 1.35 vs
       0.8 amplitude gap showed up as 160 vs 56 in power and a useless 2.44 vs 2.44
       in dB.

    Energy is integrated over a short window after each beat so we catch the body of
    the kick, not just the instant of the transient. Still a heuristic -- override with
    `downbeat_index` when a track fools it.
    """
    if len(beats) < beats_per_bar:
        return beats[:1].copy()
    grid = np.linspace(0.0, window, 8)
    strength = np.array([np.interp(b + grid, low_times, low_power).mean() for b in beats])
    scores = [strength[p::beats_per_bar].mean() for p in range(beats_per_bar)]
    return beats[int(np.argmax(scores))::beats_per_bar].copy()


def _fit_rigid(beats: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of a constant-tempo grid to tracked beats -> (bpm, phase).

    Produced music does not drift. Fitting one straight line through the tracked beats
    removes the tracker's per-beat jitter (which is bounded by its hop size) and gives
    a grid that stays locked over a whole track instead of wandering by a frame or two.
    """
    if len(beats) < 3:
        return 0.0, float(beats[0]) if len(beats) else 0.0
    k = np.arange(len(beats), dtype=float)
    # Robust-ish: drop the worst 10% of residuals and refit.
    A = np.stack([k, np.ones_like(k)], axis=-1)
    sol, *_ = np.linalg.lstsq(A, beats, rcond=None)
    resid = np.abs(beats - A @ sol)
    keep = resid <= np.percentile(resid, 90)
    if keep.sum() >= 3:
        sol, *_ = np.linalg.lstsq(A[keep], beats[keep], rcond=None)
    spb, phase = float(sol[0]), float(sol[1])
    return (60.0 / spb if spb > 1e-6 else 0.0), phase


def analyse(path: str, sr: int = 22050, bpm: float | None = None,
            tightness: float = 100.0, beats_per_bar: int = 4,
            offset: float = 0.0, trim_silence: bool = False,
            hop_length: int = 128, rigid: bool = True, auto_align: bool = True,
            downbeat_index: int | None = None) -> BeatGrid:
    """Analyse an audio file into a BeatGrid.

    bpm            - force a tempo instead of estimating (fixes half/double-time errors)
    tightness      - higher keeps the tracker closer to a steady tempo; lower lets it
                     follow rubato. Raise it for electronic, lower for live playing.
    hop_length     - onset envelope resolution. librosa's default of 512 quantises every
                     beat to a 23ms grid, which is a visible timing error at 24fps.
                     128 gives ~5.8ms. Cheap; leave it.
    rigid          - refit a constant-tempo grid through the tracked beats. Correct for
                     anything produced to a click, which is all pop/electronic. Turn it
                     off for live or rubato material.
    auto_align     - slide the finished grid to the lag that puts the most onset
                     energy on the beats, cancelling the analysis window's latency.
    offset         - seconds added to every beat; nudges a grid that is consistently
                     early or late against the audio.
    downbeat_index - force the bar phase (0..beats_per_bar-1) instead of estimating.
    """
    import librosa

    y, sr = librosa.load(path, sr=sr, mono=True)
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=40)
    duration = float(len(y) / sr)

    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, aggregate=np.median)
    onset_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, bpm=bpm,
        tightness=tightness, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    if rigid and len(beats) >= 3:
        fit_bpm, phase = _fit_rigid(beats)
        if fit_bpm > 1e-6:
            tempo = fit_bpm
            spb = 60.0 / tempo
            # Extend the fitted line across the whole track, including any intro or
            # outro the tracker skipped.
            k0 = int(np.floor(-phase / spb))
            k1 = int(np.ceil((duration - phase) / spb))
            beats = phase + np.arange(k0, k1 + 1) * spb
            beats = beats[(beats >= -1e-9) & (beats <= duration)]

    if auto_align and len(beats) >= 4:
        # Spectral-flux envelopes peak a frame or two AFTER the actual transient, so a
        # fitted grid sits systematically late by a fixed amount. Slide the whole grid
        # over a small lag range and keep the lag that maximises energy on the beats.
        lags = np.arange(-0.060, 0.0601, 0.001)
        score = [np.interp(beats + L, onset_times, onset_env,
                           left=0.0, right=0.0).sum() for L in lags]
        beats = beats + float(lags[int(np.argmax(score))])

    beats = beats + offset

    # Low-band power for downbeat phase (kick, not snare). Linear, not dB -- see
    # _estimate_downbeats.
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length,
                                         n_mels=32, fmax=400)
    low_power = mel[:5].mean(axis=0)
    low_times = librosa.times_like(low_power, sr=sr, hop_length=hop_length)

    if downbeat_index is not None and len(beats):
        downbeats = beats[int(downbeat_index) % beats_per_bar::beats_per_bar].copy()
    else:
        downbeats = _estimate_downbeats(beats, low_power, low_times, beats_per_bar)

    return BeatGrid(tempo=tempo, beats=beats, downbeats=downbeats, duration=duration,
                    sr=sr, onset_env=onset_env, onset_times=onset_times,
                    beats_per_bar=beats_per_bar)


def synthetic(tempo: float, duration: float, beats_per_bar: int = 4,
              offset: float = 0.0) -> BeatGrid:
    """A perfectly rigid grid. Use when you already know the BPM of a produced track
    and would rather not let the tracker second-guess it."""
    spb = 60.0 / tempo
    beats = np.arange(offset, duration, spb)
    return BeatGrid(tempo=tempo, beats=beats, downbeats=beats[::beats_per_bar].copy(),
                    duration=duration, beats_per_bar=beats_per_bar,
                    onset_env=np.zeros(1), onset_times=np.zeros(1))
