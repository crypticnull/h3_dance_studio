"""Beat analysis node: audio -> DK_BEATGRID."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import _dk
from ._dk import beatgrid, cache_pose, grid_report  # noqa: F401


class DKBeatGrid:
    """Analyse a song into a hard timing grid (tempo, beats, downbeats)."""

    CATEGORY = "dancekit"
    RETURN_TYPES = ("DK_BEATGRID", "STRING")
    RETURN_NAMES = ("beat_grid", "report")
    FUNCTION = "analyse"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the song (mp3/wav/flac/...). Ignored when the "
                               "optional AUDIO input is connected."}),
                "bpm": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 400.0, "step": 0.1,
                    "tooltip": "Force the tempo instead of estimating it. 0 = auto. "
                               "The tracker's classic failure is half/double time "
                               "(60 vs 120 BPM) -- if the report shows half or double "
                               "what you expect, set this. Not a failure of the tool; "
                               "it's genuinely ambiguous from onsets alone."}),
                "tightness": ("FLOAT", {
                    "default": 100.0, "min": 1.0, "max": 1000.0, "step": 1.0,
                    "tooltip": "How strongly the tracker is held to a steady tempo. "
                               "Raise for electronic/produced music, lower to let it "
                               "follow rubato in live playing. Mostly irrelevant when "
                               "rigid is on, which refits a constant tempo anyway."}),
                "offset": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001,
                    "tooltip": "Seconds added to every beat. Use to nudge a grid that "
                               "is consistently early or late against the audio. "
                               "auto_align already cancels the analysis latency, so "
                               "you rarely need this."}),
                "beats_per_bar": ("INT", {
                    "default": 4, "min": 2, "max": 12,
                    "tooltip": "Time signature numerator. Downbeats are every Nth "
                               "beat; phrase structure in Compose builds on bars."}),
                "downbeat_index": ("INT", {
                    "default": -1, "min": -1, "max": 11,
                    "tooltip": "Force which beat (0..beats_per_bar-1) is beat 1 of the "
                               "bar. -1 = auto-detect from low-band (kick) energy. The "
                               "detector deliberately scores the kick, not the snare -- "
                               "the snare is louder but marks beats 2 and 4. If the "
                               "grid lands on beat 3, set this instead of fighting it."}),
                "rigid": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Refit one constant-tempo line through the tracked "
                               "beats. Anything produced to a click (all pop and "
                               "electronic) does not drift, and this removes the "
                               "tracker's per-beat jitter entirely. Turn OFF only for "
                               "live or rubato material."}),
                "auto_align": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Spectral-flux onset envelopes peak a frame or two "
                               "AFTER the real transient, so a fitted grid sits "
                               "systematically late. This slides the whole grid to the "
                               "lag that puts the most onset energy on the beats."}),
                "trim_silence": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Strip leading/trailing silence before analysis. Only "
                               "use if you will also trim the audio you feed the video "
                               "model, or the grid will be offset from the file."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "ComfyUI AUDIO input (e.g. from LoadAudio). Takes "
                               "priority over audio_path when connected."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def analyse(self, audio_path, bpm, tightness, offset, beats_per_bar,
                downbeat_index, rigid, auto_align, trim_silence,
                audio=None, unique_id=None):
        path = self._resolve_audio(audio, audio_path)
        grid = beatgrid.analyse(
            str(path),
            bpm=(bpm if bpm > 0 else None),
            tightness=tightness,
            beats_per_bar=beats_per_bar,
            offset=offset,
            trim_silence=trim_silence,
            rigid=rigid,
            auto_align=auto_align,
            downbeat_index=(downbeat_index if downbeat_index >= 0 else None),
        )
        return (grid, grid_report(grid))

    @staticmethod
    def _resolve_audio(audio, audio_path) -> Path:
        if audio is not None:
            import soundfile as sf
            wav = audio["waveform"]
            sr = int(audio["sample_rate"])
            arr = wav[0] if wav.ndim == 3 else wav      # (C, S)
            arr = np.asarray(arr.detach().cpu().numpy() if hasattr(arr, "detach") else arr)
            if arr.ndim == 2:
                arr = arr.mean(axis=0)                  # mono
            tmp = Path(tempfile.gettempdir()) / "dancekit_audio_in.wav"
            sf.write(str(tmp), arr.astype(np.float32), sr)
            return tmp
        if not audio_path or not Path(audio_path).is_file():
            raise FileNotFoundError(
                f"No audio: connect an AUDIO input or set audio_path "
                f"(got {audio_path!r}).")
        return Path(audio_path)


NODE_CLASS_MAPPINGS = {"DKBeatGrid": DKBeatGrid}
NODE_DISPLAY_NAME_MAPPINGS = {"DKBeatGrid": "DanceKit Beat Grid"}
