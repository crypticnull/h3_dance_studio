# dancekit

Beat-locked pose control for video generation. Turns a song into a hard timing grid, then
produces an OpenPose skeleton sequence that lands on it — so choreography is a constraint
your video model obeys rather than something it might coincidentally notice.

Three ways in, depending on where the choreography comes from:

| Path | Choreography from | Command |
|---|---|---|
| **Generated** (no source) | a pose vocabulary + the song's own structure | `compose` |
| **ML-generated** | a music→motion model (EDGE, AtomicDance, OpenDance) | `smpl` |
| **Transferred** | an existing dance clip | `extract` → `retime` |

Plus `harvest`, which turns a folder of clips into a pose vocabulary the generated path
composes from — that's how you get *style* rather than generic movement.

Output is always the same thing: OpenPose BODY_18 JSON in ComfyUI's `POSE_KEYPOINT` format,
plus a rendered skeleton MP4 you can drop straight into a ControlNet / pose-conditioned
I2V graph.

---

## Why this exists

A LoRA cannot fix timing. It's a weight delta on a diffusion transformer — there's no
planner in it, no clock, no representation of a beat grid, and all frames denoise in
parallel from noise. A dance LoRA can absolutely fix *movement quality* (amplitude, weight,
follow-through, the difference between committed choreography and vague swaying) but the
moment you need motion to hit on the 1, timing has to live in an explicit control signal.
That's what this produces.

Second thing, and it's the one people miss: **what makes movement read as intentional is
motif repetition, not timing accuracy.** A sequence of random poses landing perfectly on
every beat still looks like flailing. A dancer states a phrase, repeats it on the other
side, and brings it back when the chorus returns. `compose` is built around that — it
detects musical sections, writes one motif per section, and replays it on every return with
side-flips and tail variations.

---

## Install

```bash
pip install -r requirements.txt
# optional, only for `extract` (video -> pose):
pip install rtmlib onnxruntime-gpu
# optional, only for `prep` from a URL:
pip install yt-dlp
```

Needs `ffmpeg` on PATH. Everything else is CPU-only and fast — no GPU needed for any of
this; your 5090 stays free for generation.

---

## 1. Generated choreography (no source clip)

```bash
python -m dancekit compose song.mp3 -o out/ --seed 3
```

Writes `out/pose.json`, `out/skeleton.mp4` (with the song muxed in so you can check sync by
eye), and `out/compose.json` showing the motifs and section map it chose.

Useful knobs:

```
--subdivision 1     pose changes per beat. 1 = quarters (calmer), 2 = eighths (busy,
                    closer to actual TikTok choreo density)
--snap 0.65         fraction of each interval spent moving. Lower = the shape arrives
                    early and holds = sharper, more "hit"-like. 0.45-0.6 for popping,
                    0.8-1.0 for sustained/flowy
--overshoot 0.12    slight past-the-pose travel that settles back. 0.15-0.25 reads as
                    snap; above 0.4 looks broken
--sections 3        how many distinct musical sections to write material for
--variation 0.5     probability of flipping sides / varying the tail on repeats
--bounce 0.012      per-beat vertical pulse. Real dancers never fully stop, and a dead
                    hold between hits is the biggest tell of synthetic motion
--seed N            change this to get a different dance for the same song
```

Audit the pose vocabulary:

```bash
python -m dancekit poses -o library.png
```

The built-in library is 19 poses (plus mirrors) and is deliberately a *starting*
vocabulary — generic, not stylish. The real win is replacing it with your own; see
"Custom vocabulary" below.

## 2. ML-generated choreography

Music→motion models output 3D SMPL joints, and pose-conditioned video models want a 2D
OpenPose image. Nothing off the shelf bridges those, so:

```bash
# EDGE / AtomicDance / OpenDance -> .npy or .pkl of (T, 24, 3) joints
python -m dancekit smpl motion.npy -o out/ --azimuth 0 --audio song.mp3
```

- **EDGE** (Stanford-TML) — CVPR 2023, code and weights public, arbitrary length,
  supports joint-wise conditioning and in-betweening. The mature option.
- **AtomicDance** — ECCV 2026, represents choreography as reusable atomic movements,
  16 GB VRAM. Code is up; pretrained checkpoints were still marked "will be released"
  when I checked, so verify before planning around it.
- **OpenDance** — most controllable (music + text + keypoints + trajectory), 101 hours
  of training data. Check release status.

`--azimuth` / `--elevation` set the camera. Projection is orthographic on purpose:
these models emit motion in a canonical space with no camera, and inventing a focal
length adds distortion the video model then has to fight.

## 3. Transfer from an existing dance

```bash
python -m dancekit prep "https://..." -o clips/ --fps 24
python -m dancekit extract clips/x_norm.mp4 -o pose.json
python -m dancekit retime pose.json my_song.mp3 -o out/ --src-fps 24 --loop
```

`extract` also runs a slow-motion check — worth heeding. A large fraction of dance clips
online are conformed slow motion or speed-ramped, which produces weightless, dreamy
movement, and it's equally poison as a retiming source and as LoRA training data.

`retime` finds the held shapes in the source (speed minima), pins each to a grid point on
your song, and warps the timing between them — so you keep the source's inner detail and
only replace its clock. `--mode nearest` handles loose/freestyle sources; `--stride 2` maps
keyposes to every other grid point when the source is half-time.

If you already have poses from ComfyUI, skip `extract` — `DWPreprocessor` →
`SavePoseKpsAsJsonFile` writes exactly the format `retime` reads.

---

## Wiring into ComfyUI

`skeleton.mp4` (or the `--frames` PNG sequence) is a standard OpenPose rendering — the
canonical 18-colour palette and filled-ellipse limbs, which matters because ControlNet
OpenPose models were trained on precisely that rendering.

- **Load Video (Upload)** → frames → **Apply ControlNet** with an OpenPose model, into
  whatever I2V graph you're running your character through.
- For **SteadyDancer** (Wan 2.1 I2V, pose + reference image), feed the skeleton frames in
  place of the driving video's extracted pose.
- Match the frame count to your model's requirement. For MiniMax H3 that's
  `frames % 17 == 5` — so 22, 39, 56, 73, 90, 107, 124. Set `--fps 24` and cut to length,
  or generate in chunks and stitch.
- Render at your generation resolution (`--width 832 --height 1472` etc.). Multiples of 32.

Sanity check before you spend GPU time: play `skeleton.mp4` with the audio on. If the
skeleton hits with the track there, the timing problem is solved and anything still off in
the final render is the video model's conditioning strength, not the control signal.

---

## Harvesting a vocabulary

The built-in poses are generic. Style comes from the vocabulary, so point `harvest` at a
folder of clips — real footage, your own generated output, or a mix:

```bash
python -m dancekit harvest clips/ -o vocab/ --skip-slowmo
python -m dancekit compose song.mp3 -o out/ --library vocab/vocabulary.npz
```

It accepts video files (`.mp4/.mov/.avi/.mkv/.webm`, needs rtmlib) and ComfyUI pose JSON
side by side in the same folder, walks subdirectories, and writes `vocabulary.npz`, a
`vocabulary.png` contact sheet, and a `manifest.json` recording which clip and frame every
shape came from.

**Be clear on what it keeps.** It harvests *shapes*. Sequence, timing and phrase structure
are discarded and regenerated by `compose`, so a vocabulary built from real footage does
not reproduce anyone's routine.

What happens to each clip:

1. **Keyposes** — held shapes only (speed minima). Transitions are not vocabulary.
2. **Quality gate** — rejects frames missing core joints (a cropped frame is not a shape),
   frames with anatomically implausible limb lengths (detector failures), and figures too
   small in frame. Rejections are counted per reason so you can see what your footage is
   costing you.
3. **Slow-motion check** per clip, printed as a warning; `--skip-slowmo` drops them.
   Conformed slow motion teaches weightless movement and is worth excluding.
4. **Canonicalisation** — every shape is rebuilt on one standard body, keeping its angles.
   Without this, shapes from clips at different distances and framings make the figure
   grow and shrink between every hit. `--foreshorten` (default 0.6) controls how much
   observed limb proportion is blended back; that's the only depth cue a 2D skeleton has,
   so 0 is flattest and most stable, 1 keeps the most depth.
5. **Dedupe** — hierarchical clustering on weighted bone-angle distance, keeping each
   cluster's medoid. Angles rather than joint positions, because two dancers of different
   builds hitting the same shape differ a lot in xy and barely at all in angle.
   Mirror-invariant by default, since `compose` generates mirrors itself.
6. **Ranking** — shapes that recurred across the footage sort above one-off outliers, and
   energies are rank-normalised so the composer gets a full dynamic range regardless of
   how the raw scores bunched up.

Tuning:

```
--prominence 0.12    lower finds more shapes per clip
--min-distance 0.30  dedupe threshold, radians of mean bone angle. Lower = richer,
                     noisier vocabulary; higher = fewer, more distinct shapes
--max-poses 32       cap on vocabulary size
--min-count 2        require a shape to recur before it earns a slot -- the best filter
                     for noisy footage, since detector glitches don't repeat
--drop-neutral 0.18  discard shapes too close to plain standing
--keep-mirrors       treat a pose and its mirror as distinct (rarely what you want)
```

Always look at `vocabulary.png` before composing with it. A vocabulary with three near
identical shapes and one broken skeleton produces exactly the dance you'd expect.

The Python API is there too if you want to filter or hand-edit the set:

```python
from dancekit import harvest, beatgrid, compose

cands, report = harvest.harvest_sequence(pose_seq, fps=30, source="clip.mp4")
lib, meta = harvest.build_vocabulary(cands, max_poses=24, min_distance=0.28)
harvest.save_library("vocab.npz", lib, meta)

lib, meta, emap = harvest.load_library("vocab.npz")     # adds mirrors
grid = beatgrid.analyse("song.mp3")
pose, info = compose.compose(grid, library=lib, energy_map=emap, seed=5)
```

You can equally hand-author poses in bone-angle space — see `poselib.NEUTRAL` and
`LIBRARY`.

---

## Beat detection notes

Defaults are tuned for produced music, which is what you'll be using:

- `hop_length=128` — librosa's default of 512 quantises every beat to a 23 ms grid, which
  is over half a frame at 24 fps. 128 gives ~5.8 ms. Costs nothing.
- `rigid=True` — refits one constant-tempo line through the tracked beats. Anything cut to
  a click doesn't drift, and this removes the tracker's per-beat jitter entirely. Turn it
  off (`--no-rigid`) for live or rubato material.
- `auto_align=True` — spectral-flux envelopes peak a frame or two *after* the actual
  transient, so a fitted grid sits systematically late. This slides the grid to the lag
  that puts the most onset energy on the beats.

Measured on synthetic tracks with known ground truth: tempo exact to ±0.01 BPM, all beats
recovered, mean beat error ~11 ms (about a quarter of a frame at 24 fps), downbeat phase
correct.

If tempo comes out half or double, pass `--bpm`. If the downbeat lands on beat 3, pass
`--downbeat-index`. Both are common and neither is a failure of the tool.

---

## Tests

```bash
pip install -e ".[test]"
pytest
```

439 tests, about 30 seconds, no GPU. The beat-analysis tests measure against synthetic
tracks with known beat times, so the accuracy figures quoted above are checked rather
than remembered. The two load-bearing ones are worth knowing about:

- `test_keyposes_land_on_the_grid` — every detected keypose ends up at the beat time it
  was assigned. That is the whole premise of `retime`.
- `test_the_dance_is_exactly_on_the_shape_at_every_anchor` — the composed equivalent.

`ffmpeg` on PATH enables the video-rendering tests; without it they skip. The `rtmlib`
video paths are not covered — see the last bullet below.

## Limits, honestly

- **The built-in vocabulary is plain.** 19 poses of generic movement. It will produce
  clearly beat-locked, structured motion, but it won't produce *style*. Style comes from a
  harvested or hand-authored library.
- **2D only, front-facing.** Poses are authored for a camera-facing figure. Turns,
  travelling, and floor work aren't represented. The `smpl` path handles 3D properly if you
  need rotation.
- **Downbeat detection is a heuristic.** It scores low-band power (kick, not snare —
  scoring broadband onset strength reliably picks the snare, which is exactly wrong). It's
  right on most 4/4 pop and overridable when it isn't.
- **Section detection is energy-based**, not harmonic. A quiet chorus and a quiet verse
  will get the same material.
- **Loop points jump.** `retime --loop` restarts the phrase; feed it a clean 8-count or
  you'll see the seam.
- **Harvest keeps shapes, not sequences.** By design — but it means a vocabulary alone
  won't give you a specific routine, only that footage's movement palette.
- **`extract` and `harvest` on video are untested against real DWPose output here** — it's written against
  rtmlib's `Body` API but were validated on synthetic pose data, not a live model run.
  Expect to adjust if rtmlib's return shapes have moved. The JSON path is fully tested.

## A note on sources

Pulling clips off TikTok means other people's likenesses and licensed music, and bulk
downloading is against their terms. Fine for personal experiments. Worth thinking about
before anything ships — which is part of why the `compose` path exists.
