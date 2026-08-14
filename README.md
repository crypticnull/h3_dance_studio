# h3_dance_studio

Beat-locked pose control for AI video generation. Turn a song into a hard timing grid,
generate or harvest choreography against it, and drive MiniMax H3 / Wan / ControlNet
workflows with a skeleton sequence that actually lands on the beat.

Two pieces:

| | |
|---|---|
| **[`dancekit/`](dancekit/)** | The library and CLI. Beat analysis, choreography generation, pose harvesting, retiming, SMPL projection, OpenPose rendering. CPU-only, no GPU needed. |
| **[`comfyui-dancekit/`](comfyui-dancekit/)** | ComfyUI node pack wrapping all of it, with in-node skeleton preview, frame scrubbing, beat markers, and a clickable pose-library browser. |

## The problem this solves

A LoRA cannot fix timing. It's a weight delta on a diffusion transformer — no planner, no
clock, no representation of a beat grid, and every frame denoises in parallel from noise.
A dance LoRA genuinely improves *movement quality* (amplitude, weight, follow-through,
the difference between committed choreography and vague swaying), but the moment you need
motion to hit on the 1, timing has to live in an explicit control signal.

Second, less obvious: **what makes movement read as intentional is motif repetition, not
timing accuracy.** Random poses landing perfectly on every beat still look like flailing.
A dancer states a phrase, repeats it on the other side, and brings it back when the chorus
returns. The composer is built around that.

## Three ways to get choreography

| Path | Choreography from | Command |
|---|---|---|
| **Generated** | a pose vocabulary + the song's own structure | `compose` |
| **ML-generated** | a music→motion model (EDGE, AtomicDance, OpenDance) | `smpl` |
| **Transferred** | an existing dance clip | `extract` → `retime` |

Plus `harvest`, which turns a folder of clips — real footage, your own generated output, or
both — into a pose vocabulary the generated path composes from. That's how you get *style*
rather than generic movement.

Output is always the same: OpenPose BODY_18 in ComfyUI's `POSE_KEYPOINT` format, plus a
rendered skeleton image sequence you can feed straight into a video graph.

## Quick start

```bash
pip install -r dancekit/requirements.txt

# generate a dance for your song
python -m dancekit compose song.mp3 -o out/ --seed 3

# build a style vocabulary from clips, then compose with it
python -m dancekit harvest clips/ -o vocab/ --skip-slowmo --min-count 2
python -m dancekit compose song.mp3 -o out/ --library vocab/vocabulary.npz
```

Play `out/skeleton.mp4` with the audio on. If the skeleton hits with the track there, the
timing problem is solved.

## ComfyUI

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/crypticnull/h3_dance_studio
ln -s h3_dance_studio/comfyui-dancekit ./comfyui-dancekit
pip install -e h3_dance_studio/dancekit
```

See [`comfyui-dancekit/install.md`](comfyui-dancekit/install.md) for details and
[`docs/h3-integration.md`](docs/h3-integration.md) for wiring into a MiniMax H3 R2V graph.

## Docs

- [`dancekit/README.md`](dancekit/README.md) — full CLI reference, tuning, and the honest
  limits
- [`docs/h3-integration.md`](docs/h3-integration.md) — dropping pose reference into an
  existing H3 workflow

## A note on sources

Harvesting from footage you didn't make means other people's likenesses, and bulk
downloading from most platforms is against their terms. Fine for personal experiments;
worth thinking about before anything ships. Which is part of why the fully generated path
exists.
