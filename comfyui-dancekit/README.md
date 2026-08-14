# comfyui-dancekit

ComfyUI nodes for [dancekit](https://github.com/): beat-locked OpenPose skeleton
sequences for pose-conditioned video generation (ControlNet, SteadyDancer, MiniMax
H3). Song in, timing grid out, choreography composed / transferred / projected onto
it, skeleton IMAGE batch out — ready to feed Apply ControlNet.

Install: see [install.md](install.md).

## Nodes (category: `dancekit`)

| Node | Does |
|---|---|
| **DanceKit Beat Grid** | audio (path or AUDIO input) → `DK_BEATGRID` + report. Rigid constant-tempo refit, kick-based downbeat detection, auto lag alignment. |
| **DanceKit Compose** | `DK_BEATGRID` (+ optional `DK_LIBRARY`) → `DK_POSE`. Generates original choreography: one motif per detected musical section, replayed with side-flips on every return. |
| **DanceKit Harvest** | folder/file of clips or pose JSONs (+ optional `DK_POSE`) → `DK_LIBRARY` + contact-sheet IMAGE + manifest. Keyposes only, quality-gated, canonicalised, deduped in bone-angle space. |
| **DanceKit Save / Load Library** | `DK_LIBRARY` ↔ `vocabulary.npz` (same file the dancekit CLI writes). |
| **DanceKit Load / Save Pose JSON** | interop with `DWPreprocessor` → `SavePoseKpsAsJsonFile` (`POSE_KEYPOINT` format, pixel or normalised coords). |
| **DanceKit Retime** | `DK_POSE` + `DK_BEATGRID` → `DK_POSE`. Pins the source's held shapes to grid points, warps timing between them in bone space. |
| **DanceKit SMPL to OpenPose** | EDGE / AtomicDance / OpenDance motion file → `DK_POSE` (orthographic projection, whole-clip framing). |
| **DanceKit Render Skeleton** | `DK_POSE` → IMAGE batch (B,H,W,3 float 0..1), canonical OpenPose colours — this feeds ControlNet. |
| **DanceKit Pose Info** | `DK_POSE` → diagnostics string (keyposes, implied BPM, slow-mo check). |
| **DanceKit Trim / Frame Count** | trim/pad to a frame count, with a MiniMax H3 snap (`frames % 17 == 5`: 22, 39, 56, 73, 90, 107, 124). |

`DK_BEATGRID`, `DK_POSE`, `DK_LIBRARY` are plain Python objects passed along wires.
`DK_POSE` carries its fps and (when known) the beat grid it was made against, so
downstream previews can show beat markers.

## Typical graphs

**Generated choreography**

```
LoadAudio → Beat Grid → Compose → Trim (h3_frame_grid) → Render Skeleton → Apply ControlNet → ...
                            ↑
       Harvest (clips/) ─ library
```

**Transfer an existing dance**

```
Load Pose JSON → Retime ← Beat Grid ← LoadAudio
                    ↓
             Render Skeleton → ...
```

## Frontend widgets

- **Skeleton preview** on Compose / Retime / Render / SMPL / Load Pose JSON / Trim:
  canvas skeleton, frame scrubber, play/pause at the pose's fps, frame counter.
  Beat ticks under the scrubber (tall = downbeat) so you can see hits line up.
  Data is cached server-side per node at execution time — scrub freely, nothing
  recomputes. Run the graph once to populate.
- **Library browser** on Harvest / Load Library: scrollable grid of the vocabulary,
  energy per pose, click to toggle keep/drop. Dropped names go into the node's
  `dropped` widget (so the selection survives graph save/load) and are excluded on
  the next run. Note: Harvest names are positional (`h00`, `h01`, ...) — they stay
  stable only while the harvest inputs are unchanged.
- Tooltips on every adjustable widget, both native (`tooltip` in the input specs)
  and JS-side for older frontends.

## Notes

- Beat analysis, composing and rendering are CPU-only and fast; the GPU stays free
  for generation.
- Sanity check before spending GPU time: scrub the preview and confirm hits land on
  the beat ticks. If the skeleton is on-beat there, anything off in the final render
  is conditioning strength, not the control signal.
- `selftest.py` runs the whole node chain headless (no ComfyUI needed).
