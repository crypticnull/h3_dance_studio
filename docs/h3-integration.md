# Wiring dancekit into a MiniMax H3 R2V workflow

Written against a `MiniMaxH3ReferenceToVideo` graph that already has `LoadAudio` →
`TrimAudioDuration` → `ref_audios.ref_audio_0`, a `VHS_LoadVideo` feeding
`ref_videos.ref_video_0`, and `ResolutionSelector` driving width/height.

## The short version

`MiniMaxH3ReferenceToVideo` takes its reference videos as **`IMAGE` batches**, and
**DanceKit Render Skeleton outputs an `IMAGE` batch**. They connect directly. No adapter,
no conversion node.

```
LoadAudio ─┬─> TrimAudioDuration ──> ref_audios.ref_audio_0
           │
           └─> DanceKit Beat Grid ──> DanceKit Compose ──> DanceKit Trim / Frame Count
                                                                    │
                                            DanceKit Render Skeleton ┘
                                                     │
                                                     └──> ref_videos.ref_video_1
```

Feed the **same audio file** to `LoadAudio` and to `DanceKit Beat Grid`. That's the whole
trick — the choreography is then locked to the exact track H3 is already conditioning on,
rather than to a track that merely sounds similar.

Most R2V graphs leave `ref_video_1` empty, so the skeleton can go there without giving up
the reference video already in slot 0.

## Frame count

H3 requires `frames % 17 == 5` → 22, 39, 56, 73, 90, 107, 124.

Set **DanceKit Trim / Frame Count** to `h3_frame_grid = true` and it snaps to the nearest
valid count. Then make `MiniMaxH3ReferenceToVideo`'s `length` match that number. If the
two disagree, H3 will resample or truncate your skeleton and the beat alignment you just
built goes out the window.

At 24 fps, 124 frames is 5.17 seconds — about 11 beats at 128 BPM. Use
`--seconds` / `max_seconds` on Compose, or Trim's `start_frame`, to pick which slice of a
longer composed dance you actually render.

## Resolution

Render the skeleton at the same dimensions H3 is generating at. With `ResolutionSelector`
on `9:16 (Portrait Widescreen)`, read the width/height it outputs and set the same values
on **DanceKit Render Skeleton**. Multiples of 32.

Mismatched aspect ratios are the most common cause of "the pose reference did nothing" —
a letterboxed or stretched skeleton reads as a different body shape.

## Set expectations: this is soft conditioning

Worth being straight about. H3's `ref_videos` are *references*, not ControlNet. They carry
motion and style as guidance the model weighs against everything else — they are not a
per-frame constraint the way a pose ControlNet is. So:

- You will get noticeably better rhythm and more deliberate, structured movement.
- You will **not** get frame-exact pose adherence. The character won't hit your skeleton's
  shapes precisely.

If you need hard pose lock, that requires a genuine pose-ControlNet path — Wan 2.1/2.2 with
an OpenPose ControlNet, or SteadyDancer, both of which take a skeleton as an actual
constraint. The same **Render Skeleton** node feeds those graphs unchanged; only the
consumer differs.

A reasonable hybrid: hard pose control on a Wan pass to establish the motion, then H3 for
the final look with that output as `ref_video_0`.

## Checking it before you spend GPU time

1. Run just the Beat Grid → Compose → Render chain.
2. Scrub the skeleton preview. Beat ticks appear under the scrubber — short for beats,
   tall for downbeats.
3. Confirm the shapes land on the ticks.

If the skeleton hits the beat in the preview, any remaining drift in the final video is
H3's conditioning strength, not your control signal — and that's a different problem with
different fixes (reference weighting, prompt, LoRA).

## Where a LoRA still helps

None of this replaces a movement LoRA — it complements it. Pose reference supplies
*timing and structure*; a LoRA trained on good dance footage supplies *quality* — real limb
extension, weight transfer, hair and fabric follow-through. The two operate on different
failure modes. Feed the skeleton as reference and run your dance LoRA through
`Power Lora Loader` in the same graph.
