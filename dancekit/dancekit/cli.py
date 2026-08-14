"""dancekit command line.

    python -m dancekit beats    song.mp3
    python -m dancekit compose  song.mp3 -o out/       # generated choreography
    python -m dancekit harvest  clips/ -o vocab/       # clips -> pose vocabulary
    python -m dancekit poses    -o library.png         # audit the vocabulary
    python -m dancekit extract  clip.mp4 -o pose.json  # video -> pose (needs rtmlib)
    python -m dancekit retime   pose.json song.mp3 -o out/
    python -m dancekit smpl     motion.npy -o out/     # EDGE/AtomicDance -> OpenPose
    python -m dancekit render   pose.json -o out.mp4
    python -m dancekit prep     <url|file> -o clips/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import (beatgrid, compose as composer, harvest as harvester, keypose, poseio,
               poselib, render, retime, smpl2d)


def _grid(a):
    return beatgrid.analyse(a.audio, bpm=a.bpm, tightness=a.tightness, rigid=not a.no_rigid,
                            auto_align=not a.no_align, offset=a.offset,
                            beats_per_bar=a.beats_per_bar, downbeat_index=a.downbeat_index)


def _add_audio_args(p):
    p.add_argument("--bpm", type=float, default=None, help="force tempo")
    p.add_argument("--tightness", type=float, default=100.0)
    p.add_argument("--offset", type=float, default=0.0, help="shift grid, seconds")
    p.add_argument("--beats-per-bar", type=int, default=4)
    p.add_argument("--downbeat-index", type=int, default=None)
    p.add_argument("--no-rigid", action="store_true", help="don't refit constant tempo")
    p.add_argument("--no-align", action="store_true", help="don't auto-correct grid lag")


def _add_render_args(p):
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=1472)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--thickness", type=float, default=1.0)
    p.add_argument("--frames", action="store_true", help="also write a PNG sequence")


# --- commands --------------------------------------------------------------------------

def cmd_beats(a):
    g = _grid(a)
    out = g.as_dict()
    out["beat_interval_s"] = round(60.0 / g.tempo, 4)
    print(json.dumps({k: v for k, v in out.items() if k not in ("beats", "downbeats")},
                     indent=2))
    print(f"beats: {len(g.beats)}  downbeats: {len(g.downbeats)}  "
          f"first downbeat {g.downbeats[0]:.3f}s" if len(g.downbeats) else "")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"-> {a.out}")


def cmd_compose(a):
    g = _grid(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    lib = emap = None
    if a.library:
        lib, meta, emap = harvester.load_library(a.library)
        print(f"vocabulary: {len(meta)} shapes (+mirrors) from {a.library}")

    pose, info = composer.compose(
        g, fps=a.fps, subdivision=a.subdivision, phrase_beats=a.phrase_beats,
        sections=a.sections, seed=a.seed, snap=a.snap, overshoot=a.overshoot,
        variation=a.variation, max_seconds=a.seconds, bounce=a.bounce,
        library=lib, energy_map=emap)

    poseio.save_pose_json(out / "pose.json", pose,
                          {"canvas_width": a.width, "canvas_height": a.height})
    (out / "compose.json").write_text(json.dumps(info, indent=2))
    render.render_video(pose, out / "skeleton.mp4", fps=a.fps, width=a.width,
                        height=a.height, thickness=a.thickness,
                        audio=a.audio if not a.no_audio else None)
    if a.frames:
        render.render_frames(pose, out / "frames", a.width, a.height, a.thickness)

    print(f"tempo {info['tempo']}  {info['frames']} frames @ {a.fps}fps  "
          f"({info['duration_s']}s)  phrases {info['phrases']}")
    print(f"sections: {info['section_sequence']}")
    print(f"-> {out}/pose.json, {out}/skeleton.mp4")


def cmd_poses(a):
    lib = {k: v for k, v in poselib.all_poses().items()
           if a.mirrors or not k.endswith("_m")}
    render.contact_sheet(lib, a.out, cell=a.cell, cols=a.cols)
    print(f"{len(lib)} poses -> {a.out}")


def cmd_extract(a):
    try:
        from rtmlib import Body
    except ImportError:
        sys.exit("Needs rtmlib: pip install rtmlib onnxruntime-gpu\n"
                 "Or export pose from ComfyUI (DWPreprocessor -> SavePoseKpsAsJsonFile) "
                 "and skip this step.")
    import cv2

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = Body(mode=a.mode, backend="onnxruntime", device=a.device)
    frames, n = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        kps, scores = model(img)
        if len(kps) == 0:
            frames.append(np.zeros((18, 3)))
        else:
            best = int(np.argmax(scores.sum(axis=1)))
            k = np.concatenate([kps[best], scores[best][:, None]], axis=-1)[:18]
            k[:, 0] /= W
            k[:, 1] /= H
            frames.append(k)
        n += 1
        if a.max_frames and n >= a.max_frames:
            break
    cap.release()

    pose = np.stack(frames)
    poseio.save_pose_json(a.out, pose, {"canvas_width": W, "canvas_height": H})
    rep = keypose.detect_slowmo(pose, fps)
    print(f"{n} frames @ {fps:.3f}fps -> {a.out}")
    print(f"slow-motion check: {rep}")


def cmd_retime(a):
    pose, meta = poseio.load_pose_json(a.pose)
    g = _grid(a)
    keys = keypose.detect_keyposes(pose, fps=a.src_fps, prominence=a.prominence,
                                   min_gap_s=a.min_gap, max_count=a.max_keyposes)
    print(f"source: {pose.shape[0]} frames, {len(keys)} keyposes")
    print(f"  {keypose.phrase_report(pose, keys, a.src_fps)}")

    grid_times = g.subdivide(a.subdivision)
    out_pose, info = retime.retime_to_grid(
        pose, a.src_fps, keys, grid_times, out_fps=a.fps, snap=a.snap,
        overshoot=a.overshoot, stride=a.stride, start_index=a.start_index,
        mode=a.mode, root_damping=a.root_damping, loop=a.loop)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    poseio.save_pose_json(out / "pose.json", out_pose,
                          {"canvas_width": a.width, "canvas_height": a.height})
    (out / "retime.json").write_text(json.dumps(info, indent=2))
    render.render_video(out_pose, out / "skeleton.mp4", fps=a.fps, width=a.width,
                        height=a.height, thickness=a.thickness,
                        audio=a.audio if not a.no_audio else None)
    print(f"{info['anchors']} anchors, {info['out_frames']} frames -> {out}")


def cmd_smpl(a):
    j = smpl2d.load_joints(a.motion)
    if j.ndim == 2:
        j = j.reshape(j.shape[0], -1, 3)
    pose = smpl2d.smpl_to_openpose(j, azimuth=a.azimuth, elevation=a.elevation,
                                   up_axis=a.up, headroom=a.headroom)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    poseio.save_pose_json(out / "pose.json", pose,
                          {"canvas_width": a.width, "canvas_height": a.height})
    render.render_video(pose, out / "skeleton.mp4", fps=a.fps, width=a.width,
                        height=a.height, thickness=a.thickness, audio=a.audio)
    print(f"{j.shape} -> {pose.shape} -> {out}")


def cmd_harvest(a):
    src = Path(a.source)
    files = list(harvester.iter_sources(src))
    if not files:
        sys.exit(f"no videos or pose json found under {src}")

    extractor = None
    if any(f.suffix.lower() in harvester.VIDEO_EXT for f in files):
        try:
            extractor = harvester.make_rtmlib_extractor(mode=a.mode, device=a.device)
        except ImportError:
            print("! rtmlib not installed -- videos will be skipped, json still processed",
                  file=sys.stderr)

    cands, reports = [], []
    for f in files:
        try:
            pose, fps = harvester.poses_from_file(f, extractor, max_frames=a.max_frames)
        except Exception as exc:
            print(f"  skip {f.name}: {exc}", file=sys.stderr)
            continue
        if a.src_fps:
            fps = a.src_fps
        c, rep = harvester.harvest_sequence(
            pose, fps=fps, prominence=a.prominence, min_gap_s=a.min_gap,
            conf_thresh=a.conf, foreshorten=a.foreshorten, source=f.name)
        if rep["slowmo"]["likely_slowmo"]:
            print(f"  ! {f.name}: looks like slow motion "
                  f"(dup ratio {rep['slowmo']['duplicate_frame_ratio']})"
                  + ("  -- skipped" if a.skip_slowmo else "  -- kept, check it"))
            if a.skip_slowmo:
                reports.append(rep)
                continue
        print(f"  {f.name}: {rep['keyposes']} keyposes -> {rep['kept']} usable"
              + (f"  rejected {rep['rejected']}" if rep["rejected"] else ""))
        cands.extend(c)
        reports.append(rep)

    if not cands:
        sys.exit("no usable poses harvested")

    lib, meta = harvester.build_vocabulary(
        cands, max_poses=a.max_poses, min_distance=a.min_distance,
        drop_near_neutral=a.drop_neutral, mirror_invariant=not a.keep_mirrors,
        min_cluster_size=a.min_count)

    if not lib:
        sys.exit(
            f"{len(cands)} shapes harvested but the filters discarded all of them.\n"
            f"--min-count {a.min_count} needs a shape to recur that many times; "
            f"--min-distance {a.min_distance} may be merging everything into clusters "
            f"too small to survive it.\n"
            "Try --min-count 1, or a lower --min-distance, or more footage.")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    harvester.save_library(out / "vocabulary.npz", lib, meta)
    render.contact_sheet({m["name"]: lib[m["name"]] for m in meta},
                         out / "vocabulary.png", cell=a.cell, cols=a.cols)
    (out / "manifest.json").write_text(json.dumps(
        {"clips": reports, "vocabulary": meta}, indent=2, default=str))

    print(f"\n{len(cands)} candidate shapes -> {len(meta)} vocabulary entries")
    print(f"-> {out}/vocabulary.npz  {out}/vocabulary.png")
    print(f"Use it:  python -m dancekit compose song.mp3 -o out/ "
          f"--library {out}/vocabulary.npz")


def cmd_render(a):
    pose, meta = poseio.load_pose_json(a.pose)
    render.render_video(pose, a.out, fps=a.fps, width=a.width, height=a.height,
                        thickness=a.thickness, audio=a.audio)
    print(f"{pose.shape[0]} frames -> {a.out}")


def cmd_prep(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src = a.source
    if src.startswith("http"):
        subprocess.run(["yt-dlp", "-f", "bv*+ba/b", "-o", str(out / "%(id)s.%(ext)s"), src],
                       check=True)
        cands = sorted(out.glob("*.mp4")) + sorted(out.glob("*.webm"))
        src = str(cands[-1])

    stem = Path(src).stem
    dst = out / f"{stem}_norm.mp4"
    vf = (f"scale={a.width}:{a.height}:force_original_aspect_ratio=increase,"
          f"crop={a.width}:{a.height}")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
           "-vf", vf, "-r", str(a.fps),
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
    if a.strip_audio:
        cmd += ["-an"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += [str(dst)]
    subprocess.run(cmd, check=True)
    print(f"-> {dst}  ({a.width}x{a.height} @ {a.fps}fps)")
    print("Note: fps is RESAMPLED, not retagged. Check for slow motion with "
          "`dancekit extract` before training on this.")


# --- parser ------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="dancekit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("beats", help="analyse a song into a beat grid")
    p.add_argument("audio"); p.add_argument("-o", "--out")
    _add_audio_args(p); p.set_defaults(fn=cmd_beats)

    p = sub.add_parser("compose", help="generate choreography for a song")
    p.add_argument("audio"); p.add_argument("-o", "--out", default="out")
    p.add_argument("--subdivision", type=int, default=1,
                   help="pose changes per beat: 1=quarters, 2=eighths")
    p.add_argument("--phrase-beats", type=int, default=8)
    p.add_argument("--sections", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snap", type=float, default=0.65,
                   help="fraction of each interval spent moving; lower = sharper hits")
    p.add_argument("--overshoot", type=float, default=0.12)
    p.add_argument("--variation", type=float, default=0.5)
    p.add_argument("--bounce", type=float, default=0.012)
    p.add_argument("--seconds", type=float, default=None, help="cap output length")
    p.add_argument("--library", default=None,
                   help="harvested vocabulary .npz (default: built-in poses)")
    p.add_argument("--no-audio", action="store_true")
    _add_audio_args(p); _add_render_args(p); p.set_defaults(fn=cmd_compose)

    p = sub.add_parser("poses", help="contact sheet of the pose vocabulary")
    p.add_argument("-o", "--out", default="poses.png")
    p.add_argument("--cell", type=int, default=280); p.add_argument("--cols", type=int, default=5)
    p.add_argument("--mirrors", action="store_true"); p.set_defaults(fn=cmd_poses)

    p = sub.add_parser("extract", help="video -> OpenPose json (rtmlib/DWPose)")
    p.add_argument("video"); p.add_argument("-o", "--out", default="pose.json")
    p.add_argument("--mode", default="balanced", choices=["performance", "lightweight", "balanced"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0); p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("retime", help="warp an existing dance onto a song's grid")
    p.add_argument("pose"); p.add_argument("audio")
    p.add_argument("-o", "--out", default="out")
    p.add_argument("--src-fps", type=float, default=30.0)
    p.add_argument("--prominence", type=float, default=0.15)
    p.add_argument("--min-gap", type=float, default=0.12)
    p.add_argument("--max-keyposes", type=int, default=None)
    p.add_argument("--subdivision", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--mode", default="sequential", choices=["sequential", "nearest"])
    p.add_argument("--snap", type=float, default=0.7)
    p.add_argument("--overshoot", type=float, default=0.0)
    p.add_argument("--root-damping", type=float, default=0.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--no-audio", action="store_true")
    _add_audio_args(p); _add_render_args(p); p.set_defaults(fn=cmd_retime)

    p = sub.add_parser("smpl", help="3D SMPL motion -> OpenPose skeleton")
    p.add_argument("motion"); p.add_argument("-o", "--out", default="out")
    p.add_argument("--azimuth", type=float, default=0.0)
    p.add_argument("--elevation", type=float, default=0.0)
    p.add_argument("--up", default="y", choices=["y", "z"])
    p.add_argument("--headroom", type=float, default=0.10)
    p.add_argument("--audio", default=None)
    _add_render_args(p); p.set_defaults(fn=cmd_smpl)

    p = sub.add_parser("harvest", help="folder of clips -> pose vocabulary")
    p.add_argument("source", help="video/json file, or a folder of them")
    p.add_argument("-o", "--out", default="vocab")
    p.add_argument("--src-fps", type=float, default=None, help="override detected fps")
    p.add_argument("--prominence", type=float, default=0.12,
                   help="lower finds more shapes per clip")
    p.add_argument("--min-gap", type=float, default=0.10)
    p.add_argument("--conf", type=float, default=0.3, help="joint confidence floor")
    p.add_argument("--foreshorten", type=float, default=0.6,
                   help="0 = fully canonical body, 1 = keep observed proportions")
    p.add_argument("--max-poses", type=int, default=32)
    p.add_argument("--min-distance", type=float, default=0.30,
                   help="dedupe threshold, radians of mean bone angle")
    p.add_argument("--drop-neutral", type=float, default=0.18,
                   help="discard shapes this close to plain standing")
    p.add_argument("--min-count", type=int, default=1,
                   help="require a shape to recur this many times")
    p.add_argument("--keep-mirrors", action="store_true",
                   help="treat a pose and its mirror as distinct")
    p.add_argument("--skip-slowmo", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--mode", default="balanced",
                   choices=["performance", "lightweight", "balanced"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--cell", type=int, default=240)
    p.add_argument("--cols", type=int, default=6)
    p.set_defaults(fn=cmd_harvest)

    p = sub.add_parser("render", help="pose json -> skeleton video")
    p.add_argument("pose"); p.add_argument("-o", "--out", default="skeleton.mp4")
    p.add_argument("--audio", default=None)
    _add_render_args(p); p.set_defaults(fn=cmd_render)

    p = sub.add_parser("prep", help="download and normalise a source clip")
    p.add_argument("source"); p.add_argument("-o", "--out", default="clips")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=1472)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--strip-audio", action="store_true"); p.set_defaults(fn=cmd_prep)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    main()
