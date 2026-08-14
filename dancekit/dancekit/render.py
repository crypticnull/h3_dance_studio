"""Render pose sequences as OpenPose-style skeleton frames.

The colours and the filled-ellipse limb style are not cosmetic -- ControlNet OpenPose
models were trained on exactly this rendering, so matching it is what makes the
conditioning actually bite.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .skeleton import COLORS, LIMB_PAIRS, NUM_JOINTS


def draw_pose(pose: np.ndarray, width: int, height: int, thickness: float = 1.0,
              conf_thresh: float = 0.05, background=(0, 0, 0)) -> np.ndarray:
    """Draw one (18,3) normalised pose. Returns an RGB uint8 image."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = background[::-1]  # cv2 works in BGR

    stick = max(1, int(round(4 * thickness * min(width, height) / 512)))
    dot = max(1, int(round(4 * thickness * min(width, height) / 512)))

    pts = pose[:, :2].copy()
    pts[:, 0] *= width
    pts[:, 1] *= height
    ok = pose[:, 2] > conf_thresh

    for i, (a, b) in enumerate(LIMB_PAIRS):
        if not (ok[a] and ok[b]):
            continue
        pa, pb = pts[a], pts[b]
        mid = (pa + pb) / 2.0
        length = float(np.linalg.norm(pa - pb))
        if length < 1e-3:
            continue
        angle = float(np.degrees(np.arctan2(pa[1] - pb[1], pa[0] - pb[0])))
        poly = cv2.ellipse2Poly((int(mid[0]), int(mid[1])),
                                (int(length / 2), stick), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, COLORS[i % len(COLORS)][::-1])

    for j in range(NUM_JOINTS):
        if not ok[j]:
            continue
        cv2.circle(canvas, (int(pts[j, 0]), int(pts[j, 1])), dot,
                   COLORS[j % len(COLORS)][::-1], thickness=-1)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def render_frames(pose_seq: np.ndarray, out_dir: str | Path, width: int = 832,
                  height: int = 1472, thickness: float = 1.0) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in range(pose_seq.shape[0]):
        img = draw_pose(pose_seq[t], width, height, thickness)
        cv2.imwrite(str(out_dir / f"{t:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out_dir


def render_video(pose_seq: np.ndarray, out_path: str | Path, fps: float = 24.0,
                 width: int = 832, height: int = 1472, thickness: float = 1.0,
                 audio: str | None = None, crf: int = 16) -> Path:
    """Render straight to H.264 via an ffmpeg pipe. Pass `audio` to mux the song in,
    which makes eyeballing the sync trivial."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-"]
    if audio:
        cmd += ["-i", str(audio), "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(out_path)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for t in range(pose_seq.shape[0]):
            proc.stdin.write(draw_pose(pose_seq[t], width, height, thickness).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    return out_path


def contact_sheet(poses: dict[str, np.ndarray], out_path: str | Path, cell: int = 220,
                  cols: int = 6) -> Path:
    """Grid of every pose in a library, labelled. Useful for auditing a vocabulary
    before composing with it."""
    names = list(poses.keys())
    rows = int(np.ceil(len(names) / cols))
    sheet = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for i, n in enumerate(names):
        r, c = divmod(i, cols)
        img = draw_pose(poses[n], cell, cell, thickness=0.8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, n[:18], (4, cell - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = img
    cv2.imwrite(str(out_path), sheet)
    return Path(out_path)
