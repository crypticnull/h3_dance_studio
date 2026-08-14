"""Shared plumbing for the dancekit ComfyUI nodes.

- Locates and imports dancekit (pip-installed, vendored inside this pack, or a
  sibling checkout in custom_nodes/), with a useful error when it is missing.
- Defensive imports of torch / comfy.utils.ProgressBar / folder_paths so the node
  modules can be imported and tested outside a running ComfyUI.
- A server-side preview cache keyed by node id, which the /dancekit/* routes serve
  to the frontend widgets. Cached at execution time so scrubbing the preview never
  recomputes anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# dancekit import
# --------------------------------------------------------------------------------------

_PACK_DIR = Path(__file__).resolve().parent.parent

_DANCEKIT_ERR = (
    "dancekit is not importable. The dancekit toolkit must be installed for these "
    "nodes to work. Either:\n"
    "  1. pip install -e /path/to/dancekit        (in ComfyUI's python environment)\n"
    "  2. copy or symlink the `dancekit` package folder next to this node pack\n"
    "     (custom_nodes/comfyui-dancekit/dancekit/ or custom_nodes/dancekit/)\n"
    f"See {_PACK_DIR / 'install.md'} for details."
)


def _import_dancekit():
    try:
        import dancekit  # noqa: F401
        return dancekit
    except ImportError:
        pass

    # Vendored / sibling layouts. Each candidate is a directory whose CHILD named
    # `dancekit` is the package, so the candidate itself goes on sys.path.
    candidates = [
        _PACK_DIR,                       # custom_nodes/comfyui-dancekit/dancekit/
        _PACK_DIR.parent,                # custom_nodes/dancekit/  (bare package)
        _PACK_DIR.parent / "dancekit",   # custom_nodes/dancekit/dancekit/ (repo checkout)
    ]
    for cand in candidates:
        pkg = cand / "dancekit" / "__init__.py"
        if pkg.is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            try:
                import dancekit  # noqa: F401
                return dancekit
            except ImportError:
                continue
    raise ImportError(_DANCEKIT_ERR)


dancekit = _import_dancekit()

from dancekit import beatgrid, compose, harvest, keypose, poseio, poselib  # noqa: E402
from dancekit import render as dkrender  # noqa: E402
from dancekit import retime as dkretime  # noqa: E402
from dancekit import skeleton, smpl2d  # noqa: E402


# --------------------------------------------------------------------------------------
# torch (lazy) and comfy shims
# --------------------------------------------------------------------------------------

def get_torch():
    """Import torch on first use, with a clear error if it is absent."""
    try:
        import torch
        return torch
    except ImportError as e:  # pragma: no cover - torch is present in any ComfyUI
        raise ImportError(
            "torch is required for IMAGE outputs. It ships with every ComfyUI "
            "install; if you are running these nodes standalone, pip install torch."
        ) from e


try:
    from comfy.utils import ProgressBar
except Exception:  # not running inside ComfyUI
    class ProgressBar:  # minimal stand-in with the same surface
        def __init__(self, total):
            self.total = total
            self.current = 0

        def update(self, n=1):
            self.current += n

        def update_absolute(self, value, total=None, preview=None):
            self.current = value


def get_output_dir() -> Path:
    """ComfyUI's output directory if available, else cwd."""
    try:
        import folder_paths
        return Path(folder_paths.get_output_directory())
    except Exception:
        return Path.cwd()


def resolve_out_path(path: str, default_name: str) -> Path:
    """Relative paths land in ComfyUI's output directory."""
    p = Path(path) if path else Path(default_name)
    if not p.is_absolute():
        p = get_output_dir() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------
# Preview cache (served by the /dancekit/* routes)
# --------------------------------------------------------------------------------------

# node_id (str) -> {"pose": (T,18,3) ndarray, "fps": float,
#                   "beats": ndarray|None, "downbeats": ndarray|None}
POSE_CACHE: dict[str, dict] = {}

# node_id (str) -> {"lib": {name: (18,3)}, "meta": [dict]}
LIBRARY_CACHE: dict[str, dict] = {}


def cache_pose(unique_id, pose: np.ndarray, fps: float, grid=None) -> None:
    if unique_id is None:
        return
    entry = {"pose": np.asarray(pose, dtype=np.float32), "fps": float(fps),
             "beats": None, "downbeats": None}
    if grid is not None:
        entry["beats"] = np.asarray(grid.beats, dtype=float)
        entry["downbeats"] = np.asarray(grid.downbeats, dtype=float)
    POSE_CACHE[str(unique_id)] = entry


def cache_library(unique_id, lib: dict, meta: list) -> None:
    if unique_id is None:
        return
    LIBRARY_CACHE[str(unique_id)] = {"lib": dict(lib), "meta": list(meta)}


# --------------------------------------------------------------------------------------
# DK_* container helpers
# --------------------------------------------------------------------------------------
# DK_POSE   = {"pose": (T,18,3) float ndarray, "fps": float, "meta": dict, "grid": BeatGrid|None}
# DK_LIBRARY= {"lib": {name: (18,3)} WITHOUT mirrors, "meta": [dict], "energy_map": {name: float}|None}

def make_pose(pose: np.ndarray, fps: float, meta: dict | None = None, grid=None) -> dict:
    pose = np.asarray(pose, dtype=float)
    if pose.ndim != 3 or pose.shape[1:] != (18, 3):
        raise ValueError(f"DK_POSE expects a (T,18,3) array, got {pose.shape}")
    return {"pose": pose, "fps": float(fps), "meta": meta or {}, "grid": grid}


def make_library(lib: dict, meta: list | None = None, energy_map: dict | None = None) -> dict:
    return {"lib": dict(lib), "meta": list(meta or []), "energy_map": energy_map}


def strip_mirrors(lib: dict, emap: dict | None):
    """Keep only base poses (drop *_m); compose regenerates mirrors itself."""
    base = {n: p for n, p in lib.items() if not n.endswith("_m")}
    e = None
    if emap is not None:
        e = {n: v for n, v in emap.items() if n in base}
    return base, e


def library_for_compose(dklib: dict | None):
    """Expand a DK_LIBRARY into (lib_with_mirrors, energy_map) for compose()."""
    if dklib is None:
        return None, None
    lib, emap = {}, {}
    meta_e = {m.get("name"): m.get("energy") for m in dklib.get("meta") or []}
    src_e = dklib.get("energy_map") or {}
    for n, p in dklib["lib"].items():
        e = src_e.get(n, meta_e.get(n))
        if e is None:
            e = harvest.pose_energy(np.asarray(p))
        lib[n] = np.asarray(p, dtype=float)
        lib[n + "_m"] = poselib.mirror(np.asarray(p, dtype=float))
        emap[n] = float(e)
        emap[n + "_m"] = float(e)
    # Rank-normalise if energies did not come pre-normalised 0..1
    vals = [v for v in emap.values()]
    if vals and (max(vals) > 1.0 + 1e-6 or min(vals) < -1e-6):
        names = list(emap.keys())
        ranked = harvest.rank_normalise([emap[n] for n in names])
        emap = dict(zip(names, ranked))
    return lib, emap


def apply_dropped(dklib: dict, dropped_json: str) -> dict:
    """Remove poses the user toggled off in the library browser widget."""
    dropped = parse_dropped(dropped_json)
    if not dropped:
        return dklib
    lib = {n: p for n, p in dklib["lib"].items() if n not in dropped}
    meta = [m for m in dklib.get("meta") or [] if m.get("name") not in dropped]
    emap = dklib.get("energy_map")
    if emap is not None:
        emap = {n: v for n, v in emap.items()
                if n not in dropped and (n[:-2] if n.endswith("_m") else n) not in dropped}
    return make_library(lib, meta, emap)


def parse_dropped(dropped_json: str) -> set:
    if not dropped_json or not dropped_json.strip():
        return set()
    try:
        val = json.loads(dropped_json)
        if isinstance(val, list):
            return {str(x) for x in val}
    except (ValueError, TypeError):
        pass
    # Fall back to comma-separated names
    return {s.strip() for s in dropped_json.split(",") if s.strip()}


def pose_to_image_batch(pose: np.ndarray, width: int, height: int, thickness: float,
                        progress: bool = True):
    """Render every frame and stack into a ComfyUI IMAGE tensor (B,H,W,C) float32 0..1."""
    torch = get_torch()
    T = pose.shape[0]
    pbar = ProgressBar(T) if progress else None
    out = np.empty((T, height, width, 3), dtype=np.float32)
    for t in range(T):
        img = dkrender.draw_pose(pose[t], width, height, thickness)
        out[t] = img.astype(np.float32) / 255.0
        if pbar is not None:
            pbar.update(1)
    return torch.from_numpy(out)


def grid_report(grid) -> str:
    first_db = float(grid.downbeats[0]) if len(grid.downbeats) else float("nan")
    lines = [
        f"tempo: {grid.tempo:.2f} BPM  (beat every {60.0 / max(grid.tempo, 1e-9):.4f}s)",
        f"beats: {len(grid.beats)}   downbeats: {len(grid.downbeats)} "
        f"({grid.beats_per_bar}/bar)",
        f"first downbeat: {first_db:.3f}s",
        f"duration: {grid.duration:.2f}s",
    ]
    return "\n".join(lines)
