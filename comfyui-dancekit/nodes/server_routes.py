"""HTTP routes serving pose frames, beat times and library thumbnails to the
frontend widgets.

Everything is served from the in-memory caches in _dk.py, filled at node execution
time and keyed by node id -- so scrubbing the preview slider never recomputes
anything. Registration is a no-op outside a running ComfyUI (no `server` module).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _dk


def _pose_payload(entry: dict) -> dict:
    pose = entry["pose"]
    # Round to 4 decimals: sub-pixel at any sane canvas size, third of the bytes.
    frames = np.round(pose, 4).tolist()
    out = {"fps": entry["fps"], "frames": frames, "count": int(pose.shape[0])}
    if entry.get("beats") is not None:
        out["beats"] = np.round(entry["beats"], 4).tolist()
        out["downbeats"] = np.round(entry["downbeats"], 4).tolist()
    return out


def _library_payload(lib: dict, meta: list) -> dict:
    e_by_name = {m.get("name"): m.get("energy") for m in meta}
    poses = {}
    for n, p in lib.items():
        poses[n] = np.round(np.asarray(p, dtype=float), 4).tolist()
    return {
        "names": list(lib.keys()),
        "poses": poses,
        "energies": {n: e_by_name.get(n) for n in lib.keys()},
        "meta": meta,
    }


def register_routes() -> bool:
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception:
        return False

    routes = PromptServer.instance.routes

    @routes.get("/dancekit/pose")
    async def dk_pose(request):
        node_id = request.rel_url.query.get("node_id", "")
        entry = _dk.POSE_CACHE.get(str(node_id))
        if entry is None:
            return web.json_response(
                {"error": "no cached pose for this node; run the graph first"},
                status=404)
        return web.json_response(_pose_payload(entry))

    @routes.get("/dancekit/library")
    async def dk_library(request):
        node_id = request.rel_url.query.get("node_id", "")
        entry = _dk.LIBRARY_CACHE.get(str(node_id))
        if entry is None:
            # Fall back to reading the .npz directly (Load Library nodes can browse
            # before the graph has ever run).
            path = request.rel_url.query.get("path", "")
            if path and Path(path).is_file():
                try:
                    lib, meta, emap = _dk.harvest.load_library(path)
                    base, _ = _dk.strip_mirrors(lib, emap)
                    return web.json_response(_library_payload(base, meta))
                except Exception as e:
                    return web.json_response({"error": str(e)}, status=500)
            return web.json_response(
                {"error": "no cached library for this node; run the graph first"},
                status=404)
        return web.json_response(_library_payload(entry["lib"], entry["meta"]))

    @routes.post("/dancekit/clear")
    async def dk_clear(request):
        _dk.POSE_CACHE.clear()
        _dk.LIBRARY_CACHE.clear()
        return web.json_response({"ok": True})

    return True


ROUTES_REGISTERED = register_routes()
