/**
 * dancekit frontend extension.
 *
 * - Skeleton preview widget (canvas + frame scrubber + play/pause + beat ticks)
 *   on Compose / Retime / Render / SMPL / Load Pose JSON / Trim nodes.
 * - Library browser widget (clickable pose grid with energies, keep/drop toggling
 *   persisted into the node's `dropped` widget) on Harvest / Load Library nodes.
 * - Tooltips on adjustable widgets (in addition to the native `tooltip` entries in
 *   the Python input specs, for older frontends that don't render those).
 *
 * Pose data comes from the backend routes /dancekit/pose and /dancekit/library,
 * served from a cache filled at execution time -- scrubbing never recomputes.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Canonical OpenPose BODY_18 rendering tables.
// MUST match dancekit/skeleton.py exactly (selftest.py asserts this).
const LIMB_PAIRS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
    [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
    [0, 15], [15, 17],
];

const COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85],
];

const PREVIEW_NODES = new Set([
    "DKCompose", "DKRetime", "DKRenderSkeleton", "DKSMPLToOpenPose",
    "DKLoadPoseJSON", "DKTrim",
]);
const LIBRARY_NODES = new Set(["DKHarvest", "DKLoadLibrary"]);

// ---------------------------------------------------------------------------------
// Tooltips (mirrors the Python `tooltip` entries so older frontends get them too).
// ---------------------------------------------------------------------------------

const SHARED_TIPS = {
    snap: "Fraction of each interval spent moving. Lower = the shape arrives early and holds until the next beat, which reads as a sharp 'hit'. 0.45-0.6 for popping, 0.8-1.0 for sustained/flowy.",
    overshoot: "Slight past-the-pose travel that settles back (popping/hip-hop snap). 0.15-0.25 reads as snap; above 0.4 limbs look broken.",
    prominence: "Keypose threshold relative to the clip's own speed spread, so it transfers between energetic and languid clips. Lower finds more held shapes.",
    min_gap: "Minimum seconds between detected keyposes; stops one long hold counting as several hits.",
    subdivision: "Pose changes per beat. 1 = quarters (calmer), 2 = eighths (closer to real TikTok choreo density).",
};

const TOOLTIPS = {
    DKBeatGrid: {
        bpm: "Force the tempo (0 = auto). The tracker's classic failure is half/double time -- if the report shows 60 when you expect 120, set this.",
        tightness: "How strongly the tracker sticks to a steady tempo. Raise for electronic, lower for rubato. Mostly irrelevant with rigid on.",
        offset: "Seconds added to every beat; nudges a grid that is consistently early or late.",
        beats_per_bar: "Time signature numerator; downbeats are every Nth beat.",
        downbeat_index: "Force which beat is beat 1 (-1 = auto from low-band kick energy -- the snare is louder but marks 2 and 4, so it is deliberately ignored).",
        rigid: "Refit one constant-tempo line through the tracked beats. Correct for anything produced to a click; removes per-beat jitter entirely. Off for live/rubato material.",
        auto_align: "Onset envelopes peak slightly AFTER the real transient, so fitted grids sit late. This slides the grid to the lag with the most onset energy on the beats.",
        trim_silence: "Strip leading/trailing silence first. Only if you also trim the audio you feed the video model.",
        audio_path: "Path to the song. Ignored when the AUDIO input is connected.",
    },
    DKCompose: {
        fps: "Output frame rate; match your video model.",
        subdivision: SHARED_TIPS.subdivision,
        phrase_beats: "Beats per motif; 8 matches the 8-count dancers think in.",
        sections: "Distinct musical sections to write material for. A returning chorus gets the SAME motif back -- motif repetition, not timing accuracy, is what reads as choreography.",
        seed: "Different seed = different dance for the same song.",
        snap: SHARED_TIPS.snap,
        overshoot: SHARED_TIPS.overshoot,
        variation: "Probability of side-flipping / tail-varying a repeated motif.",
        bounce: "Per-beat vertical pulse. Real dancers never fully stop; a dead hold between hits is the biggest tell of synthetic motion.",
        max_seconds: "Compose only the first N seconds (0 = full track).",
    },
    DKRetime: {
        src_fps: "Source clip frame rate (0 = use the DK_POSE's fps). Wrong values scale every keypose time off-beat.",
        prominence: SHARED_TIPS.prominence,
        min_gap: SHARED_TIPS.min_gap,
        max_keyposes: "Cap keyposes, keeping the deepest holds (0 = no cap).",
        subdivision: "Grid density to pin keyposes to; 2 = eighths, where most TikTok choreo hits.",
        stride: "Map keyposes to every Nth grid point; 2 when the source is half-time.",
        start_index: "First grid point used; shift to start on a downbeat.",
        mode: "sequential preserves the phrase exactly (source is on-beat); nearest snaps each keypose to the closest grid point (loose/freestyle sources).",
        snap: SHARED_TIPS.snap,
        overshoot: SHARED_TIPS.overshoot,
        root_damping: "Pulls global travel toward the clip's mean position so the dancer stays in frame.",
        loop: "Repeat the phrase to cover the song. Feed a clean 8-count or the seam jumps.",
        out_fps: "Output frame rate; match your video model.",
    },
    DKHarvest: {
        source_path: "Folder (recursive) or file. Pose JSON and video side by side; video needs rtmlib.",
        prominence: SHARED_TIPS.prominence,
        min_gap: SHARED_TIPS.min_gap,
        conf: "Confidence core joints need. A pose missing an ankle is a cropped frame, not a shape.",
        foreshorten: "Blend of observed limb proportions kept after rebuilding on the standard body -- the only depth cue a 2D skeleton has. 0 = flattest/most stable, 1 = most depth/least stable.",
        max_poses: "Vocabulary size cap; recurring shapes outrank one-off outliers.",
        min_distance: "Dedupe threshold in radians of mean weighted bone angle (angles, not xy: different builds hitting the same shape differ in xy, barely in angle). Lower = richer, noisier vocabulary.",
        drop_neutral: "Discard shapes this close to plain standing.",
        min_count: "Require a shape to recur before it earns a slot -- the best noise filter; detector glitches don't repeat.",
        mirror_invariant: "Treat a pose and its mirror as one shape. Compose generates mirrors itself, so keeping both wastes slots.",
        skip_slowmo: "Drop clips flagged as conformed slow motion -- it teaches weightless, dreamy movement.",
        src_fps: "Override fps for pose JSONs that don't record one (0 = auto/30).",
        dropped: "Pose names excluded on the next run. Click poses in the browser below instead of editing this by hand.",
    },
    DKLoadLibrary: {
        path: "vocabulary.npz written by Harvest / Save Library / the dancekit CLI.",
        dropped: "Pose names excluded from the loaded library. Click poses in the browser below.",
    },
    DKSaveLibrary: { path: "Relative paths land in ComfyUI's output directory." },
    DKLoadPoseJSON: {
        path: "ComfyUI POSE_KEYPOINT JSON (DWPreprocessor -> SavePoseKpsAsJsonFile). Pixel or normalised coords both load.",
        fps: "Frame rate of the source clip -- the JSON doesn't record it and Retime needs it.",
        person: "-1 = most confidently detected body (almost always the dancer, not a background figure).",
    },
    DKSavePoseJSON: {
        path: "Relative paths land in ComfyUI's output directory.",
        canvas_width: "Canvas size recorded in the JSON (used for aspect by re-renderers).",
        canvas_height: "Canvas height recorded in the JSON.",
        normalised: "Write 0..1 coords (default) instead of pixels; loaders sniff either.",
    },
    DKRenderSkeleton: {
        width: "Match your generation resolution (multiples of 32).",
        height: "832x1472 suits 9:16 dance content.",
        thickness: "Limb thickness multiplier. Colours/limb style stay canonical -- ControlNet OpenPose models were trained on exactly this rendering.",
    },
    DKSMPLToOpenPose: {
        motion_path: ".npy/.npz/.pkl of (T,24+,3) SMPL joints from EDGE / AtomicDance / OpenDance.",
        fps: "Frame rate the motion was generated at (EDGE = 30).",
        azimuth: "Camera yaw. Orthographic on purpose: canonical-space motion has no camera, and an invented focal length adds distortion the video model must fight.",
        elevation: "Camera pitch; keep small, pose conditioning dislikes top-down views.",
        up_axis: "SMPL is y-up; if the skeleton lies on its side, switch to z.",
        headroom: "Top margin. Scale is fixed over the whole clip so raising an arm doesn't zoom the camera.",
        floor: "Bottom margin fraction.",
    },
    DKTrim: {
        target_frames: "Desired frame count (0 = keep length, useful with the H3 grid snap).",
        start_frame: "Drop N frames from the front, e.g. to start on a downbeat.",
        pad_mode: "hold freezes the last frame; loop repeats from the start (seam jumps unless the clip cycles cleanly).",
        h3_frame_grid: "Snap to the nearest valid MiniMax H3 count (frames % 17 == 5): 22, 39, 56, 73, 90, 107, 124.",
    },
    DKPoseInfo: {
        prominence: SHARED_TIPS.prominence,
        min_gap: SHARED_TIPS.min_gap,
    },
};

// ---------------------------------------------------------------------------------
// Skeleton drawing
// ---------------------------------------------------------------------------------

function rgb(c) {
    return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
}

/** Draw one (18,3) frame (normalised coords) onto a 2D context. */
function drawSkeleton(ctx, frame, w, h, confThresh = 0.05) {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);
    if (!frame) return;
    const stick = Math.max(1, Math.round(4 * Math.min(w, h) / 512));
    ctx.lineCap = "round";
    for (let i = 0; i < LIMB_PAIRS.length; i++) {
        const [a, b] = LIMB_PAIRS[i];
        if (frame[a][2] <= confThresh || frame[b][2] <= confThresh) continue;
        ctx.strokeStyle = rgb(COLORS[i % COLORS.length]);
        ctx.lineWidth = stick * 2;
        ctx.beginPath();
        ctx.moveTo(frame[a][0] * w, frame[a][1] * h);
        ctx.lineTo(frame[b][0] * w, frame[b][1] * h);
        ctx.stroke();
    }
    for (let j = 0; j < 18; j++) {
        if (frame[j][2] <= confThresh) continue;
        ctx.fillStyle = rgb(COLORS[j % COLORS.length]);
        ctx.beginPath();
        ctx.arc(frame[j][0] * w, frame[j][1] * h, stick, 0, Math.PI * 2);
        ctx.fill();
    }
}

// ---------------------------------------------------------------------------------
// Pose preview widget
// ---------------------------------------------------------------------------------

const CSS = `
.dk-preview { background:#111; border-radius:6px; padding:6px; font:11px sans-serif; color:#bbb; }
.dk-preview canvas.dk-view { width:100%; display:block; background:#000; border-radius:4px; }
.dk-row { display:flex; align-items:center; gap:6px; margin-top:4px; }
.dk-row input[type=range] { flex:1; min-width:0; }
.dk-btn { background:#333; color:#ddd; border:1px solid #555; border-radius:4px;
          cursor:pointer; padding:1px 8px; font-size:11px; }
.dk-btn:hover { background:#444; }
.dk-ticks { width:100%; height:14px; display:block; }
.dk-lib { display:grid; grid-template-columns:repeat(auto-fill, minmax(84px,1fr));
          gap:4px; max-height:280px; overflow-y:auto; margin-top:4px; }
.dk-cell { position:relative; border:1px solid #333; border-radius:4px; cursor:pointer;
           background:#000; text-align:center; }
.dk-cell canvas { width:100%; display:block; }
.dk-cell .dk-label { font-size:10px; color:#9a9; padding:1px 2px; }
.dk-cell.dk-dropped { opacity:0.28; border-color:#a33; }
.dk-cell.dk-dropped::after { content:"dropped"; position:absolute; top:2px; left:2px;
                              color:#f66; font-size:9px; }
.dk-msg { color:#777; font-size:11px; padding:4px; }
`;

let cssInjected = false;
function injectCSS() {
    if (cssInjected) return;
    cssInjected = true;
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);
}

function addPosePreview(node) {
    injectCSS();
    const container = document.createElement("div");
    container.className = "dk-preview";
    container.title = "Skeleton preview. Runs from cached results -- execute the graph once to populate.";

    const view = document.createElement("canvas");
    view.className = "dk-view";
    view.width = 216;
    view.height = 384;

    const ticks = document.createElement("canvas");
    ticks.className = "dk-ticks";
    ticks.width = 216;
    ticks.height = 14;
    ticks.title = "Beat markers: short ticks are beats, tall ticks are downbeats. The white line is the current frame -- check your hits land on ticks.";

    const row = document.createElement("div");
    row.className = "dk-row";
    const playBtn = document.createElement("button");
    playBtn.className = "dk-btn";
    playBtn.textContent = "▶";
    playBtn.title = "Play / pause at the pose's own fps";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = 0;
    slider.max = 0;
    slider.value = 0;
    slider.title = "Scrub frames";
    const label = document.createElement("span");
    label.textContent = "no data";

    row.appendChild(playBtn);
    row.appendChild(slider);
    row.appendChild(label);
    container.appendChild(view);
    container.appendChild(ticks);
    container.appendChild(row);

    const state = { data: null, frame: 0, playing: false, lastT: 0, acc: 0, raf: null };

    function drawTicks() {
        const ctx = ticks.getContext("2d");
        ctx.clearRect(0, 0, ticks.width, ticks.height);
        const d = state.data;
        if (!d || !d.count) return;
        const dur = d.count / d.fps;
        if (d.beats) {
            const downs = new Set((d.downbeats || []).map((t) => t.toFixed(3)));
            ctx.fillStyle = "#5af";
            for (const t of d.beats) {
                if (t < 0 || t > dur) continue;
                const x = (t / dur) * ticks.width;
                const tall = downs.has(t.toFixed(3));
                ctx.fillStyle = tall ? "#fc3" : "#5af";
                ctx.fillRect(x, tall ? 0 : 5, 1.5, tall ? 14 : 9);
            }
        }
        // playhead
        ctx.fillStyle = "#fff";
        ctx.fillRect((state.frame / Math.max(d.count - 1, 1)) * ticks.width - 0.5, 0, 1, 14);
    }

    function drawFrame() {
        const ctx = view.getContext("2d");
        const d = state.data;
        drawSkeleton(ctx, d ? d.frames[state.frame] : null, view.width, view.height);
        if (d) {
            const t = state.frame / d.fps;
            label.textContent = state.frame + "/" + (d.count - 1) + "  " + t.toFixed(2) + "s";
        } else {
            label.textContent = "no data";
        }
        drawTicks();
    }

    function setFrame(f) {
        const d = state.data;
        if (!d) return;
        state.frame = Math.max(0, Math.min(d.count - 1, f | 0));
        slider.value = state.frame;
        drawFrame();
    }

    function tick(ts) {
        if (!state.playing) return;
        const d = state.data;
        if (!d) { state.playing = false; return; }
        if (state.lastT) {
            state.acc += (ts - state.lastT) / 1000;
            const step = 1 / d.fps;
            while (state.acc >= step) {
                state.acc -= step;
                state.frame = (state.frame + 1) % d.count;
            }
            slider.value = state.frame;
            drawFrame();
        }
        state.lastT = ts;
        state.raf = requestAnimationFrame(tick);
    }

    playBtn.addEventListener("click", () => {
        state.playing = !state.playing;
        playBtn.textContent = state.playing ? "⏸" : "▶";
        state.lastT = 0;
        state.acc = 0;
        if (state.playing) state.raf = requestAnimationFrame(tick);
    });
    slider.addEventListener("input", () => {
        state.playing = false;
        playBtn.textContent = "▶";
        setFrame(parseInt(slider.value, 10));
    });

    node.dancekitRefresh = async function () {
        try {
            const res = await api.fetchApi("/dancekit/pose?node_id=" + node.id);
            if (!res.ok) { state.data = null; drawFrame(); return; }
            const data = await res.json();
            state.data = data;
            // keep the preview canvas at roughly the pose's aspect (portrait default)
            slider.max = Math.max(0, data.count - 1);
            setFrame(Math.min(state.frame, data.count - 1));
        } catch (e) {
            state.data = null;
            drawFrame();
        }
    };

    if (node.addDOMWidget) {
        node.addDOMWidget("dk_preview", "dk_preview", container, {
            serialize: false,
            getMinHeight: () => 300,
        });
    }
    drawFrame();
    // Populate from the cache if the node already ran this session.
    node.dancekitRefresh();
}

// ---------------------------------------------------------------------------------
// Library browser widget
// ---------------------------------------------------------------------------------

function getDroppedSet(node) {
    const w = (node.widgets || []).find((x) => x.name === "dropped");
    if (!w || !w.value) return new Set();
    try {
        const v = JSON.parse(w.value);
        if (Array.isArray(v)) return new Set(v.map(String));
    } catch (e) { /* fall through */ }
    return new Set(String(w.value).split(",").map((s) => s.trim()).filter(Boolean));
}

function setDroppedSet(node, set) {
    const w = (node.widgets || []).find((x) => x.name === "dropped");
    if (!w) return;
    w.value = JSON.stringify([...set].sort());
    node.graph?.setDirtyCanvas(true, false);
}

function addLibraryBrowser(node) {
    injectCSS();
    const container = document.createElement("div");
    container.className = "dk-preview";
    const msg = document.createElement("div");
    msg.className = "dk-msg";
    msg.textContent = "Library browser: run the graph (or set a valid path) to load poses.";
    const grid = document.createElement("div");
    grid.className = "dk-lib";
    grid.title = "Click a pose to toggle keep/drop. Dropped poses are excluded on the NEXT run (stored in the `dropped` widget, saved with the graph).";
    container.appendChild(msg);
    container.appendChild(grid);

    function rebuild(data) {
        grid.innerHTML = "";
        const dropped = getDroppedSet(node);
        const names = data.names || [];
        msg.textContent = names.length
            ? names.length + " poses -- click to toggle keep/drop (applies on next run)"
            : "Library is empty.";
        for (const name of names) {
            const cell = document.createElement("div");
            cell.className = "dk-cell" + (dropped.has(name) ? " dk-dropped" : "");
            const cv = document.createElement("canvas");
            cv.width = 96;
            cv.height = 96;
            drawSkeleton(cv.getContext("2d"), data.poses[name], 96, 96);
            const lab = document.createElement("div");
            lab.className = "dk-label";
            const e = data.energies ? data.energies[name] : null;
            lab.textContent = name + (e != null ? "  e=" + Number(e).toFixed(2) : "");
            cell.title = "energy " + (e != null ? Number(e).toFixed(3) : "?") +
                " -- 0 = contained, 1 = full extension. Compose follows the track's dynamics with this.";
            cell.appendChild(cv);
            cell.appendChild(lab);
            cell.addEventListener("click", () => {
                const cur = getDroppedSet(node);
                if (cur.has(name)) cur.delete(name); else cur.add(name);
                setDroppedSet(node, cur);
                cell.classList.toggle("dk-dropped", cur.has(name));
            });
            grid.appendChild(cell);
        }
    }

    node.dancekitRefresh = async function () {
        try {
            let url = "/dancekit/library?node_id=" + node.id;
            const pathW = (node.widgets || []).find((x) => x.name === "path");
            if (pathW && pathW.value) url += "&path=" + encodeURIComponent(pathW.value);
            const res = await api.fetchApi(url);
            if (!res.ok) return;
            rebuild(await res.json());
        } catch (e) { /* server not ready */ }
    };

    if (node.addDOMWidget) {
        node.addDOMWidget("dk_library", "dk_library", container, {
            serialize: false,
            getMinHeight: () => 220,
        });
    }
    node.dancekitRefresh();
}

// ---------------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------------

app.registerExtension({
    name: "dancekit.widgets",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        const name = nodeData.name;
        const isPreview = PREVIEW_NODES.has(name);
        const isLibrary = LIBRARY_NODES.has(name);
        const tips = TOOLTIPS[name];
        if (!isPreview && !isLibrary && !tips) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (tips) {
                for (const w of this.widgets || []) {
                    if (tips[w.name]) {
                        w.tooltip = tips[w.name];
                        if (w.options) w.options.tooltip = tips[w.name];
                    }
                }
            }
            if (isPreview) addPosePreview(this);
            if (isLibrary) addLibraryBrowser(this);
            return r;
        };

        if (isPreview || isLibrary) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);
                if (this.dancekitRefresh) this.dancekitRefresh();
            };
        }
    },
});
