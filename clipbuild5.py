#!/usr/bin/env python3
"""Short-form clip builder v5.
Same STILL-camera deadband tracking Daniel settled on in clipbuild4 (lock on
the speaker, ignore small movement, one slow smoothstep glide when he truly
relocates) — but detection is now YOLOv8 person detection instead of the old
OpenCV HOG/Haar stack, which missed constantly under stage lighting and made
the lock point drift. Also probes source dimensions instead of hardcoding.
Usage: clipbuild5.py specs.json
spec: {src, start, end, hook, platform, date, cues:[[s,e,text],...]}
"""
import json, os, shutil, statistics, subprocess, sys, tempfile

import cv2
from ultralytics import YOLO

FF = os.path.expanduser("~/.local/bin/ffmpeg")

PLAT = {
    "Shorts":   dict(crop_ar=1080/1920, ow=1080, oh=1920, fs=150, margin=520),
    "Facebook": dict(crop_ar=1080/1350, ow=1080, oh=1350, fs=54, margin=90),
}

FPS = 3                 # detection sample rate
DEAD_FRAC = 0.04        # deadband: ~4% of source width
SUSTAIN_S = 1.5         # seconds outside deadband before believing a move
GLIDE = 1.3             # seconds for the re-center move
DET_W = 1280            # detection frame width
CROP_H_FRAC = 0.78      # crop height as fraction of source height
CROP_Y_ANCHOR = 0.10    # crop sits near the top so the bottom edge clears audience heads
MAX_JUMP_FRAC = 0.20    # max believable subject move per sample (fraction of width)
STAGE_TOP_FRAC = 0.45   # speaker's box must start above this fraction of frame height

_model = None
def model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model

def probe_dims(src):
    out = subprocess.run([FF, "-hide_banner", "-i", src], capture_output=True, text=True).stderr
    import re
    m = re.search(r"Video:.* (\d{3,5})x(\d{3,5})", out)
    return int(m.group(1)), int(m.group(2))

def detect_centers(src, start, dur, tmp, src_w):
    """(t, person_cx) in source coords. Ultralytics ByteTrack: persistent IDs
    across frames, so we follow ONE track (the on-stage speaker) instead of
    re-guessing per frame. Audience heads/backs along the bottom edge are
    rejected; the speaker's box starts near the top of the frame."""
    fdir = os.path.join(tmp, "frames"); os.makedirs(fdir, exist_ok=True)
    subprocess.run([FF, "-y", "-v", "error", "-ss", str(start), "-t", str(dur),
                    "-i", src, "-vf", f"fps={FPS},scale={DET_W}:-2",
                    os.path.join(fdir, "f%05d.jpg")], check=True)
    scale = src_w / DET_W
    xs, last, lock_id = [], None, None
    names = sorted(os.listdir(fdir))
    m = model()
    for n in names:
        r = m.track(os.path.join(fdir, n), classes=[0], conf=0.35, persist=True,
                    tracker="bytetrack.yaml", verbose=False)[0]
        fh = r.orig_shape[0]
        stage = {}
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            if y1 < fh * STAGE_TOP_FRAC:   # audience boxes start low; a full-body speaker may touch the bottom edge
                tid = int(b.id[0]) if b.id is not None else -1
                stage[tid] = ((x1 + x2) / 2 * scale, (x2 - x1) * (y2 - y1))
        cx = None
        if lock_id in stage:
            cx = stage[lock_id][0]
        elif stage:
            if last is None:
                lock_id = max(stage, key=lambda k: stage[k][1])
            else:
                lock_id = min(stage, key=lambda k: abs(stage[k][0] - last))
            cx = stage[lock_id][0]
        # a person can't cross a fifth of the stage in one sample: a jump that
        # big is a false lock (ID swap, a screen, someone at the edge) — hold
        if cx is not None and last is not None and abs(cx - last) > src_w * MAX_JUMP_FRAC:
            cx = None
            lock_id = None
        if cx is None:
            cx = last                     # None until the first real detection
        last = cx
        xs.append(cx)
    shutil.rmtree(fdir)
    # never guess frame-center: back-fill leading misses from the first real
    # detection (a clip that opens with the speaker undetected used to start
    # off-screen), and default to center only if he was never seen at all
    first = next((x for x in xs if x is not None), src_w / 2)
    xs = [first if x is None else x for x in xs]
    med = [statistics.median(xs[max(0, j - 2):j + 3]) for j in range(len(xs))]
    return [(j / FPS, med[j]) for j in range(len(med))]

def follow_knots(centers, cw, src_w):
    """GoPro-style: subject dead-center every frame. Heavy smoothing (~1.7s
    moving average over the median-filtered track) kills jitter and sway while
    the crop continuously follows."""
    lo, hi = 0, src_w - cw
    if not centers:
        return [(0.0, (src_w - cw) / 2)]
    win = max(1, int(2.2 * FPS / 2))
    xs = [c for _, c in centers]
    sm = [statistics.mean(xs[max(0, i - win):i + win + 1]) for i in range(len(xs))]
    return [(t, min(max(s - cw / 2, lo), hi)) for (t, _), s in zip(centers, sm)]

def deadband_knots(centers, cw, src_w):
    lo, hi = 0, src_w - cw
    dead = src_w * DEAD_FRAC
    sustain = max(1, int(SUSTAIN_S * FPS))
    def clampx(c): return min(max(c - cw / 2, lo), hi)
    if not centers:
        return [(0.0, (src_w - cw) / 2)]
    lock = clampx(statistics.median([c for _, c in centers[:max(1, 2 * FPS)]]))
    knots = [(0.0, lock)]
    outside = 0
    for i, (t, c) in enumerate(centers):
        x = clampx(c)
        if abs(x - lock) > dead:
            outside += 1
            if outside >= sustain:
                tgt = clampx(statistics.median(
                    [cc for _, cc in centers[i - sustain + 1:i + 1]]))
                t0 = max(knots[-1][0], t - GLIDE)
                knots += [(t0, lock), (t, tgt)]
                lock = tgt
                outside = 0
        else:
            outside = 0
    knots.append((centers[-1][0] + 5, lock))
    return knots

def linear_expr(knots):
    """Piecewise-LINEAR x(t). For dense smoothed follow tracks: per-segment
    smoothstep easing makes the camera stop/restart at every sample (a visible
    lunge each 1/3 s); linear segments keep velocity continuous — the heavy
    moving-average smoothing already shapes the accelerations."""
    expr = f"{knots[0][1]:.1f}"
    for (t0, x0), (t1, x1) in zip(knots, knots[1:]):
        if t1 <= t0 or abs(x1 - x0) < 0.05:
            continue
        p = f"clip((t-{t0:.3f})/{t1 - t0:.3f}\\,0\\,1)"
        expr += f"+{x1 - x0:.2f}*{p}"
    return expr

def knots_expr(knots):
    expr = f"{knots[0][1]:.1f}"
    for (t0, x0), (t1, x1) in zip(knots, knots[1:]):
        if abs(x1 - x0) < 0.5 or t1 <= t0:
            continue
        p = f"clip((t-{t0:.2f})/{t1 - t0:.2f}\\,0\\,1)"
        expr += f"+{x1 - x0:.1f}*{p}*{p}*(3-2*{p})"
    return expr

def esc_ass(s):
    return s.replace("{", "").replace("}", "")

def build_ass(cues, path, ow, oh, fs, margin):
    cues = sorted(cues)
    for i in range(len(cues) - 1):
        if cues[i][1] > cues[i + 1][0]:
            cues[i] = (cues[i][0], cues[i + 1][0] - 0.01, cues[i][2])
    def ts(t):
        t = max(t, 0)
        return f"{int(t//3600)}:{int(t%3600//60):02d}:{int(t%60):02d}.{int(t*100%100):02d}"
    with open(path, "w") as f:
        f.write(f"""[Script Info]
ScriptType: v4.00+
PlayResX: {ow}
PlayResY: {oh}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial Black,{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,8,3,2,60,60,{margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        for s, e, txt in cues:
            if e <= s:
                continue
            f.write(f"Dialogue: 0,{ts(s)},{ts(e)},Cap,,0,0,0,,{esc_ass(txt)}\n")

def build_clip(spec):
    src, start, end = spec["src"], spec["start"], spec["end"]
    p = PLAT[spec["platform"]]
    dur = end - start
    src_w, src_h = probe_dims(src)
    # crop: punch in and keep the crop's bottom edge above the audience —
    # Daniel's rule: NEVER show people's heads along the bottom of the frame
    ch = int(src_h * CROP_H_FRAC) & ~1
    cw = int(ch * p["crop_ar"]) & ~1
    dest = os.path.expanduser(spec.get("dest", "~/Downloads"))
    out = os.path.join(dest, f"{spec['date']} - {spec['hook']} - {spec['platform']}.mp4")
    tmp = tempfile.mkdtemp(dir=dest)
    try:
        centers = detect_centers(src, start, dur, tmp, src_w)
        if spec.get("track") == "follow":
            knots = follow_knots(centers, cw, src_w)
            xexpr = linear_expr(knots)
        else:
            knots = deadband_knots(centers, cw, src_w)
            xexpr = knots_expr(knots)
        nmoves = sum(1 for a, b in zip(knots, knots[1:]) if abs(b[1] - a[1]) >= 0.5)
        y = int((src_h - ch) * CROP_Y_ANCHOR)
        ass = os.path.join(tmp, "caps.ass")
        build_ass([tuple(c) for c in spec["cues"]], ass, p["ow"], p["oh"], p["fs"], p["margin"])
        assf = ass.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        vf = (f"crop={cw}:{ch}:'{xexpr}':{y},"
              f"scale={p['ow']}:{p['oh']},format=yuv420p,"
              f"ass=filename='{assf}'")
        subprocess.run([FF, "-y", "-v", "error",
                        "-ss", str(start), "-t", str(dur), "-i", src,
                        "-ss", str(start + 0.1), "-t", str(dur), "-i", src,
                        "-map", "0:v:0", "-map", "1:a:0",
                        "-vf", vf, "-c:v", "libx264", "-preset", "fast",
                        "-crf", "19", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                        "-t", str(dur), "-movflags", "+faststart", out], check=True)
        print(f"BUILT ({nmoves} camera moves) {out}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        specs = json.load(f)
    for spec in specs:
        build_clip(spec)
    print("ALL CLIPS DONE", flush=True)
