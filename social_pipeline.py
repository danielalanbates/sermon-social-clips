#!/usr/bin/env python3
"""Social — find short-form clip material in a long video, edit it, save it.
Copyright (c) 2026 Daniel Bates / BatesAI (batesai.org). All rights reserved.

Pipeline: extract audio -> mlx-whisper (word timestamps) -> Claude picks the
best self-contained moments -> clipbuild5 renders each one (YOLO deadband
reframe, 9:16, captions burned last) into the destination the user chose.

Usage: social_pipeline.py <video> <dest_folder> [--per-video 5] [--platform Shorts]
Transcripts are cached in ~/Library/Application Support/Social/cache so
re-runs on the same video skip straight to selection.
"""
import argparse, datetime, json, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clipbuild5

CACHE = os.path.expanduser("~/Library/Application Support/Social/cache")
WHISPER = os.path.expanduser("~/Library/Python/3.9/bin/mlx_whisper")
FF = os.path.expanduser("~/.local/bin/ffmpeg")

PROMPT = """You are selecting short-form social clips from a long-form video transcript
(church service / sermon / talk). Pick the {n} best self-contained moments, each 30-60
seconds long (up to 75 if the point needs it), that would work as a vertical short: a strong
hook in the first 3 seconds, one clear idea, emotionally or spiritually punchy, quotable, and
understandable with zero context. Avoid announcements, scripture reading alone, or housekeeping.
CRITICAL: every clip must END on the note the speaker was building to — the payoff line,
the point of the story, the application — never mid-story or before the conclusion lands.
Include the full setup AND the resolution. Snap start/end to segment boundaries.

Transcript segments (start_sec | end_sec | text):
{segments}

Respond with ONLY a JSON array, no prose, each item:
{{"start_sec": float, "end_sec": float, "title": "short punchy title (<=60 chars)",
  "hook": "why this works as a Short (1 sentence)", "score": 1-10}}"""


def run_claude(prompt):
    # strip Claude Code session vars so a nested `claude -p` doesn't hang when run from inside a session
    UNSET = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_MESSAGING_SOCKET",
             "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_BRIDGE_SESSION_ID",
             "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_PID")
    env = {k: v for k, v in os.environ.items() if k not in UNSET}
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    last = None
    for attempt in range(3):
        try:
            r = subprocess.run(["claude", "-p", "--output-format", "json", "--model",
                                "sonnet", "--disallowedTools", "*"],
                               input=prompt, capture_output=True, text=True,
                               timeout=300, env=env, cwd=os.path.expanduser("~"))
            if r.returncode != 0:
                raise RuntimeError(r.stderr[:500])
            break
        except subprocess.TimeoutExpired as e:
            last = e; print("  retrying after timeout...", flush=True)
    else:
        raise last
    out = json.loads(r.stdout)["result"].strip()
    if out.startswith("```"):
        out = out.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(out)


def transcribe(video):
    os.makedirs(CACHE, exist_ok=True)
    key = re.sub(r"\W+", "_", os.path.basename(video)) + f"_{os.path.getsize(video)}"
    jpath = os.path.join(CACHE, key + ".json")
    if os.path.exists(jpath):
        print("transcript cached:", jpath, flush=True)
        return jpath
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        print("extracting audio...", flush=True)
        subprocess.run([FF, "-y", "-v", "error", "-i", video, "-vn", "-ac", "1",
                        "-ar", "16000", wav], check=True)
        print("transcribing (mlx-whisper large-v3-turbo)...", flush=True)
        subprocess.run([WHISPER, wav, "--model", "mlx-community/whisper-large-v3-turbo",
                        "--output-dir", td, "--output-format", "json",
                        "--word-timestamps", "True"], check=True)
        os.replace(os.path.join(td, "a.json"), jpath)
    return jpath


def sentence_segments(segs, gap=0.7, max_words=25):
    """Whisper often emits one word per segment; regroup into sentence-ish
    lines (split on pauses / punctuation) so the picker sees real structure."""
    out, cur = [], []
    def flush():
        if cur:
            out.append(dict(start=cur[0]["start"], end=cur[-1]["end"],
                            text=" ".join(w["text"].strip() for w in cur)))
    for w in segs:
        if cur and (w["start"] - cur[-1]["end"] > gap or len(cur) >= max_words
                    or cur[-1]["text"].strip().endswith((".", "?", "!"))):
            flush(); cur = []
        cur.append(w)
    flush()
    return out


def pick_clips(jpath, n):
    segs = sentence_segments(json.load(open(jpath))["segments"])
    lines = [f"{s['start']:.1f} | {s['end']:.1f} | {s['text'].strip()}" for s in segs]
    chunks, cur, size = [], [], 0
    for ln in lines:                       # ~20KB per call; larger prompts hang the CLI
        if size + len(ln) > 20000 and cur:
            chunks.append(cur); cur, size = [], 0
        cur.append(ln); size += len(ln) + 1
    if cur:
        chunks.append(cur)
    cands = []
    for i, ch in enumerate(chunks, 1):
        print(f"selecting: chunk {i}/{len(chunks)}", flush=True)
        cands += run_claude(PROMPT.format(n=n, segments="\n".join(ch)))
    if len(chunks) > 1 and len(cands) > n:
        print("final pick across chunks...", flush=True)
        cands = run_claude(
            f"From these candidate short-form moments (all from one video), pick the {n} "
            f"strongest, favoring variety of topic. Respond with ONLY a JSON array of the "
            f"chosen items, unchanged:\n{json.dumps(cands)}")
    return cands


def word_cues(jpath, start, end):
    """One-word cues, timestamps rebased to clip start, non-overlapping; each word
    stays up until the next one starts."""
    words = []
    for seg in json.load(open(jpath))["segments"]:
        for w in seg.get("words", []) or []:
            if w["start"] >= start - 0.2 and w["end"] <= end + 0.2:
                words.append(w)
    cues = [[w] for w in words]          # one word at a time
    out = []
    for c in cues:
        s = max(0.0, c[0]["start"] - start)
        e = max(s + 0.3, c[-1]["end"] - start)
        if out:                          # hold previous word until this one starts
            out[-1] = (out[-1][0], s - 0.01, out[-1][2])
        out.append((s, e, "".join(w["word"] for w in c).strip().upper()))
    return out


NOTES_PROMPT = """You write YouTube Shorts / Instagram Reels / Facebook Reels metadata for a church
sermon clip. Speaker: the pastor at New Heights Church, Richland WA. Tone: warm, direct, not
clickbaity, no emojis in the title. Here is the clip transcript:

{text}

Respond with ONLY a JSON object:
{{"title": "YouTube title (<=70 chars, hook first)",
  "description": "2-3 sentence YouTube description ending with an invitation to watch the full
sermon, then a blank line and 5-8 relevant hashtags",
  "pinned_comment": "one friendly pinned comment that asks a question to spark replies",
  "instagram_caption": "1-2 sentence caption + hashtags",
  "facebook_caption": "1-2 sentence caption, no hashtags",
  "quote": "the single most quotable line from the transcript, verbatim"}}"""


def write_notes(video_path, clip_text, dest, hook):
    """Companion .txt next to each clip: pre-filled title, description, pinned
    comment, captions for each platform, and the full transcript."""
    meta = run_claude(NOTES_PROMPT.format(text=clip_text))
    txt = os.path.splitext(video_path)[0] + ".txt"
    with open(txt, "w") as f:
        f.write(f"VIDEO: {os.path.basename(video_path)}\n")
        f.write("\n")
        f.write("=== YOUTUBE TITLE ===\n" + meta["title"] + "\n\n")
        f.write("=== YOUTUBE DESCRIPTION ===\n" + meta["description"] + "\n\n")
        f.write("=== PINNED COMMENT ===\n" + meta["pinned_comment"] + "\n\n")
        f.write("=== INSTAGRAM CAPTION ===\n" + meta["instagram_caption"] + "\n\n")
        f.write("=== FACEBOOK CAPTION ===\n" + meta["facebook_caption"] + "\n\n")
        f.write("=== QUOTE ===\n" + meta["quote"] + "\n\n")
        f.write("=== TRANSCRIPT ===\n" + clip_text + "\n")
    return txt


def clip_text(jpath, start, end):
    words = [w["word"] for seg in json.load(open(jpath))["segments"]
             for w in seg.get("words", []) or [] if w["start"] >= start - 0.2 and w["end"] <= end + 0.2]
    return "".join(words).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("dest")
    ap.add_argument("--per-video", type=int, default=5)
    ap.add_argument("--platform", default="Shorts", choices=list(clipbuild5.PLAT))
    ap.add_argument("--min-score", type=int, default=7)
    ap.add_argument("--track", default="follow", choices=["follow", "deadband"],
                    help="follow = GoPro-style always-centered; deadband = still camera")
    a = ap.parse_args()
    dest = os.path.expanduser(a.dest)
    os.makedirs(dest, exist_ok=True)

    jpath = transcribe(a.video)
    clips = [c for c in pick_clips(jpath, a.per_video) if int(c.get("score", 0)) >= a.min_score]
    if not clips:
        print("No clip-worthy moments found."); return
    date = datetime.date.fromtimestamp(os.path.getmtime(a.video)).strftime("%-m-%-d-%y")
    built = []
    for c in clips:
        s, e = float(c["start_sec"]), float(c["end_sec"])
        hook = re.sub(r'[/:\\"]', "", c.get("title") or "clip").strip()[:60]
        spec = dict(src=a.video, start=s, end=e, hook=hook, platform=a.platform,
                    date=date, dest=dest, track=a.track, cues=word_cues(jpath, s, e))
        print(f"rendering: {hook} ({s:.0f}-{e:.0f}s)", flush=True)
        clipbuild5.build_clip(spec)
        outp = os.path.join(dest, f"{date} - {hook} - {a.platform}.mp4")
        built.append(outp)
        print("writing notes...", flush=True)
        write_notes(outp, clip_text(jpath, s, e), dest, hook)
    for f in built:
        subprocess.run(["open", f])
    subprocess.run(["osascript", "-e",
                    f'display notification "Made {len(built)} clip(s) in {os.path.basename(dest)}" with title "Social"'])
    print(f"DONE — {len(built)} clip(s) in {dest}", flush=True)


if __name__ == "__main__":
    main()
