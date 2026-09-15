# Social

Mine a long video (sermon, service, talk) for short-form social clips, edit them
automatically, and save them wherever you choose.

**Social.app** (in /Applications, built from `main.applescript` via `osacompile`)
asks for a source video and a destination folder, then opens a Terminal window
running `social_pipeline.py`, which:

1. Extracts audio and transcribes with mlx-whisper large-v3-turbo (word
   timestamps). Transcripts cache in `~/Library/Application Support/Social/cache`
   keyed by filename+size, so re-runs skip straight to selection.
2. Asks Claude (`claude -p`, transcript chunked ≤20KB) to pick the best 30–60s
   self-contained moments, then a final cross-chunk pick.
3. Renders each clip with `clipbuild5.py`: YOLOv8 person detection feeding the
   still-camera deadband reframe (lock on the speaker, ignore small movement,
   one slow smoothstep glide when he truly relocates), 9:16 punch-in, captions
   burned AFTER the crop as the last filter, word-grouped non-overlapping cues.
4. Opens every finished clip and posts a notification. Nothing is uploaded.

Rebuild the app after editing the AppleScript:
`osacompile -o /Applications/Social.app main.applescript`

Deps: `pip3 install --user ultralytics mlx-whisper opencv-python`;
ffmpeg at `~/.local/bin/ffmpeg` (imageio-ffmpeg symlink); `claude` CLI.

History: extracted 2026-09-06 from the Weeklies pipeline (now archived in
`Code/Archive/weeklies-archived-2026-09-06`) and the Drive shorts workspace
(`av@nhcrichland Drive → daniel/code/shorts`, where the sermon-library batch
scripts and videos.db remain).
