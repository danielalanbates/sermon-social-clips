#!/bin/bash
# Rebuild /Applications/Social.app whenever the source changes (launchd WatchPaths).
REPO="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Code/social"
SRC="$REPO/main.applescript"
APP="/Applications/Social.app"
[ -f "$SRC" ] || exit 0
# skip if the app is already newer than the source
[ "$APP/Contents/Resources/Scripts/main.scpt" -nt "$SRC" ] && exit 0
if osacompile -o "$APP" "$SRC" 2>/tmp/social_rebuild.err; then
  osascript -e 'display notification "Social.app rebuilt from Code/social" with title "Social"'
else
  osascript -e "display notification \"Social.app rebuild FAILED — see /tmp/social_rebuild.err\" with title \"Social\""
fi
