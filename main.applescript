-- Social.app: pick a long video, pick a destination, mine + edit short-form clips.
-- Copyright (c) 2026 Daniel Bates / BatesAI (batesai.org). All rights reserved.
on run
	set repoPath to (POSIX path of (path to home folder)) & "Library/Mobile Documents/com~apple~CloudDocs/Code/social"
	set vid to POSIX path of (choose file with prompt "Social: pick the source video to mine for clips" of type {"public.movie"})
	set dest to POSIX path of (choose folder with prompt "Social: save the finished clips where?")
	set appDir to (POSIX path of (path to home folder)) & "Library/Application Support/Social"
	do shell script "mkdir -p " & quoted form of appDir
	set runner to appDir & "/run_social.command"
	set cmd to "#!/bin/bash" & linefeed & "cd " & quoted form of repoPath & linefeed & "/usr/bin/python3 social_pipeline.py " & quoted form of vid & " " & quoted form of dest & linefeed & "echo; echo 'Social: done — you can close this window.'"
	do shell script "cat > " & quoted form of runner & " <<'EOF'" & linefeed & cmd & linefeed & "EOF"
	do shell script "chmod +x " & quoted form of runner
	do shell script "open " & quoted form of runner
end run
