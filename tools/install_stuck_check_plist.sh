#!/bin/bash
# Install the stuck-transcription check as a LaunchAgent (every 30 min).
# Runs tools/check_stuck_transcriptions.py --notify. Pure Python, no LLM.
#
# Usage:
#   bash tools/install_stuck_check_plist.sh            # install + load
#   bash tools/install_stuck_check_plist.sh --unload   # stop + unload only
set -euo pipefail

LABEL="com.user.check-stuck-transcriptions"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_PLIST="$SCRIPT_DIR/launchagents/$LABEL.plist"
DEST_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UID_NUM="$(id -u)"

if [[ "${1:-}" == "--unload" ]]; then
    launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
    echo "Unloaded $LABEL (plist left on disk at $DEST_PLIST)"
    exit 0
fi

mkdir -p "$HOME/Library/Logs/MeetingRecorder"
/Users/Matthias/Repos/MeetingMemory/venv/bin/python -c \
    "import py_compile; py_compile.compile('$SCRIPT_DIR/check_stuck_transcriptions.py', doraise=True)"

cp "$SRC_PLIST" "$DEST_PLIST"
launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$DEST_PLIST"
echo "Installed and loaded: $LABEL"
echo "Log: ~/Library/Logs/MeetingRecorder/check-stuck-transcriptions.log"
