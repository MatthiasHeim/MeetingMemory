#!/bin/bash
# Install the deterministic Linear-task reconciliation sweep as a LaunchAgent.
#
# Runs tools/reconcile_meeting_tasks.py (pure Python — NOT a `claude -p` job, so
# it can never hit a Claude session limit) twice a day: 18:30 (≈30 min after
# transcript-sync) and 03:00 (nightly safety net). See reconcile_meeting_tasks.py
# and docs on the 2026-07-10 dropped-actions incident.
#
# NOT loaded automatically by any build — run this script explicitly.
#
# Usage:
#   bash tools/install_reconcile_plist.sh            # install + load
#   bash tools/install_reconcile_plist.sh --unload   # stop + unload only

set -euo pipefail

LABEL="com.user.reconcile-meeting-tasks"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_PLIST="$SCRIPT_DIR/launchagents/$LABEL.plist"
DEST_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UID_NUM="$(id -u)"

if [[ "${1:-}" == "--unload" ]]; then
    launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
    echo "Unloaded $LABEL (plist left on disk at $DEST_PLIST)"
    exit 0
fi

# Ensure the (non-TCC) log directory exists before launchd tries to open std-io.
mkdir -p "$HOME/Library/Logs/MeetingRecorder"

# Sanity: the sweep must at least import cleanly with the repo venv.
VENV_PY="/Users/Matthias/Repos/MeetingMemory/venv/bin/python"
if [[ -x "$VENV_PY" ]]; then
    "$VENV_PY" -c "import py_compile; py_compile.compile('$SCRIPT_DIR/reconcile_meeting_tasks.py', doraise=True)" \
        && echo "reconcile_meeting_tasks.py compiles OK with the repo venv"
else
    echo "WARNING: repo venv python not found at $VENV_PY — create it first (see CLAUDE.md)."
fi

cp "$SRC_PLIST" "$DEST_PLIST"
echo "Copied plist → $DEST_PLIST"

launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$DEST_PLIST"

echo "Installed and loaded: $LABEL"
echo "Status: $(launchctl list | grep reconcile-meeting-tasks || echo 'not found in launchctl list')"
echo
echo "Run once now to verify:"
echo "  launchctl kickstart -p gui/$UID_NUM/$LABEL"
echo "  tail -f ~/Library/Logs/MeetingRecorder/reconcile-meeting-tasks.log"
