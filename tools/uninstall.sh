#!/bin/bash
# MeetingRecorder Uninstallation Script
# Removes LaunchAgents and optionally cleans up data

set -e

LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
MEETING_RECORDER_DIR="$HOME/Documents/MeetingRecorder"

echo "=========================================="
echo "MeetingRecorder Uninstallation"
echo "=========================================="
echo ""

# Stop and unload LaunchAgents
echo "Stopping services..."
launchctl unload "$LAUNCH_AGENTS_DIR/com.user.meetingrecorder.plist" 2>/dev/null || true
launchctl unload "$LAUNCH_AGENTS_DIR/com.user.transcribewatcher.plist" 2>/dev/null || true

echo "Removing LaunchAgents..."
rm -f "$LAUNCH_AGENTS_DIR/com.user.meetingrecorder.plist"
rm -f "$LAUNCH_AGENTS_DIR/com.user.transcribewatcher.plist"

echo ""
echo "LaunchAgents removed."
echo ""

# Ask about data
read -p "Do you want to delete recordings and transcripts? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing $MEETING_RECORDER_DIR..."
    rm -rf "$MEETING_RECORDER_DIR"
    echo "Data removed."
else
    echo "Keeping data at $MEETING_RECORDER_DIR"
fi

echo ""
echo "=========================================="
echo "Uninstallation Complete"
echo "=========================================="
echo ""
echo "Note: The tools directory and Python scripts are still in place."
echo "To completely remove, delete: $(dirname "${BASH_SOURCE[0]}")"
echo ""
