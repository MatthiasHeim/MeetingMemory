#!/bin/bash
# MeetingRecorder Installation Script
# Sets up the recording and transcription automation tools

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOSCRIBE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$NOSCRIBE_DIR/venv"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
MEETING_RECORDER_DIR="$HOME/Documents/MeetingRecorder"

echo "=========================================="
echo "MeetingRecorder Installation"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Create/update virtual environment
echo ""
echo "Setting up virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating new virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "Using existing virtual environment at $VENV_DIR"
fi

# Activate venv and install dependencies
echo ""
echo "Installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip > /dev/null
pip install -r "$SCRIPT_DIR/requirements.txt"

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p "$MEETING_RECORDER_DIR/Recordings"
mkdir -p "$MEETING_RECORDER_DIR/Transcripts"
mkdir -p "$MEETING_RECORDER_DIR/logs"

# Copy config if it doesn't exist
if [ ! -f "$MEETING_RECORDER_DIR/config.yaml" ]; then
    echo "Creating default config at $MEETING_RECORDER_DIR/config.yaml"
    # Config was already created, but ensure it exists
    if [ -f "$MEETING_RECORDER_DIR/config.yaml" ]; then
        echo "Config already exists"
    fi
fi

# Install LaunchAgents
echo ""
echo "Installing LaunchAgents..."
mkdir -p "$LAUNCH_AGENTS_DIR"

# Copy and update plist files with correct paths
cp "$SCRIPT_DIR/launchagents/com.user.meetingrecorder.plist" "$LAUNCH_AGENTS_DIR/"
cp "$SCRIPT_DIR/launchagents/com.user.transcribewatcher.plist" "$LAUNCH_AGENTS_DIR/"

echo "LaunchAgents installed to $LAUNCH_AGENTS_DIR"

# Load LaunchAgents
echo ""
echo "Loading LaunchAgents..."

# Unload first if already loaded (ignore errors)
launchctl unload "$LAUNCH_AGENTS_DIR/com.user.meetingrecorder.plist" 2>/dev/null || true
launchctl unload "$LAUNCH_AGENTS_DIR/com.user.transcribewatcher.plist" 2>/dev/null || true

# Load the agents
launchctl load "$LAUNCH_AGENTS_DIR/com.user.meetingrecorder.plist"
launchctl load "$LAUNCH_AGENTS_DIR/com.user.transcribewatcher.plist"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "What's next:"
echo ""
echo "1. Look for the 🎙️ icon in your menu bar"
echo "   Click it to start/stop recording"
echo ""
echo "2. Configure your settings:"
echo "   open $MEETING_RECORDER_DIR/config.yaml"
echo ""
echo "3. For combined mic + system audio recording:"
echo "   - Install BlackHole: brew install blackhole-2ch"
echo "   - See BLACKHOLE_SETUP.md for detailed instructions"
echo ""
echo "4. Add your n8n webhook URL to the config file"
echo ""
echo "Recordings will be saved to:"
echo "   $MEETING_RECORDER_DIR/Recordings/"
echo ""
echo "Transcripts will be saved to:"
echo "   $MEETING_RECORDER_DIR/Transcripts/"
echo ""
echo "To manually run the tools:"
echo "   python $SCRIPT_DIR/meeting_recorder.py"
echo "   python $SCRIPT_DIR/transcribe_watcher.py"
echo ""
