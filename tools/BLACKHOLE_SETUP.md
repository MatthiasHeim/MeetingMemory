# BlackHole Audio Setup Guide

This guide explains how to set up BlackHole on macOS to record both your microphone and system audio (e.g., from Zoom, Teams, or other meeting apps) simultaneously.

## What is BlackHole?

BlackHole is a free, open-source virtual audio driver for macOS that allows applications to pass audio to other applications with zero latency.

## Installation

### Step 1: Install BlackHole

```bash
brew install blackhole-2ch
```

Or download from: https://existential.audio/blackhole/

After installation, you may need to restart your computer.

### Step 2: Open Audio MIDI Setup

1. Open **Audio MIDI Setup** (found in `/Applications/Utilities/Audio MIDI Setup.app`)
2. Or use Spotlight: Press `Cmd + Space` and type "Audio MIDI Setup"

### Step 3: Create a Multi-Output Device

This allows you to hear audio while it's also being routed to BlackHole.

1. Click the **+** button in the bottom left corner
2. Select **Create Multi-Output Device**
3. Check both:
   - **BlackHole 2ch**
   - **Your speakers/headphones** (e.g., "MacBook Pro Speakers" or "External Headphones")
4. Make sure **BlackHole 2ch** is listed FIRST (drag to reorder if needed)
5. Right-click on the Multi-Output Device and select **Use This Device For Sound Output**

**Important:** The order matters! BlackHole should be first to ensure audio is properly routed.

### Step 4: Create an Aggregate Device

This combines your microphone input with BlackHole (system audio) into a single input device.

1. Click the **+** button again
2. Select **Create Aggregate Device**
3. Check both:
   - **BlackHole 2ch**
   - **Your microphone** (e.g., "MacBook Pro Microphone" or your external mic)
4. Rename it to **"Aggregate Device"** (double-click the name to edit)

### Step 5: Configure MeetingRecorder

Edit your config file:

```bash
open ~/Documents/MeetingRecorder/config.yaml
```

Change the audio device setting:

```yaml
audio:
  device: "Aggregate Device"  # Changed from "default"
  sample_rate: 16000
  channels: 1
```

### Step 6: Configure Your Meeting App

Set your meeting app (Zoom, Teams, etc.) to use the Multi-Output Device for audio output:

**Zoom:**
1. Open Zoom Settings → Audio
2. Set **Speaker** to "Multi-Output Device"
3. Keep **Microphone** as your regular mic

**Microsoft Teams:**
1. Click your profile → Settings → Devices
2. Set **Speaker** to "Multi-Output Device"

**Google Meet:**
1. Click the three dots → Settings → Audio
2. Set **Speakers** to "Multi-Output Device"

## Verification

To verify your setup is working:

1. Play some audio on your computer
2. Run this command to see audio levels:

```bash
python -c "
import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    volume = np.linalg.norm(indata) * 10
    print('|' + '#' * int(volume) + ' ' * (50 - int(volume)) + '|')

device = None
for i, d in enumerate(sd.query_devices()):
    if 'Aggregate' in d['name']:
        device = i
        break

if device:
    print(f'Using device: {sd.query_devices(device)[\"name\"]}')
    print('Play some audio and speak into your mic. You should see levels:')
    with sd.InputStream(device=device, callback=callback):
        sd.sleep(10000)
else:
    print('Aggregate Device not found!')
"
```

You should see volume bars when:
- You play system audio (music, video, meeting audio)
- You speak into your microphone

## Troubleshooting

### No audio from Aggregate Device
- Ensure BlackHole is properly installed (restart after installation)
- Verify the Multi-Output Device is set as the system output
- Check that both devices are checked in the Aggregate Device

### Can't hear audio anymore
- Make sure you're using the Multi-Output Device (not BlackHole directly) as your system output
- The Multi-Output Device routes audio to both BlackHole AND your speakers

### Recording is silent
- Check that "Aggregate Device" is spelled correctly in config.yaml
- Run `python meeting_recorder.py --list-devices` to see available devices

### Audio is distorted or has echo
- Mute your microphone in the meeting app if you don't need to speak
- Use headphones instead of speakers to prevent feedback

## Audio Device Reference

After setup, you should have these devices:

| Device | Type | Purpose |
|--------|------|---------|
| BlackHole 2ch | Input/Output | Virtual audio loopback |
| Multi-Output Device | Output | Speakers + BlackHole combined |
| Aggregate Device | Input | Microphone + BlackHole combined |

## Resetting to Default

If you want to stop using BlackHole:

1. Open System Settings → Sound
2. Set Output back to your speakers/headphones
3. Edit `config.yaml` and set `device: "default"`

To uninstall BlackHole:
```bash
brew uninstall blackhole-2ch
```
