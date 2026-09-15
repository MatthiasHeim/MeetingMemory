#!/bin/zsh
# Build the foreground-capable ScreenCaptureKit app bundle used by the native
# recorder. This only creates a local bundle; it does not install anything,
# change TCC, or begin capture.

set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
output_bundle="${repo_root}/build/MeetingNativeCapture.app"
run_self_test=0

usage() {
  print "usage: ${0:t} [--output PATH.app] [--self-test]"
}

while (( $# > 0 )); do
  case "$1" in
    --output)
      (( $# >= 2 )) || { print -u2 -- "--output requires a .app path"; exit 2; }
      output_bundle="$2"
      shift 2
      ;;
    --self-test)
      run_self_test=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      print -u2 -- "unknown argument: $1"
      usage >&2
      exit 2
      ;;
  esac
done

[[ "${output_bundle:e}" == "app" ]] || {
  print -u2 -- "output bundle must end in .app: ${output_bundle}"
  exit 2
}

swiftc_path="$(xcrun --sdk macosx --find swiftc)"
sdk_path="$(xcrun --sdk macosx --show-sdk-path)"
stage_root="$(mktemp -d "${TMPDIR:-/tmp}/meeting-native-capture.XXXXXX")"
stage_bundle="${stage_root}/MeetingNativeCapture.app"
cleanup() { rm -rf -- "${stage_root}"; }
trap cleanup EXIT

mkdir -p "${stage_bundle}/Contents/MacOS"
plist="${stage_bundle}/Contents/Info.plist"
/usr/bin/plutil -create xml1 "${plist}"
/usr/libexec/PlistBuddy -c "Add :CFBundleDevelopmentRegion string en" "${plist}"
/usr/libexec/PlistBuddy -c "Add :CFBundleExecutable string native_capture" "${plist}"
/usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string com.lailix.meetingmemory.nativecapture" "${plist}"
/usr/libexec/PlistBuddy -c "Add :CFBundleInfoDictionaryVersion string 6.0" "${plist}"
/usr/libexec/PlistBuddy -c "Add :CFBundleName string MeetingNativeCapture" "${plist}"
/usr/libexec/PlistBuddy -c "Add :CFBundlePackageType string APPL" "${plist}"
/usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string 1.0" "${plist}"
/usr/libexec/PlistBuddy -c "Add :CFBundleVersion string 1" "${plist}"
# Keep this a regular, foreground-capable app. A background-only helper cannot
# reliably present the microphone/screen-recording permission prompts.
/usr/libexec/PlistBuddy -c "Add :LSUIElement bool false" "${plist}"
/usr/libexec/PlistBuddy -c "Add :NSMicrophoneUsageDescription string MeetingMemory records your microphone as a separate local audio track." "${plist}"
/usr/libexec/PlistBuddy -c "Add :NSScreenCaptureUsageDescription string MeetingMemory captures system audio as a separate local audio track." "${plist}"
/usr/libexec/PlistBuddy -c "Add :NSAudioCaptureUsageDescription string MeetingMemory captures system audio locally for meeting transcription." "${plist}"
/usr/bin/plutil -lint "${plist}"

"${swiftc_path}" \
  -swift-version 5 \
  -target arm64-apple-macosx15.0 \
  -sdk "${sdk_path}" \
  -O \
  -framework AppKit \
  -framework AVFoundation \
  -framework AudioToolbox \
  -framework CoreMedia \
  -framework CoreVideo \
  -framework ScreenCaptureKit \
  "${repo_root}/tools/native_capture.swift" \
  -o "${stage_bundle}/Contents/MacOS/native_capture"

# Bind each executable to the source it was built from, including local edits.
mkdir -p "${stage_bundle}/Contents/Resources"
/usr/bin/python3 - "${repo_root}" "${stage_bundle}/Contents/Resources/build-info.json" <<'PY'
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
root, output = map(Path, sys.argv[1:])
def git(*args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
files = ['tools/native_capture.swift', 'tools/native_capture_bridge.py', 'tools/build_native_capture.sh']
output.write_text(json.dumps({
    'git_revision': git('rev-parse', 'HEAD'),
    'working_tree_dirty': bool(git('status', '--porcelain')),
    'built_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
}, sort_keys=True, indent=2) + '\n')
PY

# An ad-hoc signature is enough for local development. Set
# NATIVE_CAPTURE_SIGN_IDENTITY to a Developer ID identity for distributable
# builds. The runtime option preserves the bundle identity used by macOS TCC.
sign_identity="${NATIVE_CAPTURE_SIGN_IDENTITY:--}"
/usr/bin/codesign --force --sign "${sign_identity}" --options runtime --timestamp=none "${stage_bundle}"
/usr/bin/codesign --verify --strict --verbose=2 "${stage_bundle}"

mkdir -p "${output_bundle:h}"
if [[ -e "${output_bundle}" ]]; then
  # This target is an explicitly named generated app bundle. Replacing it is
  # intentional and leaves source files, permissions, and installed apps alone.
  rm -rf -- "${output_bundle}"
fi
mv "${stage_bundle}" "${output_bundle}"

if (( run_self_test )); then
  self_test_dir="$(mktemp -d "${TMPDIR:-/tmp}/meeting-native-capture-self-test.XXXXXX")"
  trap 'rm -rf -- "${stage_root}" "${self_test_dir}"' EXIT
  "${output_bundle}/Contents/MacOS/native_capture" --output-dir "${self_test_dir}" --self-test
  [[ -f "${self_test_dir}/capture-manifest.json" ]] || {
    print -u2 -- "self-test did not write capture-manifest.json"
    exit 1
  }
  owner_test_dir="${self_test_dir}/owner-exit"
  "${output_bundle}/Contents/MacOS/native_capture" --output-dir "${owner_test_dir}" --self-test-owner-exit
  [[ -f "${owner_test_dir}/capture-manifest.json" ]] || {
    print -u2 -- "owner-exit self-test did not write capture-manifest.json"
    exit 1
  }
  /usr/bin/python3 - "${self_test_dir}/capture-manifest.json" "${owner_test_dir}/capture-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

timeline_path, owner_path = map(Path, sys.argv[1:])
timeline = json.loads(timeline_path.read_text(encoding='utf-8'))
assert timeline['synthetic'] is True
assert timeline['status'] == 'degraded'
assert timeline['finalized'] is True
assert isinstance(timeline['build_provenance']['source_sha256'], dict)
tracks = {track['source_id']: track for track in timeline['tracks']}
system = tracks['system']
segments = system['segments']
assert len(segments) == 2, segments
assert segments[0]['path'] != segments[1]['path']
assert abs(segments[0]['start_seconds'] - 3.25) < 0.000001
assert abs(segments[1]['start_seconds'] - 4.0) < 0.000001
assert segments[1]['start_seconds'] > segments[0]['start_seconds'] + segments[0]['duration_seconds']
assert any(
    gap['reason'] == 'timestamp_gap'
    and abs(gap['begin_seconds'] - 3.27) < 0.000001
    and abs(gap['end_seconds'] - 4.0) < 0.000001
    for gap in system['gaps']
)
for segment in segments:
    assert (timeline_path.parent / segment['path']).is_file()
    assert (timeline_path.parent / segment['timing_path']).is_file()

owner = json.loads(owner_path.read_text(encoding='utf-8'))
assert owner['synthetic'] is True
assert owner['status'] == 'complete'
assert owner['finalized'] is True
assert isinstance(owner['build_provenance']['source_sha256'], dict)
assert owner['owner_process']['monitor_kind'] == 'kqueue_evfilt_proc_note_exit'
assert owner['owner_process']['exit_observed'] is True
assert any(event['event'] == 'synthetic_owner_process_exit_monitor_verified' for event in owner['events'])
PY
  /usr/bin/plutil -lint -- "${self_test_dir}/capture-manifest.json" >/dev/null 2>&1 || true
  print "self-test manifests: ${self_test_dir}/capture-manifest.json and ${owner_test_dir}/capture-manifest.json"
fi

print "built signed foreground app bundle: ${output_bundle}"
