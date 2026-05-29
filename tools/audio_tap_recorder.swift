// audio_tap_recorder.swift — Core Audio process-tap recorder for MeetingMemory.
//
// Captures system audio (all process output, stereo mixdown) AND the default
// microphone into a SINGLE clocked aggregate device, written to one multichannel
// WAV. One IO proc / one clock => no drift between the two sources (the property
// that a two-process capture can't guarantee).
//
// Channel layout of the output WAV (sub-devices first, then taps):
//   ch0  = microphone (Matthias)
//   ch1  = system audio Left  (remote participants)
//   ch2  = system audio Right (remote participants)
// Downstream split (Phase 5): ch0 -> host, ch1/ch2 -> remote.
//
// Build:   swiftc -O tools/audio_tap_recorder.swift -o build/audio_tap_recorder
// Usage:   audio_tap_recorder <out.wav> [seconds]   # omit seconds => record until SIGINT
//
// ── STATUS (2026-05-29) ───────────────────────────────────────────────────────
// SOLVED: the macOS `kTCCServiceAudioCapture` permission. An unsigned/unbundled
//   CLI never triggers TCC (macOS silently feeds it -91 dB). Wrapping the binary
//   in a signed .app bundle with NSAudioCaptureUsageDescription DID register +
//   grant the permission (TCC now shows com.lailix.meetingmemory.audiotap = allowed).
//   => deploy this inside a signed .app bundle (see build steps below + docs).
//
// REMAINING BUG: combining a sub-device (mic) AND a tap in ONE aggregate produces
//   a MULTI-STREAM input — mic on input stream 0 (1ch), system tap on stream 1
//   (2ch). The IO proc below reads only mBuffers.mBuffers[0] and the format query
//   only inspects element 0, so it currently captures the MIC STREAM ONLY (the
//   aggregate reports "1 ch" and the system audio is dropped). To finish: query
//   per-stream formats, handle the multi-buffer AudioBufferList (loop all buffers),
//   and write all channels interleaved (mic + sysL + sysR). See
//   docs/recording-architecture.md "Option C status".
//
//   Build a signed test bundle:
//     swiftc -O tools/audio_tap_recorder.swift -o build/audio_tap_recorder
//     mkdir -p build/AudioTapRecorder.app/Contents/MacOS
//     cp build/audio_tap_recorder build/AudioTapRecorder.app/Contents/MacOS/
//     # + Info.plist with NSAudioCaptureUsageDescription, then:
//     codesign --force --deep --options runtime -s - build/AudioTapRecorder.app
// ──────────────────────────────────────────────────────────────────────────────
import Foundation
import CoreAudio
import AudioToolbox

func chk(_ s: OSStatus, _ what: String) {
    if s != noErr {
        FileHandle.standardError.write("ERR \(what): OSStatus \(s)\n".data(using: .utf8)!)
        exit(2)
    }
}
func logln(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let args = CommandLine.arguments
guard args.count >= 2 else { print("usage: audio_tap_recorder <out.wav> [seconds]"); exit(1) }
let outURL = URL(fileURLWithPath: args[1])
let duration: Double? = args.count >= 3 ? Double(args[2]) : nil

// ── Default input (microphone) UID ────────────────────────────────────────────
func defaultInputUID() -> String? {
    var devID = AudioObjectID(0)
    var sz = UInt32(MemoryLayout<AudioObjectID>.size)
    var a = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultInputDevice,
                                       mScope: kAudioObjectPropertyScopeGlobal,
                                       mElement: kAudioObjectPropertyElementMain)
    if AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &sz, &devID) != noErr { return nil }
    var uid: Unmanaged<CFString>? = nil
    var usz = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    var ua = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyDeviceUID,
                                        mScope: kAudioObjectPropertyScopeGlobal,
                                        mElement: kAudioObjectPropertyElementMain)
    if AudioObjectGetPropertyData(devID, &ua, 0, nil, &usz, &uid) != noErr { return nil }
    return uid?.takeRetainedValue() as String?   // property returns a +1 CFString
}

// ── 1) Global stereo tap of all processes, unmuted, private ────────────────────
let desc = CATapDescription()
desc.name = "MeetingMemory system tap"
desc.processes = []
desc.isExclusive = true       // exclude none => include all processes
desc.isMono = false
desc.isMixdown = true
desc.isPrivate = true
desc.muteBehavior = .unmuted  // user still hears the call
var tapID = AudioObjectID(0)
chk(AudioHardwareCreateProcessTap(desc, &tapID), "AudioHardwareCreateProcessTap")
let tapUID = desc.uuid.uuidString
logln("tap id=\(tapID) uid=\(tapUID)")

// ── 2) Aggregate: mic sub-device (clock master) + system tap, single clock ─────
let micUID = defaultInputUID()
logln("default mic uid=\(micUID ?? "<none>")")
var subdevices: [[String: Any]] = []
if let m = micUID { subdevices.append(["uid": m, "drift": 1]) }   // drift-compensate the mic against the tap clock
let aggUID = "com.meetingmemory.recagg." + UUID().uuidString
var aggDict: [String: Any] = [
    "uid": aggUID,
    "name": "MeetingMemory Recording Aggregate",
    "private": true,
    "tapautostart": true,
    "subdevices": subdevices,
    "taps": [["uid": tapUID, "drift": 0]],
]
if let m = micUID { aggDict["master"] = m }   // mic is the clock master
var aggID = AudioObjectID(0)
chk(AudioHardwareCreateAggregateDevice(aggDict as CFDictionary, &aggID), "CreateAggregateDevice")
logln("aggregate id=\(aggID)")

// ── 3) Input stream format of the aggregate ────────────────────────────────────
var fmt = AudioStreamBasicDescription()
var fsz = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
var fa = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreamFormat,
                                    mScope: kAudioObjectPropertyScopeInput,
                                    mElement: kAudioObjectPropertyElementMain)
chk(AudioObjectGetPropertyData(aggID, &fa, 0, nil, &fsz, &fmt), "GetStreamFormat")
logln("agg input format: \(fmt.mSampleRate) Hz, \(fmt.mChannelsPerFrame) ch, flags=\(fmt.mFormatFlags)")

// ── 4) Output WAV (16-bit PCM) ─────────────────────────────────────────────────
var fileFmt = AudioStreamBasicDescription(
    mSampleRate: fmt.mSampleRate, mFormatID: kAudioFormatLinearPCM,
    mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
    mBytesPerPacket: 2 * fmt.mChannelsPerFrame, mFramesPerPacket: 1,
    mBytesPerFrame: 2 * fmt.mChannelsPerFrame, mChannelsPerFrame: fmt.mChannelsPerFrame,
    mBitsPerChannel: 16, mReserved: 0)
var extFile: ExtAudioFileRef?
chk(ExtAudioFileCreateWithURL(outURL as CFURL, kAudioFileWAVEType, &fileFmt, nil,
        AudioFileFlags.eraseFile.rawValue, &extFile), "ExtAudioFileCreateWithURL")
chk(ExtAudioFileSetProperty(extFile!, kExtAudioFileProperty_ClientDataFormat,
        UInt32(MemoryLayout<AudioStreamBasicDescription>.size), &fmt), "SetClientDataFormat")

// ── 5) IO proc ─────────────────────────────────────────────────────────────────
var frames: Int64 = 0
let bpf = fmt.mBytesPerFrame
var ioProcID: AudioDeviceIOProcID?
let block: AudioDeviceIOBlock = { (_, inInputData, _, _, _) in
    if let f = extFile, bpf > 0 {
        let n = inInputData.pointee.mBuffers.mDataByteSize / bpf
        if n > 0 { _ = ExtAudioFileWrite(f, n, inInputData); frames += Int64(n) }
    }
}
chk(AudioDeviceCreateIOProcIDWithBlock(&ioProcID, aggID, nil, block), "CreateIOProcIDWithBlock")

// ── 6) Run until duration elapses or SIGINT ────────────────────────────────────
var running = true
let sig = DispatchSource.makeSignalSource(signal: SIGINT)
sig.setEventHandler { running = false }
sig.resume()
signal(SIGINT, SIG_IGN)
chk(AudioDeviceStart(aggID, ioProcID), "AudioDeviceStart")
logln("recording\(duration.map { " \($0)s" } ?? " until SIGINT")...")
if let d = duration {
    Thread.sleep(forTimeInterval: d)
} else {
    while running { Thread.sleep(forTimeInterval: 0.2) }
}

// ── 7) Teardown ────────────────────────────────────────────────────────────────
AudioDeviceStop(aggID, ioProcID)
if let p = ioProcID { AudioDeviceDestroyIOProcID(aggID, p) }
ExtAudioFileDispose(extFile!)
AudioHardwareDestroyAggregateDevice(aggID)
AudioHardwareDestroyProcessTap(tapID)
logln("done. wrote \(frames) frames (\(fmt.mChannelsPerFrame)ch @ \(Int(fmt.mSampleRate))Hz) to \(outURL.path)")
