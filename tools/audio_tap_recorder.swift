// audio_tap_recorder.swift — Core Audio process-tap recorder for MeetingMemory.
//
// Captures system audio (all process output, stereo mixdown) AND the default
// microphone into a SINGLE clocked aggregate device, written to one multichannel
// WAV. One IO proc / one clock => no drift between the two sources (the property
// that a two-process capture can't guarantee).
//
// Output WAV channel layout (input streams in aggregate order: sub-devices, then taps):
//   ch0  = microphone (Matthias)
//   ch1  = system audio Left  (remote participants)
//   ch2  = system audio Right (remote participants)
// Downstream split (Phase 5): ch0 -> host, ch1/ch2 -> remote.
//
// Build:   swiftc -O tools/audio_tap_recorder.swift -o build/audio_tap_recorder
// Usage:   audio_tap_recorder <out.wav> [seconds]   # omit seconds => record until SIGINT
//
// ── STATUS: ARCHITECTURE PROVEN END-TO-END (2026-05-29) ───────────────────────
// Both capture halves verified working with this exact code (3-channel, single
// clocked aggregate, multi-stream interleave):
//   • SYSTEM audio (ch1/ch2): captured at -18.9 dB when launched via LaunchServices
//     (`open AudioTapRecorder.app`) so the process is attributed to the bundle id
//     that holds kTCCServiceAudioCapture.
//   • MIC (ch0): captured at -24 dB when the running process holds
//     kTCCServiceMicrophone (e.g. run directly under a terminal that has mic access).
//
// DEPLOYMENT — the ONE remaining step is granting BOTH TCC permissions to the
// signed bundle that runs this:
//   - kTCCServiceAudioCapture  (system audio / process tap)  — Info.plist needs
//     NSAudioCaptureUsageDescription; granted on first capture.
//   - kTCCServiceMicrophone     (the mic sub-device)          — Info.plist needs
//     NSMicrophoneUsageDescription; a background LSUIElement app can't show the
//     prompt, so grant it from a FOREGROUND app (the MeetingRecorder menu-bar app
//     this will be integrated into) or toggle it in System Settings > Privacy &
//     Security > Microphone.
// Notes: unsigned/unbundled CLIs never trigger TCC at all (silent -91 dB); the
// process must be launched AS the signed bundle (LaunchServices) for attribution.
//     mkdir -p build/AudioTapRecorder.app/Contents/MacOS
//     cp build/audio_tap_recorder build/AudioTapRecorder.app/Contents/MacOS/
//     # Info.plist: CFBundleIdentifier + NSAudioCaptureUsageDescription + NSMicrophoneUsageDescription
//     codesign --force --deep --options runtime -s - build/AudioTapRecorder.app
//     open build/AudioTapRecorder.app --args /tmp/out.wav 8
// ──────────────────────────────────────────────────────────────────────────────
import Foundation
import CoreAudio
import AudioToolbox
import AVFoundation

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
let extraArgs = Array(args.dropFirst(2))
// --system-only: tap-only aggregate (system audio, 2ch). Needs ONLY
// kTCCServiceAudioCapture (no mic, no microphone permission). Used by the hybrid
// pipeline where the mic is captured separately by the Python recorder.
let systemOnly = extraArgs.contains("--system-only")
let duration: Double? = extraArgs.compactMap { Double($0) }.first

// ── Microphone permission ─────────────────────────────────────────────────────
// The mic sub-device needs kTCCServiceMicrophone. Request it explicitly via
// AVFoundation so the app registers in System Settings > Privacy > Microphone
// and (if undetermined) shows the prompt. The system-audio tap uses a separate
// permission (kTCCServiceAudioCapture) granted on first tap read.
func ensureMicPermission(timeout: Double = 60) {
    let status = AVCaptureDevice.authorizationStatus(for: .audio)
    logln("mic authorization status (pre): \(status.rawValue)")
    if status == .authorized { return }
    var done = false
    AVCaptureDevice.requestAccess(for: .audio) { granted in
        logln("mic access granted: \(granted)")
        done = true
    }
    // Spin the MAIN run loop while waiting — the permission prompt + its
    // completion handler are delivered on the main run loop, so blocking it
    // (e.g. on a semaphore) deadlocks the prompt and it never appears.
    let deadline = Date(timeIntervalSinceNow: timeout)
    while !done && Date() < deadline {
        RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.1))
    }
}
if !systemOnly { ensureMicPermission() }

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
// Use the dedicated initializer (empty exclude list => tap ALL processes). The
// plain init() + property-setting path produced a tap that delivered silence —
// the global-tap initializers set internal device/stream targeting that manual
// property assignment does not replicate.
let desc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
desc.name = "MeetingMemory system tap"
desc.isPrivate = true
desc.muteBehavior = .unmuted  // user still hears the call
var tapID = AudioObjectID(0)
chk(AudioHardwareCreateProcessTap(desc, &tapID), "AudioHardwareCreateProcessTap")
let tapUID = desc.uuid.uuidString
logln("tap id=\(tapID) uid=\(tapUID)")

// ── 2) Aggregate: mic sub-device (clock master) + system tap, single clock ─────
let micUID = systemOnly ? nil : defaultInputUID()
logln("system-only=\(systemOnly) default mic uid=\(micUID ?? "<none>")")
var subdevices: [[String: Any]] = []
if let m = micUID { subdevices.append(["uid": m, "drift": 1]) }   // drift-compensate mic against tap clock
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

// ── 3) Enumerate ALL input streams (mic + tap are SEPARATE streams) ────────────
// The aggregate exposes one input stream per member: mic stream (1ch) then tap
// stream (2ch). The IO proc receives one AudioBufferList buffer per stream, so we
// must sum their channels and interleave them into the output file.
func inputStreams(_ dev: AudioObjectID) -> [AudioStreamID] {
    var a = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreams,
                                       mScope: kAudioObjectPropertyScopeInput,
                                       mElement: kAudioObjectPropertyElementMain)
    var dsz: UInt32 = 0
    if AudioObjectGetPropertyDataSize(dev, &a, 0, nil, &dsz) != noErr { return [] }
    let n = Int(dsz) / MemoryLayout<AudioStreamID>.size
    var ids = [AudioStreamID](repeating: 0, count: n)
    if AudioObjectGetPropertyData(dev, &a, 0, nil, &dsz, &ids) != noErr { return [] }
    return ids
}
func streamChannels(_ stream: AudioStreamID) -> (UInt32, Double) {
    var f = AudioStreamBasicDescription()
    var fsz = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    var a = AudioObjectPropertyAddress(mSelector: kAudioStreamPropertyVirtualFormat,
                                       mScope: kAudioObjectPropertyScopeGlobal,
                                       mElement: kAudioObjectPropertyElementMain)
    if AudioObjectGetPropertyData(stream, &a, 0, nil, &fsz, &f) != noErr { return (0, 48000) }
    return (f.mChannelsPerFrame, f.mSampleRate)
}

let streams = inputStreams(aggID)
var perStreamChannels: [Int] = []
var sampleRate = 48000.0
for s in streams {
    let (ch, sr) = streamChannels(s)
    perStreamChannels.append(Int(ch))
    if sr > 0 { sampleRate = sr }
}
let totalChannels = perStreamChannels.reduce(0, +)
logln("input streams=\(streams.count) channels-per-stream=\(perStreamChannels) total=\(totalChannels) @ \(Int(sampleRate))Hz")
guard totalChannels > 0 else { logln("no input channels — aborting"); exit(3) }

// ── 4) Output WAV (16-bit PCM, totalChannels) + client format (float interleaved)
var clientFmt = AudioStreamBasicDescription(
    mSampleRate: sampleRate, mFormatID: kAudioFormatLinearPCM,
    mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
    mBytesPerPacket: UInt32(4 * totalChannels), mFramesPerPacket: 1,
    mBytesPerFrame: UInt32(4 * totalChannels), mChannelsPerFrame: UInt32(totalChannels),
    mBitsPerChannel: 32, mReserved: 0)
var fileFmt = AudioStreamBasicDescription(
    mSampleRate: sampleRate, mFormatID: kAudioFormatLinearPCM,
    mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
    mBytesPerPacket: UInt32(2 * totalChannels), mFramesPerPacket: 1,
    mBytesPerFrame: UInt32(2 * totalChannels), mChannelsPerFrame: UInt32(totalChannels),
    mBitsPerChannel: 16, mReserved: 0)
var extFile: ExtAudioFileRef?
chk(ExtAudioFileCreateWithURL(outURL as CFURL, kAudioFileWAVEType, &fileFmt, nil,
        AudioFileFlags.eraseFile.rawValue, &extFile), "ExtAudioFileCreateWithURL")
chk(ExtAudioFileSetProperty(extFile!, kExtAudioFileProperty_ClientDataFormat,
        UInt32(MemoryLayout<AudioStreamBasicDescription>.size), &clientFmt), "SetClientDataFormat")

// Reusable scratch interleave buffer (float). Sized for a generous block.
let maxFrames = 16384
let scratch = UnsafeMutablePointer<Float>.allocate(capacity: maxFrames * totalChannels)
let pscPerStream = perStreamChannels   // capture for the block

// ── 5) IO proc: interleave per-stream buffers -> scratch -> file ───────────────
var frames: Int64 = 0
var ioProcID: AudioDeviceIOProcID?
let totalCh = totalChannels
let block: AudioDeviceIOBlock = { (_, inInputData, _, _, _) in
    let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inInputData))
    guard abl.count > 0 else { return }
    // Frames this cycle = bytes / (4 bytes * channels) of the first buffer.
    let b0 = abl[0]
    let ch0 = max(Int(b0.mNumberChannels), 1)
    let n = Int(b0.mDataByteSize) / (4 * ch0)
    if n <= 0 || n > maxFrames { return }
    // Zero the scratch (defensive — handles channel-count drift).
    scratch.update(repeating: 0, count: n * totalCh)
    var chOffset = 0
    for i in 0..<abl.count {
        let buf = abl[i]
        let bch = Int(buf.mNumberChannels)
        guard bch > 0, let src = buf.mData?.assumingMemoryBound(to: Float.self) else { continue }
        // Copy this stream's bch interleaved channels into [chOffset ..< chOffset+bch].
        for f in 0..<n {
            for c in 0..<bch {
                let outIdx = f * totalCh + chOffset + c
                if outIdx < n * totalCh { scratch[outIdx] = src[f * bch + c] }
            }
        }
        chOffset += bch
        if chOffset >= totalCh { break }
    }
    var outABL = AudioBufferList(
        mNumberBuffers: 1,
        mBuffers: AudioBuffer(mNumberChannels: UInt32(totalCh),
                              mDataByteSize: UInt32(n * totalCh * 4),
                              mData: scratch))
    if let f = extFile {
        if ExtAudioFileWrite(f, UInt32(n), &outABL) == noErr { frames += Int64(n) }
    }
    _ = pscPerStream
}
chk(AudioDeviceCreateIOProcIDWithBlock(&ioProcID, aggID, nil, block), "CreateIOProcIDWithBlock")

// ── 6) Run the MAIN RUN LOOP until SIGINT (or duration) ────────────────────────
// CoreAudio device IO + aggregate/tap delivery is reliable when the main run
// loop is actually running; a bare `Thread.sleep`/poll loop intermittently
// starves the IO proc (observed: 0 frames). Stop by stopping the run loop from
// the SIGINT handler (clean teardown => valid WAV header) or a duration timer.
let sig = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
sig.setEventHandler { CFRunLoopStop(CFRunLoopGetMain()) }
sig.resume()
signal(SIGINT, SIG_IGN)
chk(AudioDeviceStart(aggID, ioProcID), "AudioDeviceStart")
logln("recording\(duration.map { " \($0)s" } ?? " until SIGINT")...")
if let d = duration {
    DispatchQueue.main.asyncAfter(deadline: .now() + d) { CFRunLoopStop(CFRunLoopGetMain()) }
}
CFRunLoopRun()

// ── 7) Teardown ────────────────────────────────────────────────────────────────
AudioDeviceStop(aggID, ioProcID)
if let p = ioProcID { AudioDeviceDestroyIOProcID(aggID, p) }
ExtAudioFileDispose(extFile!)
AudioHardwareDestroyAggregateDevice(aggID)
AudioHardwareDestroyProcessTap(tapID)
scratch.deallocate()
logln("done. wrote \(frames) frames (\(totalCh)ch @ \(Int(sampleRate))Hz) to \(outURL.path)")
