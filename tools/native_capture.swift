// native_capture.swift
//
// A deliberately small ScreenCaptureKit capture boundary for MeetingMemory.
// It keeps microphone and system audio as distinct, lossless CAF tracks and
// records their original CMSampleBuffer timing instead of treating the first
// sample of each source as time zero.
//
// This executable is intended to run from MeetingNativeCapture.app through
// LaunchServices.  The app bundle is built by tools/build_native_capture.sh.
// Do not use this as a background daemon: the foreground-capable bundle is
// part of macOS's permission attribution and makes permission failures visible.

import AppKit
@preconcurrency import AVFoundation
import AudioToolbox
import CoreMedia
import CoreVideo
import Darwin
import Foundation
@preconcurrency import ScreenCaptureKit

private let manifestName = "capture-manifest.json"
private let segmentSeconds = 20.0
private let readinessSeconds = 15.0
private let maxPendingPackets = 256
private let maxFramesPerPacket = 96_000
private let gapToleranceSeconds = 0.020
private let backwardsToleranceSeconds = 0.050

private enum NativeCaptureError: LocalizedError {
    case usage(String)
    case invalidSample(String)
    case writer(String)

    var errorDescription: String? {
        switch self {
        case .usage(let message), .invalidSample(let message), .writer(let message):
            return message
        }
    }
}

private struct Options {
    let outputDirectory: URL
    let duration: Double?
    let sessionToken: String
    let selfTest: Bool
    let selfTestOwnerExit: Bool
    let ownerPID: pid_t?

    static func parse(_ arguments: [String]) throws -> Options {
        var outputDirectory: URL?
        var duration: Double?
        var sessionToken = UUID().uuidString
        var selfTest = false
        var selfTestOwnerExit = false
        var ownerPID: pid_t?
        var index = 1

        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--output-dir":
                index += 1
                guard index < arguments.count else {
                    throw NativeCaptureError.usage("--output-dir requires a path")
                }
                outputDirectory = URL(fileURLWithPath: arguments[index], isDirectory: true)
            case "--duration":
                index += 1
                guard index < arguments.count, let value = Double(arguments[index]), value > 0 else {
                    throw NativeCaptureError.usage("--duration requires a positive number of seconds")
                }
                duration = value
            case "--session-token":
                index += 1
                guard index < arguments.count, !arguments[index].isEmpty else {
                    throw NativeCaptureError.usage("--session-token requires a non-empty value")
                }
                sessionToken = arguments[index]
            case "--self-test":
                selfTest = true
            case "--self-test-owner-exit":
                selfTestOwnerExit = true
            case "--owner-pid":
                index += 1
                guard index < arguments.count,
                      let parsedPID = Int32(arguments[index]),
                      parsedPID > 0 else {
                    throw NativeCaptureError.usage("--owner-pid requires a positive process ID")
                }
                ownerPID = pid_t(parsedPID)
            case "--help", "-h":
                throw NativeCaptureError.usage("usage: native_capture --output-dir DIR [--duration SECONDS] [--session-token TOKEN] [--owner-pid PID] [--self-test | --self-test-owner-exit]")
            default:
                throw NativeCaptureError.usage("unknown argument: \(argument)")
            }
            index += 1
        }

        guard let outputDirectory else {
            throw NativeCaptureError.usage("--output-dir is required")
        }
        guard !(selfTest && selfTestOwnerExit) else {
            throw NativeCaptureError.usage("--self-test and --self-test-owner-exit cannot be combined")
        }
        return Options(
            outputDirectory: outputDirectory.standardizedFileURL,
            duration: duration,
            sessionToken: sessionToken,
            selfTest: selfTest,
            selfTestOwnerExit: selfTestOwnerExit,
            ownerPID: ownerPID
        )
    }
}

private func iso8601Now() -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter.string(from: Date())
}

private func rawTime(_ time: CMTime) -> [String: Any] {
    var result: [String: Any] = [
        "value": time.value,
        "timescale": time.timescale,
        "flags": time.flags.rawValue,
        "epoch": time.epoch,
    ]
    if time.isValid && !time.isIndefinite {
        result["seconds"] = CMTimeGetSeconds(time)
    } else {
        result["seconds"] = NSNull()
    }
    return result
}

private func seconds(_ time: CMTime) -> Double? {
    guard time.isValid, !time.isIndefinite else { return nil }
    let value = CMTimeGetSeconds(time)
    return value.isFinite ? value : nil
}

/// The stream-to-host mapping is evidence about clock conversion, but its
/// anchors are not a session-start boundary: ScreenCaptureKit can legitimately
/// report zero-valued anchors while sample PTS values are already host-clock
/// times. Use the host clock captured when this controller was created as the
/// one shared timeline epoch, and reject a zero/invalid value rather than
/// silently producing a multi-day offset.
private func usableHostClockEpoch(_ hostClockStarted: CMTime) -> CMTime? {
    guard let value = seconds(hostClockStarted), value > 0 else { return nil }
    return hostClockStarted
}

private func hostRelativeSeconds(_ hostPTS: CMTime, epoch: CMTime) -> Double? {
    guard let usableEpoch = usableHostClockEpoch(epoch),
          let epochSeconds = seconds(usableEpoch),
          let packetSeconds = seconds(hostPTS) else {
        return nil
    }
    let relative = packetSeconds - epochSeconds
    return relative.isFinite ? relative : nil
}

private func mappingAnchorState(_ anchor: CMTime) -> String {
    guard let value = seconds(anchor) else { return "unusable" }
    return value > 0 ? "usable" : "zero_or_nonpositive"
}

private func relativePath(_ url: URL, from root: URL) -> String {
    let rootPath = root.standardizedFileURL.path
    let path = url.standardizedFileURL.path
    guard path.hasPrefix(rootPath + "/") else { return path }
    return String(path.dropFirst(rootPath.count + 1))
}

private func jsonData(_ object: Any) throws -> Data {
    guard JSONSerialization.isValidJSONObject(object) else {
        throw NativeCaptureError.writer("attempted to write an invalid JSON manifest")
    }
    return try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
}

private func writeJSONAtomically(_ object: Any, to url: URL) throws {
    let data = try jsonData(object)
    try data.write(to: url, options: .atomic)
}

private func readBuildProvenance() -> [String: Any] {
    guard let url = Bundle.main.url(forResource: "build-info", withExtension: "json"),
          let data = try? Data(contentsOf: url),
          let info = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
        return ["available": false]
    }
    return info
}

private final class PendingPacketBudget {
    enum ClaimResult {
        case accepted
        case closed
        case overflow
    }

    private let lock = NSLock()
    private let limit: Int
    private var pending = 0
    private var closed = false
    private var overflowReported = false

    init(limit: Int) {
        self.limit = limit
    }

    func claim() -> ClaimResult {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return .closed }
        guard pending < limit else {
            if !overflowReported {
                overflowReported = true
                closed = true
                return .overflow
            }
            return .closed
        }
        pending += 1
        return .accepted
    }

    func release() {
        lock.lock()
        pending = max(0, pending - 1)
        lock.unlock()
    }

    func close() {
        lock.lock()
        closed = true
        lock.unlock()
    }

    func pendingCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return pending
    }

}

/// Watches one already-running process through the kernel rather than polling
/// a reusable numeric PID.  An EVFILT_PROC registration is bound to the
/// process that existed when it was installed, so a later PID reuse cannot be
/// mistaken for continued ownership.
private final class OwnerProcessExitMonitor: @unchecked Sendable {
    private let ownerPID: pid_t
    private let queue: DispatchQueue
    private let onExit: () -> Void
    private let lock = NSLock()
    private var descriptor: Int32 = -1
    private var source: DispatchSourceRead?
    private var cancelled = false

    init(ownerPID: pid_t, queue: DispatchQueue, onExit: @escaping () -> Void) {
        self.ownerPID = ownerPID
        self.queue = queue
        self.onExit = onExit
    }

    func start() throws {
        guard ownerPID > 0 else {
            throw NativeCaptureError.writer("owner process ID is invalid")
        }
        let newDescriptor = kqueue()
        guard newDescriptor >= 0 else {
            throw NativeCaptureError.writer("cannot create owner process monitor: \(String(cString: strerror(errno)))")
        }

        var change = kevent(
            ident: UInt(ownerPID),
            filter: Int16(EVFILT_PROC),
            flags: UInt16(EV_ADD | EV_ENABLE | EV_ONESHOT),
            fflags: UInt32(NOTE_EXIT),
            data: 0,
            udata: nil
        )
        guard kevent(newDescriptor, &change, 1, nil, 0, nil) == 0 else {
            let message = String(cString: strerror(errno))
            close(newDescriptor)
            throw NativeCaptureError.writer("cannot monitor owner process \(ownerPID): \(message)")
        }

        let newSource = DispatchSource.makeReadSource(fileDescriptor: newDescriptor, queue: queue)
        newSource.setEventHandler { [weak self] in
            self?.consumeExitEvent()
        }
        newSource.setCancelHandler {
            close(newDescriptor)
        }

        lock.lock()
        guard !cancelled, source == nil else {
            lock.unlock()
            newSource.cancel()
            throw NativeCaptureError.writer("owner process monitor was already started or cancelled")
        }
        descriptor = newDescriptor
        source = newSource
        lock.unlock()
        newSource.resume()
    }

    func cancel() {
        lock.lock()
        guard !cancelled else {
            lock.unlock()
            return
        }
        cancelled = true
        let activeSource = source
        source = nil
        descriptor = -1
        lock.unlock()
        activeSource?.cancel()
    }

    private func consumeExitEvent() {
        lock.lock()
        guard !cancelled, descriptor >= 0 else {
            lock.unlock()
            return
        }
        let activeDescriptor = descriptor
        lock.unlock()

        var event = kevent()
        let received = kevent(activeDescriptor, nil, 0, &event, 1, nil)
        guard received == 1,
              event.filter == Int16(EVFILT_PROC),
              (event.fflags & UInt32(NOTE_EXIT)) != 0 else {
            return
        }
        cancel()
        onExit()
    }
}

private struct CapturedPacket {
    let sourceID: String
    let pts: CMTime
    let sampleDuration: CMTime
    let frames: Int
    let sourceFormat: AVAudioFormat
    let formatMetadata: [String: Any]
    let buffers: [Data]
}

private func formatMetadata(_ description: CMFormatDescription, format: AVAudioFormat) throws -> [String: Any] {
    guard let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description) else {
        throw NativeCaptureError.invalidSample("audio sample has no AudioStreamBasicDescription")
    }
    let streamFormat = asbd.pointee
    return [
        "sample_rate": streamFormat.mSampleRate,
        "channels": Int(streamFormat.mChannelsPerFrame),
        "format_id": String(format: "0x%08x", streamFormat.mFormatID),
        "format_flags": streamFormat.mFormatFlags,
        "bits_per_channel": Int(streamFormat.mBitsPerChannel),
        "bytes_per_frame": Int(streamFormat.mBytesPerFrame),
        "frames_per_packet": Int(streamFormat.mFramesPerPacket),
        "interleaved": (streamFormat.mFormatFlags & kAudioFormatFlagIsNonInterleaved) == 0,
        "av_audio_format": format.description,
    ]
}

private func copyAudioBuffers(from sampleBuffer: CMSampleBuffer) throws -> [Data] {
    var sizeNeeded = 0
    var retainedBlockBuffer: CMBlockBuffer?
    let firstStatus = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer,
        bufferListSizeNeededOut: &sizeNeeded,
        bufferListOut: nil,
        bufferListSize: 0,
        blockBufferAllocator: nil,
        blockBufferMemoryAllocator: nil,
        flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
        blockBufferOut: &retainedBlockBuffer
    )
    guard firstStatus == noErr, sizeNeeded > 0 else {
        throw NativeCaptureError.invalidSample("cannot inspect audio buffer list (OSStatus \(firstStatus))")
    }

    let rawList = UnsafeMutableRawPointer.allocate(
        byteCount: sizeNeeded,
        alignment: MemoryLayout<AudioBufferList>.alignment
    )
    defer { rawList.deallocate() }
    let list = rawList.bindMemory(to: AudioBufferList.self, capacity: 1)
    var blockBuffer: CMBlockBuffer?
    let copyStatus = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer,
        bufferListSizeNeededOut: nil,
        bufferListOut: list,
        bufferListSize: sizeNeeded,
        blockBufferAllocator: nil,
        blockBufferMemoryAllocator: nil,
        flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
        blockBufferOut: &blockBuffer
    )
    guard copyStatus == noErr else {
        throw NativeCaptureError.invalidSample("cannot copy audio buffer list (OSStatus \(copyStatus))")
    }

    let audioBuffers = UnsafeMutableAudioBufferListPointer(list)
    var copied: [Data] = []
    copied.reserveCapacity(audioBuffers.count)
    for buffer in audioBuffers {
        guard let data = buffer.mData, buffer.mDataByteSize > 0 else {
            throw NativeCaptureError.invalidSample("audio buffer contains no samples")
        }
        copied.append(Data(bytes: data, count: Int(buffer.mDataByteSize)))
    }
    return copied
}

private func snapshot(_ sampleBuffer: CMSampleBuffer, sourceID: String) throws -> CapturedPacket {
    guard CMSampleBufferDataIsReady(sampleBuffer) else {
        throw NativeCaptureError.invalidSample("audio sample data is not ready")
    }
    let frameCount = CMSampleBufferGetNumSamples(sampleBuffer)
    guard frameCount > 0, frameCount <= maxFramesPerPacket else {
        throw NativeCaptureError.invalidSample("audio packet has invalid frame count \(frameCount)")
    }
    let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
    guard seconds(pts) != nil else {
        throw NativeCaptureError.invalidSample("audio packet has no valid presentation timestamp")
    }
    guard let description = CMSampleBufferGetFormatDescription(sampleBuffer) else {
        throw NativeCaptureError.invalidSample("audio packet has no format description")
    }
    let format = AVAudioFormat(cmAudioFormatDescription: description)
    guard format.sampleRate > 0,
          format.channelCount > 0 else {
        throw NativeCaptureError.invalidSample("audio packet has no usable PCM format")
    }
    return CapturedPacket(
        sourceID: sourceID,
        pts: pts,
        sampleDuration: CMSampleBufferGetDuration(sampleBuffer),
        frames: frameCount,
        sourceFormat: format,
        formatMetadata: try formatMetadata(description, format: format),
        buffers: try copyAudioBuffers(from: sampleBuffer)
    )
}

private final class SegmentWriter {
    private let root: URL
    private let sourceID: String
    private let index: Int
    private let inputFormat: AVAudioFormat
    private let outputFormat: AVAudioFormat
    private let converter: AVAudioConverter
    private var file: AVAudioFile?
    private let partialAudioURL: URL
    private let finalAudioURL: URL
    private let partialTimingURL: URL
    private let finalTimingURL: URL
    private let startPTS: CMTime
    private let startHostPTS: CMTime
    private let startSeconds: Double
    private var frames: Int64 = 0
    private var lastPTS: CMTime
    private var lastHostPTS: CMTime
    private var timings: [[String: Any]] = []

    init(
        root: URL,
        sourceID: String,
        index: Int,
        firstPacket: CapturedPacket,
        firstHostPTS: CMTime,
        startSeconds: Double
    ) throws {
        self.root = root
        self.sourceID = sourceID
        self.index = index
        self.inputFormat = firstPacket.sourceFormat
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: firstPacket.sourceFormat.sampleRate,
            channels: firstPacket.sourceFormat.channelCount,
            interleaved: true
        ), let converter = AVAudioConverter(from: firstPacket.sourceFormat, to: outputFormat) else {
            throw NativeCaptureError.writer("cannot create a float32 CAF converter for \(sourceID)")
        }
        self.outputFormat = outputFormat
        self.converter = converter
        self.startPTS = firstPacket.pts
        self.startHostPTS = firstHostPTS
        self.startSeconds = startSeconds
        self.lastPTS = firstPacket.pts
        self.lastHostPTS = firstHostPTS

        let name = String(format: "%@-%04d", sourceID, index)
        let segmentsDirectory = root.appendingPathComponent("segments", isDirectory: true)
        self.partialAudioURL = segmentsDirectory.appendingPathComponent("\(name).partial.caf")
        self.finalAudioURL = segmentsDirectory.appendingPathComponent("\(name).caf")
        self.partialTimingURL = segmentsDirectory.appendingPathComponent("\(name).timing.partial.json")
        self.finalTimingURL = segmentsDirectory.appendingPathComponent("\(name).timing.json")
        self.file = try AVAudioFile(
            forWriting: partialAudioURL,
            settings: outputFormat.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: true
        )
    }

    var durationSeconds: Double {
        Double(frames) / outputFormat.sampleRate
    }

    var isAtLimit: Bool {
        durationSeconds >= segmentSeconds
    }

    func append(_ packet: CapturedPacket, hostPTS: CMTime, relativePTS: Double) throws {
        guard sameFormat(packet.sourceFormat, inputFormat) else {
            throw NativeCaptureError.writer("source format changed within a segment")
        }
        guard let pcm = AVAudioPCMBuffer(
            pcmFormat: inputFormat,
            frameCapacity: AVAudioFrameCount(packet.frames)
        ) else {
            throw NativeCaptureError.writer("cannot allocate PCM buffer")
        }
        // AVAudioPCMBuffer initially advertises a zero byte length even when
        // it has allocated frame capacity. Set the intended frame length
        // before validating/copying the realtime snapshot's buffers.
        pcm.frameLength = AVAudioFrameCount(packet.frames)
        let destination = UnsafeMutableAudioBufferListPointer(pcm.mutableAudioBufferList)
        guard destination.count == packet.buffers.count else {
            throw NativeCaptureError.writer("input buffer layout changed before conversion")
        }
        for (buffer, bytes) in zip(destination, packet.buffers) {
            guard let destinationBytes = buffer.mData,
                  Int(buffer.mDataByteSize) >= bytes.count else {
                throw NativeCaptureError.writer("PCM destination is smaller than captured buffer")
            }
            bytes.withUnsafeBytes { source in
                guard let sourceBase = source.baseAddress else { return }
                memcpy(destinationBytes, sourceBase, bytes.count)
            }
        }
        let outputCapacity = AVAudioFrameCount(packet.frames + 32)
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputCapacity) else {
            throw NativeCaptureError.writer("cannot allocate converted PCM buffer")
        }
        var suppliedInput = false
        var conversionError: NSError?
        let conversionStatus = converter.convert(to: output, error: &conversionError) { _, status in
            if suppliedInput {
                status.pointee = .noDataNow
                return nil
            }
            suppliedInput = true
            status.pointee = .haveData
            return pcm
        }
        if conversionStatus == .error {
            throw NativeCaptureError.writer("PCM conversion failed: \(conversionError?.localizedDescription ?? "unknown error")")
        }
        guard output.frameLength > 0 else {
            throw NativeCaptureError.writer("PCM conversion produced no frames")
        }
        guard let file else {
            throw NativeCaptureError.writer("segment file is already closed")
        }
        try file.write(from: output)
        frames += Int64(output.frameLength)
        lastPTS = packet.pts
        lastHostPTS = hostPTS
        timings.append([
            "pts": rawTime(packet.pts),
            "host_pts": rawTime(hostPTS),
            "sample_duration": rawTime(packet.sampleDuration),
            "frames": packet.frames,
            "written_frames": Int(output.frameLength),
            "start_seconds": relativePTS,
        ])
    }

    func finalize() throws -> [String: Any] {
        file = nil // closes the CAF before it becomes visible under its final name
        let timingDocument: [String: Any] = [
            "schema_version": 1,
            "source_id": sourceID,
            "segment_index": index,
            "start_raw_pts": rawTime(startPTS),
            "last_raw_pts": rawTime(lastPTS),
            "start_host_pts": rawTime(startHostPTS),
            "last_host_pts": rawTime(lastHostPTS),
            "entries": timings,
        ]
        try writeJSONAtomically(timingDocument, to: partialTimingURL)
        let manager = FileManager.default
        if manager.fileExists(atPath: finalAudioURL.path) || manager.fileExists(atPath: finalTimingURL.path) {
            throw NativeCaptureError.writer("refusing to replace completed segment \(finalAudioURL.lastPathComponent)")
        }
        try manager.moveItem(at: partialTimingURL, to: finalTimingURL)
        try manager.moveItem(at: partialAudioURL, to: finalAudioURL)
        return [
            "path": relativePath(finalAudioURL, from: root),
            "timing_path": relativePath(finalTimingURL, from: root),
            "start_seconds": startSeconds,
            "duration_seconds": durationSeconds,
            "frames": frames,
            "sample_rate": outputFormat.sampleRate,
            "channels": Int(outputFormat.channelCount),
            "format": "caf/pcm_f32le",
            "source_format": formatDictionary(inputFormat),
            "raw_pts_start": rawTime(startPTS),
            "raw_pts_end": rawTime(lastPTS),
            "host_pts_start": rawTime(startHostPTS),
            "host_pts_end": rawTime(lastHostPTS),
        ]
    }

    private func sameFormat(_ lhs: AVAudioFormat, _ rhs: AVAudioFormat) -> Bool {
        lhs.sampleRate == rhs.sampleRate &&
            lhs.channelCount == rhs.channelCount &&
            lhs.isInterleaved == rhs.isInterleaved &&
            lhs.commonFormat == rhs.commonFormat
    }

    private func formatDictionary(_ format: AVAudioFormat) -> [String: Any] {
        [
            "sample_rate": format.sampleRate,
            "channels": Int(format.channelCount),
            "interleaved": format.isInterleaved,
            "common_format": String(describing: format.commonFormat),
            "description": format.description,
        ]
    }
}

private final class SourceState {
    let sourceID: String
    var firstBufferReady = false
    var disabled = false
    var firstPTS: CMTime?
    var firstHostPTS: CMTime?
    var expectedEndSeconds: Double?
    var formatMetadata: [String: Any]?
    var sampleRate: Double?
    var channels: Int?
    var currentSegment: SegmentWriter?
    var nextSegmentIndex = 1
    var segments: [[String: Any]] = []
    var gaps: [[String: Any]] = []
    var receivedFrames: Int64 = 0
    var deviceID: String?

    init(sourceID: String) {
        self.sourceID = sourceID
    }
}

private final class NativeCaptureController: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    private let options: Options
    private let manifestURL: URL
    private let segmentsURL: URL
    private let writerQueue = DispatchQueue(label: "com.lailix.meetingmemory.native-capture.writer", qos: .userInitiated)
    // ScreenCaptureKit invokes the sample handler on this queue. It must stay
    // serial: concurrent snapshots can reach `writerQueue` out of order for
    // one source, turning ordinary production arrival order into a false
    // timestamp discontinuity. The bounded packet budget still limits work
    // before any callback-owned data reaches the writer.
    private let callbackQueue = DispatchQueue(label: "com.lailix.meetingmemory.native-capture.sample", qos: .userInitiated)
    private let packetBudget = PendingPacketBudget(limit: maxPendingPackets)
    private let lifecycleLock = NSLock()
    private let sessionID = UUID().uuidString
    private let startedWallTime = iso8601Now()
    private let hostClockStarted = CMClockGetTime(CMClockGetHostTimeClock())
    private let mic = SourceState(sourceID: "mic")
    private let system = SourceState(sourceID: "system")

    private var stream: SCStream?
    private var streamClock: CMClock?
    private var streamEpoch: CMTime?
    private var hostEpoch: CMTime?
    private var streamClockMapping: [String: Any]?
    private var preEpochPackets: [CapturedPacket] = []
    private var streamStarted = false
    private var stopRequested = false
    private var startupAborted = false
    // `finalizationStarted` prevents duplicate shutdown paths. `finalized` is
    // deliberately published only after every queued packet and segment close
    // has completed, so a consumer can reject an active degraded capture.
    private var finalizationStarted = false
    private var finalized = false
    private var endedWallTime: String?
    private var terminalFailure = false
    private var status = "starting"
    private var errors: [[String: Any]] = []
    private var lifecycleEvents: [[String: Any]] = []
    private var routeEvents: [[String: Any]] = []
    private var readinessTimer: DispatchSourceTimer?
    private var durationTimer: DispatchSourceTimer?
    private var routeTimer: DispatchSourceTimer?
    private var ownerProcessMonitor: OwnerProcessExitMonitor?
    private var ownerMonitorState = "not_configured"
    private var ownerExitObserved = false
    private var lastManifestFlush = Date.distantPast
    private var exitStatus = 0
    private var micCaptureEnabled = true

    init(options: Options) {
        self.options = options
        self.manifestURL = options.outputDirectory.appendingPathComponent(manifestName)
        self.segmentsURL = options.outputDirectory.appendingPathComponent("segments", isDirectory: true)
        super.init()
    }

    var processExitStatus: Int {
        lifecycleLock.lock()
        defer { lifecycleLock.unlock() }
        return exitStatus
    }

    func start() {
        writerQueue.async { [weak self] in
            guard let self else { return }
            // Session creation installs the owner-death monitor before any
            // ScreenCaptureKit or microphone permission work can begin.
            self.prepareSession()
            guard !self.terminalFailure, !self.isStopRequested() else { return }
            Task { [weak self] in
                await self?.configureAndStartStream()
            }
        }
    }

    func requestStop(reason: String) {
        lifecycleLock.lock()
        guard !stopRequested else {
            lifecycleLock.unlock()
            return
        }
        stopRequested = true
        let activeStream = stream
        lifecycleLock.unlock()

        durationTimer?.cancel()
        durationTimer = nil
        if let activeStream {
            activeStream.stopCapture { [weak self] error in
                guard let self else { return }
                self.writerQueue.async {
                    if let error {
                        self.recordError(code: "stop_capture_failed", message: error.localizedDescription, sourceID: nil)
                        self.status = "failed"
                        self.terminalFailure = true
                    }
                    self.packetBudget.close()
                    self.finishWhenPacketsDrain(reason: reason)
                }
            }
        } else {
            writerQueue.async { [weak self] in
                guard let self else { return }
                self.packetBudget.close()
                self.finishWhenPacketsDrain(reason: reason)
            }
        }
    }

    private func prepareSession() {
        do {
            let manager = FileManager.default
            if manager.fileExists(atPath: manifestURL.path) {
                throw NativeCaptureError.writer("session directory already contains \(manifestName): \(manifestURL.path)")
            }
            try manager.createDirectory(at: options.outputDirectory, withIntermediateDirectories: true)
            try manager.createDirectory(at: segmentsURL, withIntermediateDirectories: true)
            try installOwnerProcessMonitor()
            writeManifest(force: true)
        } catch {
            failImmediately(code: "session_initialization_failed", message: error.localizedDescription)
        }
    }

    private func installOwnerProcessMonitor() throws {
        guard let ownerPID = options.ownerPID else {
            ownerMonitorState = "not_configured"
            return
        }
        let monitor = OwnerProcessExitMonitor(ownerPID: ownerPID, queue: writerQueue) { [weak self] in
            self?.ownerProcessExited(ownerPID: ownerPID)
        }
        do {
            try monitor.start()
            ownerProcessMonitor = monitor
            ownerMonitorState = "watching"
        } catch {
            ownerMonitorState = "install_failed"
            throw error
        }
    }

    /// Called on the serial writer queue by the kernel-bound process monitor.
    /// Preserve any completed CAFs and route shutdown through the ordinary
    /// drain/finalize path; never leave a LaunchServices app recording after
    /// the Python owner that launched it has gone away.
    private func ownerProcessExited(ownerPID: pid_t) {
        guard !finalizationStarted, !finalized else { return }
        ownerExitObserved = true
        ownerMonitorState = "owner_exited"
        recordError(
            code: "owner_process_exited",
            message: "owner process \(ownerPID) exited; stopping capture and finalizing completed segments",
            sourceID: nil
        )
        markDegraded()
        writeManifest(force: true)
        DispatchQueue.main.async { [weak self] in
            self?.requestStop(reason: "owner_process_exited")
        }
    }

    private func configureAndStartStream() async {
        guard !isStopRequested() else {
            writerQueue.async { [weak self] in self?.finishWhenPacketsDrain(reason: "stopped_before_start") }
            return
        }
        do {
            let micAllowed = await microphonePermission()
            writerQueue.sync { [weak self] in
                guard let self else { return }
                self.micCaptureEnabled = micAllowed
                if !micAllowed {
                    self.recordError(code: "microphone_permission_denied", message: "microphone permission was not granted; attempting system-only capture", sourceID: "mic")
                    self.markDegraded()
                    self.writeManifest(force: true)
                }
            }

            let content = try await SCShareableContent.current
            guard let display = content.displays.first else {
                throw NativeCaptureError.writer("ScreenCaptureKit did not provide a display for the audio stream filter")
            }
            let configuration = SCStreamConfiguration()
            configuration.capturesAudio = true
            configuration.sampleRate = 48_000
            configuration.channelCount = 2
            configuration.excludesCurrentProcessAudio = true
            configuration.queueDepth = 3
            configuration.captureMicrophone = micAllowed
            if micAllowed {
                let deviceID = AVCaptureDevice.default(for: .audio)?.uniqueID
                configuration.microphoneCaptureDeviceID = deviceID
                writerQueue.async { [weak self] in
                    self?.mic.deviceID = deviceID
                }
            }

            let filter = SCContentFilter(display: display, excludingWindows: [])
            let createdStream = SCStream(filter: filter, configuration: configuration, delegate: self)
            try createdStream.addStreamOutput(self, type: .audio, sampleHandlerQueue: callbackQueue)
            if micAllowed {
                try createdStream.addStreamOutput(self, type: .microphone, sampleHandlerQueue: callbackQueue)
            }
            let shouldStop = attachStreamIfStillRunning(createdStream)
            guard !shouldStop else {
                requestStop(reason: "stopped_before_stream_start")
                return
            }
            try await createdStream.startCapture()
            guard let synchronizationClock = createdStream.synchronizationClock else {
                throw NativeCaptureError.writer("ScreenCaptureKit started without a synchronization clock; refusing to invent a shared timestamp origin")
            }
            let hostClock = CMClockGetHostTimeClock()
            var relativeRate = 0.0
            var streamAnchor = CMTime.invalid
            var hostAnchor = CMTime.invalid
            let mappingStatus = CMSyncGetRelativeRateAndAnchorTime(
                synchronizationClock,
                relativeTo: hostClock,
                relativeRateOut: &relativeRate,
                anchorTimeOut: &streamAnchor,
                relativeToAnchorTimeOut: &hostAnchor
            )
            guard mappingStatus == noErr,
                  relativeRate.isFinite,
                  seconds(streamAnchor) != nil,
                  seconds(hostAnchor) != nil else {
                throw NativeCaptureError.writer("cannot establish ScreenCaptureKit-to-host-clock mapping (OSStatus \(mappingStatus))")
            }
            guard let sharedHostEpoch = usableHostClockEpoch(hostClockStarted) else {
                throw NativeCaptureError.writer("native host-clock startup epoch is zero or invalid; refusing to create ambiguous source offsets")
            }
            writerQueue.async { [weak self] in
                self?.didStartStream(
                    clock: synchronizationClock,
                    streamEpoch: streamAnchor,
                    hostEpoch: sharedHostEpoch,
                    mapping: [
                        "relative_rate": relativeRate,
                        "stream_anchor_raw_pts": rawTime(streamAnchor),
                        "host_anchor_raw_pts": rawTime(hostAnchor),
                        "stream_anchor_state": mappingAnchorState(streamAnchor),
                        "host_anchor_state": mappingAnchorState(hostAnchor),
                        "shared_host_epoch_source": "controller_host_clock_started",
                        "shared_host_epoch_raw": rawTime(sharedHostEpoch),
                    ]
                )
            }
        } catch {
            writerQueue.async { [weak self] in
                self?.failImmediately(code: "stream_start_failed", message: error.localizedDescription)
            }
        }
    }

    private func microphonePermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    continuation.resume(returning: granted)
                }
            }
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    private func attachStreamIfStillRunning(_ candidate: SCStream) -> Bool {
        lifecycleLock.lock()
        defer { lifecycleLock.unlock() }
        if !stopRequested && !startupAborted {
            stream = candidate
            return false
        }
        return true
    }

    private func didStartStream(
        clock: CMClock,
        streamEpoch: CMTime,
        hostEpoch: CMTime,
        mapping: [String: Any]
    ) {
        guard !terminalFailure else { return }
        streamClock = clock
        self.streamEpoch = streamEpoch
        self.hostEpoch = hostEpoch
        streamClockMapping = mapping
        streamStarted = true
        if mapping["stream_anchor_state"] as? String != "usable" ||
            mapping["host_anchor_state"] as? String != "usable" {
            recordLifecycle(
                event: "stream_mapping_anchor_not_used_as_epoch",
                message: "ScreenCaptureKit mapping anchor was zero or unusable; retaining it as raw mapping evidence while using the explicit host-clock startup epoch",
                sourceID: nil
            )
        }
        scheduleReadinessDeadline()
        scheduleDurationDeadline()
        scheduleRoutePolling()
        writeManifest(force: true)
        let heldPackets = preEpochPackets
        preEpochPackets.removeAll(keepingCapacity: true)
        for packet in heldPackets {
            consume(packet)
        }
    }

    private func scheduleReadinessDeadline() {
        let timer = DispatchSource.makeTimerSource(queue: writerQueue)
        timer.schedule(deadline: .now() + readinessSeconds)
        timer.setEventHandler { [weak self] in
            self?.readinessDeadlineReached()
        }
        readinessTimer = timer
        timer.resume()
    }

    private func scheduleDurationDeadline() {
        guard let duration = options.duration else { return }
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + duration)
        timer.setEventHandler { [weak self] in
            self?.requestStop(reason: "duration_elapsed")
        }
        durationTimer = timer
        timer.resume()
    }

    private func scheduleRoutePolling() {
        guard micCaptureEnabled else { return }
        let timer = DispatchSource.makeTimerSource(queue: writerQueue)
        timer.schedule(deadline: .now() + 2, repeating: 2)
        timer.setEventHandler { [weak self] in
            self?.pollMicrophoneRoute()
        }
        routeTimer = timer
        timer.resume()
    }

    private func pollMicrophoneRoute() {
        guard let recordedID = mic.deviceID else { return }
        let currentID = AVCaptureDevice.default(for: .audio)?.uniqueID
        guard currentID != recordedID else { return }
        routeEvents.append([
            "source_id": "mic",
            "event": "default_input_changed",
            "selected_device_id": recordedID,
            "current_default_device_id": currentID ?? NSNull(),
            "wall_time": iso8601Now(),
        ])
        // The stream is configured with the selected device ID. A default-route
        // change is evidence for review, not proof that its existing stream moved.
        writeManifest(force: true)
    }

    private func readinessDeadlineReached() {
        readinessTimer?.cancel()
        readinessTimer = nil
        guard !allRequiredSourcesReady() else { return }
        if mic.firstBufferReady || system.firstBufferReady {
            if !mic.firstBufferReady {
                recordError(code: "source_not_ready", message: "microphone delivered no buffer before readiness deadline; continuing to retain a late first PTS if it arrives", sourceID: "mic")
            }
            if !system.firstBufferReady {
                recordError(code: "source_not_ready", message: "system audio delivered no buffer before readiness deadline; continuing to retain a late first PTS if it arrives", sourceID: "system")
            }
            markDegraded()
            writeManifest(force: true)
            return
        }
        failImmediately(code: "readiness_timeout", message: "neither microphone nor system audio delivered a buffer before the readiness deadline")
    }

    private func allRequiredSourcesReady() -> Bool {
        system.firstBufferReady && (!micCaptureEnabled || mic.firstBufferReady)
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of outputType: SCStreamOutputType) {
        let sourceID: String
        switch outputType {
        case .audio:
            sourceID = "system"
        case .microphone:
            sourceID = "mic"
        default:
            return
        }
        switch packetBudget.claim() {
        case .accepted:
            break
        case .overflow:
            writerQueue.async { [weak self] in
                self?.failAndRequestStop(code: "writer_queue_overflow", message: "the bounded audio writer queue reached \(maxPendingPackets) packets; capture stopped without silently dropping samples", sourceID: sourceID)
            }
            return
        case .closed:
            // A callback racing normal shutdown is intentionally ignored. The
            // capture already stopped accepting packets and will finalize the
            // packets claimed before close; this is not queue overflow.
            return
        }
        do {
            let packet = try snapshot(sampleBuffer, sourceID: sourceID)
            writerQueue.async { [weak self] in
                guard let self else { return }
                defer { self.packetBudget.release() }
                self.consume(packet)
            }
        } catch {
            packetBudget.release()
            writerQueue.async { [weak self] in
                self?.failAndRequestStop(code: "sample_snapshot_failed", message: error.localizedDescription, sourceID: sourceID)
            }
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        let expectedStop = isStopRequested()
        writerQueue.async { [weak self] in
            guard let self, !self.finalizationStarted else { return }
            if expectedStop {
                self.recordLifecycle(event: "stream_stopped_after_requested_stop", message: error.localizedDescription, sourceID: nil)
                self.writeManifest(force: true)
                return
            }
            self.recordError(code: "stream_stopped", message: error.localizedDescription, sourceID: nil)
            self.status = "failed"
            self.terminalFailure = true
            self.packetBudget.close()
            self.finishWhenPacketsDrain(reason: "stream_stopped")
        }
    }

    private func consume(_ packet: CapturedPacket) {
        guard !terminalFailure, !finalizationStarted else { return }
        guard let clock = streamClock, let hostEpoch else {
            // `startCapture()` may deliver an early callback before its async
            // completion exposes synchronizationClock. Keep a small bounded
            // pre-epoch buffer instead of fabricating a host-time conversion.
            guard preEpochPackets.count < maxPendingPackets else {
                failAndRequestStop(code: "timeline_epoch_unavailable", message: "received more than \(maxPendingPackets) audio packets before ScreenCaptureKit exposed its shared synchronization clock", sourceID: packet.sourceID)
                return
            }
            preEpochPackets.append(packet)
            return
        }
        let hostPTS = CMSyncConvertTime(packet.pts, from: clock, to: CMClockGetHostTimeClock())
        guard let relativePTS = hostRelativeSeconds(hostPTS, epoch: hostEpoch) else {
            failAndRequestStop(code: "timeline_epoch_invalid", message: "received audio with an invalid host-clock epoch or presentation timestamp", sourceID: packet.sourceID)
            return
        }
        let sourceDuration: CMTime
        if seconds(packet.sampleDuration) != nil {
            sourceDuration = packet.sampleDuration
        } else {
            sourceDuration = CMTime(
                seconds: Double(packet.frames) / packet.sourceFormat.sampleRate,
                preferredTimescale: 1_000_000_000
            )
        }
        let sourceEndPTS = CMTimeAdd(packet.pts, sourceDuration)
        let hostEndPTS = CMSyncConvertTime(sourceEndPTS, from: clock, to: CMClockGetHostTimeClock())
        guard let relativeEnd = hostRelativeSeconds(hostEndPTS, epoch: hostEpoch) else {
            failAndRequestStop(code: "timeline_end_invalid", message: "could not convert an audio packet end time onto the host clock", sourceID: packet.sourceID)
            return
        }
        guard relativeEnd >= relativePTS else {
            failAndRequestStop(code: "timeline_non_forwarding_clock", message: "host-clock conversion moved an audio packet end before its start", sourceID: packet.sourceID)
            return
        }

        consumeResolved(packet, hostPTS: hostPTS, relativePTS: relativePTS, relativeEnd: relativeEnd)
    }

    /// The writer-side half of packet ingestion. Keeping it separate from the
    /// Core Media clock conversion lets the self-test exercise the same gap,
    /// segment, CAF, timing-sidecar, and manifest path as production packets.
    private func consumeResolved(
        _ packet: CapturedPacket,
        hostPTS: CMTime,
        relativePTS: Double,
        relativeEnd: Double
    ) {
        guard !terminalFailure, !finalizationStarted else { return }
        let source = packet.sourceID == "mic" ? mic : system
        guard !source.disabled else { return }

        if relativePTS < 0, source.firstPTS == nil {
            // Keep rather than clamp a buffer which predates our requested
            // common epoch. An assembler can make that explicit instead of
            // silently losing media near startup.
            lifecycleEvents.append([
                "event": "pre_epoch_first_packet",
                "source_id": packet.sourceID,
                "start_seconds": relativePTS,
                "raw_pts": rawTime(packet.pts),
                "wall_time": iso8601Now(),
            ])
        }

        if let existing = source.formatMetadata,
           !sameFormatMetadata(existing, packet.formatMetadata) {
            source.disabled = true
            recordError(code: "source_format_changed", message: "\(packet.sourceID) changed sample rate, channel layout, or PCM format; retaining completed segments and stopping this source to avoid a silent timeline misalignment", sourceID: packet.sourceID)
            routeEvents.append([
                "source_id": packet.sourceID,
                "event": "format_changed",
                "previous_format": existing,
                "new_format": packet.formatMetadata,
                "raw_pts": rawTime(packet.pts),
                "wall_time": iso8601Now(),
            ])
            closeCurrentSegment(source)
            markDegraded()
            writeManifest(force: true)
            return
        }
        if source.formatMetadata == nil {
            source.formatMetadata = packet.formatMetadata
            source.sampleRate = packet.sourceFormat.sampleRate
            source.channels = Int(packet.sourceFormat.channelCount)
        }

        if !source.firstBufferReady {
            source.firstBufferReady = true
            source.firstPTS = packet.pts
            source.firstHostPTS = hostPTS
            if relativePTS > gapToleranceSeconds {
                appendGap(
                    source,
                    begin: 0,
                    end: relativePTS,
                    reason: "initial_no_samples",
                    startRawPTS: nil,
                    endRawPTS: packet.pts
                )
                // A late first buffer is the startup defect this boundary is
                // designed to expose. It remains degraded even if both tracks
                // later become ready; the saved coordinate/gap is authoritative.
                markDegraded()
            }
            if allRequiredSourcesReady() && status == "starting" {
                status = "recording"
                readinessTimer?.cancel()
                readinessTimer = nil
            }
            writeManifest(force: true)
        }

        if let expected = source.expectedEndSeconds {
            let delta = relativePTS - expected
            if delta > gapToleranceSeconds {
                appendGap(
                    source,
                    begin: expected,
                    end: relativePTS,
                    reason: "timestamp_gap",
                    startRawPTS: nil,
                    endRawPTS: packet.pts
                )
                markDegraded()
                // A CAF holds contiguous samples. Writing this late packet to
                // the existing file would collapse the declared wall-clock
                // hole when a downstream reader concatenates the file's
                // frames. Finalize the earlier immutable segment first, then
                // let the append path below open a new segment at the late
                // packet's shared timestamp. The <=20ms jitter policy remains
                // deliberately unchanged and stays in one segment.
                closeCurrentSegment(source)
                guard !terminalFailure else {
                    writeManifest(force: true)
                    return
                }
            } else if delta < -backwardsToleranceSeconds {
                source.disabled = true
                recordError(code: "non_monotonic_pts", message: "\(packet.sourceID) delivered a presentation timestamp \(String(format: "%.3f", -delta))s behind its expected timeline; source stopped rather than writing ambiguous overlapping audio", sourceID: packet.sourceID)
                closeCurrentSegment(source)
                markDegraded()
                writeManifest(force: true)
                return
            }
        }

        do {
            if source.currentSegment == nil {
                source.currentSegment = try SegmentWriter(
                    root: options.outputDirectory,
                    sourceID: source.sourceID,
                    index: source.nextSegmentIndex,
                    firstPacket: packet,
                    firstHostPTS: hostPTS,
                    startSeconds: relativePTS
                )
            }
            guard let segment = source.currentSegment else { return }
            try segment.append(packet, hostPTS: hostPTS, relativePTS: relativePTS)
            source.receivedFrames += Int64(packet.frames)
            source.expectedEndSeconds = max(source.expectedEndSeconds ?? relativePTS, relativeEnd)
            if segment.isAtLimit {
                closeCurrentSegment(source)
            }
            writeManifest(force: false)
        } catch {
            failAndRequestStop(code: "segment_write_failed", message: error.localizedDescription, sourceID: packet.sourceID)
        }
    }

    private func closeCurrentSegment(_ source: SourceState) {
        guard let segment = source.currentSegment else { return }
        do {
            let record = try segment.finalize()
            source.segments.append(record)
            source.nextSegmentIndex += 1
            source.currentSegment = nil
            // The CAF and timing sidecar are now immutable and recoverable.
            // Checkpoint their manifest entry immediately rather than letting
            // the normal one-second progress throttle leave a closed segment
            // invisible after a process crash. This remains on the serial
            // writer queue, never an audio callback.
            writeManifest(force: true)
        } catch {
            failAndRequestStop(code: "segment_finalize_failed", message: error.localizedDescription, sourceID: source.sourceID)
        }
    }

    private func appendGap(
        _ source: SourceState,
        begin: Double,
        end: Double,
        reason: String,
        startRawPTS: CMTime?,
        endRawPTS: CMTime?
    ) {
        guard end > begin else { return }
        var gap: [String: Any] = [
            "begin_seconds": begin,
            "end_seconds": end,
            "duration_seconds": end - begin,
            "reason": reason,
        ]
        if let startRawPTS { gap["raw_pts_start"] = rawTime(startRawPTS) }
        if let endRawPTS { gap["raw_pts_end"] = rawTime(endRawPTS) }
        source.gaps.append(gap)
    }

    private func sameFormatMetadata(_ lhs: [String: Any], _ rhs: [String: Any]) -> Bool {
        let keys = ["sample_rate", "channels", "format_id", "format_flags", "bits_per_channel", "bytes_per_frame", "interleaved"]
        return keys.allSatisfy { String(describing: lhs[$0]) == String(describing: rhs[$0]) }
    }

    private func markDegraded() {
        if status != "failed" && status != "complete" {
            status = "degraded"
        }
    }

    private func recordError(code: String, message: String, sourceID: String?) {
        var error: [String: Any] = [
            "code": code,
            "message": message,
            "wall_time": iso8601Now(),
        ]
        if let sourceID { error["source_id"] = sourceID }
        errors.append(error)
    }

    private func recordLifecycle(event: String, message: String, sourceID: String?) {
        var entry: [String: Any] = [
            "event": event,
            "message": message,
            "wall_time": iso8601Now(),
        ]
        if let sourceID { entry["source_id"] = sourceID }
        lifecycleEvents.append(entry)
    }

    private func failAndRequestStop(code: String, message: String, sourceID: String?) {
        guard !terminalFailure else { return }
        recordError(code: code, message: message, sourceID: sourceID)
        status = "failed"
        terminalFailure = true
        writeManifest(force: true)
        DispatchQueue.main.async { [weak self] in
            self?.requestStop(reason: code)
        }
    }

    private func failImmediately(code: String, message: String) {
        guard !finalizationStarted else { return }
        recordError(code: code, message: message, sourceID: nil)
        status = "failed"
        terminalFailure = true
        packetBudget.close()
        lifecycleLock.lock()
        startupAborted = true
        lifecycleLock.unlock()
        writeManifest(force: true)
        // A readiness/startup failure may occur after SCStream has begun. Route
        // every fatal path through requestStop so the stream is stopped before
        // writers publish their terminal manifest; if no stream exists yet it
        // still schedules safe finalization and prevents later startup work.
        DispatchQueue.main.async { [weak self] in
            self?.requestStop(reason: code)
        }
    }

    private func finishWhenPacketsDrain(reason: String) {
        guard !finalizationStarted else { return }
        if packetBudget.pendingCount() > 0 {
            writerQueue.asyncAfter(deadline: .now() + .milliseconds(10)) { [weak self] in
                self?.finishWhenPacketsDrain(reason: reason)
            }
            return
        }
        finalizationStarted = true
        readinessTimer?.cancel()
        readinessTimer = nil
        routeTimer?.cancel()
        routeTimer = nil
        ownerProcessMonitor?.cancel()
        ownerProcessMonitor = nil
        if options.ownerPID != nil, !ownerExitObserved {
            ownerMonitorState = "stopped"
        }
        closeCurrentSegment(mic)
        closeCurrentSegment(system)

        if !terminalFailure {
            if !mic.firstBufferReady && !system.firstBufferReady {
                recordError(code: "stopped_before_first_buffer", message: "capture stopped before either source delivered audio", sourceID: nil)
                status = "failed"
                terminalFailure = true
            } else if status == "starting" {
                markDegraded()
            } else if status == "recording" {
                status = "complete"
            }
        }
        recordLifecycle(event: "capture_finalized", message: "capture finalized after \(reason)", sourceID: nil)
        endedWallTime = iso8601Now()
        finalized = true
        writeManifest(force: true)

        lifecycleLock.lock()
        exitStatus = terminalFailure ? 1 : 0
        let finalExitStatus = exitStatus
        lifecycleLock.unlock()
        // CFRunLoopStop alone did not reliably terminate an NSApplication
        // launched through LaunchServices after a declined TCC prompt: it left
        // an orphaned, already-finalized helper alive. All segment writers and
        // the atomic terminal manifest are complete above, so terminate on the
        // main queue rather than relying on RunLoop.main.run() to return.
        DispatchQueue.main.async {
            exit(Int32(finalExitStatus))
        }
    }

    private func writeManifest(force: Bool) {
        let now = Date()
        guard force || now.timeIntervalSince(lastManifestFlush) >= 1 else { return }
        lastManifestFlush = now
        do {
            try writeJSONAtomically(manifestDocument(), to: manifestURL)
            emitStatusJSON()
        } catch {
            // There is no safe recovery if the manifest itself cannot be made
            // durable. Keep the in-memory error and stop at the next opportunity.
            terminalFailure = true
            status = "failed"
            errors.append([
                "code": "manifest_write_failed",
                "message": error.localizedDescription,
                "wall_time": iso8601Now(),
            ])
            DispatchQueue.main.async { [weak self] in
                self?.requestStop(reason: "manifest_write_failed")
            }
        }
    }

    private func manifestDocument() -> [String: Any] {
        let tracks = [trackDocument(mic), trackDocument(system)]
        let buildProvenance = readBuildProvenance()
        let finalDuration = tracks
            .compactMap { ($0["end_seconds"] as? Double) }
            .max() ?? 0
        var document: [String: Any] = [
            "schema_version": 2,
            "capture_backend": "screencapturekit",
            "build_provenance": buildProvenance,
            "status": status,
            "session_id": sessionID,
            "session_token": options.sessionToken,
            "started_wall_time": startedWallTime,
            "ended_wall_time": (endedWallTime as Any?) ?? NSNull(),
            "updated_wall_time": iso8601Now(),
            "finalized": finalized,
            "timeline": "host_clock",
            "timeline_metadata": [
                "host_clock_started_raw": rawTime(hostClockStarted),
                "epoch_raw_pts": (streamEpoch.map(rawTime) as Any?) ?? NSNull(),
                "host_clock_epoch_raw": (hostEpoch.map(rawTime) as Any?) ?? NSNull(),
                "epoch_clock": "SCStream.synchronizationClock",
                "host_clock_epoch_source": "controller_host_clock_started",
                "mapping_anchor_policy": "retained as stream-to-host conversion evidence; never selected as the shared host-clock epoch",
                "stream_to_host_clock": (streamClockMapping as Any?) ?? NSNull(),
                "start_seconds_definition": "CMSampleBuffer.presentationTimeStamp converted from SCStream.synchronizationClock to host clock, minus the one shared host_clock_epoch_raw; individual tracks are never independently zeroed",
            ],
            "process": [
                "pid": ProcessInfo.processInfo.processIdentifier,
                "bundle_identifier": (Bundle.main.bundleIdentifier as Any?) ?? NSNull(),
                "executable": (CommandLine.arguments.first as Any?) ?? NSNull(),
            ],
            "owner_process": [
                "pid": (options.ownerPID.map { Int($0) } as Any?) ?? NSNull(),
                "monitor": ownerMonitorState,
                "monitor_kind": "kqueue_evfilt_proc_note_exit",
                "exit_observed": ownerExitObserved,
            ],
            "readiness": [
                "mic": mic.firstBufferReady,
                "system": system.firstBufferReady,
                "all_required_sources_ready": allRequiredSourcesReady(),
                "ready_deadline_seconds": readinessSeconds,
            ],
            "tracks": tracks,
            "events": lifecycleEvents,
            "route_events": routeEvents,
            "errors": errors,
            "final_duration_seconds": finalDuration,
        ]
        if options.selfTest || options.selfTestOwnerExit {
            document["synthetic"] = true
        }
        return document
    }

    private func trackDocument(_ source: SourceState) -> [String: Any] {
        let segmentEnd = source.segments.compactMap { segment -> Double? in
            guard let start = segment["start_seconds"] as? Double,
                  let duration = segment["duration_seconds"] as? Double else { return nil }
            return start + duration
        }.max() ?? 0
        let gapEnd = source.gaps.compactMap { $0["end_seconds"] as? Double }.max() ?? 0
        var track: [String: Any] = [
            "source_id": source.sourceID,
            "first_buffer_ready": source.firstBufferReady,
            "segments": source.segments,
            "gaps": source.gaps,
            "received_frames": source.receivedFrames,
            "end_seconds": max(segmentEnd, gapEnd),
        ]
        if let firstPTS = source.firstPTS { track["first_raw_pts"] = rawTime(firstPTS) }
        if let firstHostPTS = source.firstHostPTS { track["first_host_pts"] = rawTime(firstHostPTS) }
        if let formatMetadata = source.formatMetadata { track["source_format"] = formatMetadata }
        if let sampleRate = source.sampleRate { track["sample_rate"] = sampleRate }
        if let channels = source.channels { track["channels"] = channels }
        if let deviceID = source.deviceID { track["device_id"] = deviceID }
        if source.disabled { track["disabled"] = true }
        return track
    }

    private func emitStatusJSON() {
        let event: [String: Any] = [
            "event": "status",
            "status": status,
            "session_id": sessionID,
            "manifest_path": manifestURL.path,
            "first_buffer_ready": ["mic": mic.firstBufferReady, "system": system.firstBufferReady],
            "degraded": status == "degraded",
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: event, options: [.sortedKeys]) else { return }
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0A]))
    }

    /// Exercise production-side packet handling without ScreenCaptureKit or a
    /// TCC prompt. The packets enter after clock conversion, which is the
    /// only part that cannot exist without a live SCStream; gap detection,
    /// CAF conversion, segment finalization, timing sidecars, and manifests
    /// are exactly the same writer path used by captured packets.
    fileprivate func runSyntheticTimelineSelfTest() throws {
        var result: Result<Void, Error> = .success(())
        writerQueue.sync {
            do {
                try self.runSyntheticTimelineSelfTestOnWriterQueue()
            } catch {
                result = .failure(error)
            }
        }
        try result.get()
    }

    private func runSyntheticTimelineSelfTestOnWriterQueue() throws {
        let manager = FileManager.default
        guard !manager.fileExists(atPath: manifestURL.path) else {
            throw NativeCaptureError.writer("self-test output directory already contains \(manifestName)")
        }
        try manager.createDirectory(at: options.outputDirectory, withIntermediateDirectories: true)
        try manager.createDirectory(at: segmentsURL, withIntermediateDirectories: true)

        let timescale: CMTimeScale = 1_000_000_000
        // Reproduce the live ScreenCaptureKit behavior where the mapping
        // anchors are both zero even though converted sample PTS values are
        // host-clock times around 220,000 seconds after boot. The shared epoch
        // must be the explicit native host-clock start, never either anchor.
        let zeroMappingAnchor = CMTime.zero
        let origin = CMTime(value: 220_792_444_913_416, timescale: timescale)
        guard usableHostClockEpoch(zeroMappingAnchor) == nil,
              let selectedHostEpoch = usableHostClockEpoch(origin),
              mappingAnchorState(zeroMappingAnchor) == "zero_or_nonpositive" else {
            throw NativeCaptureError.writer("synthetic zero mapping anchor did not require an explicit host-clock epoch")
        }
        func time(at relativeSeconds: Double) -> CMTime {
            CMTimeAdd(selectedHostEpoch, CMTime(seconds: relativeSeconds, preferredTimescale: timescale))
        }
        func approximatelyEqual(_ value: Double?, _ expected: Double) -> Bool {
            guard let value else { return false }
            return abs(value - expected) < 0.000_001
        }
        let observedSystemPTS = CMTime(value: 220_792_624_044_458, timescale: timescale)
        let observedMicPTS = CMTime(value: 10_598_049_999, timescale: 48_000)
        guard approximatelyEqual(hostRelativeSeconds(observedSystemPTS, epoch: selectedHostEpoch), 0.179_131_042),
              approximatelyEqual(hostRelativeSeconds(observedMicPTS, epoch: selectedHostEpoch), 0.263_399_084) else {
            throw NativeCaptureError.writer("synthetic zero mapping anchor produced an implausible host-clock offset")
        }
        func ingest(_ packet: CapturedPacket) throws {
            guard let relativePTS = hostRelativeSeconds(packet.pts, epoch: selectedHostEpoch),
                  let relativeEnd = hostRelativeSeconds(
                      CMTimeAdd(packet.pts, packet.sampleDuration),
                      epoch: selectedHostEpoch
                  ) else {
                throw NativeCaptureError.writer("synthetic packet could not be projected onto the explicit host-clock epoch")
            }
            consumeResolved(packet, hostPTS: packet.pts, relativePTS: relativePTS, relativeEnd: relativeEnd)
        }

        // This is intentionally a shared origin with a pre-epoch microphone
        // packet and a late system source. The third system packet introduces
        // a timestamp gap after a real CAF has already been opened.
        streamEpoch = zeroMappingAnchor
        hostEpoch = selectedHostEpoch
        streamClockMapping = [
            "synthetic": true,
            "relative_rate": 1.0,
            "stream_anchor_raw_pts": rawTime(zeroMappingAnchor),
            "host_anchor_raw_pts": rawTime(zeroMappingAnchor),
            "stream_anchor_state": mappingAnchorState(zeroMappingAnchor),
            "host_anchor_state": mappingAnchorState(zeroMappingAnchor),
            "shared_host_epoch_source": "controller_host_clock_started",
            "shared_host_epoch_raw": rawTime(selectedHostEpoch),
        ]
        micCaptureEnabled = true
        func rawTimeValue(_ value: Any?) -> Int64? {
            guard let raw = value as? [String: Any] else { return nil }
            if let value = raw["value"] as? Int64 { return value }
            if let value = raw["value"] as? NSNumber { return value.int64Value }
            return nil
        }
        let syntheticTimeline = manifestDocument()["timeline_metadata"] as? [String: Any]
        guard syntheticTimeline?["host_clock_epoch_source"] as? String == "controller_host_clock_started",
              rawTimeValue(syntheticTimeline?["host_clock_epoch_raw"]) == selectedHostEpoch.value,
              rawTimeValue(syntheticTimeline?["epoch_raw_pts"]) == 0 else {
            throw NativeCaptureError.writer("synthetic zero mapping anchor was not represented with the explicit host-clock epoch")
        }

        let micFirst = try syntheticPacket(sourceID: "mic", pts: time(at: -0.10), frequency: 440)
        try ingest(micFirst)
        let micSecond = try syntheticPacket(sourceID: "mic", pts: time(at: -0.09), frequency: 440)
        try ingest(micSecond)

        let systemFirst = try syntheticPacket(sourceID: "system", pts: time(at: 3.25), frequency: 880)
        try ingest(systemFirst)
        let systemSecond = try syntheticPacket(sourceID: "system", pts: time(at: 3.26), frequency: 880)
        try ingest(systemSecond)
        let systemAfterGap = try syntheticPacket(sourceID: "system", pts: time(at: 4.00), frequency: 880)
        try ingest(systemAfterGap)

        guard !terminalFailure else {
            throw NativeCaptureError.writer("synthetic production packet path reported a terminal failure")
        }
        closeCurrentSegment(mic)
        closeCurrentSegment(system)
        guard !terminalFailure else {
            throw NativeCaptureError.writer("synthetic production packet path could not finalize its CAF segments")
        }
        writeManifest(force: true)
        guard system.segments.count == 2,
              let first = system.segments.first,
              let second = system.segments.last,
              let firstStart = first["start_seconds"] as? Double,
              let firstDuration = first["duration_seconds"] as? Double,
              let secondStart = second["start_seconds"] as? Double,
              let firstPath = first["path"] as? String,
              let secondPath = second["path"] as? String,
              manager.fileExists(atPath: options.outputDirectory.appendingPathComponent(firstPath).path),
              manager.fileExists(atPath: options.outputDirectory.appendingPathComponent(secondPath).path),
              approximatelyEqual(firstStart, 3.25),
              firstDuration > 0,
              approximatelyEqual(secondStart, 4.00),
              secondStart > firstStart + firstDuration,
              system.gaps.contains(where: {
                  ($0["reason"] as? String) == "timestamp_gap" &&
                      approximatelyEqual($0["begin_seconds"] as? Double, 3.27) &&
                      approximatelyEqual($0["end_seconds"] as? Double, 4.00)
              }) else {
            throw NativeCaptureError.writer("synthetic timestamp gap did not rotate to a new CAF segment")
        }

        status = "degraded"
        recordLifecycle(
            event: "synthetic_timestamp_gap_segment_rotation_verified",
            message: "late system packet was written to a new CAF segment after a declared timestamp gap",
            sourceID: "system"
        )
        endedWallTime = iso8601Now()
        finalized = true
        writeManifest(force: true)
    }

    private func isStopRequested() -> Bool {
        lifecycleLock.lock()
        defer { lifecycleLock.unlock() }
        return stopRequested || startupAborted
    }
}

private func syntheticPacket(sourceID: String, pts: CMTime, frequency: Float) throws -> CapturedPacket {
    let sampleRate = 48_000.0
    let frames = 480
    guard let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: sampleRate,
        channels: 1,
        interleaved: true
    ) else {
        throw NativeCaptureError.writer("cannot allocate synthetic PCM format")
    }
    var samples = [Float](repeating: 0, count: frames)
    for frame in 0..<frames {
        samples[frame] = sinf(2 * .pi * frequency * Float(frame) / Float(sampleRate))
    }
    let audioData = samples.withUnsafeBufferPointer { buffer in
        Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
    }
    return CapturedPacket(
        sourceID: sourceID,
        pts: pts,
        sampleDuration: CMTime(seconds: Double(frames) / sampleRate, preferredTimescale: 1_000_000_000),
        frames: frames,
        sourceFormat: format,
        formatMetadata: [
            "sample_rate": sampleRate,
            "channels": 1,
            "format_id": "synthetic_lpcm_f32le",
            "format_flags": 0,
            "bits_per_channel": 32,
            "bytes_per_frame": 4,
            "frames_per_packet": 1,
            "interleaved": true,
            "av_audio_format": format.description,
        ],
        buffers: [audioData]
    )
}

private func runSelfTest(options: Options) throws {
    let controller = NativeCaptureController(options: options)
    try controller.runSyntheticTimelineSelfTest()
}

private func runOwnerExitMonitorSelfTest(options: Options) throws {
    let manager = FileManager.default
    let manifestURL = options.outputDirectory.appendingPathComponent(manifestName)
    guard !manager.fileExists(atPath: manifestURL.path) else {
        throw NativeCaptureError.writer("owner-exit self-test output directory already contains \(manifestName)")
    }
    try manager.createDirectory(at: options.outputDirectory, withIntermediateDirectories: true)

    // The helper process is local and inert. It exists only long enough to
    // prove that EVFILT_PROC/NOTE_EXIT wakes the native monitor without any
    // ScreenCaptureKit stream, audio device, or permission request.
    let owner = Process()
    owner.executableURL = URL(fileURLWithPath: "/bin/sh")
    owner.arguments = ["-c", "sleep 0.2"]
    try owner.run()
    let ownerPID = owner.processIdentifier
    let exitObserved = DispatchSemaphore(value: 0)
    let monitor = OwnerProcessExitMonitor(
        ownerPID: ownerPID,
        queue: DispatchQueue(label: "com.lailix.meetingmemory.native-capture.owner-self-test")
    ) {
        exitObserved.signal()
    }
    defer {
        monitor.cancel()
        if owner.isRunning {
            owner.terminate()
            owner.waitUntilExit()
        }
    }
    try monitor.start()
    guard exitObserved.wait(timeout: .now() + 3) == .success else {
        throw NativeCaptureError.writer("owner process exit monitor did not receive NOTE_EXIT during self-test")
    }
    owner.waitUntilExit()

    let now = iso8601Now()
    try writeJSONAtomically([
        "schema_version": 2,
        "capture_backend": "screencapturekit",
        "build_provenance": readBuildProvenance(),
        "synthetic": true,
        "status": "complete",
        "session_id": UUID().uuidString,
        "session_token": options.sessionToken,
        "started_wall_time": now,
        "ended_wall_time": now,
        "updated_wall_time": now,
        "finalized": true,
        "timeline": "host_clock",
        "process": [
            "pid": ProcessInfo.processInfo.processIdentifier,
            "executable": (CommandLine.arguments.first as Any?) ?? NSNull(),
        ],
        "owner_process": [
            "pid": Int(ownerPID),
            "monitor": "owner_exited",
            "monitor_kind": "kqueue_evfilt_proc_note_exit",
            "exit_observed": true,
        ],
        "readiness": ["mic": false, "system": false, "all_required_sources_ready": false],
        "tracks": [],
        "events": [[
            "event": "synthetic_owner_process_exit_monitor_verified",
            "message": "kqueue EVFILT_PROC NOTE_EXIT was observed for a synthetic owner process",
            "wall_time": now,
        ]],
        "route_events": [],
        "errors": [],
        "final_duration_seconds": 0.0,
    ], to: manifestURL)
    let event: [String: Any] = [
        "event": "status",
        "status": "complete",
        "manifest_path": manifestURL.path,
        "synthetic": true,
    ]
    let data = try JSONSerialization.data(withJSONObject: event, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
}

do {
    let options = try Options.parse(CommandLine.arguments)
    if options.selfTest {
        try runSelfTest(options: options)
        exit(0)
    }
    if options.selfTestOwnerExit {
        try runOwnerExitMonitorSelfTest(options: options)
        exit(0)
    }
    guard #available(macOS 15.0, *) else {
        throw NativeCaptureError.usage("native_capture requires macOS 15 or later")
    }
    _ = NSApplication.shared
    NSApp.setActivationPolicy(.regular)
    signal(SIGINT, SIG_IGN)
    let controller = NativeCaptureController(options: options)
    let signalSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
    signalSource.setEventHandler {
        controller.requestStop(reason: "sigint")
    }
    signalSource.resume()
    controller.start()
    RunLoop.main.run()
    _ = signalSource
    exit(Int32(controller.processExitStatus))
} catch {
    let event: [String: Any] = [
        "event": "fatal",
        "status": "failed",
        "error": error.localizedDescription,
    ]
    if let data = try? JSONSerialization.data(withJSONObject: event, options: [.sortedKeys]) {
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0A]))
    }
    FileHandle.standardError.write(Data(("native_capture: \(error.localizedDescription)\n").utf8))
    exit(2)
}
