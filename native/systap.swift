// systap: stream all system output audio to stdout as mono float32 little-endian at the tap's
// sample rate, using a CoreAudio process tap (macOS 14.4+). No virtual driver needed.
//
// The tap delivers nothing while the system is silent, so a wall-clock timer pads the gap with
// zeros: the stream stays aligned with the microphone track and its length equals real time.
//
// Usage: systap            (runs until killed; prints "rate=<Hz>" to stderr once running)
import CoreAudio
import Foundation

func fail(_ msg: String, _ status: OSStatus = 0) -> Never {
    FileHandle.standardError.write("systap: \(msg)\(status != 0 ? " (OSStatus \(status))" : "")\n".data(using: .utf8)!)
    exit(1)
}
func check(_ s: OSStatus, _ what: String) { if s != noErr { fail(what, s) } }

let desc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
desc.uuid = UUID()
desc.name = "meeting-notes system audio"
desc.muteBehavior = .unmuted   // the user keeps hearing the meeting

var tapID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateProcessTap(desc, &tapID), "cannot create the system audio tap (permission or macOS < 14.4?)")

var fmt = AudioStreamBasicDescription()
var addr = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat,
                                      mScope: kAudioObjectPropertyScopeGlobal,
                                      mElement: kAudioObjectPropertyElementMain)
var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
check(AudioObjectGetPropertyData(tapID, &addr, 0, nil, &size, &fmt), "cannot read the tap format")
let channels = max(1, Int(fmt.mChannelsPerFrame))
let rate = fmt.mSampleRate

let aggDict: [String: Any] = [
    kAudioAggregateDeviceNameKey: "meeting-notes tap",
    kAudioAggregateDeviceUIDKey: UUID().uuidString,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceTapListKey: [[kAudioSubTapUIDKey: desc.uuid.uuidString,
                                       kAudioSubTapDriftCompensationKey: true]],
]
var aggID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateAggregateDevice(aggDict as CFDictionary, &aggID), "cannot create the aggregate device")

let lock = NSLock()
var pending = [Float]()   // mono samples not written yet

var procID: AudioDeviceIOProcID?
check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, DispatchQueue(label: "systap.io")) { _, inData, _, _, _ in
    let list = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inData))
    for b in list {
        guard let p = b.mData else { continue }
        let n = Int(b.mDataByteSize) / MemoryLayout<Float>.size
        let frames = n / channels
        let src = p.assumingMemoryBound(to: Float.self)
        var mono = [Float](repeating: 0, count: frames)
        for i in 0..<frames {
            var s: Float = 0
            for c in 0..<channels { s += src[i * channels + c] }
            mono[i] = s / Float(channels)
        }
        lock.lock(); pending.append(contentsOf: mono); lock.unlock()
    }
}, "cannot create the IO proc")

func cleanup() {
    if let p = procID { AudioDeviceStop(aggID, p); AudioDeviceDestroyIOProcID(aggID, p) }
    AudioHardwareDestroyAggregateDevice(aggID)
    AudioHardwareDestroyProcessTap(tapID)
}
signal(SIGTERM, SIG_IGN); signal(SIGINT, SIG_IGN)
let sigs = [SIGTERM, SIGINT].map { sig -> DispatchSourceSignal in
    let s = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    s.setEventHandler { cleanup(); exit(0) }
    s.resume()
    return s
}
_ = sigs

check(AudioDeviceStart(aggID, procID), "cannot start the tap")
FileHandle.standardError.write("rate=\(Int(rate))\n".data(using: .utf8)!)

let t0 = DispatchTime.now().uptimeNanoseconds
var written = 0
let timer = DispatchSource.makeTimerSource(queue: .main)
timer.schedule(deadline: .now(), repeating: .milliseconds(20))
timer.setEventHandler {
    lock.lock(); let chunk = pending; pending.removeAll(keepingCapacity: true); lock.unlock()
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
    let target = Int(elapsed * rate)
    let gap = target - (written + chunk.count)
    var out = [Float]()
    if gap > Int(rate / 20) { out = [Float](repeating: 0, count: gap) }   // > 50 ms behind: silence
    out.append(contentsOf: chunk)
    if out.isEmpty { return }
    written += out.count
    out.withUnsafeBufferPointer { buf in
        var off = 0
        let total = buf.count * MemoryLayout<Float>.size
        let raw = UnsafeRawPointer(buf.baseAddress!)
        while off < total {
            let n = write(1, raw + off, total - off)
            if n <= 0 { cleanup(); exit(0) }   // reader went away
            off += n
        }
    }
}
timer.resume()
dispatchMain()
