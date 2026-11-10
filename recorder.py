"""Record one or more input devices (e.g. microphone + BlackHole system audio)
and mix them into a single 16 kHz mono track for Whisper."""

import queue
import threading
import time
from math import gcd
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

import speakers

TARGET_SR = 16000
STALL_SEC = 2.0   # a track this far behind the others is padded with silence, not waited for
SYNC_SEC = 30.0   # how often a track's WAV header is refreshed on disk
STOP_TIMEOUT_SEC = 5.0   # how long to wait for an audio stream to close before abandoning it


def list_input_devices():
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devices.append({
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "samplerate": int(d["default_samplerate"]),
            })
    return devices


class _Track:
    """One input device, streamed to a temporary WAV file on disk."""

    def __init__(self, device_index: int, path: Path):
        info = sd.query_devices(device_index)
        self.name = info["name"]
        self.path = path
        self.sr = int(info["default_samplerate"])
        self.channels = max(1, min(2, info["max_input_channels"]))
        self.role = speakers.role_of(self.name)   # "mic" or "system"
        self.env = speakers.RunningEnvelope()     # loudness over time, fed by drain_mixed()
        self.buf16 = np.zeros(0, dtype=np.float32)  # drained, resampled, not yet handed out by drain_mixed()
        self.level = 0.0
        self.peak = 0.0  # loudest sample heard since the recording started
        self.collect = False  # when True, also keep samples in memory for drain()
        self._pending: list[np.ndarray] = []
        self._plock = threading.Lock()
        self._q: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._file = sf.SoundFile(str(path), mode="w", samplerate=self.sr,
                                  channels=1, subtype="PCM_16")
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._stream = sd.InputStream(device=device_index, samplerate=self.sr,
                                      channels=self.channels, dtype="float32",
                                      callback=self._callback)

    def _callback(self, indata, frames, time_info, status):
        mono = indata.mean(axis=1)
        block = mono.copy()
        self._q.put(block)
        if self.collect:  # read-only from here on, so sharing the block is safe
            with self._plock:
                self._pending.append(block)
        self.level = float(np.sqrt(np.mean(mono ** 2)) if len(mono) else 0.0)
        if len(mono):
            self.peak = max(self.peak, float(np.abs(mono).max()))

    def drain(self) -> np.ndarray:
        """Return (and forget) everything captured since the last drain, at this device's rate."""
        with self._plock:
            blocks, self._pending = self._pending, []
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)

    def _write_loop(self):
        last_sync = time.monotonic()
        while True:
            block = self._q.get()
            if block is None:
                break
            self._file.write(block)
            if time.monotonic() - last_sync > SYNC_SEC:
                self._file.flush()   # rewrites the WAV header, so a crash leaves a readable file
                last_sync = time.monotonic()

    def start(self):
        self._writer.start()
        self._stream.start()

    def stop(self):
        # PortAudio/CoreAudio can deadlock inside stream.stop() when the device changes state
        # mid-recording (seen when the mic stalled): the call never returns and the whole save
        # hangs. Close the stream on a helper thread and give up after STOP_TIMEOUT_SEC. The audio
        # is already on disk, so abandoning a stuck stream beats losing the meeting.
        def _close_stream():
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

        closer = threading.Thread(target=_close_stream, daemon=True)
        closer.start()
        closer.join(STOP_TIMEOUT_SEC)
        if closer.is_alive():
            print(f"[recorder] {self.name}: stream did not stop within {STOP_TIMEOUT_SEC}s, abandoning it")
        self._q.put(None)
        self._writer.join()
        self._file.close()


def _to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != TARGET_SR and len(audio):
        g = gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)
    return audio


def _load_resampled(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return _to_16k(audio, sr)


def _normalise(x: np.ndarray, target_rms: float = 0.05, min_rms: float = 1e-5) -> np.ndarray:
    """Bring each track to a similar loudness so a quiet mic isn't drowned out.
    Tracks quieter than min_rms are left alone (don't boost pure noise)."""
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    if rms < min_rms:
        return x
    return x * min(target_rms / rms, 20.0)


def _mix(parts: list[np.ndarray]) -> np.ndarray:
    length = max((len(p) for p in parts), default=0)
    mix = np.zeros(length, dtype=np.float32)
    for p in parts:
        mix[: len(p)] += p
    peak = float(np.max(np.abs(mix))) if length else 0.0
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix


def _combine_envelopes(roles: list[str], envs: list[np.ndarray]) -> dict | None:
    """Loudness over time per role. None unless both a microphone and a system-audio
    track were recorded (otherwise there is nothing to compare)."""
    by_role: dict[str, list[np.ndarray]] = {"mic": [], "system": []}
    for role, env in zip(roles, envs):
        by_role[role].append(env)
    if not by_role["mic"] or not by_role["system"]:
        return None
    return {"mic_env": speakers.merge_envelopes(by_role["mic"]),
            "sys_env": speakers.merge_envelopes(by_role["system"])}


def _role_envelopes(roles: list[str], raw: list[np.ndarray]) -> dict | None:
    """Same, from the un-normalised 16 kHz audio of each track."""
    return _combine_envelopes(roles, [speakers.frame_rms(a) for a in raw])


class Recorder:
    def __init__(self):
        self.tracks: list[_Track] = []
        self.started_at: float | None = None
        self.tail = np.zeros(0, dtype=np.float32)  # audio still undrained when stop() ran
        # Loudness over time of the mic and the system track ({"mic_env", "sys_env"}), set by
        # stop() when both kinds of input were recorded; used to label who is speaking.
        self.envelopes: dict | None = None
        # Un-normalised 16 kHz audio of the microphone, kept by stop() next to the envelopes so a
        # voiceprint can score what was said; None unless both a mic and a system track were recorded.
        self.mic_audio: np.ndarray | None = None
        self.sys_audio: np.ndarray | None = None   # likewise the system-audio track, the echo reference
        self._drain_lock = threading.Lock()

    def release_audio(self) -> None:
        """Drop the audio kept by stop() once the meeting has been processed (it can be hundreds of MB)."""
        self.tail = np.zeros(0, dtype=np.float32)
        self.envelopes = None
        self.mic_audio = None
        self.sys_audio = None

    @property
    def is_recording(self) -> bool:
        return self.started_at is not None

    def start(self, device_indexes: list[int], workdir: Path, live: bool = False):
        """live=True keeps audio in memory so drain_mixed() can feed a live transcriber."""
        if self.is_recording:
            raise RuntimeError("Already recording")
        workdir.mkdir(parents=True, exist_ok=True)
        self.tail = np.zeros(0, dtype=np.float32)
        self.envelopes = None
        self.mic_audio = None
        self.sys_audio = None
        tracks = []
        try:
            for n, idx in enumerate(dict.fromkeys(device_indexes)):
                t = _Track(idx, workdir / f"track{n}.wav")
                t.collect = live
                t.start()
                tracks.append(t)
        except Exception:
            for t in tracks:
                t.stop()
            raise
        self.tracks = tracks
        self.started_at = time.time()

    def levels(self):
        return [{"name": t.name, "level": round(t.level, 4)} for t in self.tracks]

    def elapsed(self) -> float:
        return time.time() - self.started_at if self.started_at else 0.0

    def heard_audio(self) -> bool:
        """False if every selected device has delivered nothing but digital silence so far."""
        return any(t.peak > 1e-4 for t in self.tracks)

    def drain_mixed(self, flush: bool = False) -> np.ndarray:
        """Audio captured since the previous call, mixed to 16 kHz mono (live mode only).
        Loudness is normalised per call, so quiet stretches are not boosted into noise.

        Each call returns the same number of samples for every track (the shortest one;
        the surplus of the others waits for the next call). Padding the shorter track
        instead would stretch the live timeline by a few milliseconds per call, a few
        seconds over a 5-minute meeting, and the transcript times would drift away from
        the real audio. A track that stalls for more than STALL_SEC is padded with silence
        so it cannot hold the others back. flush=True takes everything (used at stop)."""
        with self._drain_lock:
            for t in self.tracks:
                t.buf16 = np.concatenate([t.buf16, _to_16k(t.drain(), t.sr)])
            lens = [len(t.buf16) for t in self.tracks]
            longest = max(lens, default=0)
            n = longest if flush else max(min(lens, default=0), longest - int(STALL_SEC * TARGET_SR))
            raw = []
            for t in self.tracks:
                take, t.buf16 = t.buf16[:n], t.buf16[n:]
                if len(take) < n:
                    take = np.pad(take, (0, n - len(take)))
                t.env.add(take)   # un-normalised, so mic and system stay comparable
                raw.append(take)
        return _mix([_normalise(r, min_rms=1e-3) for r in raw])

    def live_envelopes(self) -> dict | None:
        """Loudness of mic and system audio so far (up to the last drain_mixed call), for
        labelling live captions. After stop() it is the final result, if any."""
        with self._drain_lock:
            tracks = list(self.tracks)
            if not tracks:
                return self.envelopes
            return _combine_envelopes([t.role for t in tracks], [t.env.array() for t in tracks])

    def stop(self, out_path: Path, keep_sec: float | None = None) -> tuple[np.ndarray, float]:
        """Stop recording, mix tracks, save FLAC. Returns (16 kHz audio, seconds).
        Anything the live transcriber has not drained yet is left in self.tail.
        keep_sec keeps only the first keep_sec seconds of every track (auto-stop uses it to drop
        the silence recorded after the meeting app released the mic); the tail is left whole."""
        if not self.is_recording:
            raise RuntimeError("Not recording")
        for t in self.tracks:
            t.stop()
        self.tail = self.drain_mixed(flush=True)
        raw = [_load_resampled(t.path) for t in self.tracks]
        if keep_sec is not None and keep_sec > 0:
            raw = [r[: int(keep_sec * TARGET_SR)] for r in raw]
        self.envelopes = _role_envelopes([t.role for t in self.tracks], raw)
        self.mic_audio = next((r for t, r in zip(self.tracks, raw) if t.role == "mic"), None) \
            if self.envelopes else None
        self.sys_audio = next((r for t, r in zip(self.tracks, raw) if t.role == "system"), None) \
            if self.envelopes else None
        mix = _mix([_normalise(r) for r in raw])
        length = len(mix)
        sf.write(str(out_path), mix, TARGET_SR, format="FLAC")
        for t in self.tracks:
            t.path.unlink(missing_ok=True)
        self.tracks = []
        self.started_at = None
        return mix, length / TARGET_SR


class VoiceSampler:
    """Records one input device into memory, to enrol the user's voice. Stops collecting by
    itself after MAX_SECONDS so a forgotten session cannot grow without limit."""

    MAX_SECONDS = 120

    def __init__(self, device_index: int):
        info = sd.query_devices(device_index)
        self.name = info["name"]
        self.sr = int(info["default_samplerate"])
        self.level = 0.0
        self._blocks: list[np.ndarray] = []
        self._n = 0
        self._lock = threading.Lock()
        self._stream = sd.InputStream(device=device_index, samplerate=self.sr,
                                      channels=max(1, min(2, info["max_input_channels"])),
                                      dtype="float32", callback=self._callback)
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        mono = indata.mean(axis=1)
        self.level = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
        with self._lock:
            if self._n < self.MAX_SECONDS * self.sr:
                self._blocks.append(mono.copy())
                self._n += len(mono)

    @property
    def seconds(self) -> float:
        return self._n / self.sr

    def stop(self) -> np.ndarray:
        """Close the device and return everything recorded, as 16 kHz mono."""
        self.close()
        with self._lock:
            audio = np.concatenate(self._blocks) if self._blocks else np.zeros(0, dtype=np.float32)
        return _to_16k(audio, self.sr)

    def close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


class _Meter:
    """One input device opened only to read its loudness. Nothing is recorded or kept."""

    def __init__(self, device_index: int):
        info = sd.query_devices(device_index)
        self.name = info["name"]
        self.level = 0.0   # loudness of the latest block (RMS)
        self.peak = 0.0    # loudest sample since the test started
        self._stream = sd.InputStream(device=device_index, samplerate=int(info["default_samplerate"]),
                                      channels=max(1, min(2, info["max_input_channels"])),
                                      dtype="float32", callback=self._callback)
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        mono = indata.mean(axis=1)
        if len(mono):
            self.level = float(np.sqrt(np.mean(mono ** 2)))
            self.peak = max(self.peak, float(np.abs(mono).max()))

    def close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


class LevelMonitor:
    """Lets the user check that every chosen input hears sound before recording.
    Stops by itself after MAX_SECONDS so a forgotten test cannot hold the microphone."""

    MAX_SECONDS = 60

    def __init__(self, device_indexes: list[int]):
        self._meters: list[_Meter] = []
        self._started = time.time()
        try:
            for idx in dict.fromkeys(device_indexes):
                self._meters.append(_Meter(idx))
        except Exception:
            self.stop()
            raise

    @property
    def expired(self) -> bool:
        return time.time() - self._started > self.MAX_SECONDS

    def snapshot(self) -> dict:
        return {"seconds": round(time.time() - self._started, 1),
                "levels": [{"name": m.name, "level": round(m.level, 4), "heard": m.peak > 1e-4} for m in self._meters]}

    def stop(self):
        for m in self._meters:
            m.close()
        self._meters = []
