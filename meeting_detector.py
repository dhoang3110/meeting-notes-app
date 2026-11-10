"""Notice when a meeting app (Microsoft Teams by default) starts using the microphone.

macOS 14.4+ lets any process ask CoreAudio which other processes currently have an input
stream running (the same data behind the orange mic indicator), with their bundle ids.
No extra permission is needed. A meeting is "started" when a matching app has held the
mic for DEBOUNCE_POLLS polls in a row (ignores the short mic blips of a device test);
the detector re-arms once the app releases the mic, so each meeting prompts only once.
A meeting is "over" once the app has released the mic for end_polls polls: on_end(released_at)
gets the time it let go, so the caller can drop the silent grace period from the recording. A gap
between `release` and `end_polls` polls fires on_start again without an end: from the mic
alone, back-to-back calls and a reconnect look the same, so the caller decides (app.py asks).
Muting in Teams keeps its input stream running, so a long mute does not count as the end.

  probe_apps()      -> bundle ids of processes using an input device right now
  MeetingDetector   -> polls a probe, calls on_start(bundle_id) once per meeting"""

import ctypes
import sys
import threading
import time

DEFAULT_APPS = "com.microsoft.teams"   # prefix match: covers com.microsoft.teams2 (new Teams)
POLL_SEC = 2.0
DEBOUNCE_POLLS = 2                     # ~4 s of continuous mic use
RELEASE_POLLS = 3                      # ~6 s without mic use before a new meeting can trigger


def _fourcc(s: str) -> int:
    return int.from_bytes(s.encode("ascii"), "big")


class _Addr(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]


_CA = _CF = None


def _load():
    global _CA, _CF
    if _CA is None:
        _CA = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        _CF = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        _CF.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        _CF.CFStringGetCString.restype = ctypes.c_bool
        _CF.CFRelease.argtypes = [ctypes.c_void_p]
        _CA.AudioObjectGetPropertyDataSize.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        _CA.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    return _CA, _CF


def _get(obj: int, selector: str, ctype):
    """Read one fixed-size property; None if the object does not have it."""
    ca, _ = _load()
    addr = _Addr(_fourcc(selector), _fourcc("glob"), 0)
    value, size = ctype(), ctypes.c_uint32(ctypes.sizeof(ctype))
    if ca.AudioObjectGetPropertyData(obj, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(value)) != 0:
        return None
    return value


def probe_apps() -> set[str]:
    """Bundle ids of all processes that currently have an audio input stream running."""
    if sys.platform != "darwin":
        return set()
    ca, cf = _load()
    addr = _Addr(_fourcc("prs#"), _fourcc("glob"), 0)   # kAudioHardwarePropertyProcessObjectList
    size = ctypes.c_uint32(0)
    if ca.AudioObjectGetPropertyDataSize(1, ctypes.byref(addr), 0, None, ctypes.byref(size)) != 0:
        return set()   # macOS older than 14.4: no per-process info
    count = size.value // 4
    ids = (ctypes.c_uint32 * count)()
    if count == 0 or ca.AudioObjectGetPropertyData(1, ctypes.byref(addr), 0, None, ctypes.byref(size), ids) != 0:
        return set()
    found = set()
    for obj in ids:
        running = _get(obj, "piri", ctypes.c_uint32)   # kAudioProcessPropertyIsRunningInput
        if not running or not running.value:
            continue
        ref = _get(obj, "pbid", ctypes.c_void_p)       # kAudioProcessPropertyBundleID (CFString)
        if not ref or not ref.value:
            continue
        buf = ctypes.create_string_buffer(256)
        if cf.CFStringGetCString(ref.value, buf, 256, 0x08000100):   # kCFStringEncodingUTF8
            found.add(buf.value.decode("utf-8", "replace"))
        cf.CFRelease(ref.value)
    return found


def parse_apps(spec: str | None) -> list[str]:
    return [p.strip().lower() for p in (spec or DEFAULT_APPS).split(",") if p.strip()]


def match(found: set[str], prefixes: list[str]) -> str | None:
    for bundle in sorted(found):
        if any(bundle.lower().startswith(p) for p in prefixes):
            return bundle
    return None


class MeetingDetector:
    def __init__(self, on_start, apps: list[str] | None = None, probe=probe_apps,
                 debounce: int = DEBOUNCE_POLLS, release: int = RELEASE_POLLS,
                 on_end=None, end_polls: int = 20, clock=time.time):
        self.on_start = on_start
        self.on_end = on_end
        self.end_polls = end_polls   # polls without mic use before the meeting counts as over (default ~40 s)
        self._open = False
        self.ended = False
        self.apps = apps or parse_apps(None)
        self.probe = probe
        self.debounce, self.release = debounce, release
        self._active = 0      # consecutive polls with a matching app on the mic
        self._idle = 0        # consecutive polls without one
        self._armed = True
        self.clock = clock
        self.released_at: float | None = None   # wall-clock time of the last poll that found the mic free
        self._stop = threading.Event()
        self._thread = None

    def poll(self) -> str | None:
        """One check. Returns the bundle id when a new meeting is detected, else None."""
        bundle = match(self.probe(), self.apps)
        if bundle:
            self._idle = 0
            self._active += 1
            if self._armed and self._active >= self.debounce:
                self._armed = False
                self._open = True
                return bundle
        else:
            if self._idle == 0:
                self.released_at = self.clock()   # at most one poll after the app really let go
            self._active = 0
            self._idle += 1
            if self._idle >= self.release:
                self._armed = True
            if self._open and self._idle >= self.end_polls:
                self._open = False
                self.ended = True
        return None

    def app_on_mic(self) -> bool:
        """True while a detected meeting is on and the app held the mic at the last poll (not
        during the end grace after it let go)."""
        return self._open and self._active > 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="meeting-detector")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def step(self):
        """One poll plus callbacks; an end is reported before a start in the same poll."""
        bundle = self.poll()
        if self.ended:
            self.ended = False
            if self.on_end:
                self.on_end(self.released_at)
        if bundle:
            self.on_start(bundle)

    def _run(self):
        while not self._stop.wait(POLL_SEC):
            try:
                self.step()
            except Exception:
                import traceback
                traceback.print_exc()
