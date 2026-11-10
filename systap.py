"""System-audio capture without a virtual driver: wraps the native `systap` helper
(native/systap.swift, a CoreAudio process tap, macOS 14.4+) so it looks like a sounddevice
InputStream. The helper writes mono float32 at its own sample rate, padded with silence in real
time, so this track stays aligned with the microphone."""

import os
import platform
import subprocess
import threading
from pathlib import Path

import numpy as np

SYSTEM_TAP_INDEX = -100            # pseudo device index used in the API / recorder
SYSTEM_TAP_NAME = "System Audio (built-in)"
BLOCK_SAMPLES = 960                # 20 ms at 48 kHz
START_CHECK_SEC = 0.4              # how long start() waits to see whether the helper died at once

_DEFAULT_BIN = Path(__file__).parent / "native" / "bin" / "systap"


def helper_path() -> Path:
    return Path(os.environ.get("SYSTAP_BIN") or _DEFAULT_BIN)


def available() -> bool:
    """True on macOS 14.4+ when the helper has been built."""
    if platform.system() != "Darwin" or not helper_path().is_file():
        return False
    try:
        major, minor = (int(x) for x in (platform.mac_ver()[0].split(".") + ["0"])[:2])
    except ValueError:
        return False
    return (major, minor) >= (14, 4)


class TapStream:
    """Same surface as the part of sounddevice.InputStream that recorder.py uses."""

    def __init__(self, device=None, samplerate=48000, channels=1, dtype="float32", callback=None):
        self.samplerate = samplerate
        self._callback = callback
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self.stderr = ""

    def start(self):
        self._proc = subprocess.Popen([str(helper_path())], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
        try:
            self._proc.wait(timeout=START_CHECK_SEC)   # still running after this = started fine
        except subprocess.TimeoutExpired:
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            return
        err = self._proc.stderr.read().decode(errors="replace").strip()
        raise RuntimeError(err or "the system audio helper exited immediately")

    def _read_loop(self):
        size = BLOCK_SAMPLES * 4
        out = self._proc.stdout
        while True:
            data = out.read(size)
            if len(data) < size:
                break
            block = np.frombuffer(data, dtype="<f4").reshape(-1, 1).copy()
            if self._callback:
                self._callback(block, len(block), None, None)

    def stop(self):
        proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._thread:
            self._thread.join(timeout=2)

    def close(self):
        self.stop()
