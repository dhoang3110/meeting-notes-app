"""Machine and model performance numbers for the live performance panel.

A background thread samples CPU, RAM, memory pressure, swap, GPU, thermal state and battery once a
second. Ollama and Whisper numbers are read on demand from their own modules."""

import os
import re
import subprocess
import sys
import threading
import time

import psutil
import requests

_GB = 1024 ** 3
_lock = threading.Lock()
_latest: dict = {}
_started = False
_proc = psutil.Process(os.getpid())


def _gpu() -> dict:
    """Apple GPU utilisation from ioreg (no sudo needed). Empty dict if unavailable."""
    if sys.platform != "darwin":
        return {}
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w0", "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=2).stdout
    except Exception:
        return {}
    def grab(key):
        m = re.search(rf'"{re.escape(key)}"=(\d+)', out)
        return int(m.group(1)) if m else None
    return {"util": grab("Device Utilization %"), "mem_gb": round((grab("In use system memory") or 0) / _GB, 1)}


def _thermal() -> str:
    """'normal', or the level macOS reports once it has started throttling."""
    if sys.platform != "darwin":
        return "unknown"
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=2).stdout
    except Exception:
        return "unknown"
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    if m and int(m.group(1)) < 100:
        return f"throttled ({m.group(1)}% CPU speed)"
    m = re.search(r"(Thermal|Performance) warning level.*?(\d+)", out)
    if m and int(m.group(2)) > 0:
        return "warning"
    return "normal"


_PRESSURE = {1: "normal", 2: "warn", 4: "critical"}  # kern.memorystatus_vm_pressure_level
SWAP_RATE_WINDOW = 10    # seconds the swap traffic rate is averaged over
SWAP_RATE_WARN = 2.0     # MB/s of swap traffic: the Mac is short of RAM right now
SWAP_RATE_BAD = 20.0     # MB/s: heavy paging, everything slows down
_swap_history: list[tuple[float, int]] = []   # (time, total swap traffic in bytes)


def _pressure() -> str | None:
    """macOS memory pressure, the same green/yellow/red as Activity Monitor. None off macOS."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        value, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        if libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(value),
                             ctypes.byref(size), None, 0) != 0:
            return None
        return _PRESSURE.get(value.value)
    except Exception:
        return None


def parse_vm_stat(text: str) -> int | None:
    """Total bytes swapped in + out since boot, from `vm_stat` output. psutil's swap_memory().sin
    is no use on macOS: it counts every page-in (files too), not just swap."""
    page = re.search(r"page size of (\d+) bytes", text)
    ins = re.search(r"^Swapins:\s+(\d+)", text, re.M)
    outs = re.search(r"^Swapouts:\s+(\d+)", text, re.M)
    if not (page and ins and outs):
        return None
    return (int(ins.group(1)) + int(outs.group(1))) * int(page.group(1))


def _swap_traffic() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        return parse_vm_stat(subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2).stdout)
    except Exception:
        return None


def swap_rate(history: list[tuple[float, int]]) -> float | None:
    """MB/s of swap traffic between the oldest and newest sample, or None with too little data."""
    if len(history) < 2 or history[-1][0] <= history[0][0]:
        return None
    (t0, b0), (t1, b1) = history[0], history[-1]
    return round(max(0, b1 - b0) / (t1 - t0) / 1024 ** 2, 1)


def memory_status(pressure: str | None, swap_mb_s: float | None) -> str:
    """'ok', 'warn' or 'bad' for the RAM warning. Judged on what is happening now (macOS memory
    pressure and the current swap traffic), not on swap used: that is only what an earlier squeeze
    left on disk, and it stays high until those apps quit or the Mac restarts, even with plenty of
    free RAM."""
    rate = swap_mb_s or 0.0
    if pressure == "critical" or rate >= SWAP_RATE_BAD:
        return "bad"
    if pressure == "warn" or rate >= SWAP_RATE_WARN:
        return "warn"
    return "ok"


def _sample():
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    bat = psutil.sensors_battery()
    now, traffic = time.monotonic(), _swap_traffic()
    if traffic is not None:
        _swap_history.append((now, traffic))
        while _swap_history and now - _swap_history[0][0] > SWAP_RATE_WINDOW:
            _swap_history.pop(0)
    rate, pressure = swap_rate(_swap_history), _pressure()
    snap = {
        "cpu": psutil.cpu_percent(interval=None),
        "ram_used_gb": round((vm.total - vm.available) / _GB, 1),
        "ram_total_gb": round(vm.total / _GB, 1),
        "swap_used_gb": round(sw.used / _GB, 1),
        "swap_mb_s": rate,
        "mem_pressure": pressure,
        "mem_status": memory_status(pressure, rate),
        "app_rss_gb": round(_proc.memory_info().rss / _GB, 2),
        "gpu": _gpu(),
        "battery": {"percent": round(bat.percent), "plugged": bat.power_plugged} if bat else None,
    }
    return snap


def _loop():
    n = 0
    thermal = "normal"
    while True:
        try:
            snap = _sample()
            if n % 5 == 0:  # thermal state changes slowly
                thermal = _thermal()
            snap["thermal"] = thermal
            with _lock:
                _latest.clear()
                _latest.update(snap)
        except Exception:
            pass
        n += 1
        time.sleep(1.0)


def start():
    global _started
    if _started:
        return
    _started = True
    psutil.cpu_percent(interval=None)  # first call primes the counter
    threading.Thread(target=_loop, daemon=True).start()


def system() -> dict:
    with _lock:
        return dict(_latest)


def ollama_models() -> list[dict] | None:
    """Models installed in Ollama (smallest first), or None if Ollama is not running."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        tags = requests.get(f"{host}/api/tags", timeout=2).json().get("models", [])
    except Exception:
        return None
    return sorted(({"name": m["name"], "size_gb": round(m.get("size", 0) / _GB, 1)} for m in tags),
                  key=lambda m: m["size_gb"])


def ollama_loaded() -> dict | None:
    """What Ollama currently holds in memory (from /api/ps), or None if it is not running."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        models = requests.get(f"{host}/api/ps", timeout=1).json().get("models", [])
    except Exception:
        return None
    if not models:
        return {"model": None}
    m = models[0]
    size, vram = m.get("size", 0), m.get("size_vram", 0)
    return {"model": m.get("name"), "size_gb": round(size / _GB, 1),
            "gpu_pct": round(100 * vram / size) if size else None,
            "context": m.get("context_length"), "expires_at": m.get("expires_at")}
