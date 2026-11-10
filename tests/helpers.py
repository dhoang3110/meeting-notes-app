"""Shared helpers for the tests and the benchmark: golden dataset loading and scoring."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "golden"
sys.path.insert(0, str(ROOT))

STOPWORDS = set("""a about after again all also am an and any are as at be because been before being but by can could
did do does doing down for from get got had has have he her here him his how i if in into is it its just like me more
most my no not of off on one only or other our out over really she so some than that the their them then there these
they this those to too up us very was we were what when where which who why will with would yeah you your okay going
know think want need make right well kind thing things lot way say said""".split())


def golden_items() -> list[dict]:
    """Manifest entries whose audio is present on this machine."""
    mf = GOLDEN / "manifest.json"
    if not mf.exists():
        return []
    items = json.loads(mf.read_text())["recordings"]
    return [i for i in items if (GOLDEN / i["name"] / "audio.flac").exists()]


def audio_path(item: dict) -> Path:
    return GOLDEN / item["name"] / "audio.flac"


def reference_segments(item: dict) -> list[dict]:
    return json.loads((GOLDEN / item["name"] / "reference.json").read_text())


# ---------- text scoring ----------

def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).replace("'", "").split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: (substitutions + deletions + insertions) / reference words."""
    r, h = words(reference), words(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def max_repeat_run(segments: list[dict]) -> int:
    """Longest run of identical consecutive segments (a Whisper stuck-in-a-loop symptom)."""
    best = run = 0
    prev = None
    for s in segments:
        key = " ".join(words(s["text"]))
        run = run + 1 if key == prev else 1
        prev = key
        best = max(best, run)
    return best


def term_coverage(text: str, terms: list[str]) -> float:
    """Share of the expected key terms that appear in text (case-insensitive)."""
    if not terms:
        return 1.0
    low = text.lower()
    return sum(t.lower() in low for t in terms) / len(terms)


def salient_terms(text: str, n: int = 25) -> list[str]:
    """Most frequent non-trivial words: a starting point for choosing key terms by hand."""
    counts: dict[str, int] = {}
    for w in words(text):
        if len(w) > 3 and w not in STOPWORDS and not w.isdigit():
            counts[w] = counts.get(w, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


def notes_text(notes: dict) -> str:
    """Flatten the summariser's JSON into plain text for scoring."""
    return json.dumps(notes, ensure_ascii=False)


# ---------- running the real pipelines ----------

def metrics_ollama():
    """What Ollama has loaded, or None when it is not running."""
    import metrics
    return metrics.ollama_loaded()


def transcribe_full(item: dict) -> list[dict]:
    import soundfile as sf
    import transcriber
    audio, _ = sf.read(audio_path(item), dtype="float32")
    return transcriber.transcribe(audio, language="en")


def transcribe_live(item: dict, chunk_sec: float = 10.0, partial_sec: float = 0.0):
    """Replay a recording through LiveTranscriber, much faster than real time.
    Returns (segments, LiveTranscriber) so callers can read its stats."""
    import time
    import numpy as np
    import soundfile as sf
    import live
    audio, sr = sf.read(audio_path(item), dtype="float32")
    step = int(10 * sr)

    class Replay:  # hands out 10 s of audio per drain, like a recording running ~20x fast
        pos = 0

        def drain_mixed(self):
            out = audio[self.pos : self.pos + step]
            self.pos += len(out)
            return out

    rec = Replay()
    lt = live.LiveTranscriber(rec, language="en", chunk_sec=chunk_sec, partial_sec=partial_sec)
    lt.start()
    while rec.pos < len(audio) - step:
        time.sleep(0.25)
    tail = audio[rec.pos:]
    rec.pos = len(audio)
    return lt.finish(tail), lt


class ResourceSampler:
    """Samples RAM, swap and GPU while a block runs. Use as a context manager."""

    def __init__(self, every: float = 1.0):
        self.every = every
        self.peak_ram_gb = self.peak_app_gb = 0.0
        self.swap_growth_gb = 0.0
        self._gpu: list[int] = []

    def __enter__(self):
        import threading
        import psutil
        self._ps = psutil
        self._proc = psutil.Process()
        self._swap0 = psutil.swap_memory().used
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        import metrics
        gb = 1024 ** 3
        while not self._stop.is_set():
            vm = self._ps.virtual_memory()
            self.peak_ram_gb = max(self.peak_ram_gb, (vm.total - vm.available) / gb)
            self.peak_app_gb = max(self.peak_app_gb, self._proc.memory_info().rss / gb)
            self.swap_growth_gb = max(self.swap_growth_gb, (self._ps.swap_memory().used - self._swap0) / gb)
            util = metrics._gpu().get("util")
            if util is not None:
                self._gpu.append(util)
            self._stop.wait(self.every)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self) -> dict:
        return {"peak_ram_gb": round(self.peak_ram_gb, 1), "peak_app_gb": round(self.peak_app_gb, 2),
                "swap_growth_gb": round(self.swap_growth_gb, 1),
                "gpu_avg_pct": round(sum(self._gpu) / len(self._gpu)) if self._gpu else None}
