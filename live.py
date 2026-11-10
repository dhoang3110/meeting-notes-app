"""Transcribe a recording while it is still running.

A background thread pulls the audio captured so far from the recorder twice a second.

* Provisional captions: every PARTIAL_SEC of new audio, the not-yet-final audio is run
  through Whisper and shown as grey, replaceable text, so words appear a few seconds
  after they are spoken.
* Final text: once about CHUNK_SEC seconds have piled up, it cuts at the quietest
  moment near the end (so words are not split), transcribes that chunk, and keeps it.
  The provisional text for that stretch is replaced by this more accurate version.

When recording stops only the last partial chunk is left, so the transcript is ready
almost immediately."""

import threading
import time

import numpy as np

import speakers
import transcriber

SR = 16000
SEARCH_SEC = 3.0     # look for a pause in the last N seconds of the buffer
FRAME_SEC = 0.1
MIN_RMS = 5e-4       # audio quieter than this is treated as silence (skips Whisper)


def find_cut(buf: np.ndarray, search_sec: float = SEARCH_SEC) -> int:
    """Index near the end of buf where the audio is quietest (a pause between words)."""
    frame = int(FRAME_SEC * SR)
    start = max(0, len(buf) - int(search_sec * SR))
    n = (len(buf) - start) // frame
    if n < 2:
        return len(buf)
    frames = buf[start : start + n * frame].reshape(n, frame)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    # prefer the latest frame among the quietest, so chunks stay close to CHUNK_SEC
    best = int(np.flatnonzero(energy <= energy.min() * 1.05 + 1e-9)[-1])
    return start + best * frame + frame // 2


class LiveTranscriber:
    def __init__(self, recorder, language=None, vocabulary=None,
                 chunk_sec: float = 10.0, partial_sec: float = 2.0, engine: str | None = None,
                 label_speakers: bool = True):
        self._rec = recorder
        self._label = label_speakers
        self._language = language
        self._vocabulary = vocabulary
        self.engine = engine            # e.g. "parakeet:mlx-community/parakeet-tdt-0.6b-v2"; None = default
        self.chunk_sec = chunk_sec
        self._chunk = int(chunk_sec * SR)
        self._partial_n = int(partial_sec * SR)   # 0 = no provisional captions
        self._lock = threading.Lock()
        self._segments: list[dict] = []
        self._partial = {"start": 0.0, "text": "", "speaker": None, "seq": 0}
        self._consumed = 0          # samples already finalised
        self._pending = 0           # samples drained but not finalised yet
        self._stats = {"final_runs": 0, "final_audio": 0.0, "final_proc": 0.0,
                       "partial_runs": 0, "last_partial_audio": 0.0, "last_partial_proc": 0.0,
                       "last_final_audio": 0.0, "last_final_proc": 0.0}
        self.error: str | None = None
        self._stop = threading.Event()
        self._tail = np.zeros(0, dtype=np.float32)
        self._thread = threading.Thread(target=self._run, daemon=True)

    # ---------- public ----------

    def start(self):
        self._thread.start()

    def segments(self, after: int = 0) -> list[dict]:
        with self._lock:
            return list(self._segments[after:])

    def partial(self) -> dict:
        """Provisional text for the audio after the last final segment."""
        with self._lock:
            return dict(self._partial)

    @property
    def seconds(self) -> float:
        """How much of the meeting has been transcribed (finalised) so far."""
        return self._consumed / SR

    def stats(self) -> dict:
        """Whisper speed numbers. rtf = processing time / audio time (below 1 keeps up)."""
        with self._lock:
            s = dict(self._stats)
        rtf = lambda proc, audio: round(proc / audio, 2) if audio else None
        return {
            "engine": transcriber.engine_label(self.engine),
            "final_runs": s["final_runs"], "partial_runs": s["partial_runs"],
            "final_rtf": rtf(s["final_proc"], s["final_audio"]),
            "last_final_proc": round(s["last_final_proc"], 2),
            "last_partial_proc": round(s["last_partial_proc"], 2),
            "last_partial_rtf": rtf(s["last_partial_proc"], s["last_partial_audio"]),
            "unfinalised_sec": round(self._pending / SR, 1),
        }

    def finish(self, tail: np.ndarray) -> list[dict]:
        """Hand over the audio left at stop time, wait for the last chunk, return all segments.
        Raises if live transcription failed (caller should fall back to the saved audio)."""
        self._tail = tail
        self._stop.set()
        self._thread.join()
        if self.error:
            raise RuntimeError(self.error)
        return transcriber.drop_repeats(self.segments())  # also catches loops that span chunks

    # ---------- worker ----------

    def _run(self):
        carry = np.zeros(0, dtype=np.float32)
        since_partial = 0
        try:
            while not self._stop.wait(0.5):
                new = self._rec.drain_mixed()
                if len(new):
                    carry = np.concatenate([carry, new])
                    since_partial += len(new)
                self._pending = len(carry)
                if len(carry) >= self._chunk:
                    cut = find_cut(carry)
                    self._final(carry[:cut])
                    carry = carry[cut:]
                    self._pending = len(carry)
                    since_partial = len(carry)
                elif self._partial_n and since_partial >= self._partial_n:
                    self._provisional(carry)
                    since_partial = 0
            rest = np.concatenate([carry, self._tail])
            if len(rest):
                self._final(rest)
            self._set_partial("")
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def _whisper(self, audio: np.ndarray) -> tuple[list[dict], float]:
        """Run Whisper on audio; returns (segments with chunk-relative times, seconds taken)."""
        if float(np.sqrt(np.mean(audio ** 2))) < MIN_RMS:  # silence: Whisper would hallucinate
            return [], 0.0
        t0 = time.time()
        segs = transcriber.transcribe(audio, language=self._language, vocabulary=self._vocabulary,
                                      engine=self.engine)
        return segs, time.time() - t0

    def _speaker(self, spans: list[dict]) -> list[dict]:
        """Add "speaker" (Me / Others) to segments with absolute times, if the recorder can
        tell the mic from the system audio. Provisional: the final pass in app.py relabels
        everything once the whole recording (and so a better leakage estimate) is known."""
        env = getattr(self._rec, "live_envelopes", lambda: None)() if self._label else None
        return speakers.label_segments(spans, **env) if env and spans else spans

    def _final(self, audio: np.ndarray):
        offset = self._consumed / SR
        segs, took = self._whisper(audio)
        for s in segs:
            s["start"] = round(s["start"] + offset, 2)
            s["end"] = round(s["end"] + offset, 2)
        segs = self._speaker(segs)
        with self._lock:
            self._segments.extend(segs)
            if took:
                self._stats["final_runs"] += 1
                self._stats["final_audio"] += len(audio) / SR
                self._stats["final_proc"] += took
                self._stats["last_final_proc"] = took
        self._consumed += len(audio)
        self._set_partial("")  # the final text replaces the provisional text

    def _provisional(self, audio: np.ndarray):
        segs, took = self._whisper(audio)
        text = " ".join(s["text"] for s in segs).strip()
        start = self._consumed / SR
        speaker = None
        if text:  # the whole draft is one caption, so it gets one label
            speaker = self._speaker([{"start": start, "end": start + len(audio) / SR}])[0].get("speaker")
        with self._lock:
            if took:
                self._stats["partial_runs"] += 1
                self._stats["last_partial_audio"] = len(audio) / SR
                self._stats["last_partial_proc"] = took
        self._set_partial(text, speaker)

    def _set_partial(self, text: str, speaker: str | None = None):
        with self._lock:
            self._partial = {"start": round(self._consumed / SR, 2), "text": text,
                             "speaker": speaker, "seq": self._partial["seq"] + 1}
