"""The user's voiceprint: recognise "my voice" from a short enrolment sample.

Enrolment: the user reads aloud for ~30-60 s into the microphone. The speech is cut into
overlapping windows, each is turned into a speaker embedding (a vector that is close for
the same voice and far for different voices) and their average becomes the profile.

Later, any stretch of mic audio can be scored: cosine similarity between its embedding and
the profile. The loudspeakers' sound (other participants) does not resemble the user's
voice, so this works without headphones and even if the mic is muted in the meeting app.

Embeddings come from a small ONNX model run by sherpa-onnx (CPU, no torch needed).
The profile is biometric data: it lives outside the repository (VOICEPRINT_DIR, default
~/.meeting-notes), readable only by the user, and never leaves this machine."""

import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

import speakers

SR = 16000
MODEL_NAME = "wespeaker_en_voxceleb_resnet34_LM.onnx"
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             f"speaker-recongition-models/{MODEL_NAME}")
MODEL_BYTES = 26_530_550

MAX_SCORE_SEC = 8.0      # only the first part of a long segment is scored (saves time, enough to recognise a voice)
MIN_SEC = 1.5            # less speech than this gives an unreliable embedding: no score
WINDOW_SEC = 4.0         # enrolment windows
HOP_SEC = 2.0
MIN_VOICED = 0.5         # an enrolment window needs this share of speech frames
MIN_WINDOWS = 7          # about 16 s of speech in total
SILENT_RMS = 1e-4        # quieter than this has no voice to score
DEFAULT_THRESHOLD = 0.45

_lock = threading.Lock()
_extractor = None


def data_dir() -> Path:
    return Path(os.getenv("VOICEPRINT_DIR", "~/.meeting-notes")).expanduser()


def model_path() -> Path:
    return data_dir() / "models" / MODEL_NAME


def profile_path() -> Path:
    return data_dir() / "voiceprint.npz"


def model_ready() -> bool:
    p = model_path()
    return p.exists() and p.stat().st_size == MODEL_BYTES


def ensure_model() -> Path:
    """Download the embedding model once (about 26 MB from the sherpa-onnx GitHub release)."""
    if model_ready():
        return model_path()
    import requests
    path = model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(MODEL_URL, stream=True, timeout=30) as r:
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
            if Path(tmp).stat().st_size != MODEL_BYTES:
                raise RuntimeError("The downloaded speaker model has an unexpected size; try again.")
            os.replace(tmp, path)
        finally:
            Path(tmp).unlink(missing_ok=True)
    return path


def _get_extractor():
    global _extractor
    with _lock:
        if _extractor is None:
            import sherpa_onnx
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(ensure_model()), num_threads=2)
            _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        return _extractor


def _unit(v: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(v))
    return (v / norm).astype(np.float32) if norm > 0 else None


def embed(audio: np.ndarray) -> np.ndarray | None:
    """Unit-length speaker embedding of 16 kHz mono audio, or None if there is too little to use."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < MIN_SEC * SR or float(np.sqrt(np.mean(audio ** 2))) < SILENT_RMS:
        return None
    audio = audio / max(1.0, float(np.abs(audio).max()))   # the mic can exceed full scale
    ex = _get_extractor()
    with _lock:   # one stream at a time keeps the native extractor simple
        stream = ex.create_stream()
        stream.accept_waveform(sample_rate=SR, waveform=audio)
        stream.input_finished()
        if not ex.is_ready(stream):
            return None
        return _unit(np.asarray(ex.compute(stream), dtype=np.float32))


# ---------- enrolment ----------

def build_profile(audio: np.ndarray, embed_fn=None) -> dict:
    """Profile from an enrolment recording. Raises ValueError when there is too little speech."""
    embed_fn = embed_fn or embed
    win, hop = int(WINDOW_SEC * SR), int(HOP_SEC * SR)
    env = speakers.frame_rms(audio)
    loud = env > max(1e-3, 0.25 * float(np.percentile(env, 90))) if len(env) else env > 1
    per_win = int(WINDOW_SEC / speakers.HOP_SEC)
    embs = []
    for a in range(0, max(0, len(audio) - win + 1), hop):
        i = a // int(SR * speakers.HOP_SEC)
        if loud[i : i + per_win].mean() < MIN_VOICED:
            continue
        e = embed_fn(audio[a : a + win])
        if e is not None:
            embs.append(e)
    if len(embs) < MIN_WINDOWS:
        raise ValueError("Not enough speech in the recording. Read aloud for about 30 seconds, "
                         "close to the microphone, in a quiet room.")
    embs = np.stack(embs)
    mean = _unit(embs.mean(axis=0))
    # How alike windows of the same voice are, each compared with the profile of the *other* windows.
    loo = [float(embs[k] @ _unit(np.delete(embs, k, axis=0).mean(axis=0))) for k in range(len(embs))]
    p10 = float(np.percentile(loo, 10))
    return {"embedding": mean, "threshold": float(np.clip(0.6 * p10, 0.30, 0.50)),
            "self_p10": p10, "windows": len(embs), "created": time.time()}


def save_profile(profile: dict) -> None:
    path = profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in profile.items()})
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def load_profile() -> dict | None:
    path = profile_path()
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files}


def delete_profile() -> bool:
    path = profile_path()
    if path.exists():
        path.unlink()
        return True
    return False


# ---------- scoring ----------

def threshold(profile: dict) -> float:
    env = os.getenv("VOICEPRINT_THRESHOLD")
    return float(env) if env else float(profile.get("threshold", DEFAULT_THRESHOLD))


def score(audio: np.ndarray, profile: dict, embed_fn=None) -> float | None:
    """Cosine similarity between this audio and the user's voice, or None if it is too short."""
    e = (embed_fn or embed)(audio)
    return None if e is None else float(e @ profile["embedding"])


def score_segments(segments: list[dict], mic_audio: np.ndarray, profile: dict, embed_fn=None) -> list[float | None]:
    """Voice score of every transcript segment, from the microphone audio of that stretch."""
    out = []
    for s in segments:
        a, b = max(0, int(s["start"] * SR)), int(s["end"] * SR)
        b = min(b, a + int(MAX_SCORE_SEC * SR))
        out.append(score(mic_audio[a:b], profile, embed_fn) if b > a else None)
    return out
