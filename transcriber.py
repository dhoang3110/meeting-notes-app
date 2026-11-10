"""Local speech-to-text on Apple Silicon. Two engines, both running on the Mac's GPU via MLX:

  whisper:<repo>    mlx-whisper. Many languages, takes a vocabulary hint.
  parakeet:<repo>   parakeet-mlx (NVIDIA Parakeet). English, usually faster and more
                    accurate on English, ignores the vocabulary hint.

An engine is named by one string such as "whisper:mlx-community/whisper-large-v3-turbo".
A bare Hugging Face repo name is treated as Whisper."""

import os
import re
import tempfile
import threading
from pathlib import Path

WHISPER_DEFAULT = "mlx-community/whisper-large-v3-turbo"
PARAKEET_DEFAULT = "mlx-community/parakeet-tdt-0.6b-v2"

ENGINES = [
    {"id": f"parakeet:{PARAKEET_DEFAULT}", "label": "Parakeet TDT 0.6B v2 (English, faster)"},
    {"id": f"whisper:{WHISPER_DEFAULT}", "label": "Whisper large-v3 turbo (multilingual)"},
]

_stt_lock = threading.Lock()  # one speech-to-text run at a time (live chunks, uploads, retries)
_parakeet_cache: dict = {}    # repo -> loaded model (only the latest is kept, they are ~2.5 GB)


def release_memory(unload_models: bool = True) -> None:
    """Hand memory back to the OS once transcription is finished: drop the loaded STT model
    (reloaded from disk on the next use) and empty MLX's GPU buffer cache, which otherwise keeps
    every buffer it ever allocated. Waits for any running transcription so a model is never
    freed under it."""
    import gc

    with _stt_lock:
        if unload_models:
            _parakeet_cache.clear()
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:  # MLX not installed or an older version: nothing to free
            pass


def default_engine() -> str:
    """TRANSCRIBE_MODEL from .env, else Parakeet. Whisper is only used if TRANSCRIBE_MODEL names it."""
    return os.getenv("TRANSCRIBE_MODEL") or f"parakeet:{PARAKEET_DEFAULT}"


def split_engine(spec: str | None) -> tuple[str, str]:
    """'parakeet:repo' -> ('parakeet', 'repo'); 'repo' or 'whisper:repo' -> ('whisper', 'repo')."""
    spec = spec or default_engine()
    kind, sep, repo = spec.partition(":")
    if sep and kind == "parakeet":
        return "parakeet", repo or PARAKEET_DEFAULT
    if sep and kind == "whisper":
        return "whisper", repo or WHISPER_DEFAULT
    return "whisper", spec


def engine_label(spec: str | None) -> str:
    kind, repo = split_engine(spec)
    return f"{kind} · {repo.rsplit('/', 1)[-1]}"


def transcribe(audio, language: str | None = None, vocabulary: str | None = None, engine: str | None = None):
    """audio: path to an audio file, or a 16 kHz mono float32 numpy array.
    Returns a list of {start, end, text} segments."""
    kind, repo = split_engine(engine)
    with _stt_lock:
        try:
            if kind == "parakeet":
                segments = _parakeet(audio, repo)
            else:
                segments = _whisper(audio, repo, language, vocabulary)
        finally:
            _trim_gpu_cache()
    return drop_repeats(segments)


def _cap_gpu_cache(limit_bytes: int = 512 * 1024 * 1024) -> None:
    """Belt and braces for _trim_gpu_cache: also stop the cache growing within a single long run."""
    try:
        import mlx.core as mx
        mx.set_cache_limit(limit_bytes)
    except Exception:
        pass


def _trim_gpu_cache() -> None:
    """Live transcription runs every couple of seconds on audio of a different length each time, and
    MLX keeps every freed GPU buffer in a cache that is never emptied by default. Left alone it grew
    to ~10 GB (mostly swapped out) in a 12-minute recording. Emptying it after each run costs a few
    ms of re-allocation."""
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:  # MLX not installed or an older version
        pass


def _whisper(audio, repo, language, vocabulary):
    import mlx_whisper  # imported lazily so the web UI starts fast

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=repo,
        language=language or None,
        initial_prompt=vocabulary or None,
        condition_on_previous_text=False,  # avoids repeated-phrase loops on long audio
    )
    segments = []
    for s in result.get("segments", []):
        text = s["text"].strip()
        if text:
            segments.append({"start": round(s["start"], 2), "end": round(s["end"], 2), "text": text})
    return segments


def _parakeet(audio, repo):
    from parakeet_mlx import from_pretrained  # lazy: the first use downloads ~2.5 GB

    model = _parakeet_cache.get(repo)
    if model is None:
        _parakeet_cache.clear()
        model = _parakeet_cache[repo] = from_pretrained(repo)
        _cap_gpu_cache()

    tmp = None
    if isinstance(audio, (str, Path)):
        path = str(audio)
        import soundfile as sf
        try:
            seconds = sf.info(path).duration
        except Exception:
            seconds = 0.0
    else:  # a numpy array from the live recorder: Parakeet only reads files
        import soundfile as sf
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, audio, 16000)
        path, seconds = tmp.name, len(audio) / 16000
    try:
        # long audio is cut into overlapping windows so memory stays flat
        result = model.transcribe(path, chunk_duration=120.0 if seconds > 300 else None)
    finally:
        if tmp:
            os.unlink(tmp.name)
    # Parakeet emits "<unk>" for audio it cannot decode (noise, clipping); strip it so it never
    # reaches the transcript, and drop segments that were nothing but "<unk>".
    out = []
    for s in result.sentences:
        text = re.sub(r"\s+", " ", s.text.replace("<unk>", " ")).strip()
        if text:
            out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
    return out


def drop_repeats(segments, max_run: int = 2):
    """Whisper sometimes gets stuck and repeats one sentence many times. Keep at most
    max_run identical segments in a row (short real replies like 'Yeah.' still survive)."""
    out, prev, run = [], None, 0
    for s in segments:
        key = re.sub(r"\W+", " ", s["text"].lower()).strip()
        run = run + 1 if key == prev else 1
        prev = key
        if run <= max_run:
            out.append(s)
    return out


def format_ts(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def segments_to_text(segments) -> str:
    """One line per segment; "[mm:ss] Me: text" when the segment carries a speaker label."""
    return "\n".join(
        f"[{format_ts(s['start'])}] {s['speaker'] + ': ' if s.get('speaker') else ''}{s['text']}"
        for s in segments)
