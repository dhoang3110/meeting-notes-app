"""Local meeting recorder + note-taker. Run: python app.py, then open http://127.0.0.1:8765"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, PlainTextResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import aec  # noqa: E402
import diarize  # noqa: E402
import live  # noqa: E402
import metrics  # noqa: E402
import speakers  # noqa: E402
import summarizer  # noqa: E402
import transcriber  # noqa: E402
import voiceprint  # noqa: E402

NOTES_DIR = Path(os.path.expanduser(os.getenv("NOTES_DIR", "~/MeetingNotes")))
NOTES_DIR.mkdir(parents=True, exist_ok=True)
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Meeting Notes")
_lock = threading.Lock()
_recorder = None
_current = {"folder": None, "title": None, "provider": None, "live": None}
_jobs: dict[str, dict] = {}  # meeting id -> {"stage": ..., "error": ...}
_last_live = None  # most recent LiveTranscriber, so its speed stats stay visible after Stop
_monitor = None   # LevelMonitor while the Test audio button is on
_sampler = None   # VoiceSampler while the user is recording a voice sample


def _get_recorder():
    global _recorder
    if _recorder is None:
        from recorder import Recorder
        _recorder = Recorder()
    return _recorder


def _slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text or "").strip()
    return re.sub(r"\s+", "-", s)[:50] or "meeting"


def _new_folder(title: str | None) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = NOTES_DIR / f"{stamp}_{_slug(title or 'meeting')}"
    folder.mkdir(parents=True)
    return folder


def _fmt_duration(sec: float) -> str:
    if sec < 60:
        return "<1 min"
    m = int(round(sec / 60))
    return f"{m // 60}h {m % 60}m" if m >= 60 else f"{m} min"


def _write_meta(folder: Path, **updates):
    path = folder / "meta.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta.update(updates)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def _check_stt(spec: str | None):
    """Only offer engines the app knows (a free-form model name would let a web page pick any download)."""
    if spec and spec not in {e["id"] for e in transcriber.ENGINES} and spec != transcriber.default_engine():
        raise HTTPException(400, f"Unknown speech-to-text model: {spec!r}")


def _split_provider(spec: str) -> tuple[str, str | None]:
    """'claude' -> ('claude', None); 'ollama' -> ('ollama', model from .env);
    'ollama:qwen2.5:7b' -> ('ollama', 'qwen2.5:7b'). Anything else is a 400."""
    kind, _, model = (spec or "").partition(":")
    if kind == "claude":
        return "claude", None
    if kind == "ollama":
        return "ollama", model or os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    raise HTTPException(400, f"Unknown summary provider: {spec!r}")


def _summarize_and_save(folder: Path, provider: str):
    mid = folder.name
    meta = json.loads((folder / "meta.json").read_text())
    segments = json.loads((folder / "transcript.json").read_text())
    text = transcriber.segments_to_text(segments)
    kind, model = _split_provider(provider)
    label = "Claude" if kind == "claude" else f"Ollama ({model})"
    _jobs[mid] = {"stage": f"summarising with {label}", "error": None}
    notes = summarizer.summarize(text, provider=kind, hint_title=meta.get("user_title"), model=model)
    (folder / "notes.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False))
    usage = summarizer.last_run_usage(kind, model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"))
    _write_meta(folder, title=notes.get("title"), provider=kind, model=model, usage=usage)
    _render_notes(folder)


def _render_notes(folder: Path):
    """(Re)write notes.md from notes.json, meta.json and the current transcript.json, without
    calling the LLM again (used when the transcript's speaker labels change after the summary)."""
    meta = _read_meta(folder)
    notes = json.loads((folder / "notes.json").read_text())
    segments = json.loads((folder / "transcript.json").read_text())
    kind, model = meta.get("provider") or "ollama", meta.get("model")
    (folder / "notes.md").write_text(summarizer.render_markdown(notes, {
        "date": meta.get("date", ""),
        "duration": _fmt_duration(meta.get("duration_sec", 0)),
        "provider": "Claude" if kind == "claude" else f"Ollama ({model})",
        "usage": meta.get("usage"),
    }, transcriber.segments_to_text(segments)))


def _diarize_job(folder: Path, sys_audio):
    """Background step after the notes are saved: split "Others" into Speaker 1, 2, ... by voice and
    update transcript.json and the transcript in notes.md. Failing soft: the Me/Others labels stay."""
    mid = folder.name
    try:
        started = time.time()
        turns = diarize.merge_small(diarize.diarize(sys_audio))
        segments = json.loads((folder / "transcript.json").read_text())
        labelled = diarize.label_segments(segments, turns)
        if labelled != segments:
            (folder / "transcript.json").write_text(json.dumps(labelled, indent=2, ensure_ascii=False))
            _write_meta(folder, diarize={"speakers": len({t[2] for t in turns}),
                                         "seconds": round(time.time() - started, 1)})
            if (folder / "notes.json").exists():
                _render_notes(folder)
    except Exception:
        traceback.print_exc()
    finally:
        sys_audio = None
        _jobs[mid] = {"stage": "done", "error": None}
        _release_memory()


def _discard_audio(folder: Path):
    """Privacy: once the notes are saved the recording is no longer needed, so delete it.
    Keep it when KEEP_AUDIO=1 or when the meeting's meta.json has "keep_audio": true
    (hand-set on the few meetings kept as QA material, see tests/build_golden.py)."""
    if os.getenv("KEEP_AUDIO", "0") == "1" or _read_meta(folder).get("keep_audio"):
        return
    for p in folder.glob("audio.*"):
        try:
            p.unlink()
        except OSError:
            traceback.print_exc()


def _label_speakers(folder: Path, segments: list[dict], envelopes: dict, mic_audio, sys_audio) -> list[dict]:
    """Label every segment "Me" or "Others". Layers, each optional and failing soft:
    echo cancellation (the system track removed from the mic), the user's voiceprint on what is
    left, and the loudness rules as the base."""
    echo, residual_env, voice_audio = None, None, mic_audio
    have_both = mic_audio is not None and sys_audio is not None
    if have_both and os.getenv("ECHO_CANCEL", "1") != "0":
        try:
            echo = aec.cancel(mic_audio, sys_audio)
            if echo.ok:
                voice_audio = echo.residual
                residual_env = speakers.frame_rms(echo.residual)
                _write_meta(folder, echo={"lag_sec": round(echo.lag_sec, 3), "erle_db": round(echo.erle_db, 1)})
        except Exception:  # without it the loudness rules and the voiceprint on the raw mic still work
            traceback.print_exc()
    if have_both and os.getenv("KEEP_TRACKS", "0") == "1":
        _keep_tracks(folder, mic_audio, sys_audio, echo.residual if echo is not None and echo.ok else None)
    voice, threshold = None, 0.45
    profile = voiceprint.load_profile() if voice_audio is not None else None
    if profile is not None:
        try:
            voice = voiceprint.score_segments(segments, voice_audio, profile)
            threshold = voiceprint.threshold(profile)
        except Exception:
            traceback.print_exc()
    labelled = speakers.label_segments(segments, **envelopes, voice=voice, voice_threshold=threshold,
                                       residual=residual_env)
    # Loudness only (no speech): kept so the labelling can be checked and tuned on real meetings.
    levels = {"mic": envelopes["mic_env"], "system": envelopes["sys_env"]}
    if residual_env is not None:
        levels["residual"] = residual_env
    np.savez_compressed(folder / "speaker_levels.npz", **levels)
    return labelled


def _keep_tracks(folder: Path, mic_audio, sys_audio, residual=None):
    """KEEP_TRACKS=1: save the raw microphone and system audio (and the echo-cancelled mic) next to
    the notes so the labelling can be tuned on a real meeting. This is real meeting audio on disk;
    it is never deleted automatically."""
    import soundfile as sf
    out = folder / "tracks"
    out.mkdir(exist_ok=True)
    scales = {}
    for name, x in (("mic", mic_audio), ("system", sys_audio), ("residual", residual)):
        if x is None:
            continue
        scale = max(1.0, float(abs(x).max()))   # FLAC cannot hold values beyond full scale
        sf.write(str(out / f"{name}.flac"), x / scale, 16000, format="FLAC")
        scales[name] = scale
    (out / "scales.json").write_text(json.dumps(scales))


def _process(folder: Path, audio, provider: str, live_tx=None, tail=None, stt: str | None = None,
             envelopes: dict | None = None, mic_audio=None, sys_audio=None, keep_sec: float | None = None):
    """Transcribe (unless a live transcriber already did most of it) and summarise.
    envelopes (mic and system loudness over time, from the recorder) lets us label who spoke;
    mic_audio and sys_audio (the echo reference) let echo cancellation and the user's voiceprint,
    if enrolled, make that call."""
    mid = folder.name
    try:
        _jobs[mid] = {"stage": "transcribing", "error": None}
        segments = None
        if live_tx is not None:
            try:
                segments = live_tx.finish(tail)
            except Exception:  # live transcription failed: redo it from the saved audio
                traceback.print_exc()
        if segments is None:
            segments = transcriber.transcribe(
                audio,
                language=os.getenv("WHISPER_LANGUAGE") or None,
                vocabulary=os.getenv("WHISPER_VOCABULARY") or None,
                engine=stt,
            )
        if keep_sec:   # the audio was cut at stop; live captions may still cover the silence after it
            segments = trim_segments(segments, keep_sec)
        if envelopes and os.getenv("LABEL_SPEAKERS", "1") != "0":
            segments = _label_speakers(folder, segments, envelopes, mic_audio, sys_audio)
        (folder / "transcript.json").write_text(json.dumps(segments, indent=2, ensure_ascii=False))
        if segments and not _read_meta(folder).get("duration_sec"):
            _write_meta(folder, duration_sec=segments[-1]["end"])
        _summarize_and_save(folder, provider)
        _discard_audio(folder)
        if (sys_audio is not None and os.getenv("DIARIZE", "1") != "0"
                and os.getenv("LABEL_SPEAKERS", "1") != "0" and diarize.available()):
            _jobs[mid] = {"stage": "labelling speakers", "error": None}   # the notes are already readable
            threading.Thread(target=_diarize_job, args=(folder, sys_audio), daemon=True).start()
        else:
            _jobs[mid] = {"stage": "done", "error": None}
    except Exception as e:  # keep the audio so the user can retry
        traceback.print_exc()
        _jobs[mid] = {"stage": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        audio = mic_audio = sys_audio = tail = envelopes = None  # this frame still references the big arrays
        _release_memory()


def _release_memory():
    """After a meeting is processed, free the STT model, MLX's GPU cache and the recorder's audio so a
    long-running server does not hold gigabytes while idle. Skipped while a recording or another job is
    in progress. KEEP_MODEL_LOADED=1 keeps the model in memory for faster starts."""
    with _lock:
        rec = _recorder
        if rec is not None and rec.is_recording:
            return
        if any(j["stage"] in ("transcribing", "saving audio", "queued", "labelling speakers")
               for j in _jobs.values()):
            return
        if rec is not None:
            rec.release_audio()
    transcriber.release_memory(unload_models=os.getenv("KEEP_MODEL_LOADED", "0") != "1")


def _read_meta(folder: Path) -> dict:
    p = folder / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _folder(mid: str) -> Path:
    folder = (NOTES_DIR / mid).resolve()
    if folder.parent != NOTES_DIR.resolve() or not folder.is_dir():
        raise HTTPException(404, "Meeting not found")
    return folder


# ---------------- API ----------------

class StartBody(BaseModel):
    devices: list[int]
    title: str | None = None
    provider: str = "ollama"
    stt: str | None = None   # speech-to-text engine, e.g. "parakeet:..."; None = default from .env


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    return {
        "notes_dir": str(NOTES_DIR),
        "default_provider": "ollama",
        "claude_ready": bool(os.getenv("ANTHROPIC_API_KEY")),
        "stt_models": transcriber.ENGINES,
        "default_stt": transcriber.default_engine(),
        "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
    }


ENV_FILE = Path(__file__).parent / ".env"


class SettingsBody(BaseModel):
    anthropic_api_key: str | None = None   # None = leave unchanged, "" = remove
    claude_model: str | None = None


def _key_hint(key: str) -> str:
    return f"{key[:7]}…{key[-4:]}" if len(key) > 12 else "set"


def _save_env(name: str, value: str) -> None:
    """Persist to .env (kept private, 0600) and apply to the running process."""
    from dotenv import set_key, unset_key
    if not ENV_FILE.exists():
        ENV_FILE.touch()
    ENV_FILE.chmod(0o600)
    if value:
        set_key(str(ENV_FILE), name, value, quote_mode="never")
        os.environ[name] = value
    else:
        unset_key(str(ENV_FILE), name)
        os.environ.pop(name, None)


@app.get("/api/settings")
def get_settings():
    """The API key itself is never sent back to the browser, only a masked hint."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return {
        "has_key": bool(key),
        "key_hint": _key_hint(key) if key else "",
        "claude_model": os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
    }


@app.post("/api/settings")
def save_settings(body: SettingsBody):
    if body.anthropic_api_key is not None:
        key = body.anthropic_api_key.strip()
        if key and not key.startswith("sk-ant-"):
            raise HTTPException(400, "That does not look like an Anthropic API key (it should start with sk-ant-).")
        _save_env("ANTHROPIC_API_KEY", key)
    if body.claude_model is not None:
        model = body.claude_model.strip()
        if model and not re.fullmatch(r"[A-Za-z0-9._:-]+", model):
            raise HTTPException(400, "Invalid model id.")
        _save_env("CLAUDE_MODEL", model)
    return get_settings()


@app.post("/api/settings/test")
def test_settings(body: SettingsBody):
    """Check a key (the typed one, else the saved one) against the API. Listing models is free."""
    import anthropic
    key = (body.anthropic_api_key or "").strip() or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(400, "No API key to test.")
    try:
        anthropic.Anthropic(api_key=key, max_retries=0, timeout=15).models.list(limit=1)
    except anthropic.AuthenticationError:
        raise HTTPException(400, "Anthropic rejected this key.")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Could not reach Anthropic. Check your network.")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Anthropic returned an error ({e.status_code}).")
    return {"ok": True}


@app.get("/api/ollama/models")
def ollama_models():
    """Models installed in Ollama, for the summary dropdown. models is empty if Ollama is down."""
    found = metrics.ollama_models()
    return {"running": found is not None, "models": found or []}


class MonitorBody(BaseModel):
    devices: list[int]


def _stop_monitor():
    global _monitor
    if _monitor:
        _monitor.stop()
        _monitor = None


@app.post("/api/monitor/start")
def monitor_start(body: MonitorBody):
    """Test audio: open the chosen inputs just to show their level (nothing is recorded)."""
    global _monitor
    if not body.devices:
        raise HTTPException(400, "Choose at least one input device")
    with _lock:
        if _recorder is not None and _recorder.is_recording:
            raise HTTPException(409, "Already recording")
        _stop_monitor()
        from recorder import LevelMonitor
        try:
            _monitor = LevelMonitor(body.devices)
        except Exception as e:
            raise HTTPException(500, f"Could not open the input device: {e}")
    return {"ok": True}


@app.post("/api/monitor/stop")
def monitor_stop():
    with _lock:
        _stop_monitor()
    return {"ok": True}


@app.get("/api/monitor")
def monitor_levels():
    m = _monitor
    if m is None or m.expired:
        with _lock:
            _stop_monitor()
        return {"active": False, "seconds": 0, "levels": []}
    return {"active": True, **m.snapshot()}


# ---------------- My voice (voiceprint enrolment) ----------------

class VoiceStartBody(BaseModel):
    device: int


def _stop_sampler():
    global _sampler
    if _sampler:
        _sampler.close()
        _sampler = None


def _voice_status() -> dict:
    profile = voiceprint.load_profile()
    smp = _sampler
    return {
        "enrolled": profile is not None,
        "created": float(profile["created"]) if profile is not None else None,
        "threshold": voiceprint.threshold(profile) if profile is not None else None,
        "sampling": {"seconds": round(smp.seconds, 1), "level": round(smp.level, 4),
                     "full": smp.seconds >= smp.MAX_SECONDS, "device": smp.name} if smp else None,
    }


@app.get("/api/voice")
def voice_status():
    return _voice_status()


@app.post("/api/voice/start")
def voice_start(body: VoiceStartBody):
    """Start recording a voice sample from one microphone (downloads the speaker model the first time)."""
    global _sampler
    with _lock:
        if _recorder is not None and _recorder.is_recording:
            raise HTTPException(409, "Already recording a meeting")
        _stop_monitor()
        _stop_sampler()
    try:
        voiceprint.ensure_model()
        import sherpa_onnx  # noqa: F401  (fail now, not after a minute of reading aloud)
    except Exception as e:
        raise HTTPException(500, f"Speaker model is not available: {e}")
    from recorder import VoiceSampler
    with _lock:
        try:
            _sampler = VoiceSampler(body.device)
        except Exception as e:
            raise HTTPException(500, f"Could not open the microphone: {e}")
    return _voice_status()


@app.post("/api/voice/stop")
def voice_stop():
    """Finish the sample and build the voiceprint from it."""
    global _sampler
    with _lock:
        smp, _sampler = _sampler, None
    if smp is None:
        raise HTTPException(409, "Not recording a voice sample")
    audio = smp.stop()
    try:
        voiceprint.save_profile(voiceprint.build_profile(audio))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _voice_status()


@app.post("/api/voice/cancel")
def voice_cancel():
    with _lock:
        _stop_sampler()
    return _voice_status()


@app.delete("/api/voice")
def voice_delete():
    voiceprint.delete_profile()
    return _voice_status()


@app.get("/api/devices")
def devices():
    from recorder import list_input_devices
    return list_input_devices()


@app.post("/api/start")
def start(body: StartBody):
    if not body.devices:
        raise HTTPException(400, "Choose at least one input device")
    _split_provider(body.provider)  # reject a bad choice before recording, not after
    _check_stt(body.stt)
    with _lock:
        rec = _get_recorder()
        if rec.is_recording:
            raise HTTPException(409, "Already recording")
        _stop_monitor()  # a Test audio session must not hold the devices the recording needs
        _stop_sampler()  # nor a voice sample in progress
        folder = _new_folder(body.title)
        _write_meta(folder, user_title=body.title, date=datetime.now().strftime("%a %d %b %Y, %H:%M"),
                    created=datetime.now().isoformat(), source="recording",
                    stt=body.stt or transcriber.default_engine())
        use_live = os.getenv("LIVE_TRANSCRIBE", "1") != "0"
        try:
            rec.start(body.devices, folder / "_tracks", live=use_live)
        except Exception as e:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(500, f"Could not start recording: {e}")
        live_tx = None
        if use_live:
            live_tx = live.LiveTranscriber(
                rec,
                language=os.getenv("WHISPER_LANGUAGE") or None,
                vocabulary=os.getenv("WHISPER_VOCABULARY") or None,
                chunk_sec=float(os.getenv("LIVE_CHUNK_SEC", "10")),
                partial_sec=float(os.getenv("LIVE_PARTIAL_SEC", "2")),
                engine=body.stt,
                label_speakers=os.getenv("LABEL_SPEAKERS", "1") != "0",
            )
            live_tx.start()
            global _last_live
            _last_live = live_tx
        _current.update(folder=folder, title=body.title, provider=body.provider, live=live_tx)
        _jobs[folder.name] = {"stage": "recording", "error": None}
        _auto_started["id"] = folder.name if _meeting_on_mic() else None
    return {"id": folder.name}


def _meeting_on_mic() -> bool:
    """A meeting app (Teams) holds the mic right now, so a recording started now — by the
    "Record?" dialog or by hand — is of that meeting and is stopped when it ends."""
    d = _detector
    return bool(d is not None and d.on_end is not None and d.app_on_mic())


@app.post("/api/stop")
def stop():
    return _stop_recording()


def trim_segments(segments: list[dict], keep_sec: float) -> list[dict]:
    """Drop segments that start after keep_sec and clip the one that runs past it."""
    return [{**s, "end": min(s["end"], keep_sec)} for s in segments if s["start"] < keep_sec]


def _stop_recording(keep_sec: float | None = None):
    """Stop, save and start processing. keep_sec (auto-stop) cuts the recording at that point."""
    with _lock:
        rec = _get_recorder()
        if not rec.is_recording:
            raise HTTPException(409, "Not recording")
        folder, provider, live_tx = _current["folder"], _current["provider"], _current["live"]
        _jobs[folder.name] = {"stage": "saving audio", "error": None}
        audio, seconds = rec.stop(folder / "audio.flac", keep_sec=keep_sec)
        tail, envelopes, mic_audio, sys_audio = rec.tail, rec.envelopes, rec.mic_audio, rec.sys_audio
        shutil.rmtree(folder / "_tracks", ignore_errors=True)
        _write_meta(folder, duration_sec=seconds)
        _current.update(folder=None, title=None, provider=None, live=None)
    stt = _read_meta(folder).get("stt")  # the engine chosen at Start, used if live transcription must be redone
    threading.Thread(target=_process, args=(folder, audio, provider, live_tx, tail, stt, envelopes, mic_audio, sys_audio,
                                            keep_sec), daemon=True).start()
    return {"id": folder.name}


@app.get("/api/status")
def status():
    rec = _recorder
    lt = _current["live"]
    recording = bool(rec and rec.is_recording)
    return {
        "recording": recording,
        "elapsed": rec.elapsed() if recording else 0,
        "levels": rec.levels() if recording else [],
        "current": _current["folder"].name if _current["folder"] else None,
        "jobs": {k: v for k, v in _jobs.items() if v["stage"] not in ("done",)},
        "no_audio": bool(recording and rec.elapsed() > 8 and not rec.heard_audio()),
        "session": _session_info() if recording else None,
        "live": {"transcribed": lt.seconds, "error": lt.error, "chunk_sec": lt.chunk_sec}
                if recording and lt else None,
    }


def _session_info() -> dict | None:
    """Which models the current recording uses, so the UI can show them even after a page reload."""
    folder = _current["folder"]
    if not folder:
        return None
    kind, model = _split_provider(_current["provider"] or "ollama")
    return {"stt": transcriber.engine_label(_read_meta(folder).get("stt")),
            "summary": "Claude" if kind == "claude" else f"Ollama · {model}"}


@app.get("/api/live")
def live_transcript(after: int = 0):
    """Transcript segments produced so far for the meeting being recorded."""
    lt = _current["live"]
    return {"segments": lt.segments(after) if lt else [], "recording": lt is not None,
            "partial": lt.partial() if lt else None}


@app.get("/api/metrics")
def get_metrics():
    """Machine and model performance numbers for the performance panel."""
    metrics.start()
    rec = _recorder
    recording = bool(rec and rec.is_recording)
    return {
        "recording": recording,
        "system": metrics.system(),
        "whisper": _last_live.stats() if _last_live else None,
        "llm": summarizer.llm_stats(),
        "ollama": metrics.ollama_loaded(),
        "audio": [{"name": t.name, "peak": round(t.peak, 3)} for t in rec.tracks] if recording else [],
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), title: str = Form(""), provider: str = Form("ollama"),
                 stt: str = Form("")):
    _split_provider(provider)
    _check_stt(stt or None)
    folder = _new_folder(title or Path(file.filename or "upload").stem)
    suffix = Path(file.filename or "audio").suffix or ".m4a"
    dest = folder / f"audio{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    _write_meta(folder, user_title=title or None, date=datetime.now().strftime("%a %d %b %Y, %H:%M"),
                created=datetime.now().isoformat(), source=f"upload: {file.filename}",
                stt=stt or transcriber.default_engine())
    threading.Thread(target=_process, args=(folder, str(dest), provider, None, None, stt or None),
                     daemon=True).start()
    return {"id": folder.name}


@app.get("/api/meetings")
def meetings():
    items = []
    for folder in sorted(NOTES_DIR.iterdir(), reverse=True):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        meta = _read_meta(folder)
        job = _jobs.get(folder.name)
        items.append({
            "id": folder.name,
            "title": meta.get("title") or meta.get("user_title") or folder.name,
            "date": meta.get("date", ""),
            "duration": _fmt_duration(meta["duration_sec"]) if meta.get("duration_sec") else "",
            "has_notes": (folder / "notes.md").exists(),
            "has_transcript": (folder / "transcript.json").exists(),
            "stage": job["stage"] if job else None,
            "error": job["error"] if job else None,
        })
    return items


@app.get("/api/meetings/{mid}", response_class=PlainTextResponse)
def meeting(mid: str):
    folder = _folder(mid)
    md = folder / "notes.md"
    if not md.exists():
        raise HTTPException(404, "Notes not ready yet")
    return md.read_text()


class ResummarizeBody(BaseModel):
    provider: str = "ollama"


@app.post("/api/meetings/{mid}/summarize")
def resummarize(mid: str, body: ResummarizeBody):
    folder = _folder(mid)
    _split_provider(body.provider)

    def run():
        try:
            if (folder / "transcript.json").exists():
                _summarize_and_save(folder, body.provider)
                _jobs[mid] = {"stage": "done", "error": None}
            else:  # transcription failed earlier: retry from audio
                audio = next((p for p in folder.glob("audio.*")), None)
                if not audio:
                    raise FileNotFoundError("No audio or transcript in this meeting folder")
                _process(folder, str(audio), body.provider, stt=_read_meta(folder).get("stt"))
        except Exception as e:
            traceback.print_exc()
            _jobs[mid] = {"stage": "error", "error": f"{type(e).__name__}: {e}"}

    _jobs[mid] = {"stage": "queued", "error": None}
    threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


_BUSY = ("recording", "saving audio", "transcribing", "queued")


@app.delete("/api/meetings/{mid}")
def delete_meeting(mid: str):
    """Move a meeting folder to the macOS Trash (recoverable), unless it is still being processed."""
    folder = _folder(mid)
    stage = _jobs.get(mid, {}).get("stage", "")
    if stage in _BUSY or stage.startswith("summarising"):
        raise HTTPException(409, "This meeting is still being recorded or processed. Wait until it finishes.")
    trash = Path.home() / ".Trash"
    trash.mkdir(exist_ok=True)
    dest = trash / folder.name
    n = 1
    while dest.exists():
        n += 1
        dest = trash / f"{folder.name} {n}"
    shutil.move(str(folder), str(dest))
    _jobs.pop(mid, None)
    return {"ok": True}


@app.post("/api/meetings/{mid}/reveal")
def reveal(mid: str):
    folder = _folder(mid)
    target = folder / "notes.md" if (folder / "notes.md").exists() else folder
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(target)])
    return {"path": str(target)}


# ---------------- Meeting auto-detect ----------------

_prompt_lock = threading.Lock()
_auto_started = {"id": None}   # id of the recording to stop when the meeting ends (started during a call)
_detector = None               # the MeetingDetector, once started
END_MARGIN_SEC = 2.0           # audio kept after the meeting app released the mic


def _default_devices() -> list[int]:
    """What the UI picks by default: the system microphone plus BlackHole if installed."""
    from recorder import list_input_devices
    import sounddevice as sd
    devs = list_input_devices()
    blackhole = next((d["index"] for d in devs if "blackhole" in d["name"].lower()), None)
    mic = sd.default.device[0]
    if mic is None or mic < 0 or mic == blackhole:
        mic = next((d["index"] for d in devs if d["index"] != blackhole), None)
    return [i for i in (mic, blackhole) if i is not None]


def _ask_to_record(app_name: str) -> bool:
    """Native macOS dialog; times out (counts as No) so it never blocks forever."""
    script = (f'display dialog "{app_name} meeting detected. Record and take notes?" '
              'with title "Meeting Notes" buttons {"Not now", "Record"} default button "Record" '
              'giving up after 45')
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True).stdout
    return "button returned:Record" in out


def _ask_new_meeting(app_name: str) -> bool:
    """Asked when the meeting app takes the mic again during an auto-started recording: a
    back-to-back call and a reconnect look the same. Times out as "Same meeting" (keeps recording)."""
    script = (f'display dialog "{app_name} is using the microphone again. Is this a new meeting?" '
              'with title "Meeting Notes" buttons {"Same meeting", "New meeting"} '
              'default button "Same meeting" giving up after 45')
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True).stdout
    return "button returned:New meeting" in out


def _recording_auto_started() -> bool:
    """True while the current recording is one of a meeting (see _meeting_on_mic)."""
    cur, rec = _current["folder"], _recorder
    return bool(_auto_started["id"] and rec is not None and rec.is_recording
                and cur is not None and cur.name == _auto_started["id"])


def _on_meeting_detected(bundle: str):
    if not _prompt_lock.acquire(blocking=False):
        return  # a prompt is already open
    try:
        name = "Teams" if "teams" in bundle.lower() else bundle
        rec = _recorder
        if rec is not None and rec.is_recording:
            # A recording begun outside a call is the user's to stop; one of a call may be a finished call.
            if not _recording_auto_started() or not _ask_new_meeting(name):
                return
            # save and summarise the previous meeting (cut where the app last released the mic), then record the new one
            _on_meeting_ended(_detector.released_at if _detector else None)
        elif not _ask_to_record(name):
            return
        started = start(StartBody(devices=_default_devices(), title=f"{name} meeting", provider="ollama"))
        _auto_started["id"] = started["id"]   # also when the app let go of the mic while the dialog was open
    except HTTPException as e:
        print(f"Auto-record failed: {e.detail}")
    except Exception:
        traceback.print_exc()
    finally:
        _prompt_lock.release()


def _keep_seconds(released_at: float | None, started_at: float | None, elapsed: float) -> float | None:
    """How much of the recording to keep: up to the release plus a small margin. None (keep all) when
    the release is unknown, came before the recording started, or leaves nothing worth cutting."""
    if released_at is None or started_at is None:
        return None
    keep = released_at - started_at + END_MARGIN_SEC
    return keep if 0 < keep < elapsed - 1 else None


def _on_meeting_ended(released_at: float | None = None):
    """Teams released the mic for a while: stop the recording of that meeting (started by the dialog,
    or by hand while Teams held the mic), dropping the silence recorded after the release."""
    mine = _recording_auto_started()
    _auto_started["id"] = None
    if not mine:
        return  # not recording, or the user already stopped / started something else
    try:
        rec = _recorder
        _stop_recording(_keep_seconds(released_at, rec.started_at, rec.elapsed()))
        subprocess.Popen(["osascript", "-e",
                          'display notification "Recording stopped, summarising now." with title "Meeting Notes"'])
    except HTTPException as e:
        print(f"Auto-stop failed: {e.detail}")


@app.on_event("startup")
def _start_detector():
    if sys.platform != "darwin" or os.getenv("MEETING_DETECT", "1") == "0":
        return
    import meeting_detector
    global _detector
    _detector = meeting_detector.MeetingDetector(
        _on_meeting_detected, meeting_detector.parse_apps(os.getenv("MEETING_DETECT_APPS")),
        on_end=_on_meeting_ended if os.getenv("MEETING_AUTO_STOP", "1") != "0" else None,
        end_polls=max(1, int(float(os.getenv("MEETING_END_GRACE_SEC", "40")) / meeting_detector.POLL_SEC)),
    )
    _detector.start()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8765"))
    print(f"\n  Meeting Notes running at http://127.0.0.1:{port}\n  Notes folder: {NOTES_DIR}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
