# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local Mac (Apple Silicon) meeting recorder: records mic + system audio (via BlackHole), transcribes locally with Parakeet (default; MLX Whisper code remains but is not offered in the UI), then summarises with Ollama (default) or Claude API into Markdown notes saved under `NOTES_DIR` (default `~/MeetingNotes`). FastAPI backend + single-page UI in `static/index.html`. The README is in Vietnamese and covers user setup (BlackHole, Ollama, `.env` options); code, comments and docs written to the repo stay in English.

## Commands

```bash
./run.sh                                   # creates .venv + installs requirements on first run, copies .env.example, opens browser, runs app.py (port 8765)
.venv/bin/python app.py                    # run the server directly
.venv/bin/python -m pytest tests/test_units.py                       # fast unit tests
.venv/bin/python -m pytest tests/test_golden.py                      # golden recordings < 10 min
.venv/bin/python -m pytest tests/test_golden.py --run-slow           # + the 27-minute recording
.venv/bin/python -m pytest tests/test_golden.py -m llm  # summary tests (Ollama)
.venv/bin/python -m pytest tests/test_units.py -k <name>             # single test
.venv/bin/python tests/bench.py --llm ollama:qwen2.5:14b # benchmark STT/LLM models; writes tests/results/ and compares with the previous run
.venv/bin/python tests/build_golden.py [--refresh]                   # build golden dataset from recordings in ~/MeetingNotes
```

```bash
cd tests/e2e && npm install && npx playwright install chromium   # once
cd tests/e2e && npm test                                         # Playwright UI + API tests (84, ~20 s)
cd tests/e2e && npx playwright test specs/meetings.spec.ts -g "delete"   # subset; `npm run test:ui` for the UI runner
```

No linter or build step is configured. Requirements: `requirements.txt` (runtime), `requirements-dev.txt` (pytest).

## Architecture

Flat module layout at the repo root; `app.py` wires everything together.

- **app.py** — FastAPI app, global singleton state (`_recorder`, `_current`, `_jobs`, `_last_live`, `_monitor`) guarded by `_lock`. `load_dotenv` runs *before* the project imports (hence the `# noqa: E402` imports) because modules read env vars at import time. Pipeline: `/api/start` → recording (+ optional live transcription) → `/api/stop` → background thread `_process` (transcribe unless live already did, then `_summarize_and_save`). Progress is tracked per meeting in `_jobs[mid]` (`stage`/`error`) and polled by the UI via `/api/status`. Also upload of existing audio, re-summarise, and level monitor ("Test audio") endpoints.
- **recorder.py** — `Recorder` captures mic and system-audio devices as separate `_Track`s (sounddevice), then resamples to 16 kHz, normalises and mixes them into `audio.flac`. `LevelMonitor` drives the UI level meters.
- **live.py** — `LiveTranscriber`: background thread pulling audio from the recorder. Emits provisional (grey, replaceable) captions every `LIVE_PARTIAL_SEC` and final chunks of ~`LIVE_CHUNK_SEC`, cut at the quietest point (`find_cut`) so words aren't split. On stop only the tail remains, so the transcript is ready almost immediately. If it fails, `_process` falls back to transcribing the saved audio.
- **transcriber.py** — engine abstraction; an engine spec is `whisper:<repo>` or `parakeet:<repo>` (Parakeet is English-only). `TRANSCRIBE_MODEL` overrides the default (Parakeet); only Parakeet is listed in `ENGINES`. Also `drop_repeats` (hallucination loops) and timestamp formatting.
- **speakers.py** — labels each transcript segment `Me` / `Others` by comparing the loudness of the mic track and the system-audio (BlackHole) track over the segment's time span. The recorder computes both envelopes at `stop()` (`Recorder.envelopes`, only when a mic *and* a system track were recorded); `_process` applies them. Speaker bleed from loudspeakers into the mic (no headphones) is estimated from the recording itself. `LABEL_SPEAKERS=0` disables it.
- **diarize.py** — tells the other participants apart: after the notes are saved, `_process` starts `_diarize_job` (stage `labelling speakers`, notes already readable) which clusters voices on the system-audio track with sherpa-onnx (pyannote segmentation + the voiceprint embedding model) and rewrites `Others` as `Speaker N` in `transcript.json` and in the transcript part of `notes.md` (`_render_notes`, no new LLM call). `Me` is never touched. Models are not downloaded automatically: `.venv/bin/python diarize.py --download`; without them (or with `DIARIZE=0`) the step is skipped. Tuned on a real 13-minute Teams meeting with 4–5 other speakers: `THRESHOLD` 0.65–0.7 gives the same 6 clusters, 0.5 gives 10 and 0.85 gives 2; `merge_small` (voices under 20 s fold into the nearest bigger one) leaves 5. Feeding it peak-normalised audio changed the result (6 → 4 voices), so the audio goes in as recorded. About 7× real time on 4 threads. Speaker numbers are arbitrary labels, not names.
- **aec.py** — echo cancellation at stop: the system-audio track (BlackHole, what the loudspeakers play) is the reference; its delay (often seconds, via a Multi-Output Device) is measured with GCC-PHAT on the waveforms, then a per-frequency multi-tap least-squares echo path is fitted (STFT 4096) and subtracted from the mic. The residual is what the user said; `speakers.label_segments(residual=…)` uses its loudness, and the voiceprint is scored on it. `ECHO_CANCEL=0` disables; `KEEP_TRACKS=1` stores raw mic/system/residual audio under `<meeting>/tracks/` for tuning (real meeting audio, never auto-deleted).
- **voiceprint.py** — optional "My voice" profile: enrolment sample → speaker embeddings (sherpa-onnx + a ~26 MB ONNX model downloaded on first use) → cosine score per segment against the mic audio. When enrolled, `speakers.label_segments` uses the score (`voice` field on each segment) instead of the loudness rules; segments under 1.5 s have no score and fall back to the rules. Profile lives in `VOICEPRINT_DIR` (default `~/.meeting-notes`, outside the repo; biometric data, never commit). Applied at stop only; live captions still use the loudness rules. `/api/voice*` endpoints in `app.py` drive enrolment.
- **summarizer.py** — `summarize()` dispatches to Ollama (default) or Claude (provider spec may be `ollama:<model>`; see `_split_provider` in app.py). Ollama path chunks long transcripts (`_ollama_summarize`) with `OLLAMA_NUM_CTX`. Expects a JSON object back (`JSON_SPEC`, parsed by `_parse_json`); `render_markdown` produces `notes.md`. Tracks LLM speed stats via `llm_stats()`.
- **meeting_detector.py** — polls CoreAudio (ctypes, macOS 14.4+) for processes with a running input stream; when a `MEETING_DETECT_APPS` bundle prefix (default Teams) holds the mic ~4 s it fires once per meeting and re-arms after the mic is released. `app.py` (`_on_meeting_detected`) shows a native `osascript` dialog and, on Record, calls `start()` with the default mic + BlackHole. Auto-stop (`_on_meeting_ended`, `MEETING_AUTO_STOP`): once the app has released the mic for `MEETING_END_GRACE_SEC` (40 s) the recording of that meeting is stopped and summarised (`_auto_started`: set by `start()` whenever Teams holds the mic at that moment, so the "Record?" dialog and a manual Start during a call both count), cut at the release time (`released_at`) + `END_MARGIN_SEC` (`Recorder.stop(keep_sec=…)` trims the tracks, `trim_segments` the live captions); recordings started while no meeting app holds the mic are never auto-stopped. Teams keeps its input stream while muted (checked on a real call), so mute is not an end. A shorter gap that re-arms the detector looks the same for a back-to-back call and a reconnect, so `_ask_new_meeting` asks (timeout keeps recording). Disable with `MEETING_DETECT=0`.
- **metrics.py** — background sampler for system stats (CPU/RAM/GPU/thermal) and Ollama model status, shown in the UI.

Each meeting is a folder `NOTES_DIR/<timestamp>_<slug>/` with `audio.flac`, `transcript.json`, `notes.json`, `notes.md`, `meta.json`. The folder name doubles as the meeting id in the API.

## Tests

- `tests/test_units.py` — unit tests; `tests/test_golden.py` — real recordings through real pipelines with deliberately loose thresholds (WER, real-time factor, key-term coverage) to catch regressions, not noise.
- Golden data lives in `tests/golden/`; audio and reference transcripts are git-ignored (real meeting audio) — only `manifest.json` is tracked. References are *silver* (Whisper output, not human-verified); `key_terms` in the manifest are hand-edited and preserved on `--refresh`.
- `tests/conftest.py` adds `--run-slow` / `--llm` options and parametrises the `golden` fixture; `tests/helpers.py` puts the project root on `sys.path`.
- Golden and LLM tests need the models downloaded (and Ollama running), so they are not quick.

## Playwright e2e tests (`tests/e2e/`)

- Self-contained Node project (own `package.json`, `node_modules` git-ignored). `playwright.config.ts` starts `tests/e2e/server.py`, which resets `tests/e2e/.data/` and runs the real FastAPI app on port 8799 with a throwaway `NOTES_DIR`, a throwaway `HOME` (so "move to Trash" never touches the real Trash), `MEETING_DETECT=0`, no API key and no Ollama. It never reads the real `~/MeetingNotes` or `.env`.
- `helpers.ts`: `seedMeeting()` writes meeting folders straight into the test notes dir; `mockEnvironment()` stubs hardware/Ollama/metrics endpoints and serves marked/DOMPurify from `node_modules` instead of the CDN. Tests share one server, so every test seeds its own uniquely named meetings (never "Select all" or count-assert against the real list).
- Real server: list, open, search, delete, hash deep links, and `specs/api.spec.ts` contract tests. Mocked with `page.route`: devices, recording (`/api/start|status|live|stop`), Test audio, upload, settings (the real endpoint writes `.env`) and voice enrolment, since those need a microphone, models or the network.
- `alert()`/`confirm()` fire synchronously inside click handlers: register `nextDialog(page)` (or `page.once('dialog')`) *before* the click.

## Configuration

All via `.env` (see `.env.example`; `.env` is git-ignored and is never committed). Key knobs: `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `OLLAMA_*`, `NOTES_LANGUAGE`, `WHISPER_LANGUAGE`, `WHISPER_VOCABULARY`, `LIVE_*`, `NOTES_DIR`, `PORT`.
