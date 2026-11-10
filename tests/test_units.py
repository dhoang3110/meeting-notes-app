"""Fast tests (no models, no audio files): run with `.venv/bin/python -m pytest tests/test_units.py`."""

import numpy as np
import pytest

import helpers
import live
import metrics
import recorder
import speakers
import summarizer
import transcriber

SR = 16000


def seg(text, start=0.0, end=1.0):
    return {"start": start, "end": end, "text": text}


# ---------- scoring helpers ----------

def test_wer_identical_and_one_substitution():
    assert helpers.wer("the cat sat", "The cat, sat!") == 0.0
    assert round(helpers.wer("the cat sat", "the dog sat"), 2) == 0.33


def test_wer_counts_deletions_and_insertions():
    assert helpers.wer("a b c d", "a b") == 0.5
    assert helpers.wer("a b", "a b c d") == 1.0


# ---------- repeat-loop guard ----------

def test_drop_repeats_limits_runs_but_keeps_spread_out_repeats():
    loop = [seg("They're still looking.")] * 13
    assert len(transcriber.drop_repeats(loop)) == 2
    spread = [seg("Yeah."), seg("Right."), seg("Yeah."), seg("Right."), seg("Yeah.")]
    assert transcriber.drop_repeats(spread) == spread


def test_max_repeat_run():
    assert helpers.max_repeat_run([seg("a"), seg("a"), seg("a"), seg("b")]) == 3


# ---------- live chunking ----------

def test_find_cut_lands_in_the_silent_gap():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(SR * 10) * 0.1).astype("float32")
    x[int(SR * 8.0) : int(SR * 8.4)] = 0
    assert 8.0 <= live.find_cut(x) / SR <= 8.4


class _Feed:
    def __init__(self, audio, step):
        self.audio, self.step, self.pos = audio, step, 0

    def drain_mixed(self):
        out = self.audio[self.pos : self.pos + self.step]
        self.pos += len(out)
        return out


def test_live_transcriber_offsets_timestamps_and_keeps_order(monkeypatch):
    calls = []

    def fake(audio, language=None, vocabulary=None, engine=None):
        calls.append(len(audio) / SR)
        return [{"start": 0.0, "end": 1.0, "text": f"chunk{len(calls)}"}]

    monkeypatch.setattr(transcriber, "transcribe", fake)
    audio = (np.random.default_rng(1).standard_normal(SR * 26) * 0.1).astype("float32")
    feed = _Feed(audio, SR * 13)
    lt = live.LiveTranscriber(feed, chunk_sec=10, partial_sec=0)
    lt.start()
    import time
    while feed.pos < len(audio) - SR * 13:
        time.sleep(0.1)
    segs = lt.finish(audio[feed.pos :])
    assert [s["text"] for s in segs] == [f"chunk{i}" for i in range(1, len(segs) + 1)]
    starts = [s["start"] for s in segs]
    assert starts == sorted(starts) and starts[0] == 0.0 and starts[-1] > 8   # later chunks are shifted
    assert abs(sum(calls) - 26) < 0.2                                          # all audio transcribed once


def test_live_transcriber_skips_silence(monkeypatch):
    monkeypatch.setattr(transcriber, "transcribe", lambda *a, **k: [seg("hallucinated")])
    lt = live.LiveTranscriber(_Feed(np.zeros(SR * 12, dtype="float32"), SR * 12), chunk_sec=10, partial_sec=0)
    lt.start()
    assert lt.finish(np.zeros(0, dtype="float32")) == []


# ---------- recorder mixing ----------

def test_mix_pads_and_prevents_clipping():
    mix = recorder._mix([np.full(10, 0.8, dtype="float32"), np.full(20, 0.8, dtype="float32")])
    assert len(mix) == 20 and float(np.abs(mix).max()) <= 0.99


def test_normalise_leaves_noise_floor_alone():
    quiet = np.full(1000, 1e-4, dtype="float32")
    assert np.array_equal(recorder._normalise(quiet, min_rms=1e-3), quiet)


# ---------- summariser ----------

def test_parse_json_handles_fences_and_prose():
    assert summarizer._parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


def test_chunks_respect_size():
    text = "\n".join(["x" * 100] * 50)
    parts = list(summarizer._chunks(text, 1000))
    assert len(parts) > 1 and all(len(p) <= 1100 for p in parts)


def test_render_markdown_survives_null_fields():
    notes = {"title": "T", "summary": None,
             "action_items": [{"task": "Do it", "owner": None, "due": None}, {"task": None}, "plain string"],
             "decisions": ["d"], "topics": [{"heading": "H", "points": ["p"]}], "open_questions": []}
    md = summarizer.render_markdown(notes, {"date": "d", "duration": "1 min", "provider": "x"}, "transcript")
    assert "- [ ] Do it" in md and "- [ ] plain string" in md


# ---------- summary model choice ----------

def test_split_provider_accepts_claude_and_ollama_models(monkeypatch):
    import app
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:14b")
    assert app._split_provider("claude") == ("claude", None)
    assert app._split_provider("ollama") == ("ollama", "qwen2.5:14b")
    assert app._split_provider("ollama:qwen2.5:7b") == ("ollama", "qwen2.5:7b")   # model names contain ':'


def test_split_provider_rejects_unknown():
    import pytest
    from fastapi import HTTPException
    import app
    for bad in ("gpt", "", "ollama2:x"):
        with pytest.raises(HTTPException) as e:
            app._split_provider(bad)
        assert e.value.status_code == 400


# ---------- speech-to-text engines ----------

def test_split_engine_reads_prefixes_and_bare_repos(monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_MODEL", raising=False)
    assert transcriber.split_engine("parakeet:mlx-community/parakeet-tdt-0.6b-v2") == (
        "parakeet", "mlx-community/parakeet-tdt-0.6b-v2")
    assert transcriber.split_engine("whisper:a/b") == ("whisper", "a/b")
    assert transcriber.split_engine("a/b") == ("whisper", "a/b")      # bare repo means Whisper
    assert transcriber.split_engine(None) == ("parakeet", "mlx-community/parakeet-tdt-0.6b-v2")  # default


def test_check_stt_only_allows_known_engines(monkeypatch):
    import pytest
    from fastapi import HTTPException
    import app
    monkeypatch.delenv("TRANSCRIBE_MODEL", raising=False)
    app._check_stt(None)
    app._check_stt(transcriber.ENGINES[0]["id"])
    with pytest.raises(HTTPException):
        app._check_stt("parakeet:someone/else-model")


# ---------- Test audio (level monitor) ----------

def test_level_monitor_reports_level_and_whether_sound_was_heard(monkeypatch):
    streams = []

    class FakeStream:
        def __init__(self, device, samplerate, channels, dtype, callback):
            self.callback, self.stopped = callback, False
            streams.append(self)

        def start(self): pass
        def stop(self): self.stopped = True
        def close(self): pass

    monkeypatch.setattr(recorder.sd, "query_devices", lambda i: {"name": f"dev{i}", "default_samplerate": 48000, "max_input_channels": 2})
    monkeypatch.setattr(recorder.sd, "InputStream", FakeStream)
    mon = recorder.LevelMonitor([3, 4, 3])                       # a repeated device is opened once
    assert len(streams) == 2
    streams[0].callback(np.full((480, 2), 0.2, dtype="float32"), 480, None, None)   # dev3 hears sound
    streams[1].callback(np.zeros((480, 2), dtype="float32"), 480, None, None)       # dev4 is silent
    snap = mon.snapshot()
    assert [(l["name"], l["heard"]) for l in snap["levels"]] == [("dev3", True), ("dev4", False)]
    assert abs(snap["levels"][0]["level"] - 0.2) < 1e-3
    mon.stop()
    assert all(s.stopped for s in streams) and mon.snapshot()["levels"] == []


def test_level_monitor_expires(monkeypatch):
    monkeypatch.setattr(recorder.sd, "query_devices", lambda i: {"name": "d", "default_samplerate": 48000, "max_input_channels": 1})
    monkeypatch.setattr(recorder.sd, "InputStream", lambda **k: type("S", (), {"start": lambda s: None, "stop": lambda s: None, "close": lambda s: None})())
    mon = recorder.LevelMonitor([1])
    assert not mon.expired
    mon._started -= recorder.LevelMonitor.MAX_SECONDS + 1
    assert mon.expired


# ---------- meeting auto-detect ----------

def _detector(frames, **kw):
    import meeting_detector
    it = iter(frames)
    return meeting_detector.MeetingDetector(lambda b: None, probe=lambda: next(it), **kw)


def test_detector_triggers_once_after_debounce_and_ignores_blips():
    teams = {"com.microsoft.teams2"}
    d = _detector([teams, set(), teams, teams, teams, teams], debounce=2, release=2)
    assert [d.poll() for _ in range(6)] == [None, None, None, "com.microsoft.teams2", None, None]


def test_detector_rearms_after_mic_is_released():
    teams, off = {"com.microsoft.teams2"}, set()
    d = _detector([teams, teams, off, off, teams, teams], debounce=2, release=2)
    assert [d.poll() for _ in range(6)] == [None, "com.microsoft.teams2", None, None, None, "com.microsoft.teams2"]


def test_detector_ignores_other_apps_and_matches_prefix():
    import meeting_detector
    assert meeting_detector.match({"com.apple.Safari"}, ["com.microsoft.teams"]) is None
    assert meeting_detector.match({"com.microsoft.teams2.helper"}, ["com.microsoft.teams"])


def test_detector_reports_end_only_after_grace_and_only_once():
    teams, off = {"com.microsoft.teams2"}, set()
    d = _detector([teams, teams] + [off] * 6, debounce=2, release=2, end_polls=4)
    seen = []
    for _ in range(7):
        d.poll()
        seen.append(d.ended)
    assert seen == [False, False, False, False, False, True, True]
    d.ended = False
    d._idle = 99
    d.poll()
    assert d.ended is False   # already reported: no second "end"


def _events(frames, **kw):
    """Run the detector's step() over frames; return the callbacks in the order they fired."""
    import meeting_detector
    it, events = iter(frames), []
    d = meeting_detector.MeetingDetector(lambda b: events.append("start"), probe=lambda: next(it),
                                         on_end=lambda t: events.append("end"), **kw)
    for _ in frames:
        d.step()
    return events


def test_detector_mute_keeps_meeting_open():
    # Teams keeps its input stream while muted (checked on a real call), so the mic is held throughout.
    teams = {"com.microsoft.teams2.modulehost"}
    assert _events([teams] * 30, debounce=2, release=2, end_polls=4) == ["start"]


def test_detector_short_drop_does_not_end_meeting():
    teams, off = {"com.microsoft.teams2"}, set()
    frames = [teams, teams, off, teams, teams, teams]   # 1 poll off < release
    assert _events(frames, debounce=2, release=2, end_polls=4) == ["start"]


def test_detector_gap_between_rearm_and_grace_restarts_without_ending():
    # Back-to-back call or a reconnect: indistinguishable here, so no "end" — app.py asks the user.
    teams, off = {"com.microsoft.teams2"}, set()
    frames = [teams, teams, off, off, teams, teams, off, off, off, off]
    assert _events(frames, debounce=2, release=2, end_polls=4) == ["start", "start", "end"]


def test_detector_end_fires_even_if_start_prompt_was_declined():
    # The app ignores it (nothing auto-started), but the detector must not stay stuck "open".
    teams, off = {"com.microsoft.teams2"}, set()
    frames = [teams, teams] + [off] * 4 + [teams, teams]
    assert _events(frames, debounce=2, release=2, end_polls=4) == ["start", "end", "start"]


def _fake_recording(monkeypatch, app, folder_name):
    from pathlib import Path
    stopped = []
    rec = type("R", (), {"is_recording": True, "started_at": 1000.0, "elapsed": lambda self: 600.0})()
    monkeypatch.setattr(app, "_recorder", rec)
    monkeypatch.setitem(app._current, "folder", Path("/tmp") / folder_name if folder_name else None)
    monkeypatch.setattr(app, "_stop_recording", lambda keep_sec=None: stopped.append(True))
    monkeypatch.setattr(app.subprocess, "Popen", lambda *a, **k: None)   # no macOS notification in tests
    return stopped


def test_auto_stop_stops_only_the_recording_the_detector_started(monkeypatch):
    import app
    stopped = _fake_recording(monkeypatch, app, "2026-11-10_teams")
    monkeypatch.setitem(app._auto_started, "id", "2026-11-10_teams")
    app._on_meeting_ended()
    assert stopped == [True]
    assert app._auto_started["id"] is None


def test_auto_stop_leaves_manual_recording_alone(monkeypatch):
    import app
    stopped = _fake_recording(monkeypatch, app, "2026-11-10_manual")
    monkeypatch.setitem(app._auto_started, "id", None)
    app._on_meeting_ended()
    assert stopped == []


def test_auto_stop_leaves_a_newer_recording_alone(monkeypatch):
    # The user stopped the auto-started meeting and began another one by hand.
    import app
    stopped = _fake_recording(monkeypatch, app, "2026-11-10_manual")
    monkeypatch.setitem(app._auto_started, "id", "2026-11-10_teams")
    app._on_meeting_ended()
    assert stopped == []


# ---------- deleting a meeting ----------

def test_delete_meeting_moves_folder_to_trash_and_blocks_busy(tmp_path, monkeypatch):
    import pytest
    from fastapi import HTTPException
    import app
    notes, home = tmp_path / "notes", tmp_path / "home"
    (notes / "2026-01-01_000000_x").mkdir(parents=True)
    (notes / "2026-01-01_000000_x" / "notes.md").write_text("hi")
    home.mkdir()
    monkeypatch.setattr(app, "NOTES_DIR", notes)
    monkeypatch.setattr(app.Path, "home", classmethod(lambda cls: home))
    app._jobs["2026-01-01_000000_x"] = {"stage": "transcribing", "error": None}
    with pytest.raises(HTTPException) as e:
        app.delete_meeting("2026-01-01_000000_x")
    assert e.value.status_code == 409
    app._jobs["2026-01-01_000000_x"] = {"stage": "done", "error": None}
    app.delete_meeting("2026-01-01_000000_x")
    assert not (notes / "2026-01-01_000000_x").exists()
    assert (home / ".Trash" / "2026-01-01_000000_x" / "notes.md").read_text() == "hi"
    with pytest.raises(HTTPException):
        app.delete_meeting("../escape")


# ---------- token usage per meeting ----------

def test_summarize_tracks_token_usage_across_calls(monkeypatch):
    def fake_claude(prompt, max_tokens=4000):
        summarizer._llm_begin("claude", "claude-sonnet-5-5")
        summarizer._llm_end({"prompt_tokens": 1000, "output_tokens": 200})
        return '{"title": "T", "summary": "S"}'
    monkeypatch.setattr(summarizer, "_claude", fake_claude)
    summarizer.summarize("hello there", provider="claude")
    summarizer.summarize("hello again", provider="claude")   # a new run resets the counters
    u = summarizer.last_run_usage("claude", "claude-sonnet-5-5")
    assert (u["input_tokens"], u["output_tokens"], u["calls"]) == (1000, 200, 1)
    assert u["est_cost_usd"] == round((1000 * 2 + 200 * 10) / 1e6, 4)
    monkeypatch.setenv("USD_TO_AUD", "2")
    assert summarizer.last_run_usage("claude", "claude-sonnet-5-5")["est_cost_aud"] == round(0.004 * 2, 4)
    assert "est_cost_usd" not in summarizer.last_run_usage("ollama", "qwen2.5:14b")


def test_render_markdown_shows_token_usage():
    meta = {"date": "d", "duration": "1 min", "provider": "Claude",
            "usage": {"input_tokens": 8123, "output_tokens": 1402, "calls": 1, "est_cost_usd": 0.0303, "est_cost_aud": 0.047}}
    md = summarizer.render_markdown({"title": "T", "summary": "S"}, meta, "x")
    assert "Tokens: 8,123 in · 1,402 out · est. A$0.047" in md
    assert "Tokens:" not in summarizer.render_markdown({"title": "T"}, {"date": "d"}, "x")


# ---------- speaker labels (Me vs Others) ----------

def _envelopes(plan, bleed=0.0, noise=0.0005, seed=0):
    """Mic and system loudness (0.1 s frames) for a list of (seconds, who) turns.
    'me' = user on the mic, 'others' = system track playing (leaking into the mic
    by `bleed`), 'both' = talking over each other, 'quiet' = silence."""
    rng = np.random.default_rng(seed)
    mic, sys = [], []
    for sec, who in plan:
        n = int(sec * 10)
        s = rng.uniform(0.02, 0.06, n) if who in ("others", "both") else np.zeros(n)
        m = rng.uniform(0.03, 0.08, n) if who in ("me", "both") else np.zeros(n)
        mic.append(noise + m + bleed * s)
        sys.append(s + 1e-4)
    return np.concatenate(mic).astype(np.float32), np.concatenate(sys).astype(np.float32)


def _labels(plan, spans, **kw):
    mic, sys = _envelopes(plan, **kw)
    segs = [seg("x", a, b) for a, b in spans]
    return [s["speaker"] for s in speakers.label_segments(segs, mic, sys)]


PLAN = [(10, "others"), (8, "me"), (10, "others"), (6, "me"), (10, "others")]
SPANS = [(0, 10), (10, 18), (18, 28), (28, 34), (34, 44)]
EXPECTED = ["Others", "Me", "Others", "Me", "Others"]


def test_labels_with_headphones_no_bleed():
    assert _labels(PLAN, SPANS, bleed=0.0) == EXPECTED


def test_labels_survive_speaker_leaking_into_mic():
    # Without headphones the mic hears the loudspeakers; others must not become "Me".
    assert _labels(PLAN, SPANS, bleed=0.35) == EXPECTED
    assert _labels(PLAN, SPANS, bleed=0.6) == EXPECTED


def test_label_picks_me_when_user_talks_over_others():
    plan = [(10, "others"), (6, "both"), (10, "others")]
    assert _labels(plan, [(10, 16)], bleed=0.4) == ["Me"]


def test_label_short_segment_and_past_the_end():
    mic, sys = _envelopes(PLAN, bleed=0.3)
    out = speakers.label_segments([seg("a", 12.0, 12.0), seg("b", 999.0, 1000.0)], mic, sys)
    assert [s["speaker"] for s in out] == ["Me", "Others"]


def test_label_segments_keeps_fields_and_does_not_mutate():
    mic, sys = _envelopes(PLAN)
    segs = [seg("hello", 0.0, 5.0)]
    out = speakers.label_segments(segs, mic, sys)
    assert out[0]["text"] == "hello" and out[0]["speaker"] == "Others"
    assert "speaker" not in segs[0]


def test_label_segments_with_no_audio():
    empty = np.zeros(0, dtype=np.float32)
    assert speakers.label_segments([seg("x", 0, 1)], empty, empty)[0]["speaker"] == "Others"


def test_frame_rms_and_merge():
    audio = np.concatenate([np.full(1600, 0.5), np.zeros(1600), np.full(800, 1.0)]).astype(np.float32)
    env = speakers.frame_rms(audio)
    assert len(env) == 2 and abs(env[0] - 0.5) < 1e-6 and env[1] == 0
    merged = speakers.merge_envelopes([np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0])])
    assert merged.tolist() == [1.0, 2.0, 0.0]


def test_role_of_device_names():
    assert speakers.role_of("BlackHole 2ch") == "system"
    assert speakers.role_of("MacBook Pro Microphone") == "mic"


def test_role_envelopes_need_both_mic_and_system():
    audio = np.full(3200, 0.1, dtype=np.float32)
    assert recorder._role_envelopes(["mic"], [audio]) is None
    assert recorder._role_envelopes(["system"], [audio]) is None
    both = recorder._role_envelopes(["mic", "system"], [audio, audio])
    assert set(both) == {"mic_env", "sys_env"} and len(both["mic_env"]) == 2


def test_segments_to_text_shows_speaker_only_when_labelled():
    labelled = [{"start": 0, "end": 1, "text": "Hi", "speaker": "Me"}, seg("Hello", 65, 66)]
    lines = transcriber.segments_to_text(labelled).splitlines()
    assert lines[0].endswith("Me: Hi") and lines[1].endswith("] Hello")


# ---------- live speaker labels ----------

def test_running_envelope_matches_whole_file_however_it_is_chunked():
    audio = (np.random.default_rng(3).standard_normal(SR * 5 + 123) * 0.1).astype("float32")
    run = speakers.RunningEnvelope()
    for a in range(0, len(audio), 7919):          # odd piece size: frames straddle the pieces
        run.add(audio[a : a + 7919])
    assert np.allclose(run.array(), speakers.frame_rms(audio), atol=1e-6)


class _FakeTrack:
    def __init__(self, role, audio):
        self.role, self.sr, self.env, self._audio = role, SR, speakers.RunningEnvelope(), audio
        self.buf16 = np.zeros(0, dtype="float32")

    def drain(self):
        out, self._audio = self._audio, np.zeros(0, dtype="float32")
        return out


def test_recorder_live_envelopes_come_from_drained_audio():
    rec = recorder.Recorder()
    mic, sys = np.full(SR * 2, 0.05, dtype="float32"), np.full(SR * 2, 0.02, dtype="float32")
    rec.tracks = [_FakeTrack("mic", mic), _FakeTrack("system", sys)]
    assert rec.live_envelopes()["mic_env"].size == 0          # nothing drained yet
    rec.drain_mixed()
    env = rec.live_envelopes()
    assert len(env["mic_env"]) == 20 and abs(env["mic_env"][0] - 0.05) < 1e-4
    assert abs(env["sys_env"][0] - 0.02) < 1e-4
    rec.tracks = [_FakeTrack("mic", mic)]                      # no system track: cannot label
    assert rec.live_envelopes() is None


class _LabelFeed(_Feed):
    """Others talk for the first 10 s, then the user does."""
    def live_envelopes(self):
        others = np.r_[np.full(100, 0.04), np.zeros(200)].astype("float32")
        me = np.r_[np.zeros(100), np.full(200, 0.06)].astype("float32")
        return {"mic_env": me + 0.0005, "sys_env": others + 1e-4}


def test_live_transcriber_labels_final_segments_and_partial(monkeypatch):
    n = [0]

    def fake(audio, language=None, vocabulary=None, engine=None):
        n[0] += 1
        return [{"start": 0.0, "end": 1.0, "text": f"chunk{n[0]}"}]

    monkeypatch.setattr(transcriber, "transcribe", fake)
    audio = (np.random.default_rng(1).standard_normal(SR * 26) * 0.1).astype("float32")
    feed = _LabelFeed(audio, SR * 13)
    lt = live.LiveTranscriber(feed, chunk_sec=10, partial_sec=0)
    lt.start()
    import time
    while feed.pos < len(audio) - SR * 13:
        time.sleep(0.1)
    segs = lt.finish(audio[feed.pos :])
    assert segs[0]["speaker"] == "Others" and segs[-1]["speaker"] == "Me"

    lt._consumed = 12 * SR                                      # a draft caption in the user's part
    lt._provisional(audio[:SR * 3])
    assert lt.partial()["speaker"] == "Me"


def test_live_transcriber_without_envelopes_or_with_labels_off(monkeypatch):
    monkeypatch.setattr(transcriber, "transcribe", lambda *a, **k: [seg("hi", 0.0, 1.0)])
    audio = (np.random.default_rng(1).standard_normal(SR * 12) * 0.1).astype("float32")
    for feed, kw in ((_Feed(audio, len(audio)), {}), (_LabelFeed(audio, len(audio)), {"label_speakers": False})):
        lt = live.LiveTranscriber(feed, chunk_sec=10, partial_sec=0, **kw)
        lt.start()
        segs = lt.finish(audio)   # everything arrives as the tail
        assert segs and all("speaker" not in s for s in segs)


def _noisy_meeting(seed, sigma=0.5, bleed=0.6, n_turns=30):
    """Others talk ~75% of the time and leak into the mic with a noisy ratio, so the mic is
    almost never quiet. Returns mic/system envelopes, true speaker per turn and turn spans."""
    rng = np.random.default_rng(seed)
    mic, sysl, truth, spans, t = [], [], [], [], 0
    for _ in range(n_turns):
        me = rng.random() < 0.25
        n = int(rng.uniform(15, 60))
        if me:
            m, s = rng.uniform(0.03, 0.1, n), rng.uniform(0, 0.003, n)
        else:
            s = rng.uniform(0.02, 0.08, n)
            m = bleed * s * rng.lognormal(0, sigma, n)
        mic.append(0.0008 + m)
        sysl.append(s + 1e-4)
        truth.append("Me" if me else "Others")
        spans.append((t / 10, (t + n) / 10))
        t += n
    return np.concatenate(mic).astype("float32"), np.concatenate(sysl).astype("float32"), truth, spans


def test_labels_when_others_dominate_and_mic_is_never_quiet():
    # Regression: the noise floor used to be taken from the mic's quietest frames, which are
    # loud here (loudspeaker leakage), so the leakage was subtracted away and everything became "Me".
    ok = total = wrong_me = 0
    for seed in range(8):
        mic, sys, truth, spans = _noisy_meeting(seed)
        out = speakers.label_segments([seg("x", a, b) for a, b in spans], mic, sys)
        for o, want in zip(out, truth):
            total += 1
            ok += o["speaker"] == want
            wrong_me += want == "Others" and o["speaker"] == "Me"
    assert ok / total > 0.95 and wrong_me / total < 0.03


# ---------- live timeline must not drift ----------

def test_drain_mixed_keeps_the_live_timeline_equal_to_real_time():
    # The two devices hand over slightly different amounts per call (as real ones do). Padding the
    # shorter one every call used to stretch the timeline by several seconds over 5 minutes.
    rng = np.random.default_rng(0)
    mic, sysa = _FakeTrack("mic", np.zeros(0, dtype="float32")), _FakeTrack("system", np.zeros(0, dtype="float32"))
    rec = recorder.Recorder()
    rec.tracks = [mic, sysa]
    fed = got = 0
    for _ in range(600):                                  # 600 calls ~ 5 minutes at 2 calls/s
        a, b = SR // 2 + int(rng.integers(-150, 150)), SR // 2 + int(rng.integers(-150, 150))
        mic._audio, sysa._audio = np.full(a, 0.1, dtype="float32"), np.full(b, 0.1, dtype="float32")
        fed += min(a, b)
        got += len(rec.drain_mixed())
    assert abs(got - fed) < SR * 0.3 + 300 * 150          # never ahead of the slower track by more than a block or so
    got += len(rec.drain_mixed(flush=True))
    assert got <= max(len(mic.buf16), 0) + SR * 300 + SR   # nothing is invented
    assert abs(rec.live_envelopes()["mic_env"].size / 10 - got / SR) < 0.5


def test_drain_mixed_pads_a_stalled_track_instead_of_waiting():
    rec = recorder.Recorder()
    live_t, dead = _FakeTrack("mic", np.full(SR * 5, 0.1, dtype="float32")), _FakeTrack("system", np.zeros(0, dtype="float32"))
    rec.tracks = [live_t, dead]
    out = rec.drain_mixed()
    assert len(out) == SR * 3                              # all but the last STALL_SEC seconds
    assert len(rec.drain_mixed(flush=True)) == SR * 2


# ---------- Me / Others decision rules ----------

def _env(*parts):
    return np.concatenate([np.full(int(sec * 10), v, dtype="float32") for sec, v in parts])


def test_label_me_when_mic_is_loud_and_system_silent_even_for_one_word():
    mic = _env((10, 0.3), (5, 0.3), (10, 0.0005))
    sys = _env((10, 0.05), (5, 0.0001), (10, 0.0001))
    out = speakers.label_segments([seg("Okay.", 10.5, 10.7)], mic, sys)
    assert out[0]["speaker"] == "Me"


def test_label_tolerates_timestamp_jitter_near_system_speech():
    # The segment sits half a second after the system audio ended (timestamps are approximate,
    # and the sound reaches the mic late): the mic is loudspeaker leakage, not the user.
    mic = _env((10, 0.3), (10, 0.0005))
    sys = _env((10, 0.05), (10, 0.0001))
    out = speakers.label_segments([seg("Okay.", 10.1, 10.3)], mic, sys)
    assert out[0]["speaker"] == "Others"


def test_label_others_when_mic_follows_system_even_if_leakage_is_unusually_strong():
    rng = np.random.default_rng(5)
    sys = rng.uniform(0.02, 0.08, 600).astype("float32")
    gain = np.r_[np.full(300, 0.6), np.full(300, 2.0)]    # leakage jumps 3x (a louder voice, a turned-up volume)
    mic = (0.0008 + gain * sys).astype("float32")
    out = speakers.label_segments([seg("a", 5, 25), seg("b", 35, 55)], mic, sys)
    assert [o["speaker"] for o in out] == ["Others", "Others"]


def test_label_me_when_user_talks_over_others_loudly():
    rng = np.random.default_rng(6)
    sys = rng.uniform(0.02, 0.08, 600).astype("float32")
    me = np.r_[np.zeros(300), rng.uniform(0.2, 0.5, 300)].astype("float32")
    mic = (0.0008 + 0.5 * sys + me).astype("float32")
    out = speakers.label_segments([seg("a", 5, 25), seg("b", 35, 55)], mic, sys)
    assert [o["speaker"] for o in out] == ["Others", "Me"]


# ---------- voiceprint ----------

import voiceprint


def _tone_embed(audio):
    """Fake embedder: a unit vector set by the audio's loudness, so tests need no model.
    Loud audio = 'the user's voice' direction, quiet audio = another direction."""
    if len(audio) < voiceprint.MIN_SEC * SR:
        return None
    rms = float(np.sqrt(np.mean(np.asarray(audio) ** 2)))
    return np.array([1.0, 0.0], dtype="float32") if rms > 0.1 else np.array([0.0, 1.0], dtype="float32")


def _speech(sec, amp=0.3, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(sec * SR)) * amp).astype("float32")


def test_build_profile_averages_windows_and_sets_a_threshold():
    profile = voiceprint.build_profile(_speech(30), embed_fn=_tone_embed)
    assert np.allclose(profile["embedding"], [1.0, 0.0]) and profile["windows"] >= voiceprint.MIN_WINDOWS
    assert 0.30 <= profile["threshold"] <= 0.50


def test_build_profile_rejects_silence_and_too_short_recordings():
    import pytest
    with pytest.raises(ValueError, match="Not enough speech"):
        voiceprint.build_profile(np.zeros(SR * 40, dtype="float32"), embed_fn=_tone_embed)
    with pytest.raises(ValueError, match="Not enough speech"):
        voiceprint.build_profile(_speech(8), embed_fn=_tone_embed)


def test_profile_roundtrip_is_private_and_deletable(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEPRINT_DIR", str(tmp_path / "vp"))
    assert voiceprint.load_profile() is None
    voiceprint.save_profile(voiceprint.build_profile(_speech(30), embed_fn=_tone_embed))
    path = voiceprint.profile_path()
    assert path.stat().st_mode & 0o777 == 0o600
    loaded = voiceprint.load_profile()
    assert np.allclose(loaded["embedding"], [1.0, 0.0]) and float(loaded["created"]) > 0
    assert voiceprint.delete_profile() is True and voiceprint.load_profile() is None
    assert voiceprint.delete_profile() is False


def test_threshold_can_be_overridden_from_the_environment(monkeypatch):
    profile = {"threshold": 0.4}
    assert voiceprint.threshold(profile) == 0.4
    monkeypatch.setenv("VOICEPRINT_THRESHOLD", "0.55")
    assert voiceprint.threshold(profile) == 0.55


def test_score_segments_slices_the_mic_and_skips_short_or_silent_stretches():
    profile = {"embedding": np.array([1.0, 0.0], dtype="float32")}
    mic = np.concatenate([_speech(5, 0.3), _speech(5, 0.02), np.zeros(SR * 5, dtype="float32")])
    segs = [seg("me", 0, 4), seg("them", 5.2, 9.5), seg("short", 1, 1.5), seg("silent", 10.5, 14.5), seg("empty", 3, 3)]
    scores = voiceprint.score_segments(segs, mic, profile, embed_fn=_tone_embed)
    assert scores[0] == 1.0 and scores[1] == 0.0 and scores[2] is None and scores[4] is None
    # real embed() refuses silence and clips that are too short, before touching the model
    assert voiceprint.embed(np.zeros(SR * 3, dtype="float32")) is None
    assert voiceprint.embed(_speech(0.5)) is None


def test_label_uses_the_voice_score_over_the_loudness_rules():
    # The mic is loud while the system track plays: loudness alone says "others" (leakage),
    # but the voiceprint recognises the user, and vice versa for a clip with the system silent.
    mic = _env((20, 0.3))
    sys = _env((10, 0.05), (10, 0.0001))
    segs = [seg("a", 2, 8), seg("b", 12, 18)]
    base = [s["speaker"] for s in speakers.label_segments(segs, mic, sys)]
    assert base == ["Others", "Me"]
    out = speakers.label_segments(segs, mic, sys, voice=[0.8, 0.1], voice_threshold=0.45)
    assert [s["speaker"] for s in out] == ["Me", "Others"] and out[0]["voice"] == 0.8


def test_label_voice_threshold_is_looser_when_the_system_is_silent_and_falls_back_when_unscored():
    mic = _env((20, 0.3))
    sys = _env((10, 0.05), (10, 0.0001))
    segs = [seg("a", 2, 8), seg("b", 12, 18), seg("c", 2, 3), seg("d", 12, 13)]
    out = speakers.label_segments(segs, mic, sys, voice=[0.40, 0.40, None, None], voice_threshold=0.45)
    # 0.40 is just under the threshold: not enough while others play, enough in a silent room
    assert [s["speaker"] for s in out] == ["Others", "Me", "Others", "Me"]
    # just over the threshold is still not enough while others play (leakage can pass for a match)
    just_over = speakers.label_segments(segs[:2], mic, sys, voice=[0.50, 0.50], voice_threshold=0.45)
    assert [s["speaker"] for s in just_over] == ["Others", "Me"]
    clear = speakers.label_segments(segs[:1], mic, sys, voice=[0.60], voice_threshold=0.45)
    assert clear[0]["speaker"] == "Me"
    assert "voice" not in out[2]                       # unscored segments use the loudness rules


def test_label_voice_never_overrides_a_silent_mic():
    mic = _env((10, 0.0003))
    sys = _env((10, 0.05))
    assert speakers.label_segments([seg("x", 1, 6)], mic, sys, voice=[0.95])[0]["speaker"] == "Others"


# ---------- voice enrolment API ----------

def test_voice_api_status_stop_without_sampling_and_delete(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import app as app_module
    monkeypatch.setenv("VOICEPRINT_DIR", str(tmp_path))
    c = TestClient(app_module.app)
    assert c.get("/api/voice").json() == {"enrolled": False, "created": None, "threshold": None, "sampling": None}
    assert c.post("/api/voice/stop").status_code == 409
    voiceprint.save_profile(voiceprint.build_profile(_speech(30), embed_fn=_tone_embed))
    st = c.get("/api/voice").json()
    assert st["enrolled"] is True and st["created"] > 0 and st["threshold"] > 0
    assert c.delete("/api/voice").json()["enrolled"] is False


# ---------- system track arriving late (Multi-Output Device) ----------

def _late_system_meeting(lag_frames=19, seed=11):
    """Others talk in bursts; the mic hears them at once, the system track `lag_frames` late.
    The user talks twice, when nobody else does. Returns envelopes and (start, end, truth) turns."""
    rng = np.random.default_rng(seed)
    n = 3000
    others = np.zeros(n)
    t = 20
    turns = []
    while t < n - 200:
        d = int(rng.integers(25, 70))
        others[t : t + d] = rng.uniform(0.04, 0.12, d)
        turns.append((t / 10, (t + d) / 10, "Others"))
        t += d + int(rng.integers(30, 60))
    user = np.zeros(n)
    for a, b in ((1200, 1260), (2200, 2270)):
        if others[a - 30 : b + 30].max() == 0:
            user[a:b] = rng.uniform(0.15, 0.35, b - a)
            turns.append((a / 10, b / 10, "Me"))
    mic = 0.0006 + 0.5 * others + user                 # leakage reaches the mic immediately
    sys = np.r_[np.zeros(lag_frames), others[:-lag_frames]] + 1e-4   # BlackHole's copy arrives late
    return mic.astype("float32"), sys.astype("float32"), turns


def test_estimate_lag_finds_the_delay_of_the_system_track():
    mic, sys, _ = _late_system_meeting(lag_frames=19)
    assert speakers.estimate_lag(mic, sys) == 19
    mic0, sys0, _ = _late_system_meeting(lag_frames=0)
    assert speakers.estimate_lag(mic0, sys0) == 0
    assert speakers.estimate_lag(mic[:100], sys[:100]) == 0            # too little to measure
    flat = np.full(1000, 0.1, dtype="float32")
    assert speakers.estimate_lag(flat, flat) == 0


def test_shift_earlier_moves_a_late_curve_back():
    env = np.arange(10, dtype="float32")
    assert speakers.shift_earlier(env, 3).tolist() == [3, 4, 5, 6, 7, 8, 9, 0, 0, 0]
    assert speakers.shift_earlier(env, -2).tolist() == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
    assert speakers.shift_earlier(env, 0) is env


def test_late_system_track_does_not_turn_others_into_me(monkeypatch):
    # Regression: the mic hears others ~2 s before BlackHole delivers them, so "mic loud while the
    # system track is silent" fired on every turn and labelled other people as the user.
    mic, sys, turns = _late_system_meeting()
    segs = [seg("x", a, b) for a, b, _ in turns]
    labels = [s["speaker"] for s in speakers.label_segments(segs, mic, sys)]
    assert labels == [who for _, _, who in turns]
    # without the compensation the same recording does get others labelled "Me"
    monkeypatch.setattr(speakers, "estimate_lag", lambda *a, **k: 0)
    wrong = [s["speaker"] for s, (_, _, who) in zip(speakers.label_segments(segs, mic, sys), turns)
             if who == "Others" and s["speaker"] == "Me"]
    assert len(wrong) > 3


# ---------- echo cancellation ----------

import aec
from scipy.signal import butter, fftconvolve, lfilter


def _speechlike(n, bursts, seed):
    """Band-limited noise that is on during `bursts` (start_s, end_s) and silent otherwise."""
    rng = np.random.default_rng(seed)
    b, a = butter(2, [200, 3500], btype="band", fs=SR)
    x = lfilter(b, a, rng.standard_normal(n))
    env = np.zeros(n)
    for s, e in bursts:
        env[int(s * SR): int(e * SR)] = 1.0
    env = np.convolve(env, np.hanning(int(0.05 * SR)) / np.hanning(int(0.05 * SR)).sum(), mode="same")
    return (0.3 * x * env).astype("float32")


def _echo_meeting(lag_s=1.2, seconds=70):
    """Others talk on the loudspeakers (the mic hears them after a short room path, the system
    track `lag_s` later); the user talks twice, when the others are silent."""
    n = seconds * SR
    others_turns = [(2, 9), (12, 20), (24, 31), (46, 55), (58, 66)]
    user_turns = [(33, 40), (41, 44)]
    played = _speechlike(n, others_turns, 1)
    user = _speechlike(n, user_turns, 2)
    ir = np.zeros(int(0.12 * SR)); ir[int(0.030 * SR)] = 0.6; ir[int(0.045 * SR)] = 0.2; ir[int(0.070 * SR)] = 0.08
    heard = fftconvolve(played, ir)[:n]
    mic = (heard + user + np.random.default_rng(3).standard_normal(n) * 0.001).astype("float32")
    sys = np.zeros(n, dtype="float32")
    sys[int(lag_s * SR):] = played[: n - int(lag_s * SR)]
    return mic, sys, user, others_turns, user_turns


def _rms(x, a, b):
    return float(np.sqrt(np.mean(np.asarray(x[int(a * SR): int(b * SR)], dtype=float) ** 2)))


def test_aec_measures_the_delay_and_removes_the_loudspeaker_sound():
    mic, sys, user, others, users = _echo_meeting(lag_s=1.2)
    r = aec.cancel(mic, sys)
    assert r.ok and abs(r.lag_sec - (1.2 - 0.030)) < 0.005          # the main echo tap is 30 ms after the loudspeaker
    assert r.erle_db > 15
    for a, b in others:
        assert _rms(r.residual, a + 1, b - 1) < 0.25 * _rms(mic, a + 1, b - 1)   # echo-only: mostly gone


def test_aec_keeps_the_users_voice():
    mic, sys, user, others, users = _echo_meeting()
    r = aec.cancel(mic, sys)
    for a, b in users:
        assert _rms(r.residual, a + 0.5, b - 0.5) > 0.8 * _rms(mic, a + 0.5, b - 0.5)
    a, b = users[0]
    c = np.corrcoef(r.residual[int(a * SR): int(b * SR)], user[int(a * SR): int(b * SR)])[0, 1]
    assert c > 0.95


def test_aec_falls_back_to_the_mic_when_nothing_can_be_cancelled():
    mic, sys, _, _, _ = _echo_meeting()
    unrelated = _speechlike(len(mic), [(3, 10), (15, 25), (30, 45)], 99)
    r = aec.cancel(mic, unrelated)
    assert not r.ok and r.residual is mic or (not r.ok and np.array_equal(r.residual, mic))
    assert not aec.cancel(mic, np.zeros_like(mic)).ok
    assert not aec.cancel(mic[: SR * 5], sys[: SR * 5]).ok          # too short to measure the delay


def test_aec_advance_moves_a_late_recording_back():
    x = np.arange(6, dtype="float32")
    assert aec.advance(x, 2).tolist() == [2, 3, 4, 5, 0, 0]
    assert aec.advance(x, -2).tolist() == [0, 0, 0, 1, 2, 3]
    assert aec.advance(x, 0).tolist() == x.tolist()


def _frames(*parts):
    return np.concatenate([np.full(int(sec * 10), v, dtype="float32") for sec, v in parts])


def test_label_with_residual_decides_the_overlap_case_and_gates_the_voiceprint():
    # 10 s of others on the loudspeakers; the mic is loud throughout. After echo cancellation the
    # residual is small, except from 6 s on where the user also speaks.
    mic = _frames((10, 0.2))
    sys = _frames((10, 0.1))
    residual = _frames((6, 0.01), (4, 0.15))
    segs = [seg("a", 0, 5), seg("b", 6.5, 9.5)]
    out = speakers.label_segments(segs, mic, sys, residual=residual)
    assert [s["speaker"] for s in out] == ["Others", "Me"]
    # a voice match without a voice in the residual does not make an echo "Me"
    out = speakers.label_segments(segs, mic, sys, residual=residual, voice=[0.9, 0.9], voice_threshold=0.45)
    assert [s["speaker"] for s in out] == ["Others", "Me"]
    # and without a residual the same voice scores decide alone
    out = speakers.label_segments(segs, mic, sys, voice=[0.9, 0.9], voice_threshold=0.45)
    assert [s["speaker"] for s in out] == ["Me", "Me"]


def test_release_memory_unloads_parakeet_model_unless_asked_to_keep_it():
    transcriber._parakeet_cache["repo"] = object()
    transcriber.release_memory(unload_models=False)
    assert "repo" in transcriber._parakeet_cache
    transcriber.release_memory()
    assert transcriber._parakeet_cache == {}


def test_recorder_release_audio_drops_kept_arrays():
    rec = recorder.Recorder()
    rec.mic_audio = np.ones(16000, dtype=np.float32)
    rec.sys_audio = np.ones(16000, dtype=np.float32)
    rec.tail = np.ones(100, dtype=np.float32)
    rec.envelopes = {"mic_env": 1}
    rec.release_audio()
    assert rec.mic_audio is None and rec.sys_audio is None and rec.envelopes is None and len(rec.tail) == 0


# ---------- telling the other participants apart ----------

def test_diarize_merge_small_folds_short_voices_and_renumbers_by_first_appearance():
    import diarize
    turns = [(0, 40, "b"), (40, 42, "z"), (42, 90, "a"), (90, 100, "z"), (100, 130, "b")]
    merged = diarize.merge_small(turns, min_sec=20)
    assert [t[2] for t in merged] == [1, 1, 2, 2, 1]       # "z" joins the nearest bigger voice in time
    assert diarize.merge_small([]) == []


def test_diarize_merge_small_keeps_the_biggest_voice_when_nobody_reaches_the_minimum():
    import diarize
    merged = diarize.merge_small([(0, 3, "a"), (3, 4, "b")], min_sec=20)
    assert {t[2] for t in merged} == {1}


def test_diarize_merge_small_caps_the_number_of_speakers():
    import diarize
    turns = [(i * 30, i * 30 + 30, i) for i in range(5)]
    assert len({t[2] for t in diarize.merge_small(turns, min_sec=10, max_speakers=3)}) == 3


def test_diarize_label_segments_only_relabels_others():
    import diarize
    turns = [(0, 10, 1), (10, 20, 2)]
    segs = [{"start": 1, "end": 5, "text": "a", "speaker": "Others"},
            {"start": 9, "end": 15, "text": "b", "speaker": "Others"},     # 1s with speaker 1, 5s with speaker 2
            {"start": 2, "end": 4, "text": "c", "speaker": "Me"},
            {"start": 30, "end": 31, "text": "d", "speaker": "Others"},     # nobody covers it
            {"start": 3, "end": 4, "text": "e"}]                            # no label at all
    out = diarize.label_segments(segs, turns)
    assert [s.get("speaker") for s in out] == ["Speaker 1", "Speaker 2", "Me", "Others", None]
    assert segs[0]["speaker"] == "Others"                                   # the input is not modified


def test_diarize_job_rewrites_transcript_and_notes_and_fails_soft(tmp_path, monkeypatch):
    import json
    import app
    folder = tmp_path / "2026-01-01_000000_x"
    folder.mkdir()
    segs = [{"start": 0, "end": 5, "text": "hello", "speaker": "Others"}]
    (folder / "transcript.json").write_text(json.dumps(segs))
    (folder / "notes.json").write_text(json.dumps({"title": "T", "summary": "S"}))
    (folder / "meta.json").write_text(json.dumps({"provider": "claude", "date": "d", "duration_sec": 60}))
    monkeypatch.setattr(app.diarize, "diarize", lambda audio: [(0, 5, "k")])
    monkeypatch.setattr(app, "_release_memory", lambda: None)
    app._diarize_job(folder, object())
    assert json.loads((folder / "transcript.json").read_text())[0]["speaker"] == "Speaker 1"
    assert "Speaker 1: hello" in (folder / "notes.md").read_text()
    assert json.loads((folder / "meta.json").read_text())["diarize"]["speakers"] == 1
    assert app._jobs[folder.name]["stage"] == "done"

    def boom(audio):
        raise RuntimeError("model missing")
    (folder / "transcript.json").write_text(json.dumps(segs))
    monkeypatch.setattr(app.diarize, "diarize", boom)
    app._diarize_job(folder, object())
    assert json.loads((folder / "transcript.json").read_text())[0]["speaker"] == "Others"
    assert app._jobs[folder.name]["stage"] == "done"


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               12345.
Pageins:                               9999999.
Swapins:                                    100.
Swapouts:                                   300.
"""


def test_parse_vm_stat_counts_only_swap_traffic():
    assert metrics.parse_vm_stat(VM_STAT) == 400 * 16384   # Pageins (file reads) are not swap
    assert metrics.parse_vm_stat("garbage") is None


def test_swap_rate_over_the_window():
    mb = 1024 ** 2
    assert metrics.swap_rate([]) is None
    assert metrics.swap_rate([(0.0, 0)]) is None
    assert metrics.swap_rate([(0.0, 0), (5.0, 10 * mb), (10.0, 50 * mb)]) == 5.0
    assert metrics.swap_rate([(0.0, 10 * mb), (10.0, 0)]) == 0.0   # counter reset never goes negative


def test_memory_status_ignores_old_swap_and_follows_pressure_and_traffic():
    assert metrics.memory_status("normal", 0.0) == "ok"     # gigabytes of old swap alone are fine
    assert metrics.memory_status(None, None) == "ok"
    assert metrics.memory_status("normal", 1.9) == "ok"
    assert metrics.memory_status("normal", 2.0) == "warn"
    assert metrics.memory_status("warn", 0.0) == "warn"
    assert metrics.memory_status("normal", 20.0) == "bad"
    assert metrics.memory_status("critical", 0.0) == "bad"


def test_new_call_during_auto_recording_asks_and_splits_on_new_meeting(monkeypatch):
    import app
    stopped = _fake_recording(monkeypatch, app, "2026-11-10_teams")
    monkeypatch.setitem(app._auto_started, "id", "2026-11-10_teams")
    started = []

    def fake_start(body):
        app._recorder.is_recording = True
        started.append(body.title)
        return {"id": "2026-11-10_next"}

    def fake_stop(keep_sec=None):
        stopped.append(True)
        app._recorder.is_recording = False

    monkeypatch.setattr(app, "_stop_recording", fake_stop)
    monkeypatch.setattr(app, "start", fake_start)
    monkeypatch.setattr(app, "_default_devices", lambda: [0, 1])
    monkeypatch.setattr(app, "_ask_to_record", lambda n: pytest.fail("must not ask to record again"))
    monkeypatch.setattr(app, "_ask_new_meeting", lambda n: True)
    app._on_meeting_detected("com.microsoft.teams2")
    assert stopped == [True] and started == ["Teams meeting"]
    assert app._auto_started["id"] == "2026-11-10_next"


def test_new_call_during_auto_recording_keeps_recording_on_same_meeting(monkeypatch):
    import app
    stopped = _fake_recording(monkeypatch, app, "2026-11-10_teams")
    monkeypatch.setitem(app._auto_started, "id", "2026-11-10_teams")
    monkeypatch.setattr(app, "_ask_new_meeting", lambda n: False)
    app._on_meeting_detected("com.microsoft.teams2")
    assert stopped == []
    assert app._auto_started["id"] == "2026-11-10_teams"


def test_new_call_during_manual_recording_does_not_ask(monkeypatch):
    import app
    stopped = _fake_recording(monkeypatch, app, "2026-11-10_manual")
    monkeypatch.setitem(app._auto_started, "id", None)
    monkeypatch.setattr(app, "_ask_new_meeting", lambda n: pytest.fail("manual recordings are not split"))
    app._on_meeting_detected("com.microsoft.teams2")
    assert stopped == []


def test_detector_passes_release_time_of_the_last_release():
    import meeting_detector
    clock = iter([100.0, 200.0, 300.0])
    teams, off = {"com.microsoft.teams2"}, set()
    frames = [teams, teams, off, teams, off, off, off, off]   # brief drop, mic back, then gone for good
    it, ends = iter(frames), []
    d = meeting_detector.MeetingDetector(lambda b: None, probe=lambda: next(it), on_end=ends.append,
                                         debounce=2, release=2, end_polls=4, clock=lambda: next(clock))
    for _ in frames:
        d.step()
    assert ends == [200.0]   # the final release, not the brief drop at 100


def test_keep_seconds_cuts_after_release_plus_margin():
    import app
    assert app._keep_seconds(released_at=1300.0, started_at=1000.0, elapsed=340.0) == 300.0 + app.END_MARGIN_SEC
    assert app._keep_seconds(None, 1000.0, 340.0) is None            # release unknown: keep everything
    assert app._keep_seconds(900.0, 1000.0, 340.0) is None           # released before the recording began
    assert app._keep_seconds(1339.0, 1000.0, 340.0) is None          # nothing worth cutting


def test_auto_stop_passes_the_cut_point(monkeypatch):
    import app
    _fake_recording(monkeypatch, app, "2026-11-10_teams")
    monkeypatch.setitem(app._auto_started, "id", "2026-11-10_teams")
    cuts = []
    monkeypatch.setattr(app, "_stop_recording", lambda keep_sec=None: cuts.append(keep_sec))
    app._on_meeting_ended(released_at=1000.0 + 560.0)   # recording started at 1000, now 600 s in
    assert cuts == [560.0 + app.END_MARGIN_SEC]


def test_trim_segments_drops_and_clips_after_cut():
    import app
    segs = [{"start": 0.0, "end": 5.0, "text": "a"}, {"start": 8.0, "end": 12.0, "text": "b"},
            {"start": 12.0, "end": 14.0, "text": "c"}]
    assert app.trim_segments(segs, 10.0) == [{"start": 0.0, "end": 5.0, "text": "a"},
                                            {"start": 8.0, "end": 10.0, "text": "b"}]


def test_recorder_stop_keeps_only_keep_sec(monkeypatch, tmp_path):
    import soundfile as sf
    rec = recorder.Recorder()
    track = type("T", (), {"role": "mic", "path": tmp_path / "t.flac", "stop": lambda self: None})()
    sf.write(str(track.path), np.full(10 * recorder.TARGET_SR, 0.1, dtype=np.float32), recorder.TARGET_SR)
    rec.tracks, rec.started_at = [track], 1.0
    monkeypatch.setattr(rec, "drain_mixed", lambda flush=False: np.zeros(0, dtype=np.float32))
    monkeypatch.setattr(recorder, "_load_resampled", lambda p: sf.read(str(p), dtype="float32")[0])
    audio, seconds = rec.stop(tmp_path / "audio.flac", keep_sec=4.0)
    assert seconds == 4.0 and len(audio) == 4 * recorder.TARGET_SR
    assert sf.info(str(tmp_path / "audio.flac")).duration == 4.0


def test_detector_app_on_mic_only_while_meeting_holds_it():
    import meeting_detector
    teams, off = {"com.microsoft.teams2"}, set()
    frames = [teams, teams, teams, off, off]
    it = iter(frames)
    d = meeting_detector.MeetingDetector(lambda b: None, probe=lambda: next(it), debounce=2, release=2, end_polls=4)
    seen = []
    for _ in frames:
        d.poll()
        seen.append(d.app_on_mic())
    assert seen == [False, True, True, False, False]   # not during the debounce, not during the end grace


def test_manual_start_during_a_call_is_auto_stopped(monkeypatch):
    import app
    fake = type("D", (), {"on_end": print, "app_on_mic": lambda self: True})()
    monkeypatch.setattr(app, "_detector", fake)
    assert app._meeting_on_mic() is True
    monkeypatch.setattr(fake, "app_on_mic", lambda: False, raising=False)
    assert app._meeting_on_mic() is False                  # Teams not on the mic: recording is the user's
    fake.app_on_mic, fake.on_end = (lambda: True), None
    assert app._meeting_on_mic() is False                  # MEETING_AUTO_STOP=0
    monkeypatch.setattr(app, "_detector", None)
    assert app._meeting_on_mic() is False                  # detector off (MEETING_DETECT=0 / not macOS)


# ---------- built-in system audio tap (systap) ----------

def test_systap_is_listed_and_treated_as_system_audio(monkeypatch):
    import systap, speakers
    monkeypatch.setattr(systap, "available", lambda: True)
    monkeypatch.setattr(recorder.sd, "query_devices", lambda: [])
    devs = recorder.list_input_devices()
    tap = next(d for d in devs if d["index"] == systap.SYSTEM_TAP_INDEX)
    assert speakers.role_of(tap["name"]) == "system"
    monkeypatch.setattr(systap, "available", lambda: False)
    assert all(d["index"] != systap.SYSTEM_TAP_INDEX for d in recorder.list_input_devices())


def test_systap_stream_delivers_blocks_and_reports_a_dead_helper(tmp_path, monkeypatch):
    import sys
    import time
    import systap
    fake = tmp_path / "fake_systap.py"
    fake.write_text("import sys, struct\n"
                    "sys.stdout.buffer.write(struct.pack('<%df' % 1920, *([0.25] * 1920)))\n"
                    "sys.stdout.buffer.flush()\n"
                    "import time; time.sleep(5)\n")
    wrapper = tmp_path / "systap"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {fake}\n")
    wrapper.chmod(0o755)
    monkeypatch.setenv("SYSTAP_BIN", str(wrapper))
    got = []
    s = systap.TapStream(callback=lambda indata, frames, t, st: got.append(indata))
    s.start()
    for _ in range(50):
        if len(got) >= 2:
            break
        time.sleep(0.05)
    s.stop()
    assert len(got) >= 2 and got[0].shape == (systap.BLOCK_SAMPLES, 1) and abs(float(got[0][0, 0]) - 0.25) < 1e-6

    bad = tmp_path / "bad"
    bad.write_text("#!/bin/sh\necho 'no permission' >&2\nexit 1\n")
    bad.chmod(0o755)
    monkeypatch.setenv("SYSTAP_BIN", str(bad))
    with pytest.raises(RuntimeError, match="no permission"):
        systap.TapStream(callback=lambda *a: None).start()


def test_recorder_track_can_use_the_systap_pseudo_device(tmp_path, monkeypatch):
    import systap
    created = {}

    class FakeTap:
        def __init__(self, samplerate, channels, callback):
            created.update(sr=samplerate, ch=channels, cb=callback)
        def start(self): pass
        def stop(self): pass
        def close(self): pass

    monkeypatch.setattr(systap, "TapStream", FakeTap)
    t = recorder._Track(systap.SYSTEM_TAP_INDEX, tmp_path / "t.wav")
    assert t.role == "system" and t.sr == 48000 and created["ch"] == 1
    t.start()
    created["cb"](np.full((960, 1), 0.3, dtype="float32"), 960, None, None)
    t.stop()
    assert t.peak > 0.29


# ---------- cloud summary provider ----------

def test_cloud_device_id_is_created_once_and_private(tmp_path, monkeypatch):
    import cloud
    monkeypatch.setenv("DEVICE_ID_FILE", str(tmp_path / "sub" / "device_id"))
    a = cloud.device_id()
    assert a == cloud.device_id() and len(a) == 36
    assert (tmp_path / "sub" / "device_id").stat().st_mode & 0o777 == 0o600
    (tmp_path / "sub" / "device_id").write_text("garbage")
    assert cloud.device_id() != "garbage"             # a corrupt file is replaced


class _Resp:
    def __init__(self, status, data):
        self.status_code, self._d, self.ok = status, data, status < 400
    def json(self):
        return self._d


def test_cloud_summarize_sends_only_text_and_reports_quota_errors(tmp_path, monkeypatch):
    import cloud
    monkeypatch.setenv("DEVICE_ID_FILE", str(tmp_path / "id"))
    monkeypatch.setenv("CLOUD_API_URL", "https://proxy.example/")
    monkeypatch.setenv("CLOUD_INVITE_CODE", "beta1")
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, json=json)
        return _Resp(200, {"text": '{"title":"T"}', "usage": {"input_tokens": 5, "output_tokens": 2}})

    monkeypatch.setattr(cloud.requests, "post", fake_post)
    text, usage = cloud.summarize("Me: hi", "m1", "Standup", "English")
    assert text == '{"title":"T"}' and usage["output_tokens"] == 2
    assert sent["url"] == "https://proxy.example/v1/summarize"
    assert sent["headers"]["X-Invite-Code"] == "beta1" and len(sent["headers"]["X-Device-Id"]) == 36
    assert set(sent["json"]) == {"meeting_id", "transcript", "hint_title", "language"}   # no audio, no paths

    monkeypatch.setattr(cloud.requests, "post", lambda *a, **k: _Resp(429, {"error": "quota_exceeded", "limit": 5}))
    with pytest.raises(cloud.QuotaExceeded, match="5 free cloud summaries"):
        cloud.summarize("x", "m", None, "English")
    monkeypatch.setattr(cloud.requests, "post", lambda *a, **k: _Resp(502, {"message": "down"}))
    with pytest.raises(cloud.CloudError, match="down"):
        cloud.summarize("x", "m", None, "English")
    monkeypatch.delenv("CLOUD_API_URL")
    with pytest.raises(cloud.CloudError, match="not configured"):
        cloud.summarize("x", "m", None, "English")


def test_summarizer_cloud_provider_parses_json_and_records_usage(monkeypatch):
    import cloud
    seen = {}

    def fake(transcript, meeting_id, hint_title, language):
        seen.update(meeting_id=meeting_id, hint=hint_title)
        return 'Here: {"title": "Plan", "summary": "S", "topics": []}', {"input_tokens": 50, "output_tokens": 9}

    monkeypatch.setattr(cloud, "summarize", fake)
    notes = summarizer.summarize("Me: hello", provider="cloud", hint_title="X", meeting_id="2026_m")
    assert notes["title"] == "Plan" and notes["decisions"] == [] and seen == {"meeting_id": "2026_m", "hint": "X"}
    usage = summarizer.last_run_usage("cloud", None)
    assert usage["input_tokens"] == 50 and usage["output_tokens"] == 9 and usage["calls"] == 1


def test_app_accepts_cloud_provider_and_labels_it():
    import app
    assert app._split_provider("cloud") == ("cloud", None)
    assert app._provider_label("cloud", None) == "Cloud"
    assert app._provider_label("ollama", "qwen2.5:14b") == "Ollama (qwen2.5:14b)"


def test_librosa_shim_matches_librosa_mel_filterbank():
    librosa = pytest.importorskip("librosa")
    import _librosa_shim
    for sr, n_fft, n_mels in [(16000, 512, 80), (16000, 512, 128), (16000, 400, 80), (16000, 400, 128)]:
        ref = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=sr / 2, norm="slaney")
        got = _librosa_shim.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=sr / 2, norm="slaney")
        assert got.shape == ref.shape and got.dtype == ref.dtype
        assert np.allclose(got, ref, atol=1e-7), (sr, n_fft, n_mels, float(np.abs(got - ref).max()))
    with pytest.raises(NotImplementedError):
        _librosa_shim.mel(sr=16000, n_fft=512, htk=True)
