"""Golden-dataset tests: real recordings through the real pipelines.

    .venv/bin/python -m pytest tests/test_golden.py            # short recordings
    .venv/bin/python -m pytest tests/test_golden.py --run-slow # + the 27-minute meeting
    .venv/bin/python -m pytest tests/test_golden.py  -m llm

Thresholds are deliberately loose: they catch real regressions (broken chunking, a
model that falls behind real time, a summary that loses the main topics), not noise.
For side-by-side numbers across models use tests/bench.py."""

import json
import os
import time

import pytest

import helpers
import summarizer
import transcriber

WER_FULL_MAX = 0.03      # same model, same audio: should reproduce the reference
WER_LIVE_MAX = 0.12      # chunked live transcription loses a little context
RTF_FULL_MAX = 0.25      # must run at least 4x faster than real time
RTF_LIVE_MAX = 0.5       # live must keep up comfortably while recording
KEY_TERM_MIN = 0.6       # share of key terms a summary must keep


def text_of(segs):
    return " ".join(s["text"] for s in segs)


def test_golden_dataset_is_present():
    assert helpers.golden_items(), "no golden recordings found: run tests/build_golden.py"


def test_full_file_transcript_matches_reference(golden):
    t0 = time.time()
    segs = helpers.transcribe_full(golden)
    took = time.time() - t0
    ref = text_of(helpers.reference_segments(golden))
    assert helpers.wer(ref, text_of(segs)) <= WER_FULL_MAX
    assert helpers.max_repeat_run(segs) <= 2
    assert took / golden["duration_sec"] <= RTF_FULL_MAX, f"too slow: {took:.0f}s for {golden['duration_sec']:.0f}s"


def test_live_transcript_is_close_to_reference_and_keeps_up(golden):
    t0 = time.time()
    segs, lt = helpers.transcribe_live(golden, chunk_sec=10, partial_sec=0)
    took = time.time() - t0
    ref = text_of(helpers.reference_segments(golden))
    assert lt.error is None
    assert helpers.wer(ref, text_of(segs)) <= WER_LIVE_MAX
    assert helpers.max_repeat_run(segs) <= 2
    assert (lt.stats()["final_rtf"] or 0) <= RTF_LIVE_MAX
    starts = [s["start"] for s in segs]
    assert starts == sorted(starts) and starts[-1] <= golden["duration_sec"] + 1


@pytest.mark.llm
def test_summary_keeps_the_main_topics(golden, provider):
    if provider == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("no ANTHROPIC_API_KEY")
    if provider == "ollama" and helpers.metrics_ollama() is None:
        pytest.skip("Ollama is not running")
    text = transcriber.segments_to_text(helpers.reference_segments(golden))
    notes = summarizer.summarize(text, provider=provider)
    assert notes["summary"].strip()
    assert isinstance(notes["action_items"], list) and isinstance(notes["decisions"], list)
    cover = helpers.term_coverage(helpers.notes_text(notes), golden["key_terms"])
    assert cover >= KEY_TERM_MIN, f"only {cover:.0%} of {golden['key_terms']} found in the notes"
    md = summarizer.render_markdown(notes, {"date": "", "duration": "", "provider": provider}, text)
    assert md.startswith("# ")
