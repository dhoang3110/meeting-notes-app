"""Benchmark speech-to-text and summary models on the golden recordings.

    .venv/bin/python tests/bench.py                                  # current .env configuration
    .venv/bin/python tests/bench.py --llm ollama:qwen2.5:14b ollama:qwen2.5:7b
    .venv/bin/python tests/bench.py --stt whisper:mlx-community/whisper-large-v3-turbo parakeet:mlx-community/parakeet-tdt-0.6b-v2
    .venv/bin/python tests/bench.py --only monologue-3min --skip-llm --label after-change

Results go to tests/results/<time>_<label>.json and tests/results/latest.md. Every number is
compared with the previous run of the same configuration, so a slowdown or a quality
drop after a code change shows up as a delta.

Quality columns are proxies, not human judgement:
  wer       word error rate against the silver reference transcript (lower is better)
  coverage  share of the manifest's key terms that survive into the notes (higher is better)"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers  # noqa: E402
from helpers import ResourceSampler  # noqa: E402

import requests  # noqa: E402
import summarizer  # noqa: E402
import transcriber  # noqa: E402

RESULTS = helpers.GOLDEN.parent / "results"


def text_of(segs):
    return " ".join(s["text"] for s in segs)


# ---------------- speech to text ----------------

def bench_stt(item, model, live_mode):
    os.environ["TRANSCRIBE_MODEL"] = model   # "whisper:<repo>" or "parakeet:<repo>"
    ref = text_of(helpers.reference_segments(item))
    with ResourceSampler() as rs:
        t0 = time.time()
        if live_mode:
            segs, lt = helpers.transcribe_live(item, chunk_sec=10, partial_sec=0)
            extra = {"final_rtf": lt.stats()["final_rtf"]}
        else:
            segs, extra = helpers.transcribe_full(item), {}
        took = time.time() - t0
    return {"kind": "stt-live" if live_mode else "stt-full", "config": model, "recording": item["name"],
            "seconds": round(took, 1), "x_realtime": round(item["duration_sec"] / took, 1),
            "wer": round(helpers.wer(ref, text_of(segs)), 3),
            "repeat_run": helpers.max_repeat_run(segs), **extra, **rs.summary()}


# ---------------- summaries ----------------

def unload_ollama():
    """Free the previous model so each configuration starts cold and RAM numbers are fair."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        for m in requests.get(f"{host}/api/ps", timeout=2).json().get("models", []):
            requests.post(f"{host}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=30)
        time.sleep(2)
    except Exception:
        pass


def bench_llm(item, spec):
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        os.environ["OLLAMA_MODEL"] = model
    elif model:
        os.environ["CLAUDE_MODEL"] = model
    text = transcriber.segments_to_text(helpers.reference_segments(item))
    tokens_before = summarizer.llm_stats()["total_tokens"]
    with ResourceSampler() as rs:
        t0 = time.time()
        notes = summarizer.summarize(text, provider=provider)
        took = time.time() - t0
    st = summarizer.llm_stats()
    out_tokens = st["total_tokens"] - tokens_before
    last = st["last"] or {}
    return {"kind": "llm", "config": spec, "recording": item["name"],
            "seconds": round(took, 1), "out_tokens": out_tokens,
            "tok_s": round(out_tokens / took, 1) if took else None,   # end to end, includes reading the transcript
            "gen_tok_s": last.get("gen_tok_s"), "load_sec": last.get("load_sec"),
            "coverage": round(helpers.term_coverage(helpers.notes_text(notes), item["key_terms"]), 2),
            "action_items": len(notes.get("action_items", [])), **rs.summary()}


# ---------------- reporting ----------------

COLUMNS = {
    "stt-full": ["seconds", "x_realtime", "wer", "repeat_run", "peak_ram_gb", "gpu_avg_pct"],
    "stt-live": ["seconds", "x_realtime", "wer", "repeat_run", "final_rtf", "peak_ram_gb", "gpu_avg_pct"],
    "llm": ["seconds", "out_tokens", "tok_s", "gen_tok_s", "load_sec", "coverage", "action_items",
            "peak_ram_gb", "swap_growth_gb", "gpu_avg_pct"],
}
BETTER = {"seconds": -1, "x_realtime": 1, "wer": -1, "tok_s": 1, "gen_tok_s": 1, "coverage": 1,
          "peak_ram_gb": -1, "swap_growth_gb": -1, "load_sec": -1}


def previous_run(current_name):
    files = sorted(p for p in RESULTS.glob("*.json") if p.name != current_name)
    return json.loads(files[-1].read_text()) if files else None


def delta(row, old, col):
    new, was = row.get(col), (old or {}).get(col)
    if not isinstance(new, (int, float)) or not isinstance(was, (int, float)) or was == 0 or col not in BETTER:
        return ""
    change = (new - was) / abs(was)
    if abs(change) < 0.05:
        return ""
    arrow = "▲" if BETTER[col] * change > 0 else "▼"   # ▲ better, ▼ worse
    return f" {arrow}{abs(change):.0%}"


def report(rows, prev, meta):
    old = {(r["kind"], r["config"], r["recording"]): r for r in (prev or {}).get("rows", [])}
    out = [f"# Benchmark {meta['when']} ({meta['label']})",
           f"Machine: {meta['machine']}. ▲ better / ▼ worse than the previous run "
           f"({prev['meta']['when'] if prev else 'none yet'}), changes under 5% hidden.", ""]
    for kind, cols in COLUMNS.items():
        part = [r for r in rows if r["kind"] == kind]
        if not part:
            continue
        out += [f"## {kind}", "", "| config | recording | " + " | ".join(cols) + " |",
                "|---|---|" + "---|" * len(cols)]
        for r in part:
            if "error" in r:
                out.append(f"| {r['config']} | {r['recording']} | **error:** {r['error'][:100]} |" + " |" * (len(cols) - 1))
                continue
            o = old.get((r["kind"], r["config"], r["recording"]))
            out.append(f"| {r['config']} | {r['recording']} | " +
                       " | ".join(f"{'' if r.get(c) is None else r[c]}{delta(r, o, c)}" for c in cols) + " |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt", "--whisper", nargs="*", dest="stt", default=None,
                    help="speech-to-text engines, e.g. whisper:mlx-community/whisper-large-v3-turbo "
                         "parakeet:mlx-community/parakeet-tdt-0.6b-v2 (default: the one in .env)")
    ap.add_argument("--llm", nargs="*", default=None,
                    help="ollama:<model> or claude[:<model>] (default: the Ollama model in .env)")
    ap.add_argument("--only", nargs="*", help="recording names (default: all present)")
    ap.add_argument("--include-long", action="store_true",
                    help="also run recordings over 10 minutes (slow, keeps the GPU busy for a long time)")
    ap.add_argument("--skip-stt", action="store_true")
    ap.add_argument("--skip-live", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(helpers.ROOT / ".env")
    stt_models = args.stt or [transcriber.default_engine()]
    llms = args.llm if args.llm is not None else [f"ollama:{os.getenv('OLLAMA_MODEL', 'qwen2.5:14b')}"]
    items = [i for i in helpers.golden_items()
             if (i["name"] in args.only if args.only else args.include_long or i["duration_sec"] < 600)]
    rows = []

    def run(fn, *a):
        try:
            row = fn(*a)
        except Exception as e:  # keep going: a missing model should not hide the other results
            kind = {"bench_stt": "stt-live" if a[-1] else "stt-full", "bench_llm": "llm"}[fn.__name__]
            row = {"kind": kind, "config": a[1], "recording": a[0]["name"], "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        print("  ", {k: v for k, v in row.items() if k not in ("kind",)}, flush=True)

    if not args.skip_stt:
        for model in stt_models:
            for item in items:
                print(f"[stt-full] {model} / {item['name']}", flush=True)
                run(bench_stt, item, model, False)
                if not args.skip_live:
                    print(f"[stt-live] {model} / {item['name']}", flush=True)
                    run(bench_stt, item, model, True)
    if not args.skip_llm:
        for spec in llms:
            if spec.startswith("ollama"):
                unload_ollama()
            for item in items:
                print(f"[llm] {spec} / {item['name']}", flush=True)
                run(bench_llm, item, spec)

    import platform
    import psutil
    now = datetime.now()
    meta = {"when": now.strftime("%Y-%m-%d %H:%M"), "label": args.label,
            "machine": f"{platform.machine()}, {psutil.virtual_memory().total // 1024 ** 3} GB RAM"}
    RESULTS.mkdir(exist_ok=True)
    name = f"{now.strftime('%Y%m%d_%H%M%S')}_{args.label}.json"
    prev = previous_run(name)
    (RESULTS / name).write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    md = report(rows, prev, meta)
    (RESULTS / "latest.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
