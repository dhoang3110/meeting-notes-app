"""Build (or refresh) the golden dataset from recordings already saved in ~/MeetingNotes.

    .venv/bin/python tests/build_golden.py            # add anything missing
    .venv/bin/python tests/build_golden.py --refresh  # regenerate reference transcripts

Each recording gets: audio.flac (git-ignored, it is real meeting audio), reference.json /
reference.txt (a *silver* reference: the best-quality full-file Whisper transcript, NOT
human-verified; correct reference.txt by hand to make it a true golden), and an entry in
manifest.json. key_terms in the manifest are edited by hand and kept across refreshes."""

import json
import os
import shutil
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import GOLDEN, salient_terms  # noqa: E402

import soundfile as sf  # noqa: E402
import transcriber  # noqa: E402

NOTES_DIR = Path(os.path.expanduser(os.getenv("NOTES_DIR", "~/MeetingNotes")))
SOURCES = {  # golden name -> meeting folder it is copied from
    "monologue-3min": "2026-11-04_195655_meeting",
    "product-demo-1min": "2026-11-05_162734_meeting",
    "conversation-27min": "2026-11-05_100232_meeting",
}
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


def main(refresh: bool):
    mf = GOLDEN / "manifest.json"
    old = {r["name"]: r for r in json.loads(mf.read_text())["recordings"]} if mf.exists() else {}
    recordings = []
    for name, folder in SOURCES.items():
        dest = GOLDEN / name
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / "audio.flac").exists():
            shutil.copy(NOTES_DIR / folder / "audio.flac", dest / "audio.flac")
        entry = old.get(name, {})
        if refresh or not (dest / "reference.json").exists():
            audio, sr = sf.read(dest / "audio.flac", dtype="float32")
            t0 = time.time()
            os.environ["WHISPER_MODEL"] = WHISPER_MODEL
            segs = transcriber.transcribe(audio, language="en")
            print(f"{name}: reference made in {time.time() - t0:.0f}s ({len(segs)} segments)")
            (dest / "reference.json").write_text(json.dumps(segs, indent=2, ensure_ascii=False))
            text = " ".join(s["text"] for s in segs)
            (dest / "reference.txt").write_text(text + "\n")
            entry.update(duration_sec=round(len(audio) / sr, 1), words=len(text.split()))
            print("  candidate key terms (edit manifest by hand):", salient_terms(text))
        entry.update(name=name, source_folder=folder, language="en", whisper_model=WHISPER_MODEL,
                     reference_kind="silver: machine-made, not human-verified",
                     created=entry.get("created", str(date.today())))
        entry.setdefault("key_terms", [])
        entry.setdefault("expects_action_items", False)
        recordings.append(entry)
    mf.write_text(json.dumps({"recordings": recordings}, indent=2, ensure_ascii=False) + "\n")
    print("manifest written:", mf)


if __name__ == "__main__":
    main("--refresh" in sys.argv)
