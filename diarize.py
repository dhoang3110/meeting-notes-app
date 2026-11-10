"""Tell the other participants apart ("Speaker 1", "Speaker 2", ...).

Runs on the system-audio track only (what the loudspeakers play): it holds just the remote voices,
without the user's mic and without echo, so clustering voices there is far cleaner than on the mix.
sherpa-onnx finds speech turns (a pyannote segmentation model) and groups them by voice (the same
embedding model the voiceprint uses). The user's own speech ("Me") is decided elsewhere and never
touched here: only segments labelled "Others" are split into "Speaker N".

The number of speakers is not known in advance, so the clustering threshold decides it: too low
splits one person into several, too high merges different people. merge_small() then folds clusters
that only spoke a few seconds into the nearest bigger one.

Both models are downloaded on purpose, not in the background: `python diarize.py --download`."""

import os
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import voiceprint

SR = 16000
SEG_DIR = "sherpa-onnx-pyannote-segmentation-3-0"
SEG_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           f"speaker-segmentation-models/{SEG_DIR}.tar.bz2")
SEG_ARCHIVE_BYTES = 6_958_444

THRESHOLD = 0.7          # clustering distance: lower = more speakers (0.5 split one meeting into 10, 0.85 merged to 2)
MIN_SPEAKER_SEC = 20.0   # a voice that spoke less than this in total is merged into the nearest bigger one
MAX_SPEAKERS = 8


def seg_model_path() -> Path:
    return voiceprint.data_dir() / "models" / SEG_DIR / "model.int8.onnx"


def available() -> bool:
    """Both models are on disk (nothing is downloaded behind the user's back)."""
    return seg_model_path().exists() and voiceprint.model_ready()


def download() -> None:
    """Fetch the segmentation model (about 7 MB) and the embedding model if it is missing."""
    import requests
    voiceprint.ensure_model()
    if seg_model_path().exists():
        return
    models = seg_model_path().parent.parent
    models.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=models, suffix=".part")
    try:
        with requests.get(SEG_URL, stream=True, timeout=30) as r, os.fdopen(fd, "wb") as f:
            r.raise_for_status()
            for block in r.iter_content(1 << 20):
                f.write(block)
        if Path(tmp).stat().st_size != SEG_ARCHIVE_BYTES:
            raise RuntimeError("The downloaded segmentation model has an unexpected size; try again.")
        with tarfile.open(tmp, "r:bz2") as tar:
            wanted = [m for m in tar.getmembers() if m.isfile() and m.name.startswith(f"{SEG_DIR}/")
                      and ".." not in Path(m.name).parts]
            tar.extractall(models, members=wanted, filter="data")
    finally:
        Path(tmp).unlink(missing_ok=True)


def diarize(audio: np.ndarray, threshold: float = THRESHOLD, threads: int | None = None) -> list[tuple]:
    """[(start_sec, end_sec, speaker_id)] for 16 kHz mono audio, ordered by start. Ids are arbitrary."""
    import sherpa_onnx
    threads = threads or int(os.getenv("DIARIZE_THREADS", "4"))
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_model_path())),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(voiceprint.model_path()),
                                                              num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    # The audio goes in as recorded: scaling it to a fixed peak moved the clustering from 6 voices to 4
    # on the same meeting, so there is no normalisation here.
    audio = np.asarray(audio, dtype=np.float32)
    return [(r.start, r.end, r.speaker) for r in sd.process(audio).sort_by_start_time()]


def _gap(a: tuple, b: tuple) -> float:
    """Seconds between two (start, end, ...) intervals; 0 when they touch or overlap."""
    return max(a[0] - b[1], b[0] - a[1], 0.0)


def merge_small(turns: list[tuple], min_sec: float = MIN_SPEAKER_SEC,
                max_speakers: int = MAX_SPEAKERS) -> list[tuple]:
    """Fold voices that hardly spoke into the nearest (in time) voice that did, keep at most
    max_speakers, and renumber the speakers 1, 2, ... in order of first appearance."""
    if not turns:
        return []
    total: dict = defaultdict(float)
    for start, end, who in turns:
        total[who] += end - start
    ranked = sorted(total, key=total.get, reverse=True)
    keep = {who for who in ranked[:max_speakers] if total[who] >= min_sec} or {ranked[0]}
    anchors = [t for t in turns if t[2] in keep]
    merged = [t if t[2] in keep else (t[0], t[1], min(anchors, key=lambda a: _gap(a, t))[2]) for t in turns]
    numbers: dict = {}
    for _, _, who in sorted(merged):
        numbers.setdefault(who, len(numbers) + 1)
    return [(s, e, numbers[who]) for s, e, who in merged]


def label_segments(segments: list[dict], turns: list[tuple]) -> list[dict]:
    """Copy of segments where every "Others" segment becomes "Speaker N", N being the voice that
    talks longest during it. "Me", unlabelled segments, and "Others" segments that no turn covers
    (a short "yeah") are returned as they are."""
    out = []
    for seg in segments:
        if seg.get("speaker") != "Others":
            out.append(seg)
            continue
        talked: dict = defaultdict(float)
        for start, end, who in turns:
            overlap = min(end, seg["end"]) - max(start, seg["start"])
            if overlap > 0:
                talked[who] += overlap
        out.append({**seg, "speaker": f"Speaker {max(talked, key=talked.get)}"} if talked else seg)
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["--download"]:
        download()
        print("Models ready in", seg_model_path().parent.parent)
    else:
        print("usage: python diarize.py --download")
