"""Download the speech-to-text model with visible progress.

parakeet-mlx would fetch its ~2.5 GB model silently on first use, which looks like a hang. Here the
download is an explicit step: status() says whether the model is on disk, start() downloads it in the
background, and bytes done are measured from the Hugging Face cache folder, so the progress bar works
whatever the downloader does internally and a half-finished download resumes where it stopped."""

import threading
import time
from pathlib import Path

# Size of the default model (model.safetensors + small files), used when the Hub cannot be asked.
KNOWN_BYTES = {"mlx-community/parakeet-tdt-0.6b-v2": 2_472_000_000}

_lock = threading.RLock()   # start() calls status() while holding it
_jobs: dict[str, "_Download"] = {}


def _cache_root() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def _repo_dir(repo: str) -> Path:
    return _cache_root() / ("models--" + repo.replace("/", "--"))


def is_ready(repo: str) -> bool:
    """True when a complete snapshot (config + weights) of the model is in the cache."""
    root = _repo_dir(repo)
    try:
        ref = (root / "refs" / "main").read_text().strip()
        snap = root / "snapshots" / ref
    except OSError:
        snaps = sorted((root / "snapshots").glob("*")) if (root / "snapshots").is_dir() else []
        if not snaps:
            return False
        snap = snaps[-1]
    try:
        files = [p for p in snap.iterdir() if p.exists()]   # exists() follows the symlink: broken ones drop out
    except OSError:
        return False
    names = {p.name for p in files}
    return "config.json" in names and any(n.endswith(".safetensors") for n in names)


def bytes_on_disk(repo: str) -> int:
    blobs = _repo_dir(repo) / "blobs"
    if not blobs.is_dir():
        return 0
    total = 0
    for p in blobs.iterdir():
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _total_bytes(repo: str) -> int:
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        total = sum(s.size or 0 for s in info.siblings)
        if total:
            return total
    except Exception:
        pass
    return KNOWN_BYTES.get(repo, 0)


def _friendly(e: Exception) -> str:
    name = type(e).__name__
    text = str(e)
    if any(k in name + text for k in ("ConnectionError", "Timeout", "Offline", "Name or service", "getaddrinfo",
                                      "SSLError", "ConnectError", "NewConnection")):
        return "Could not reach the download server. Check your internet connection and try again."
    if "No space left" in text or "ENOSPC" in text:
        return "Not enough free disk space for the model (about 2.5 GB). Free some space and try again."
    return f"Download failed: {text[:200]}"


class _Download:
    def __init__(self, repo: str):
        self.repo = repo
        self.error: str | None = None
        self.running = False
        self.total = KNOWN_BYTES.get(repo, 0)
        self.started = 0.0

    def run(self):
        try:
            self.total = _total_bytes(self.repo) or self.total
            from huggingface_hub import snapshot_download
            snapshot_download(self.repo)
        except Exception as e:  # shown to the user, who can retry
            self.error = _friendly(e)
        finally:
            self.running = False


def status(repo: str) -> dict:
    with _lock:
        job = _jobs.get(repo)
    ready = is_ready(repo)
    running = bool(job and job.running)
    total = (job.total if job else 0) or KNOWN_BYTES.get(repo, 0)
    done = bytes_on_disk(repo)
    if ready and not running:
        done = total = max(done, total)
    return {"repo": repo, "ready": ready and not running, "downloading": running,
            "bytes": min(done, total) if total else done, "total": total,
            "error": job.error if job and not running else None}


def start(repo: str) -> dict:
    """Begin (or resume) the download; harmless when it is already running or done."""
    with _lock:
        job = _jobs.get(repo)
        if job and job.running:
            return status(repo)
        if is_ready(repo):
            return status(repo)
        job = _jobs[repo] = _Download(repo)
        job.running = True
        job.started = time.time()
        threading.Thread(target=job.run, daemon=True).start()
    return status(repo)
