"""Client for the Meeting Notes summary proxy (proxy/ in this repo).

Only transcript text is sent, never audio. The device id is a random UUID created on first use;
the proxy counts free meetings per device id. Configure with CLOUD_API_URL (and CLOUD_INVITE_CODE
when the proxy restricts the beta)."""

import os
import uuid
from pathlib import Path

import requests

TIMEOUT_SEC = 240


class CloudError(RuntimeError):
    pass


class QuotaExceeded(CloudError):
    pass


def base_url() -> str:
    return os.getenv("CLOUD_API_URL", "").strip().rstrip("/")


def enabled() -> bool:
    return bool(base_url())


def _device_file() -> Path:
    default = "~/Library/Application Support/MeetingNotes/device_id"
    return Path(os.getenv("DEVICE_ID_FILE", default)).expanduser()


def device_id() -> str:
    path = _device_file()
    try:
        value = path.read_text().strip()
        uuid.UUID(value)
        return value
    except (OSError, ValueError):
        pass
    value = str(uuid.uuid4())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    path.chmod(0o600)
    return value


def _headers() -> dict:
    h = {"X-Device-Id": device_id()}
    code = os.getenv("CLOUD_INVITE_CODE", "").strip()
    if code:
        h["X-Invite-Code"] = code
    return h


def quota() -> dict | None:
    """{'used', 'limit', 'remaining'}, or None when the proxy cannot be reached."""
    if not enabled():
        return None
    try:
        r = requests.get(base_url() + "/v1/quota", headers=_headers(), timeout=10)
        return r.json() if r.ok else None
    except (requests.RequestException, ValueError):
        return None


def summarize(transcript: str, meeting_id: str, hint_title: str | None, language: str) -> tuple[str, dict]:
    """Returns (raw model text, usage). Raises QuotaExceeded or CloudError with a user-readable message."""
    if not enabled():
        raise CloudError("Cloud summaries are not configured (CLOUD_API_URL is empty).")
    try:
        r = requests.post(base_url() + "/v1/summarize", headers=_headers(), timeout=TIMEOUT_SEC,
                          json={"meeting_id": meeting_id, "transcript": transcript,
                                "hint_title": hint_title, "language": language})
    except requests.RequestException as e:
        raise CloudError(f"Could not reach the summary service: {e.__class__.__name__}. "
                         "Check your connection, or pick a local model.") from None
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code == 429 and data.get("error") == "quota_exceeded":
        raise QuotaExceeded(f"You have used all {data.get('limit', 5)} free cloud summaries. "
                            "Your transcript is saved: pick a local model and press Re-summarise.")
    if not r.ok:
        raise CloudError(data.get("message") or f"The summary service returned HTTP {r.status_code}.")
    return data["text"], data.get("usage") or {}
