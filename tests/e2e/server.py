"""Isolated server for the Playwright tests.

Resets tests/e2e/.data, then runs the real FastAPI app against it: a throwaway notes folder, a throwaway
HOME (so "move to Trash" never touches the real Trash) and no meeting detector, no API key, no Ollama.
Never uses the real ~/MeetingNotes or the real .env.
"""
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / ".data"
PORT = int(os.environ.get("E2E_PORT", "8799"))

shutil.rmtree(DATA, ignore_errors=True)
(DATA / "notes").mkdir(parents=True)
(DATA / "home").mkdir()

os.environ.update(
    NOTES_DIR=str(DATA / "notes"),
    HOME=str(DATA / "home"),
    VOICEPRINT_DIR=str(DATA / "home" / ".meeting-notes"),
    MEETING_DETECT="0",
    ANTHROPIC_API_KEY="",            # set (even if empty) so load_dotenv never pulls the real key from .env
    CLOUD_API_URL="",                # likewise: no cloud summaries, whatever .env says
    DEVICE_ID_FILE=str(DATA / "home" / "device_id"),
    OLLAMA_HOST="http://127.0.0.1:9",  # nothing listens here: "Ollama not running"
    OLLAMA_MODEL="qwen2.5:14b",
    TRANSCRIBE_MODEL="parakeet:mlx-community/parakeet-tdt-0.6b-v2",
)
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import uvicorn  # noqa: E402
import app as meeting_app  # noqa: E402

uvicorn.run(meeting_app.app, host="127.0.0.1", port=PORT, log_level="warning")
