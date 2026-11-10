#!/bin/bash
# Start Meeting Notes. First run creates the virtualenv and installs packages.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
[ -f .env ] || cp .env.example .env
# Stop a previous instance still holding the port, so every run is a fresh restart.
PORT_NUM="${PORT:-$(grep -E '^PORT=' .env | tail -1 | cut -d= -f2 | tr -d ' "')}"
PORT_NUM="${PORT_NUM:-8765}"
for pid in $(lsof -ti tcp:"$PORT_NUM" -sTCP:LISTEN 2>/dev/null); do
  if ps -p "$pid" -o command= | grep -q "app.py"; then
    echo "Stopping previous instance (pid $pid)..."
    kill "$pid"
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$pid" 2>/dev/null || true
  else
    echo "Port $PORT_NUM is used by another program (pid $pid); not touching it." >&2
    exit 1
  fi
done

(sleep 2 && open "http://127.0.0.1:${PORT_NUM}") &
exec .venv/bin/python app.py
