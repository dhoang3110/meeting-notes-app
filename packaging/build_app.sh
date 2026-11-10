#!/bin/bash
# Build dist/MeetingNotes.app: Swift shell + PyInstaller server + systap helper.
#   SIGN_ID="Apple Development: ..."  sign with this identity (default: ad-hoc "-")
#   SKIP_SERVER=1                     reuse an existing packaging/dist/server
#   HARDENED=1                        enable the hardened runtime (needed for notarization)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
APP=packaging/dist/MeetingNotes.app
SIGN_ID="${SIGN_ID:--}"

echo "==> systap helper"
./native/build.sh

echo "==> server (PyInstaller)"
if [ "${SKIP_SERVER:-0}" = "1" ] && [ -d packaging/dist/server ]; then echo "(reusing packaging/dist/server)"; else
rm -rf packaging/dist/server build_tmp
mkdir -p build_tmp
.venv/bin/pyinstaller --noconfirm --onedir --name server --distpath packaging/dist \
  --workpath build_tmp --specpath build_tmp --paths . --add-data "$ROOT/static:static" \
  --collect-all mlx --collect-all parakeet_mlx --collect-all mlx_whisper --collect-all sherpa_onnx \
  --collect-all sounddevice --collect-all soundfile \
  --exclude-module torch --exclude-module torchaudio --exclude-module matplotlib --exclude-module tkinter \
  app.py > build_tmp/pyi.log 2>&1 || { tail -30 build_tmp/pyi.log; exit 1; }
fi

echo "==> shell (Swift)"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/bin"
swiftc -swift-version 5 -O packaging/shell/main.swift -o "$APP/Contents/MacOS/MeetingNotes"
cp packaging/Info.plist "$APP/Contents/Info.plist"
cp -R packaging/dist/server "$APP/Contents/Resources/server"
cp native/bin/systap "$APP/Contents/Resources/bin/systap"
# Parakeet decodes audio through an `ffmpeg` on PATH; a GUI app has no Homebrew PATH, so ship one
# (static build, system frameworks only; see requirements-dev.txt: imageio-ffmpeg).
cp "$(.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$APP/Contents/Resources/bin/ffmpeg"
chmod +x "$APP/Contents/Resources/bin/ffmpeg"

echo "==> sign ($SIGN_ID)"
OPTS=(--force --timestamp=none --sign "$SIGN_ID" --entitlements packaging/entitlements.plist)
[ "${HARDENED:-0}" = "1" ] && OPTS+=(--options runtime --timestamp)
# sign every nested Mach-O first, then the bundle
find "$APP/Contents/Resources" -type f \( -name "*.so" -o -name "*.dylib" -o -perm +111 \) -print0 |
  while IFS= read -r -d '' f; do
    file "$f" | grep -q "Mach-O" && codesign "${OPTS[@]}" "$f" >/dev/null 2>&1 || true
  done
codesign "${OPTS[@]}" "$APP"
codesign --verify --deep --strict "$APP" && echo "signature ok"
du -sh "$APP"
echo "built $APP"
