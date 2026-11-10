#!/bin/bash
# Build packaging/dist/MeetingNotes.dmg from packaging/dist/MeetingNotes.app (run build_app.sh first).
#   SIGN_ID="Developer ID Application: ..."   sign the disk image (default: skip)
set -euo pipefail
cd "$(dirname "$0")/.."
APP=packaging/dist/MeetingNotes.app
DMG=packaging/dist/MeetingNotes.dmg
[ -d "$APP" ] || { echo "build the app first: packaging/build_app.sh" >&2; exit 1; }

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"   # drag-to-install target

rm -f "$DMG"
hdiutil create -volname "Meeting Notes" -srcfolder "$STAGE" -ov -format UDZO -imagekey zlib-level=9 "$DMG" >/dev/null
[ -n "${SIGN_ID:-}" ] && codesign --force --sign "$SIGN_ID" "$DMG"
hdiutil verify "$DMG" >/dev/null && echo "dmg ok"
ls -la "$DMG" | awk '{printf "%.0f MB  %s\n", $5/1048576, $9}'
