#!/usr/bin/env bash
# ROLL=<roll_number> REFLECTION=path/to/scanned.pdf bash scripts/make_submission.sh  ->  <roll_number>.zip
set -euo pipefail
: "${ROLL:?set ROLL=<roll_number>}"; : "${REFLECTION:?set REFLECTION=<scanned handwritten pdf>}"
grep -q '\[\[' report/report.md && { echo "report.md still has [[placeholders]]"; exit 1; }
[ -f report/report.pdf ] || { echo "build report/report.pdf first"; exit 1; }
git rev-parse final >/dev/null 2>&1 || { echo "tag 'final' missing"; exit 1; }
rm -rf "/tmp/sub/$ROLL" && mkdir -p "/tmp/sub/$ROLL"
git archive final | tar -x -C "/tmp/sub/$ROLL"            # tracked files only (no models, no secrets)
cp report/report.pdf "/tmp/sub/$ROLL/report/report.pdf"
cp "$REFLECTION" "/tmp/sub/$ROLL/$ROLL.pdf"
(cd /tmp/sub && zip -qr "$OLDPWD/$ROLL.zip" "$ROLL")
echo "created $ROLL.zip"; unzip -l "$ROLL.zip" | tail -1
