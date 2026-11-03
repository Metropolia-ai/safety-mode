#!/usr/bin/env bash
# Assemble the clean deploy copy of the demo page (no .git: the Vercel Hobby plan blocks the git author).
#   demo/build.sh [DEST]          default DEST=/tmp/safety-mode-deploy
# Local run of the same copy:  cd DEST && SM_STATIC_DIR=. uvicorn api.index:app --port 8090
set -euo pipefail
# Python for demo/blur_heads.py (numpy, pillow): $PYTHON, else the workspace venv two levels up, else python3
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$(dirname "$0")/../../../.venv/bin/python" ]; then PY="$(cd "$(dirname "$0")/../../.." && pwd)/.venv/bin/python"; else PY=python3; fi
fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-/tmp/safety-mode-deploy}"
VERCEL_LINK="$DEST/.vercel"
# keep the Vercel project link between builds
TMPLINK=""
if [ -d "$VERCEL_LINK" ]; then TMPLINK="$(mktemp -d)"; cp -r "$VERCEL_LINK" "$TMPLINK/"; fi
rm -rf "$DEST"; mkdir -p "$DEST"/{media,data/zones,test-footage,demo}
[ -n "$TMPLINK" ] && cp -r "$TMPLINK/.vercel" "$DEST/" && rm -rf "$TMPLINK"

cp "$REPO/web/index.html" "$DEST/index.html"
# recorded Rule Scout runs, fetched by the page as /cache/scout/<clip>.json
mkdir -p "$DEST/cache/scout"
cp "$REPO"/web/cache/scout/*.json "$DEST/cache/scout/" 2>/dev/null || true
mkdir -p "$DEST/cache" && cp -r "$REPO/web/cache/scout" "$DEST/cache/"     # recorded Rule Scout runs, static
cp -r "$REPO/api" "$DEST/api"
rsync -a --exclude __pycache__ --exclude samples "$REPO/safety_mode" "$DEST/"
cp -r "$REPO/schemas" "$REPO/rules" "$DEST/"
mkdir -p "$DEST/test-footage/detections" "$DEST/test-footage/zones"
cp "$REPO"/test-footage/detections/*.json "$DEST/test-footage/detections/"
rm -f "$DEST/test-footage/detections/summary.json"
cp "$REPO"/test-footage/zones/*.json "$DEST/test-footage/zones/"
cp -r "$REPO/test-footage/rules" "$DEST/test-footage/"
cp -r "$REPO/demo/cache" "$DEST/demo/"
cp "$REPO"/demo/vercel/vercel.json "$REPO"/demo/vercel/requirements.txt "$REPO"/demo/vercel/.vercelignore "$DEST/"

# clips: Pexels licence, test-footage/raw is not in git (download links in test-footage/README.md).
# Streams copied, audio and data tracks dropped, moov atom moved to the front for streaming.
# Clips added 30.10 (A, B, C) come from test-footage/raw/ too (copied there from candidates/raw/).
# A (walker) and B (forklift driver) show recognisable faces: the page gets demo/blur_heads.py output (head boxes blurred), cached
# in test-footage/cache/blurred/ and rebuilt when the raw clip or the detections are newer.
declare -A CLIP=( [forklift-operator]=pexels-32243653-forklift-operator
                  [warehouse-team]=pexels-15170997-warehouse-team
                  [busy-warehouse]=pexels-6079421-forklift-busy-warehouse
                  [man-crosses-floor]=pexels-38707991-man-crosses-floor
                  [forklift-crosses-aisle]=pexels-31751344-forklift-crosses-aisle
                  [site-worker-no-vest]=pexels-32245123-site-worker-no-vest )
BLUR=( man-crosses-floor forklift-crosses-aisle )
for short in "${!CLIP[@]}"; do
  src="$REPO/test-footage/raw/${CLIP[$short]}.mp4"
  if [[ " ${BLUR[*]} " == *" $short "* ]]; then
    blurred="$REPO/test-footage/cache/blurred/${CLIP[$short]}.mp4"
    dets="$REPO/test-footage/detections/${CLIP[$short]}.video.json"
    if [ ! -f "$blurred" ] || [ "$src" -nt "$blurred" ] || [ "$dets" -nt "$blurred" ] || [ "$REPO/demo/blur_heads.py" -nt "$blurred" ]; then
      mkdir -p "$(dirname "$blurred")"
      "$PY" "$REPO/demo/blur_heads.py" "${CLIP[$short]}" "$blurred"
    fi
    src="$blurred"
  fi
  ffmpeg -loglevel error -y -i "$src" -map 0:v:0 -c:v copy -an -dn -movflags +faststart "$DEST/media/$short.mp4"
  size=$(stat -c %s "$DEST/media/$short.mp4")
  if [ "$size" -gt 41943040 ]; then   # > 40 MB: re-encode to 1280 px, H.264 CRF 26
    ffmpeg -loglevel error -y -i "$src" -map 0:v:0 -vf "scale='if(gt(iw,ih),1280,-2)':'if(gt(iw,ih),-2,1280)'" \
      -c:v libx264 -crf 26 -preset slow -pix_fmt yuv420p -an -movflags +faststart "$DEST/media/$short.mp4"
  fi
  cp "$REPO/test-footage/zones/${CLIP[$short]}.json" "$DEST/data/zones/$short.json"
done
python3 - "$DEST/data/zones/webcam.json" <<'PY'
import json, sys
json.dump({"clip": "webcam", "frame_size": [1280, 720], "floor_calibration": None,
           "zones": [{"name": "past_gate", "title": "Warehouse floor", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]},
          open(sys.argv[1], "w"))
PY
du -sh "$DEST" "$DEST/media"
