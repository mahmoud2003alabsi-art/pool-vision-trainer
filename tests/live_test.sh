#!/usr/bin/env bash
# Live browser test for pool-vision-trainer: renders the 8BP-like scene
# at four viewport sizes (simulating screen changes) plus the
# browser-about-the-game trap, screenshots each, and runs the full
# pipeline on every capture. Verifies the analyzer is screen-proof and
# that the trap capture is rejected by the coverage gate.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="${1:-/tmp/live_test}"
mkdir -p "$OUT"

if ! command -v agent-browser >/dev/null 2>&1; then
  echo "[!] agent-browser not found; npm i -g agent-browser && agent-browser install"
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then echo "[!] python3 missing"; exit 2; fi

python3 -m http.server 8791 --directory "$HERE" >/dev/null 2>&1 &
SRV=$!
sleep 1
trap 'kill $SRV 2>/dev/null' EXIT

run_case() {  # name w h url
  local name="$1" w="$2" h="$3" url="$4"
  agent-browser set viewport "$w" "$h" >/dev/null
  agent-browser open "$url" >/dev/null
  agent-browser wait 1500 >/dev/null
  agent-browser screenshot "$OUT/$name.png" >/dev/null
  echo "== $name (${w}x${h})"
  if [[ "$url" == *trap* ]]; then
    "$ROOT/build/pool-trainer" analyze "$OUT/$name.png" --out "$OUT/$name" 2>&1 \
      | grep -E "table rect|cue stick|detected|detection failed" || true
    # coverage fraction the hardened probe/gate computes
    python3 - "$OUT/$name.png" "$OUT/$name" <<'PYEOF'
import sys, json
from PIL import Image
png, outdir = sys.argv[1], sys.argv[2]
import glob, os
js = glob.glob(os.path.join(outdir, '*_result.json'))
if not js:
    print('   felt coverage = n/a (detection rejected entirely)')
    sys.exit()
j = json.load(open(js[0]))
d = j.get('detection', {})
im = Image.open(png)
w, h = im.size
x0, y0, x1, y1 = (d.get('felt_rect_x0'), d.get('felt_rect_y0'),
                  d.get('felt_rect_x1'), d.get('felt_rect_y1'))
if None in (x0, y0, x1, y1):
    print('   felt coverage = n/a')
    sys.exit()
cov = (x1 - x0) * (y1 - y0) / (w * h)
print(f'   felt coverage = {cov:.2f}  -> probe gate '
      f'{"PASS" if cov >= 0.30 else "REJECT (<0.30)"}')
PYEOF
  else
    "$ROOT/build/pool-trainer" analyze "$OUT/$name.png" --out "$OUT/$name" 2>&1 \
      | grep -E "table rect|cue stick|detected|detection failed" || true
  fi
}

run_case size_1371x770 1371 770 "http://localhost:8791/index.html"
run_case size_1920x1080 1920 1080 "http://localhost:8791/index.html"
run_case size_1280x720 1280 720  "http://localhost:8791/index.html"
run_case size_1024x576 1024 576  "http://localhost:8791/index.html"
run_case trap_browser   1920 1080 "http://localhost:8791/trap.html"
agent-browser close >/dev/null 2>&1 || true
echo "done -> $OUT"
