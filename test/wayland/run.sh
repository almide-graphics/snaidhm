#!/bin/bash
# Run snaidhm's Wayland client on a headless sway and save a screenshot.
#
#   ALMIDE_SRC=~/src/almide test/wayland/run.sh [EXAMPLE] [OUT.png]
#
# Needs Docker (on macOS: colima or Docker Desktop — both share only $HOME
# with the Linux VM, so keep this repo, ALMIDE_SRC and OUT under it). The
# compiler is built from ALMIDE_SRC inside the container, once per change,
# into a named volume; the first build takes a few minutes.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
EX=${1:-examples/wayland/main.almd}
OUT=${2:-$ROOT/wayland-capture.png}
: "${ALMIDE_SRC:?set ALMIDE_SRC to an almide checkout}"
docker build -q -t snaidhm-wayland "$HERE" >/dev/null
docker volume create snaidhm-wayland-almide >/dev/null
docker volume create snaidhm-wayland-cargo >/dev/null
docker run --rm -v "$ALMIDE_SRC":/src:ro -v snaidhm-wayland-almide:/almide \
  -v snaidhm-wayland-cargo:/root/.cargo/registry snaidhm-wayland bash -c '
    mkdir -p /almide/src && cd /src && tar --exclude=./target -cf - . | (cd /almide/src && tar -xf -) &&
    cd /almide/src && CARGO_TARGET_DIR=/almide/build cargo build --release 2>&1 | tail -1'
mkdir -p "$(dirname "$OUT")"
docker run --rm -v "$ROOT":/snaidhm:ro -v snaidhm-wayland-almide:/almide \
  -v snaidhm-wayland-cargo:/root/.cargo/registry -v "$(dirname "$OUT")":/out \
  -v "$HERE/capture.sh":/capture.sh:ro snaidhm-wayland /capture.sh "$EX" "/out/$(basename "$OUT")" 6000
echo "saved $OUT"
