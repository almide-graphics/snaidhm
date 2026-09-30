#!/bin/bash
# Inside the container: run EXAMPLE under headless sway, type into it with a
# virtual keyboard (wtype), drive a pointer over it (examples/wayland/drive.almd)
# and capture the screen to OUT.
#   capture.sh EXAMPLE OUT.png [EXAMPLE ARGS...]
# CLIENT_DEBUG=1 traces the client's messages (WAYLAND_DEBUG=1) and
# SERVER_DEBUG=server the compositor's, both printed after the run.
set -u
EX=$1; OUT=$2; shift 2
ALMIDE=/almide/build/release/almide
export XDG_RUNTIME_DIR=/tmp/xdg; mkdir -p $XDG_RUNTIME_DIR; chmod 700 $XDG_RUNTIME_DIR
mkdir -p /w && cd /snaidhm && tar --exclude=./target -cf - . | (cd /w && tar -xf -)
cd /w
$ALMIDE build "$EX" -o /tmp/app 2>&1 | tail -1
$ALMIDE build examples/wayland/drive.almd -o /tmp/drive 2>&1 | tail -1
printf 'output * bg #1e2230 solid_color\n' > /tmp/sway.cfg
WAYLAND_DEBUG=${SERVER_DEBUG:-} WLR_BACKENDS=headless WLR_RENDERER=pixman sway -c /tmp/sway.cfg > /tmp/sway.log 2>&1 &
for i in $(seq 50); do [ -S $XDG_RUNTIME_DIR/wayland-1 ] && break; sleep 0.1; done
export WAYLAND_DISPLAY=wayland-1
WAYLAND_DEBUG=${CLIENT_DEBUG:-} /tmp/app "$@" > /tmp/app.log 2>&1 &
APP=$!
sleep 2
wtype "Hello, Almide!"
/tmp/drive move 300 220 sleep 300 click sleep 200 click sleep 200 scroll 3 sleep 500
grim "$OUT"
wait $APP
echo "--- app"; cat /tmp/app.log
if [ -n "${SERVER_DEBUG:-}" ]; then echo "--- server"; grep -E "^\[" /tmp/sway.log; fi
