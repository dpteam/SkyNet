#!/bin/sh
# dos_run.sh - run the 16-bit build in DOSBox and print its output.
#
# POSIX twin of tools/dos_run.ps1, so CI can exercise the real target.
# DOSBox needs a video driver even when nobody is watching; the dummy SDL
# driver makes it headless.
#
# Usage: sh tools/dos_run.sh <workdir> <command> [outfile]
set -e

WORK=${1:?usage: dos_run.sh <workdir> <command> [outfile]}
CMD=${2:?usage: dos_run.sh <workdir> <command> [outfile]}
OUT=${3:-OUT.TXT}

mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)

cat > "$WORK/dosbox.conf" <<EOF
[sdl]
autolock=false
fullscreen=false
output=surface

[dosbox]
machine=svga_s3
memsize=16
ems=true
xms=true

[cpu]
core=auto
cycles=max

[autoexec]
mount c $WORK
c:
$CMD > $OUT
exit
EOF

export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

# -exit makes DOSBox quit when the shell exits; the timeout is a guard
# against a hang in the program under test, which is itself a result.
timeout 240 dosbox -conf "$WORK/dosbox.conf" -exit >/dev/null 2>&1 || true

if [ ! -f "$WORK/$OUT" ]; then
    echo "no $OUT produced by DOSBox" >&2
    ls -la "$WORK" >&2
    exit 1
fi

cat "$WORK/$OUT"
