#!/bin/sh
# Goes on with a GP run when the computer starts again, e.g. after a power cut, from a
# crontab line such as
#
#   @reboot /home/james/Desktop/dev/yee-lab/search/resume.sh runs/2026-10-26 "2026-10-27 19:00"
#
# The run goes on from what it had written to the disk and stops by itself at the time
# given (nothing happens after it); a run already going on in the directory is left alone.
# Its output is appended to resume.log in the run's directory.
cd "$(dirname "$0")" || exit 1
out=$1
until=$2
mkdir -p "$out"
sleep 60  # let the system settle after the start
inhibit=""
if systemd-inhibit --what=sleep:idle true 2>/dev/null; then  # no sleeping while it runs, where allowed
    inhibit="systemd-inhibit --what=sleep:idle"
fi
echo "$(date '+%F %T') the computer started: going on with the run" >> "$out/resume.log"
exec $inhibit "$HOME/.local/bin/uv" run --frozen yee-search gp --out "$out" --until "$until" >> "$out/resume.log" 2>&1
