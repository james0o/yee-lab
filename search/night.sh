#!/bin/sh
# One night of search, by hand or from a crontab line such as
#
#   0 22 * * * /home/james/Desktop/dev/yee-lab/search/night.sh 06:00
#
# and a second argument the worker processes (4 by default: each GP worker holds about 2.5
# GB of voters and remembered subtrees, so 4 take about 10 GB; 3 leave room for a browser).
# A third argument "plain" searches only the plain vocabulary (grammar.restrict_to_plain),
# "coarse" only the coarse one (grammar.restrict_to_coarse), "coarse-extra" that with D20,
# DAC and KOTH.
# First every Conditional of the library, one and two levels deep (yee-search conditional,
# about an hour and a half), then genetic programming from their best until the time given (06:00 by
# default: the report after it takes an hour), then the report of the GP run on the test
# set. Writes runs/<date>-night-conditional and runs/<date>-night-gp (-plain-conditional
# and -plain-gp with "plain", -coarse-... and -coarse-extra-... alike), the progress in
# night.log of the latter. A GP run stopped by a power cut goes on with resume.sh (give it
# the same flags).
cd "$(dirname "$0")" || exit 1
until=${1:-06:00}
workers=${2:-4}
day=$(date +%F)
vocabulary=""
name=night
case "$3" in
    plain) vocabulary=--plain; name=plain ;;
    coarse) vocabulary=--coarse; name=coarse ;;
    coarse-extra) vocabulary="--coarse --extra"; name=coarse-extra ;;
    "") ;;
    *) echo "unknown vocabulary: $3 (plain, coarse or coarse-extra)" >&2; exit 1 ;;
esac
conditional=runs/$day-$name-conditional
gp=runs/$day-$name-gp
maps=runs/2026-10-27-conditional/maps  # where each method elects whom, shared by every run
mkdir -p "$conditional" "$gp"
log=$gp/night.log
uv="$HOME/.local/bin/uv"
inhibit=""
if systemd-inhibit --what=sleep:idle true 2>/dev/null; then  # no sleeping while it runs, where allowed
    inhibit="systemd-inhibit --what=sleep:idle --who=yee-search --why=night.sh"
fi
echo "$(date '+%F %T') every Conditional: $conditional" >> "$log"
$inhibit "$uv" run --frozen yee-search conditional --out "$conditional" --maps "$maps" --workers "$workers" $vocabulary >> "$conditional/search.log" 2>&1
seed=""
if [ -f "$conditional/validation.json" ]; then
    seed="--seed-from $conditional"
fi
echo "$(date '+%F %T') genetic programming until $until $seed" >> "$log"
$inhibit "$uv" run --frozen yee-search gp --out "$gp" --until "$until" --workers "$workers" $seed $vocabulary >> "$log" 2>&1
echo "$(date '+%F %T') the report on the test set" >> "$log"
$inhibit "$uv" run --frozen yee-search report "$gp" --workers "$workers" >> "$log" 2>&1
echo "$(date '+%F %T') done" >> "$log"
