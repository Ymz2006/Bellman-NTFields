#!/bin/bash
# Emits one line per milestone of this study's runs (queue events, validations, final planning, errors). Positions persist in
# work/.watchpos, so a re-armed watcher continues where the last one stopped.
cd /home/jeffrey/bellman-ntfield
POS=jeffrey/lift_midpoint/work/.watchpos
touch $POS
get() { awk -v k="$1" '$1==k{print $2}' $POS | tail -1; }
put() { grep -v "^$1 " $POS > $POS.tmp; echo "$1 $2" >> $POS.tmp; mv $POS.tmp $POS; }
emit() {  # file key filter prefix
  local f=$1 k=$2 flt=$3 pre=$4
  [ -f "$f" ] || return
  local n=$(wc -l < "$f"); local p=$(get "$k"); p=${p:-0}
  if [ "$n" -gt "$p" ]; then sed -n "$((p+1)),${n}p" "$f" | grep -E "$flt" | cut -c1-260 | sed "s|^|$pre|"; put "$k" "$n"; fi
}
while true; do
  emit jeffrey/lift_midpoint/queue/runner.log runner '.' 'QUEUE '
  emit jeffrey/lift_midpoint/work/logs/lift_mazes.done liftdone '.' 'LIFT-ALONE '
  for q in jeffrey/lift_midpoint/queue/logs/*.log jeffrey/lift_midpoint/work/logs/lift_*.log; do
    [ -f "$q" ] || continue
    rid=$(grep -o '"id": "[^"]*"' "$q" | head -1 | cut -d'"' -f4)
    [ -n "$rid" ] || continue
    emit results/registered_$rid/log.txt "$rid" "epoch (2001|3001|4001): refresh|validate epoch|MPPI seed 0|done in|Traceback|Error|error|Killed|out of memory|RuntimeError" "$rid "
  done
  sleep 30
done
