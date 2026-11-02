#!/bin/bash
# Segment-DP chains with few levels on one GPU, sequentially: for every EDGES string, the eikonal specialists on those
# edges (run.py), then the DP + TD + bell chain on the first BASE models (run3.py, all remaining segments).
#   chain_queue.sh SCENE PREPARED_RUN GPU ARCH NEWARCH BASE "EDGES1" "EDGES2" ...     (NEWARCH '-' = decoder default)
set -u
cd /home/jeffrey/bellman-ntfield
export BELLMAN600_DERIVED_IMAGE=bellman600-env:local
SCENE=$1; PREP=$2; GPU=$3; ARCH=$4; NEWARCH=$5; BASE=$6; shift 6
for EDGES in "$@"; do
  n=$(echo "$EDGES" | tr ',' '\n' | wc -l); n=$((n-1))
  echo "=== $(date -u +%H:%M:%S) specialists $SCENE arch $ARCH edges $EDGES ($n segments)"
  out=$(python3 jeffrey/segment_dp/run.py --scene $SCENE --prepared-run $PREP --trainer eikonal --mode specialists --arch $ARCH --edges "$EDGES" --normal-weight ${NORMAL:-0} --gpu $GPU 2>&1 | tail -3)
  echo "$out"; ES=$(echo "$out" | grep -o '"id": "[^"]*"' | tail -1 | cut -d'"' -f4)
  if [ -z "$ES" ] || ! grep -q '"complete": true' results/registered_$ES/status.json 2>/dev/null; then echo "specialists failed"; continue; fi
  echo "=== $(date -u +%H:%M:%S) chain on registered_$ES ($BASE base models, $n segments)"
  extra=""; if [ "$NEWARCH" != "-" ]; then extra="--new-arch $NEWARCH"; fi
  python3 jeffrey/segment_dp/run3.py --scene $SCENE --prepared-run $PREP --base-run registered_$ES --base-models $BASE --losses dp,td,bell --last-segment $n $extra --gpu $GPU 2>&1 | tail -2
done
echo "=== $(date -u +%H:%M:%S) chain queue done"
