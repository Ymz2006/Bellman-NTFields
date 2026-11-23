#!/bin/bash
# prepare 3-D houses one after another on one GPU, then hand each to the queue:  prep3d.sh <gpu> <name>[:<pin gpu>] ...
cd /home/jeffrey/bellman-ntfield
gpu=$1; shift
for item in "$@"; do
  name=${item%%:*}; pin=""; [[ "$item" == *:* ]] && pin=${item##*:}
  until [ "$(awk '/MemAvailable/ {print int($2/1e6)}' /proc/meminfo)" -ge 5 ]; do sleep 30; done
  echo "$(date +%T) prepare $name on GPU $gpu"
  out=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs/run.py --house3 $name --label v1 2>&1 | tail -1)
  id=$(echo "$out" | sed -n 's/.*"id": "\([^"]*\)".*"state": "succeeded".*/\1/p')
  if [ -n "$id" ]; then
    ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs/queue/add3d.py $name results/registered_$id $pin
    echo "$(date +%T) done $name $id: $(grep 'queries:' results/registered_$id/log.txt | cut -c1-400)"
    grep "oracle MPPI\|straight\|done in" results/registered_$id/log.txt | cut -c1-200
  else
    echo "$(date +%T) FAILED $name: $out" | cut -c1-600
  fi
done
