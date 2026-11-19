#!/bin/bash
# lift-alone controls on the 10/13/22 mazes (jeffrey/lift_scale pipeline, unchanged), one after another on GPU 1
cd /home/jeffrey/bellman-ntfield
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
for prep in 20261102T032150Z-b0baa8e9 20261101T192504Z-13cf1d04 20261101T193403Z-cea05796; do
  while [ "$(awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo)" -lt 8 ]; do sleep 30; done
  ~/.venvs/bellman-ntfield/bin/python jeffrey/lift_scale/run.py --stage field --prepared results/registered_$prep --seed 87358 --label lift --extra '--lift time' \
    > jeffrey/lift_midpoint/work/logs/lift_$prep.log 2>&1
  echo "$prep exit $?" >> jeffrey/lift_midpoint/work/logs/lift_mazes.done
done
