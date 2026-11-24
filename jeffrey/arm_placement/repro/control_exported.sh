#!/bin/bash
# Control: the exported (original) trained checkpoints of p_d055_s46_h60, re-planned on this workstation with the original
# evaluation snapshot (296a34426335) on the exported prepared data. Separates planner/hardware noise from training noise.
cd /home/jeffrey/bellman-ntfield
export CUDA_VISIBLE_DEVICES=${GPU:-1} CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1
P=~/.venvs/bellman-ntfield/bin/python
PREP=import/arm_placement_export_2026-11-23/training_data/prepared/prep_p_d055_s46_h60
X=jeffrey/arm_placement/repro/exported
for f in lift ntrl; do
  for cfg in "forward|600" "reverse bidir|300 600" "bidir|1200 2400"; do
    pl=${cfg%%|*}; st=${cfg##*|}
    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=1G $P jeffrey/arm_placement/run.py evalfield --snap 296a34426335 \
      --prepared $PREP --fields $f=$X/${f}_p_d055_s46_h60_s87358 --label exported_${f}_p_d055_s46_h60_s87358 \
      --extra "--planner $pl --steps $st"
  done
done
