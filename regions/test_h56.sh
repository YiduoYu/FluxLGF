#!/bin/bash
# H56 locked tests (protocol_h56.md)
PY=/opt/anaconda3/envs/hypre_env/bin/python
export OMP_NUM_THREADS=1
G="128:1:2:1:0:2e-2:1 128:1:1:1:0:2e-2:1 128:1:8:0.5:0:2e-2:1 128:1:2:2:0:2e-2:1 128:1:2:1:4:2e-2:1 128:1:2:1:0:1e-1:1 128:3:2:1:0:2e-2:1 128:10:2:1:0:2e-2:1 128:10:2:1:0:2e-2:2 128:10:8:0.5:2:2e-2:3 128:20:2:1:0:2e-2:1"
for g in $G; do
  $PY seq_lock.py results/h56_${g//:/_}.csv 10 $g > results/h56_${g//:/_}.log 2>&1
done
echo done > results/h56.done
