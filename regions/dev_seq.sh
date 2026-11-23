#!/bin/bash
# H56 exploration: recycling + factorisation reuse on m=128 sequences (dev geometries, seed 0), restart 30 (memory)
PY=/opt/anaconda3/envs/hypre_env/bin/python
export OMP_NUM_THREADS=1
L=results/explore_seq128b.log
: > $L
for g in 128:1:2:1:0:2e-2:0 128:10:2:1:0:2e-2:0; do
  $PY seq.py fd 5 frozen,rec5,rec10,amg $g 1e-8 30 >> $L 2>&1
  $PY seq.py fdd 5 frozen,rec5,rec10 $g 1e-8 30 >> $L 2>&1
done
echo done >> $L
