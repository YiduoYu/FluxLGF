#!/bin/bash
# H55 exploration at m=128 + real (not locked)
PY=/opt/anaconda3/envs/hypre_env/bin/python
export OMP_NUM_THREADS=1
$PY explore_defl.py amg-3d,bt-fd:2,bt-fdd:2 geo:128:30:70:1:2e-11:2e-4 geo:128:60:66:1:2e-11:2e-4 geo:128:43:54:1:2e-11:2e-4 real > results/explore_defl128.log 2>&1
echo done >> results/explore_defl128.log
