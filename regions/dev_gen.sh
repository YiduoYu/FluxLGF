#!/bin/bash
# H56 exploration (dev cases only; test cases use other seeds / parameters)
PY=/opt/anaconda3/envs/hypre_env/bin/python
export OMP_NUM_THREADS=1
$PY explore_tol.py 1e-6,1e-1 bt-fd:2,bt-fdd:2 real geo:128:30:70:1:2e-11:2e-4 geo:128:43:54:1:2e-11:2e-4 gen:128:1:2:1:4:0:0 gen:128:10:2:1:0:0:0 > results/explore_tol128.log 2>&1
$PY explore_tol.py 1e-6 amg-3d gen:128:1:2:1:4:0:0 gen:128:10:2:1:0:0:0 >> results/explore_tol128.log 2>&1
echo done >> results/explore_tol128.log
