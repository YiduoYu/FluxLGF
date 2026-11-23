#!/bin/zsh
cd "/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/gensolver"
export OMP_NUM_THREADS=1
PY=/opt/anaconda3/envs/hypre_env/bin/python
A="DOM=50.0 STR2=1 TRANS=2 SWEEPS_Y=3"
$PY -u round3/eval3.py weak ${=A} > round3/scr_weak_A.log 2>&1
$PY -u round3/eval3.py weak ${=A} SMODE=2 > round3/scr_weak_B.log 2>&1
$PY -u round3/eval3.py dev ${=A} > round3/scr_dev_A.log 2>&1
$PY -u round3/eval3.py dev ${=A} SMODE=2 > round3/scr_dev_B.log 2>&1
$PY -u round3/eval3.py weak SEED=1 > round3/scr_weak_default.log 2>&1
