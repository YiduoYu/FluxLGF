#!/bin/zsh
cd "/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/gensolver"
export OMP_NUM_THREADS=1 SSR_LIB=libssr_dev.dylib
PY=/opt/anaconda3/envs/hypre_env/bin/python
for s in weak dev; do
  $PY -u round3/eval3.py $s SPK=5.0 SPKN=5.0 > round3/scr_${s}_G.log 2>&1
done
