#!/bin/zsh
# screening of G candidates (F + exact setup speedups) with the stored baselines; eval3 is noisier than final3
cd "/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/gensolver"
export OMP_NUM_THREADS=1
unset SSR_LIB
PY=/opt/anaconda3/envs/hypre_env/bin/python
for cfg in "G0:" "SPK5:SPK=5.0" "SPK6D12:SPK=6.0 SPDEN=12.0"; do
  tag=${cfg%%:*}; kw=${cfg#*:}
  for s in dev weak; do
    $PY -u round3/eval3.py $s ${=kw} > round3/scr_${s}_${tag}.log 2>&1
  done
done
