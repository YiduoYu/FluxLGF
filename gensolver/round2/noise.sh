#!/bin/zsh
cd "/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/gensolver"
export OMP_NUM_THREADS=1
for i in 1 2 3; do
  vm_stat | grep Swapouts > round2/noise_$i.log
  /opt/anaconda3/envs/hypre_env/bin/python -u eval_ssr.py >> round2/noise_$i.log 2>&1
  vm_stat | grep Swapouts >> round2/noise_$i.log
done
