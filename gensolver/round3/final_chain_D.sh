#!/bin/zsh
cd "/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/gensolver"
export OMP_NUM_THREADS=1
unset SSR_LIB
PY=/opt/anaconda3/envs/hypre_env/bin/python
$PY -u baseline_weak.py all --hold2 > round3/baseline_weakhold2.log 2>&1
shasum -a 256 ssr.c ssr.py libssr.dylib > round3/final_sha256_D.txt
vm_stat | grep Swapouts > round3/final_swap_D.log
$PY -u round3/final3.py weakhold2 3 D > round3/final_weakhold2_D.log 2>&1
$PY -u round3/final3.py weak 3 D > round3/final_weak_D.log 2>&1
$PY -u round3/final3.py weakhold 3 D > round3/final_weakhold_D.log 2>&1
$PY -u round3/final3.py dev 3 D > round3/final_dev_D.log 2>&1
$PY -u round3/final3.py holdout 3 D > round3/final_holdout_D.log 2>&1
vm_stat | grep Swapouts >> round3/final_swap_D.log
