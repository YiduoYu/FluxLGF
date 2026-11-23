#!/bin/zsh
cd "/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/gensolver"
export OMP_NUM_THREADS=1
PY=/opt/anaconda3/envs/hypre_env/bin/python
while pgrep -f "baseline_weak.py" > /dev/null; do sleep 20; done
shasum -a 256 ssr.c ssr.py libssr.dylib > round3/final_sha256.txt
vm_stat | grep Swapouts > round3/final_swap.log
$PY -u round3/final3.py weak 3 C > round3/final_weak_C.log 2>&1
$PY -u round3/final3.py dev 3 C > round3/final_dev_C.log 2>&1
$PY -u round3/final3.py weakhold 3 C > round3/final_weakhold_C.log 2>&1
$PY -u round3/final3.py holdout 3 C > round3/final_holdout_C.log 2>&1
vm_stat | grep Swapouts >> round3/final_swap.log
