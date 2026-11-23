#!/bin/zsh
cd "/Users/duoduo/Documents/温州肯恩大学/科研/Nonlinear Poisson/FD_LGF/gensolver"
export OMP_NUM_THREADS=1
PY=/opt/anaconda3/envs/hypre_env/bin/python
vm_stat | grep Swapouts > round2/swap_before.log
$PY -u eval_ssr.py > round2/default_after_patch.log 2>&1
B="LUMP=4 THETA=0.0 DMODE=2 NTV=1 BLOCK=2 KEEP=2 FILT=0.01"
$PY -u eval_ssr.py ${=B} > round2/cand_a.log 2>&1
$PY -u eval_ssr.py ${=B} EVERY=4 > round2/cand_b.log 2>&1
vm_stat | grep Swapouts > round2/swap_after.log
