#!/bin/zsh
export OMP_NUM_THREADS=1
P=/opt/anaconda3/envs/hypre_env/bin/python
$P -u problems.py > problems.log 2>&1
$P -u baseline.py all --holdout > baseline_holdout.log 2>&1
$P -u final.py dev 3 > final_dev.log 2>&1
$P -u final.py holdout 3 > final_holdout.log 2>&1
$P -u extras.py > extras.log 2>&1
echo CHAIN_DONE
