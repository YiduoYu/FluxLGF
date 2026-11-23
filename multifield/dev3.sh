#!/bin/bash
# H51 dev (seed 1, m = 32): low heat capacity, with/without material conduction
PY=/opt/anaconda3/envs/hypre_env/bin/python
export OMP_NUM_THREADS=1
MG=(); for ng in 2 4 8; do for s in 1e-2 1 1e2; do for l in 0 1; do MG+=("mg:32:$ng:$s:1e-2:$l:1"); done; done; done
RUN_METHODS=amg-3d,amfr,amfr2,amfr-all $PY run_mf.py results/dev3_mg.csv "${MG[@]}" > results/dev3_mg.log 2>&1
