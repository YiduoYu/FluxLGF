#!/bin/bash
# H50 dev runs (seen gen3t grid of H43/H44 + multigroup dev seed 1, m = 32, ng = 3)
PY=/opt/anaconda3/envs/hypre_env/bin/python
export OMP_NUM_THREADS=1
G3=(); for F in 2e-11 1e-6 1e-3 1e-1 1.0; do for G in 2e-4 1e-2 1.0 1e2; do G3+=("g3t:64:$F:$G"); done; done
MG=(); for s in 0.1 10; do for z in 0.1 10; do for l in 0 1; do MG+=("mg:32:3:$s:$z:$l:1"); done; done; done
RUN_METHODS=amg-3d,amfr,amfr-all $PY run_mf.py results/dev2_g3t.csv "${G3[@]}" > results/dev2_g3t.log 2>&1
RUN_METHODS=amg-3d,amfr,amfr-sep,amfr-all $PY run_mf.py results/dev2_mg.csv "${MG[@]}" > results/dev2_mg.log 2>&1
echo done > results/dev2.done
