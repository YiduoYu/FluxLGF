#!/bin/bash
# H50 locked test runs (protocol_amfr.md 3246ffda), sequential, single thread
PY=/opt/anaconda3/envs/hypre_env/bin/python
export OMP_NUM_THREADS=1
TA=(); for F in 1e-8 1e-4 3e-2 0.3 3; do for G in 1e-3 3e-2 0.3 10; do TA+=("g3t:64:$F:$G"); done; done
TB=(); for ng in 2 4 8; do for s in 1e-2 1 1e2; do for z in 1e-2 1e2; do for l in 0 1; do TB+=("mg:40:$ng:$s:$z:$l:7"); done; done; done; done
ALL=amg-3d,amg-unk,amfr,amfr-sep,amfr-noelim,amfr-all
RUN_METHODS=$ALL $PY run_mf.py results/test_A.csv "${TA[@]}" > results/test_A.log 2>&1
RUN_METHODS=$ALL $PY run_mf.py results/test_B.csv "${TB[@]}" > results/test_B.log 2>&1
RUN_METHODS=$ALL $PY run_mf.py results/test_A128.csv g3t:128:1e-8:1e-3 g3t:128:1e-8:0.3 > results/test_A128.log 2>&1
RUN_METHODS=amg-3d,amfr $PY run_mf.py results/test_C.csv real > results/test_C.log 2>&1
echo done > results/test.done
