#!/bin/bash
# H52 locked tests (protocol_btfd.md)
PY=/opt/anaconda3/envs/hypre_env/bin/python
export OMP_NUM_THREADS=1
for p in 1 2 3; do $PY run_real.py results/test_R$p.csv real amg-3d bt-v bt-fd:2 > results/test_R$p.log 2>&1; done
$PY run_ir.py results/test_IR.csv 1e-6 amg-3d bt-v bt-fd > results/test_IR.log 2>&1
for c in 43:54:1:2e-11:2e-4 25:32:1:2e-11:2e-4 60:66:1:2e-11:2e-4 30:70:1:2e-11:2e-4 43:54:0.1:2e-11:2e-4 43:54:10:2e-11:2e-4 43:54:1:1e-3:1e-2; do
  $PY run_real.py results/test_G_${c//:/_}.csv geo:128:$c amg-3d bt-v bt-fd:2 > results/test_G_${c//:/_}.log 2>&1
done
$PY features.py > results/test_features.log 2>&1
echo done > results/test.done
