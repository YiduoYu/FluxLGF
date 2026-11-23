#!/bin/bash
# H53 locked tests (protocol_btfd_lu.md)
PY=/opt/anaconda3/envs/hypre_env/bin/python
export OMP_NUM_THREADS=1
for p in 1 2 3; do $PY run_real.py results/h53_R$p.csv real amg-3d bt-v bt-fd:2 > results/h53_R$p.log 2>&1; done
$PY run_ir.py results/h53_IR.csv 1e-6 amg-3d bt-v bt-fd > results/h53_IR.log 2>&1
CASES="35:45:1:2e-11:2e-4 50:58:1:2e-11:2e-4 43:54:3:2e-11:2e-4 43:54:0.3:2e-11:2e-4 43:54:1:1e-6:1e-3 43:54:1:2e-11:2e-4 25:32:1:2e-11:2e-4 60:66:1:2e-11:2e-4 30:70:1:2e-11:2e-4 43:54:0.1:2e-11:2e-4 43:54:10:2e-11:2e-4 43:54:1:1e-3:1e-2"
L="real"
for c in $CASES; do
  $PY run_real.py results/h53_G_${c//:/_}.csv geo:128:$c amg-3d bt-v bt-fd:2 > results/h53_G_${c//:/_}.log 2>&1
  L="$L geo:128:$c"
done
$PY lin.py $L > results/h53_lin.log 2>&1
echo done > results/h53.done
