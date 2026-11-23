"""Where SSR spends time: python-side setup, C setup, one apply; BoomerAMG setup and per-iteration cost on the same case."""
import sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import numpy as np, ctypes as C
from time import process_time as pt
from problems import load
import ssr
from krylov import run, AMG
case = sys.argv[1]
A, b, _ = load(case); A = A.tocsr(); n = A.shape[0]
t0 = pt(); sym = abs(A - A.T).max() <= 1e-10 * abs(A).max(); t_sym = pt() - t0
t0 = pt(); P = ssr.SSR(A); t_all = pt() - t0
x = np.random.default_rng(0).standard_normal(n)
P(x); t0 = pt()
for _ in range(10): P(x)
t_app = (pt() - t0) / 10
t0 = pt()
for _ in range(10): A @ x
t_mv = (pt() - t0) / 10
info = np.zeros(3, dtype=np.int32); lv = []
for l in range(P.levels):
    ssr._lib.ssr_level_info(P.h, l, info.ctypes.data_as(C.POINTER(C.c_int))); lv.append(tuple(int(v) for v in info))
q = run(A, b, "amg", amg_cfg=AMG["fast.25"])
print(f"{case}: n={n} nnz={A.nnz} sym={sym} symcheck={t_sym:.3f}s ssr_setup={t_all:.3f}s apply={t_app*1e3:.1f}ms matvec={t_mv*1e3:.1f}ms apply/matvec={t_app/t_mv:.1f} levels={P.levels} cx={P.complexity:.2f}")
print(f"  boomer fast.25: its={q['its']} setup={q['setup']:.3f}s solve/it={(q['cpu']-q['setup'])/max(q['its'],1)*1e3:.1f}ms ({(q['cpu']-q['setup'])/max(q['its'],1)/t_mv:.1f} matvecs)")
print("  levels n, nF/n, nnz/n:", " ".join(f"{a}:{f/a:.2f}:{z/a:.0f}" for a, f, z in lv[:12]), "...")
