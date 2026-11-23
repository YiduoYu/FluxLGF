"""Audit follow-up: real SC26_01 under the double-double original residual ||b - Ax||/||b|| < 1e-8 (IR, inner GMRES(50)
to 1e-6). Fixed before any run, on 2026-11-05:
  every run its own process, single thread, nothing else running; all runs kept, none rejected.
  scan : the audit's 12 BoomerAMG settings once each, random order. Screening only.
  pairs: AMG3D baseline, the scan's fastest AMG, bt-fd (MUMPS LU), bt-fd-chol (CHOLMOD), 5 blocks, order shuffled in
         each block. Fresh measurements: scan times are not reused, so the picked AMG is not favoured by selection.
  Primary metric: total_s (setup + inner solves + dd residuals) to 1e-8; prep_s (DD tables) is reported next to it and
  is common to all methods. A run counts only if the dd residual reached 1e-8; inner reasons are recorded.
  Each row also has vm_stat pageout/swapout deltas and the swap in use, so memory pressure can be seen, not guessed.
usage: remeasure_real.py scan | remeasure_real.py pairs THETA AGG PMAX COARSEN"""
import csv
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "audit_remeasure"
PY = "/opt/anaconda3/envs/hypre_env/bin/python"
BASE = (0.5, 1, 4, "HMIS")
SCAN = [(t, a, 4, "HMIS") for t in (0.25, 0.5, 0.7) for a in (0, 1, 2)] + [(0.5, 1, 2, "HMIS"), (0.5, 1, 8, "HMIS"), (0.5, 1, 4, "PMIS")]


def vm():
    txt = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return (int(re.search(r"Pageouts:\s+(\d+)", txt)[1]), int(re.search(r"Swapouts:\s+(\d+)", txt)[1]),
            float(re.search(r"used = ([\d.]+)M", swap)[1]))


def one(tag, method, amg, log, keep_x):
    """Run one method in a fresh process, append its 1e-8 row with the memory counters to LOG."""
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if amg:
        keys = ("strong_threshold", "agg_nl", "P_max", "coarsen_type")
        env["PETSC_OPTIONS"] = " ".join(f"-ir_amg_pc_hypre_boomeramg_{k} {v}" for k, v in zip(keys, amg))
    out = OUT / f"{tag}.csv"
    before = vm()
    t0 = perf_counter()
    with open(OUT / f"{tag}.log", "w") as fh:
        rc = subprocess.run([PY, str(HERE / "run_ir.py"), str(out), "1e-6", method], env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE).returncode
    after = vm()
    row = dict(tag=tag, method=method, amg=":".join(map(str, amg)) if amg else "", returncode=rc, process_s=perf_counter() - t0,
               pageouts=after[0] - before[0], swapouts=after[1] - before[1], swap_mb_before=before[2], swap_mb_after=after[2])
    if rc == 0:
        with open(out) as fh:
            row.update(next(r for r in csv.DictReader(fh) if float(r["target"]) == 1e-8))
        if not keep_x:
            (OUT / f"{tag}_{method}.npz").unlink()
    with open(log, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(tag, method, row.get("total_s"), row.get("final_rel"), "pageouts", row["pageouts"], flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rng = random.Random(20261105)
    if sys.argv[1] == "scan":
        order = SCAN[:]
        rng.shuffle(order)
        for i, amg in enumerate(order):
            one(f"scan{i:02d}", "amg-3d", amg, OUT / "scan.jsonl", keep_x=False)
    else:
        best = (float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
        arms = [("base", "amg-3d", BASE), ("best", "amg-3d", best), ("lu", "bt-fd", None), ("chol", "bt-fd-chol", None)]
        for rep in range(5):
            order = arms[:]
            rng.shuffle(order)
            for name, method, amg in order:
                one(f"pair{rep}_{name}", method, amg, OUT / "pairs.jsonl", keep_x=rep == 0)
