# Boomer rescue: round 3

This directory isolates a contribution/ablation study from the first two rounds.
Start with `REPORT.md` when the campaign is complete, then `RESULTS_TABLE.md`,
`PROTOCOL.md`, `THEORY.md` and `LITERATURE.md`. Partial summaries generated during
execution are provisional; `aggregate.json` records the number of completed jobs.

The source was locked before the new cases ran. `run.py run` verifies source hashes
and resumes by unique case/method/repetition identity; it never silently overwrites
measurements. Exactly 200 planned identities, with any budget-skipped identities
explicitly recorded as unsuccessful. `LOCK.json` must not be regenerated to pass
a hash check after changing a solver. Use a new study directory for a new method.

From the FD_LGF root, using the existing environment:

```sh
/opt/anaconda3/envs/hypre_env/bin/python boomer_rescue_v3_20261116/verify.py
/opt/anaconda3/envs/hypre_env/bin/python boomer_rescue_v3_20261116/run.py run
/opt/anaconda3/envs/hypre_env/bin/python boomer_rescue_v3_20261116/summarize.py --complete
/opt/anaconda3/envs/hypre_env/bin/python boomer_rescue_v3_20261116/anchors.py
/opt/anaconda3/envs/hypre_env/bin/python boomer_rescue_v3_20261116/confirm.py
/opt/anaconda3/envs/hypre_env/bin/python boomer_rescue_v3_20261116/joint.py
/opt/anaconda3/envs/hypre_env/bin/python boomer_rescue_v3_20261116/diagnostics.py
/opt/anaconda3/envs/hypre_env/bin/python boomer_rescue_v3_20261116/audit.py
/opt/anaconda3/envs/hypre_env/bin/python boomer_rescue_v3_20261116/write_report.py
/opt/anaconda3/envs/hypre_env/bin/python boomer_rescue_v3_20261116/plot.py
```

Do not run diagnostics, plotting, or another benchmark concurrently with timings.
Dependencies: PETSc with hypre/petsc4py, NumPy, SciPy, PyAMG (5.3.0 used here),
matplotlib for figures. `study.py` imports round-2 solver and round-1 experiment,
which in turn import the existing matrix assembly and double-double residual code;
their paths and SHA-256 hashes are recorded in the lock.

Raw outputs: `results.jsonl` (formal timings), `verification.jsonl` (small numerical
checks), `diagnostics.jsonl` (untimed mechanistic checks). Reported setup excludes
matrix assembly and includes all coordinate/basis/preconditioner construction.
Final independent residual audit is outside the timed interval for every method.
Peak RSS is process-wide, including assembly and intermediates, not solely basis
storage. No claim is made about MPI, GPU, 3D or the SC26_01 competition matrix.

Timing was interrupted after record 132 and resumed on 2026-11-17. See
RESUME_NOTE.json and TIMING_RESUME_CHECK.md for separate anchor and conditional
paired replication; neither replaces the original 200 rows. JOINT_PILOT.md and
joint.py describe an exploratory structure variant tested only in untimed diagnostics.
