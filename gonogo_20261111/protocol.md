# G1 — FD-LGF go/no-go against the strongest baselines at equal error (CONFIRMATORY at n = 127)

Written 2026-11-11, before the full run. Purpose: decide whether FD-LGF can carry the first paper (diagnosis of 2026-11-11,
bottleneck 3). Locked by `protocol.sha256` (files listed there); no listed file is edited while the run is going.

## What was seen before locking (exploratory, not evidence)
Smoke tests at chi = 1e8, one run each, desktop under load: n = 31 all four cases, n = 63 cases A, C, D with
fdlgf (w = 1/32, 1/64), amg_old, amg_hmis, amg_agg, pfmg_j, pfmg_rb, pfmg_rb22, smg. Seen at n = 63 (t_err, s):
A fdlgf 0.03, best PFMG 0.12, amg_agg 0.42, amg_old 1.11; C fdlgf best 0.43, pfmg_rb 0.14, amg_agg 0.52;
D fdlgf best 0.33, pfmg_j 0.31, amg_agg 0.56. B at n = 63 and everything at n = 127 were not run.
Changes made after the smoke test, before locking: petsc4py/hypre library load moved out of the timed region (it was adding
about 1.1 s to every hypre method); pfmg_rbng (identical iterations to pfmg_rb) replaced by pfmg_rb22.

## Problem
-div(a grad u) + q u = f on (0,1)^3, zero Dirichlet, 7-point flux stencil (`three_d/ops3d.py`), cases of `three_d/cases3d.py`:
A chi^x (analytic f), B step layer, C chi^x x 27 regular balls, D chi^x x 30 random balls (seed 0).
Reference u_h: B, C, D discrete MMS (u_h = u* exactly); A FD-LGF PCG to round-off (60 iterations or residual 1e-15), checked
against BoomerAMG PCG to 1e-14 (`--prep`, must agree to 1e-10). Done before locking: residuals 6e-15 to 8e-14, agreement 6e-16 to 2.5e-14.

## Methods (one preconditioner apply per PCG iteration, same PCG loop, CSR mat-vec for all)
- FD-LGF: `three_d/solvers3d.py::Method2_3D` unchanged; A, B plain background inverse; C, D ring w = 1/32 and 1/64.
- BoomerAMG (petsc4py, hypre): amg_old (Falgout, classical, theta 0.5 = reports 01/29), amg_fal25 (theta 0.25),
  amg_hmis (HMIS, ext+i, Pmax 4, theta 0.5), amg_hmis25 (same, theta 0.25), amg_pmis (PMIS, ext+i, Pmax 4, theta 0.5),
  amg_agg (amg_hmis + 1 level aggressive coarsening).
- hypre PFMG: pfmg_j (weighted Jacobi 2/2, skip 1), pfmg_rb (symmetric red-black 1/1, skip 1), pfmg_rb22 (symmetric RB 2/2, skip 0).
- hypre SMG (1/1).
Settings were chosen from fluxmg/structured/tune.txt and hypre's 3D recommendations, not tuned on these cases.

## Criteria and measurement
x0 = 0, plain PCG, at most 500 iterations. For every run record:
- t_err: wall time from the start of setup to the first iteration with ||u_k - u_h||_2 / ||u_h||_2 <= 1e-8 (PRIMARY);
- t_res: the same for true ||f - A u_k|| / ||f|| <= 1e-10 (secondary, = the criterion of reports 01/29);
- iterations, setup time, error at the residual stop, peak RSS.
The error norm is evaluated every iteration for every method (same overhead for all, it is included in the times).
Single thread (OMP/VECLIB/OPENBLAS = 1), each run its own process, all jobs shuffled (seed 20261111), 5 repeats;
the statistic is the median over repeats. Desktop machine with other programs open: load average is logged per run.

Grid: n in {63, 127} (2.6e5 and 2.0e6 unknowns), chi in {1e4, 1e8}. A run with no t_err within 500 iterations, a crash or a
timeout (900 s) counts as failed.

## Decision rule (PRIMARY: chi = 1e8, n = 127, t_err)
Per case, R = (best median t_err over all 10 baseline settings) / (best median t_err over the FD-LGF settings).
The best-of-settings choice is made on the same data for both sides; it favours the baselines (10 settings against 1-2).
- GO: R >= 1.5 on at least 3 of the 4 cases. FD-LGF stays the core of paper 1 as planned.
- REPOSITION: R >= 1.5 on A and B only. Paper 1 becomes a method for separable / stratified media; the inclusion cases
  are reported as the limit of the method.
- NO-GO: anything else. FD-LGF does not carry a paper on its own as currently formulated.

## Predictions (mine, written down so they can fail)
1. A, B (n = 127, chi = 1e8): R >= 3 (FD-LGF needs 1-3 iterations and no hierarchy).
2. C, D: R < 1.5; the best PFMG setting is faster than FD-LGF at n = 127.
3. The best BoomerAMG setting is at least 2x faster than amg_old in total time at n = 127.
4. At the residual stop 1e-10, FD-LGF's error is larger than the best baseline's on B, C, D.
So the expected outcome is REPOSITION.

## Not covered
Parallel runs, n = 255, coefficient families other than chi^x, BoxMG, AGMG.
