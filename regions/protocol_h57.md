# H57 — same-accuracy comparison: bt-fd vs BoomerAMG vs bt-v / bt-j (locked test)

Question: when every method must reach the same double-double original residual, does bt-fd keep its advantage on small
features, and does bt + AMG replace it as the default on many inclusions? Follows audit item 1 and report 25, experiment 1.

Criterion: ||b - Ax||_2 / ||b||_2 < 1e-8 in double-double, by iterative refinement, each correction by PETSc GMRES(50),
right preconditioning, inner rtol 1e-6, at most 10 refinement steps, x0 = 0 at each step. A run counts as converged only if
the dd residual reached 1e-8 and every inner reason > 0.
Metric: total_s = setup at step 0 + inner solves + dd residuals. Matrix generation, PETSc matrix construction and DD tables
are common costs, reported beside it. Refinement steps are discrete, so final residuals differ; they are reported.

Arms (compare_h57.py; each run its own process, single thread):
  amg-base    BoomerAMG AMG3D (theta 0.5)        amg-tuned   same, theta 0.25 (fastest of the audit's real-matrix screening)
  bt-v        block triangular + BoomerAMG        bt-j        joint electron-ion + BoomerAMG
  bt-fd       bt-fd MUMPS LU, match tol 0.1       bt-fd-chol  bt-fd CHOLMOD, match tol 0.1
  real matrix only: bt-fd-1e-6, bt-fd-chol-1e-6   (match tol 1e-6, as in H53 / report 24)
Sequences: preconditioner built at step 0 from A_0 and kept. G1 G4 G7 G8 G9 G10 of H56 (m = 128, 10 steps, phase 0.15 per
step, rhs seed 100 + step), one pass, arm order shuffled (seed 57). Real SC26_01: 5 blocks, arm order shuffled in each block.
G11 is excluded: bt-fd swaps by itself on this 17 GB machine, so its time is not a comparable measurement.
Nothing is rejected or trimmed. Swapouts are recorded per step, because other desktop programs are open on this machine.

Seen before the lock: all sequences with amg and bt-fd under the solver-reported criterion (H56, report 24); the real matrix
with all arms except bt-fd match tol 0.1 under IR; m = 32 development check only for bt-v and bt-j on gen_gen geometries.

Predictions (ratios are total_s; real matrix uses the median of per-block ratios):
1. All runs converge.
2. bt-fd / amg-base <= 0.85 on each of G1, G4, G7.
3. bt-fd / amg-base >= 1 on at least 2 of G8, G9, G10.
4. bt-v faster than amg-base on all 6 sequences and on the real matrix.
5. The faster of bt-v and bt-j beats bt-fd on each of G8, G9, G10.
6. bt-fd-chol faster than bt-fd on the real matrix and on at least 5 of 6 sequences; same inner iterations within 10 %.
7. amg-tuned / amg-base within 0.85-1.15 on all 6 sequences and on the real matrix.
8. Frozen preconditioner: total inner iterations at step 9 <= step 0 + 3 in every run.
9. bt-fd (tol 0.1) / bt-fd-1e-6 <= 1.10 on the real matrix.

Decision rules, fixed now (report 25):
R1. bt + AMG becomes the default for general 3T problems if one of bt-v, bt-j beats bt-fd on at least 3 of G7, G8, G9, G10 and
    its real-matrix ratio to bt-fd-1e-6 is <= 1.3.
R2. The case for FD-LGF on small features is lost if the better of bt-fd, bt-fd-chol is not faster than amg-base on at least
    2 of G1, G4, G7.
R3. FD-LGF is overturned on SC26_01 if bt-v or bt-j is not slower than bt-fd-chol-1e-6 on the real matrix.
These rules concern serial time on this machine only; parallel behaviour is a separate experiment.
Locked: 2026-11-05T17:07:12Z
