# H40 — real SolverChallenge26_01 matrix under the solver-reported residual criterion (user choice, 2026-11-04)

Why: ||b - Ax||_2/||b||_2 has a float64 floor of ~1e-2 on this matrix (H39), so the challenge targets cannot be met as a
true residual. User chose: the solver's own residual estimate decides (PETSc GMRES(50), right preconditioning,
unpreconditioned norm; x0 = 0; rtol 1e-4 and 1e-8; MAXIT 2000; no restarts). True and diag-scaled residuals are logged
to expose false convergence. SC_CRITERION=reported. Single thread for everything: OMP_NUM_THREADS = VECLIB_MAXIMUM_THREADS
= OPENBLAS_NUM_THREADS = 1. Run sequentially, nothing else running.

Methods: amg-3d, amg (BoomerAMG), fdlgf-real (NEW, design fixed here: FDM background = fitted per-block kappa, r, T1 far-
face Robin beta, constant one-way T1<-T2 coupling e12; feature = cells whose rows deviate from the background by > 1e-6
of the row max (any field), dilated by 4 cells; MUMPS LU of A_FF (403,275 unknowns); balanced form), agg3t-w, then the
H35 list (pctl, split-sym, fdlgf-GJ, fdlgf-G).

Seen before this lock (disclosed): structure (H39 notes); BoomerAMG default + GMRES reached estimate 1.2e-11 in 56 its;
agg3t-w broke down at 350 its (true residual 1.15); fdlgf-real setup 20.5 s and 1.17 s per application with
multithreaded BLAS (not single thread). No fdlgf-real iteration has been run.

Predictions:
1. BoomerAMG (best of amg / amg-3d) meets both targets; its 1e-8 solution has diag-scaled residual < 1e-6.
2. agg3t-w does not meet 1e-4 within 2000 iterations.
3. fdlgf-real meets both targets with <= 15 iterations to 1e-8, diag-scaled residual < 1e-6, peak RSS < 9 GB.
4. fdlgf-real total time <= the best BoomerAMG total time at 1e-8.
5. H35 predictions (protocol_real.md) are evaluated as locked, under this criterion.
