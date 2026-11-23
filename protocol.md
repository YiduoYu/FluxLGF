# H31 — 方法二 (FD-LGF[FDM bg]) vs traditional LGF, GMG, hypre BoomerAMG (CONFIRMATORY)

Locked 2026-11-03 before the full run (smoke test n = 31 only). User request: "单独设一个方法二的文件夹 … 完整对比
boomer AMG、GMG 和传统的 LGF：迭代次数、准确度、时间等".
Differences from H30: AMG = hypre BoomerAMG (petsc4py, PETSc defaults) instead of PyAMG RS; new standalone
implementation `fdlgf_m2.py`; case D (random conducting + insulating discs) added; n = 1023 added; ring-width
variants w = 1/32, 1/64 on C, D; single-threaded BLAS; error after every iteration recorded.

## Methods (all through axy_lgf.solvers._pcg, rule R: ‖r‖/‖f‖ ≤ 1e-10, maxiter 1000)
LGF (DST, a0 = mean a, σ = mean q) · GMG (geometric V(2,2) damped Jacobi, Galerkin) · BoomerAMG (one V-cycle)
· M2 w = 1/16 (H30 setting) · M2 w = 1/32, 1/64 (C, D only). M2 receives the known background a1(x) and feature mask.

## Cases — `cases.py`; χ ∈ {1e2, 1e4, 1e6, 1e8}; n ∈ {127, 255, 511, 1023}
A a = χ^x continuous MMS · B step layer · C χ^x × nine discs (×1e4) · D χ^x × 20 random discs (×1e4 or ×1e-4).

## Metrics
iterations; setup / solve / total time (median of 3 if one run < 5 s); algebraic error vs u_h (direct solve for A,
discrete MMS for B–D); total error vs exact u (A); error after 5 and 10 iterations, iterations to error 1e-6/1e-8/1e-10
and time to error 1e-8 (setup + k·time per iteration) from an untimed trace of the same PCG recurrence.

## Predictions
1. A, B: M2 ≤ 3 its for every χ, n; BoomerAMG and GMG 6–12 its; LGF does not converge for A at χ ≥ 1e6.
2. A: all converged methods give the same total error to 3 significant digits; LGF's is ≥ 10× larger at χ ≥ 1e6.
3. C, D: M2 w = 1/16 ≤ 11 its and changes by ≤ 2 from n = 127 to 1023; BoomerAMG ≤ 12 its (so M2's iteration
   advantage over BoomerAMG is ≤ 3 its); GMG ≥ 40 its; LGF does not converge.
4. Time, A and B at n ≥ 511: M2 total time ≤ 1/3 of BoomerAMG's and of GMG's.
5. Time, C and D at n = 1023: M2 w = 1/16 is slower than BoomerAMG; the fastest M2 ring is within 1.5× of BoomerAMG.
6. Accuracy at rule R on C, D: M2 algebraic error is larger than BoomerAMG's (residual stop hides low-a error);
   iterations to error 1e-8 are ≤ 14 for M2 w = 1/16.
