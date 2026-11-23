# H32 — 3D version of the H31 comparison (CONFIRMATORY; smoke test results listed below were seen before locking)

Locked 2026-11-03 before the full 3D run. User request: "二维和三维的方程都试一下，分别统计各自的精确度、迭代数和时间，
用 latex 的 pdf 给我". Smoke test (exploratory, χ = 1e6 only): C/D at n = 31, 63 and C at n = 127 with BoomerAMG, GMG, M2 w = 1/16, 1/32,
1/64. Seen: n = 63 M2(1/32) 11–12 its 0.4–0.5 s vs BoomerAMG 14–16 its 1.5–1.8 s; n = 127 C: M2(1/32) 10 its but
27 s (LU setup 23 s), M2(1/64) 14 its 11.5 s, BoomerAMG 14 its 11 s, GMG 101 its 19 s.

## Setup
−∇·(a∇u) + qu = f on (0,1)³, zero Dirichlet, 7-point flux stencil with arithmetic face values (`ops3d.py`), same PCG
recurrence and stopping rule as 2D (‖r‖/‖f‖ ≤ 1e-10), maxiter 500 (1000 in 2D; reduced so that the failing LGF runs at
n = 127 fit in the time budget); error traces capped at 300 iterations for every method. Cases (`cases3d.py`): A χ^x (continuous MMS),
B step layer, C χ^x × 27 balls (r = 0.08, ×1e4), D χ^x × 30 random balls (×1e4 or ×1e-4). χ ∈ {1e2, 1e4, 1e6, 1e8},
n ∈ {31, 63, 127}. Single-threaded BLAS.

Methods (`solvers3d.py`): LGF (3D DST, a0 = mean a, σ = mean q); GMG (trilinear, Galerkin, V(2,2) Jacobi
ω = 6/7 on the Laplacian); BoomerAMG (PETSc defaults, strong threshold 0.5 as hypre recommends for 3D); 方法二 with
ring w = 1/32 and 1/64 on C, D (w = 1/16 merges the 27 balls into one region whose 3D LU is infeasible at n = 127),
plain background inverse on A, B. Reference u_h for A: 方法二-PCG to round-off (validated against a direct solve at
n ≤ 63 in the smoke test).

## Predictions
1. A, B: 方法二 ≤ 3 its for all χ, n; BoomerAMG and GMG ≤ 15 its; LGF fails (500 its) on A for χ ≥ 1e6.
2. C, D: 方法二 (w = 1/32) ≤ 15 its, changing by ≤ 3 from n = 31 to 127; BoomerAMG needs more iterations than
   方法二 w = 1/32 at n = 127; GMG ≥ 50 its.
3. Time at n = 127: on A and B 方法二 total time ≤ 1/3 of BoomerAMG's; on C and D the best 方法二 ring is within
   1.3× of BoomerAMG (no clear winner), and w = 1/32 is slower than w = 1/64 because of the 3D LU setup.
4. A: all converged methods give the same total error to 3 significant digits.
5. Accuracy at the residual stop on C, D: 方法二 algebraic error larger than BoomerAMG's; iterations to error 1e-8
   ≤ 18 for 方法二 w = 1/32.
