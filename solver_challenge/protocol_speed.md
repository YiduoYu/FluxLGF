# H37 — speeding up FD-LGF-3T (locked before the m = 128 run)

User: "Speed it up" (after H36: FD-LGF-3T 1.35–1.95× slower than BoomerAMG with 3D settings).

## Already seen (EXPLORATORY)
- Profile m = 128 (H36 code): setup 5.6 s (S A Sᵀ product 2.6 s, sparse block-scaling assembly 1.2 s); one
  preconditioner application 435 ms (three 3D DST pairs ~150 ms, smoothing ~120 ms, matvec 30 ms, coarse 30 ms).
- **The LGF part does not help on this class** (m = 64): replacing the constant-coefficient Green function by the
  exact 3×3 block-Jacobi inverse gives 32 vs 32 and 56 vs 54 its; dropping it entirely 42 / 69. Fitting the Green
  coefficient to higher percentiles (76–169 its) or masking it to diffusion-dominated cells (36–104) is worse: the
  scaled coefficients are bimodal and the diffusion-dominated part still spans ~7 decades.
- Re-implementation with array ops (no S A Sᵀ, row sums by formula, D⁻¹ precomputed, bincount restriction):
  identical iterations, m = 64 setup 0.4 s, apply 59 ms (36 ms without DST). float32 breaks down at dt = 10 (dropped).
- Tuning (m = 64, no DST): box 2, ω = 0.5, V-cycle best time: 23 / 39 its, 1.28 / 2.09 s solve (dt = 1e-3 / 10).

## Confirmatory run
`run_sc.py h37_dt<dt> 128@<dt> amg-3d fdlgf3t fdlgf3t-fast bnn3t-fast`, dt ∈ {1e-3, 1e1}, single thread, nothing
else running.

## Predictions (m = 128)
1. fdlgf3t (H36 configuration, new code): same iterations as H36 (37 / 53 to 1e-8, ±1) and total time ≤ 0.7× H36
   (≤ 18.3 s / ≤ 23.7 s).
2. fdlgf3t-fast: ≤ 30 / ≤ 50 its to 1e-8; total time ≤ 1.2× BoomerAMG (3D settings) at 1e-8 for both dt.
3. bnn3t-fast: iterations within ±3 of fdlgf3t-fast; total time ≤ BoomerAMG (3D settings) at 1e-8 for both dt.
4. bnn3t-fast is faster than fdlgf3t-fast for both dt and both targets (the DST costs time and buys nothing).
