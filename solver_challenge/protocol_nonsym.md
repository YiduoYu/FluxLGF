# H38 — nonsymmetric 3T surrogates (locked before the m = 128 run)

Real 3T matrices are often nonsymmetric (emission linearisation, row normalisation). Variants in `surrogate3t.py`:
`emis` (Te–Tr exchange as linearised emission; mildly nonsymmetric, relative asymmetry ~7e-5) and `scaled` (each
equation divided by c/dt; rows differ by up to ~1e6). agg3t-w uses the plain 3×3 block inverse; GMRES(50), right PC.

## Already seen (EXPLORATORY, m = 64)
- emis dt 1e-3: BoomerAMG 3D 11 its / 1.46 s, agg3t-w 20 its / 1.03 s (to 1e-8).
- scaled dt 1e-3: BoomerAMG default 326 / 479 its (57 / 83 s); BoomerAMG 3D no convergence in 2000 its; agg3t-w
  35 / 61 its, 1.46 / 2.42 s. BoomerAMG 3D + PETSc diagonal scaling converges in its own norm but the true residual
  stalls (0.22 at rtol 1e-4, 8e-5 at 1e-8); internal rtol 1e-12 gives true 6.5e-9 in 32 its / 3.33 s.
- scaled dt 10: BoomerAMG default no convergence in 2000 its (residual 0.45).

## Predictions (m = 128, `h38_nonsym.py`)
1. emis: agg3t-w reaches both targets for both dt; total time ≤ 0.8× BoomerAMG 3D for both targets and both dt.
2. scaled dt 1e-3: agg3t-w reaches true 1e-8 in ≤ 100 its; BoomerAMG 3D + diagonal scaling at internal rtol 1e-8
   does not reach true 1e-8, at rtol 1e-12 it does; agg3t-w (1e-8) total ≤ 0.8× the rtol-1e-12 BoomerAMG run.

## Revision before any result (2026-11-03, after locking 89261c77; h38.log was empty when the run was stopped)
On m = 64 scaled dt = 10, agg3t-w's GMRES reported convergence with true residuals 7.7e-4 / 5e-6 (the residual
estimate drifts with rows scaled by up to 1e6). `run_sc.solve` now restarts from the current x, asking 10× more of
the estimate each time, until the TRUE residual meets the target (≤ 10 restarts, timed, counted). This applies to
every solver. The internal-rtol-1e-12 BoomerAMG run is therefore dropped; scaled dt = 10 is added; MAXIT 500.
Revised prediction 2: scaled, both dt: agg3t-w reaches both true targets; BoomerAMG 3D + diagonal scaling either
fails within 500 its or is ≥ 1.25× slower than agg3t-w at 1e-8 (agg3t-w ≤ 0.8×).
Prediction 1 unchanged.
