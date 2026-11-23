# H34 — anisotropic tensor coefficients: a case where BoomerAMG degrades, and FD-LGF on it

Locked 2026-11-03. User request: "做各向异性" (answer to: extend to rotated anisotropic tensor coefficients and build
cases where BoomerAMG really struggles).

PDE −∇·(K∇u) + qu = f on (0,1)², zero Dirichlet, K = χ e eᵀ + (I − e eᵀ) (anisotropy ratio χ), symmetric 9-point
stencil (`ops_aniso.py`; SPD, not an M-matrix when K12 ≠ 0). Families `cases_aniso.py`: aligned, rot30, rot45, circ,
disk, band. Methods: BoomerAMG θ = 0.25 (PETSc default) and θ = 0.5, GMG (ω = 4/5), FD-LGF w = 1/16 and 1/64
(`fdlgf_tensor.py`: background diag(k1, k2) inverted by DST-y + tridiag-x; w = 1/64 only where there is a feature mask).

## Already seen (EXPLORATORY, before locking)
- `results/scan.csv`, n = 127, 255, 511, χ ∈ {1e2, 1e4, 1e6}. BoomerAMG iterations grow with n on circ (38→61→96 at
  χ = 1e4), disk (35→50→74), band (33→38→47), mildly on rot30/rot45 (16→22); flat on aligned (≤ 8).
  FD-LGF: disk/band 4–8 its flat in n (0.24–0.33× BoomerAMG time at n = 511, χ ≥ 1e4); rot/circ 69–874 its
  (aligned background only, κ ~ χ). GMG 2–5× BoomerAMG's iterations.
- On disk/band at χ = 1e6 the residual rule stopped FD-LGF while the error was still 1e-7–1e-6 and falling ~50× per
  iteration. Fix before locking: after the rule fires, PCG keeps iterating for the error trace only (untimed),
  so its_err_1e8 (k8) is defined for every method. Smoke check n = 127: FD-LGF k8 = 6–7, BoomerAMG 22–35.

## Confirmatory run
n = 1023 (10⁶ unknowns), all 6 families, χ ∈ {1e2, 1e4, 1e6}, all methods:
`bench_aniso.py aniso1023 1023` (hypre_env, single thread). n = 511 reference values are the scan's.

## Predictions (n = 1023)
1. BoomerAMG keeps degrading with n: for both θ, its(1023) ≥ 1.3 × its(511) on circ and disk, and its(1023) > its(511)
   on band, for χ ≥ 1e4; BoomerAMG θ = 0.25 needs > 120 iterations on circ for χ ≥ 1e4.
2. aligned: FD-LGF ≤ 3 iterations and total time ≤ 0.2× best BoomerAMG. rot30, rot45, circ (global anisotropy, FD-LGF
   has no local part): FD-LGF total time ≥ 1.5× best BoomerAMG for χ ≥ 1e4.
3. disk, band: FD-LGF w = 1/16 k8 ≤ 10 for every χ and ≤ its n = 511 value + 2; total time ≤ 0.5× best BoomerAMG for
   χ ≥ 1e4.
4. GMG needs ≥ 2× BoomerAMG θ = 0.25's iterations on aligned, circ and disk for χ ≥ 1e4.
5. The residual rule hides error: on disk and band at χ = 1e6, FD-LGF w = 1/16 stops with algebraic error > 1e-8, and
   reaches 1e-8 within 3 further iterations.
