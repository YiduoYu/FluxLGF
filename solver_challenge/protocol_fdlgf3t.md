# H36 — improved FD-LGF (FD-LGF-3T) on the 3T surrogate at full challenge size (locked before the m = 128 run)

User request: "你能不能通过相关paper和相关工作，改进FD-LGF，使得它能解决这一类问题".
Method `fdlgf3t.py` (balanced form P = Q + (I − QA) M (I − AQ)):
- M: point-block (3×3 Cholesky) Green–Jacobi LGF, i.e. exact constant-coefficient Green's function (DST-II in 3D)
  of each scaled temperature block (Ladecký et al. 2025 with the scalar Jacobi replaced by the cell's 3×3 block),
  wrapped in block-Jacobi smoothing ω = 0.6 on both sides.
- Q: box-constant coarse space (b = 4 cells per direction, one vector per temperature per box; region-constant
  deflation: Vuik et al. 1999, Aksoylu et al. 2008), applied recursively (multilevel 128³ → 32³ → 8³, W-cycle with
  2 coarse sweeps, sparse LU below 5000 unknowns).

## Already seen (EXPLORATORY)
- m = 32 / 64 (dt = 1e-3): one-level FD-LGF variants fail: background only no convergence; scalar Green–Jacobi
  1663 / 1801 its; + region coarse space ~1300 its. The 3×3 block scaling is the key step (1663 → 112 at m = 32);
  + box coarse space 36–50; + block-Jacobi smoothing 25–28 (LU coarse), 30–32 (multilevel W-cycle).
- m = 64, dt from 1e-5 to 1e3: FD-LGF-3T 18–54 its, 1.9–4.7 s; BoomerAMG 3D settings 8–16 its, 1.0–2.4 s.
- m = 128, box 8 one-level with LU coarse (no smoothing): 80 its, 7.5 s setup + 30 s solve.
  box 4 one-level LU coarse (98k unknowns): splu setup > 4 min, abandoned.

## Confirmatory run
`run_sc.py h36_dt<dt> 128@<dt> amg amg-3d fdlgf3t` for dt ∈ {1e-3, 1e1}; targets 1e-4 and 1e-8; single thread.

## Predictions (m = 128)
1. FD-LGF-3T reaches both targets for both dt; iterations to 1e-8 ≤ 1.5 × the m = 64 count (32 → ≤ 48 at dt = 1e-3,
   54 → ≤ 81 at dt = 1e1): close to mesh independence.
2. Peak memory ≤ 9 GB (the challenge limit).
3. FD-LGF-3T total time ≤ 3 × BoomerAMG (3D settings) for both targets and both dt, but not faster than it.
