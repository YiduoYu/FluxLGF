# Round 3: contribution rather than parameter search

Declared before any benchmark on the cases below. Rounds 1 and 2 are development
evidence now; their source and results stay unchanged. No training or exact-solution
information is used to construct any coarse space.

## Questions

1. With the same 96-step coarse-grid Ritz construction, does expanding two modes
   in their value coordinates improve time/iterations over using 2, 16 or 32 modes
   directly? Extraction and final residual-check costs can differ with output rank;
   this is equal Arnoldi-step budget, NOT exactly equal wall-clock setup cost.
2. Does value-coordinate localization outperform geometric overlapping partition
   of unity enrichment with approximately matched coarse dimension?
3. Is the weighted construction better than unweighted hats? This comparison also
   changes handling of the near-zero hat; do not claim a clean single-factor test.
4. How does the unchanged round-2 candidate compare with a real implementation of
   adaptive smoothed aggregation (PyAMG 5.3.0 alpha-SA, 2 candidates)?

## Fixed design

All ablations start from x=0, have no pilot/gate, use Pmax=8 Boomer as B, exactly
96 coarse A-inner-product Arnoldi steps, coarse grid approximately /4 each direction,
and the same balanced correction. Weighted/unweighted/local use 2 modes. Direct
deflation uses 2/16/32 modes, dense BLAS and Cholesky. Weighted hat bins follow the
round-2 rule. Local POU uses tensor linear hats in physical space times local
{1,v1,v2}, pivoted QR, about 2*(bins+1) columns. It is a simple geometric enrichment
control, NOT a reproduction of alpha-SA or GenEO; no smoothing is applied to it.

Actual PyAMG alpha-SA: symmetric problem, 2 candidates, 5 candidate iterations,
0 improvement iterations, symmetric strength, standard aggregation, symmetric
Gauss-Seidel pre/post, Jacobi prolongation smoothing, max_levels=10,
max_coarse=10, pinv coarse solver, random seed 91117. No tuning. Its implementation
differs from hypre; wall time is a practical configuration comparison, not a
universal ranking of the underlying algorithms.

The unchanged adaptive round-2 candidate keeps its 16-step pilot, gate and variable
coordinate budget. It is included as the practical candidate, NOT used for the
equal-budget mechanism comparison. Two plain Boomer baselines, Pmax=4 and 8.

Cases (all new parameters):
- C=[.21,.73,.35], n=511, chi=1e5, new smooth RHS4 seed=17119:
  circ, disk, ellipse, two_vortices.
- D=[.47,.19,.81], n=1023, chi=1e5, actual random forcing RHS3 seed=17123:
  circ, disk, two_vortices.
- D, n=1023, circ, chi=1e4, smooth RHS4 seed=17123 (intermediate strength).

All 10 methods run on all 8 cases: three repeats on n=511, two on n=1023 = 200
measurements. Randomized order seed=17116. One subprocess at a time, one compute
thread, timeout 240 seconds per job; all failures/timeouts retained. If a particular
method exhausts 240 seconds on its first repeat for a case, later repeats of that
same case/method are marked budget-skipped (not successes) to avoid repeated
unproductive runs. All timing includes preconditioner and coarse-space setup,
excludes assembly, final DD audit and interpreter startup. Record both CPU and wall
time. Accuracy: independently recomputed double-double relative residual <=1e-8.
Manufactured error is a separate metric. No speedup credited to failed jobs.

Report every case and negative result. Repeated measurements are NOT independent
PDE samples. Summary uses per-case median timings, with CPU ratios as sensitivity.
No claim of statistical significance with 2/3 repetitions. Prespecified practical
evidence threshold: candidate should beat the best plain Boomer median in at least
6/8 cases and median time by >=20%; equal-budget weighted should beat the fastest
direct-rank choice per case in >=6/8 cases. These are pilot research criteria, not
a proof of generality or priority. Record both passes and failures.

The 2026 diffusion-filtering paper is known only through publisher abstract and
section snippets; do NOT label any comparator as an exact reproduction of it.
