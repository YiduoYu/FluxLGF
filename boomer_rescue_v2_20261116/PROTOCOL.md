# Second-round adaptive Boomer correction study

Date: 2026-11-16. All first-round cases are now DEVELOPMENT evidence. No first-round
held-out result will be called independent validation for the second-round design.
The original experiment directory is not modified.

## Questions and candidates
- Can a Ritz-residual stopping rule buy sufficient slow-coordinate accuracy without
  imposing a fixed large budget on every problem?
- Can a time-to-solution estimate prevent expensive correction on almost solved systems?
- Do multiple coordinates or splitting disconnected supports help multiple-cell fields?
  These are candidates, not promised improvements or established novelty.
- Keep a balanced SPD correction around Boomer; compare the faster of Pmax=4/8.
- Strong additional control: the same new wrapper with a fixed 96-step coordinate.
  This separates benefits of adaptation from simply increasing the budget.

## Plan
1. Diagnose Ritz convergence and early PCG prediction on old cases only.
2. Develop and compare a small number of explicit candidates, preserving all attempts.
3. Before independent validation, write LOCK.json with source hashes and all parameters.
4. Use new geometry seeds, a crossed contrast/geometry design, smooth and random-forcing
   RHS, and scales 511/1023 plus a limited 2047 check. Include topology/control cases.
5. No parameter changes after the lock. Any later refinement is labelled post-hoc.

## Measurements
One process and one BLAS/OpenMP thread, no overlapping timed jobs. Include all pilot,
coarse operator construction, Ritz work, basis/factorization and solver costs; exclude
common original matrix assembly and independent audit. Require double-double relative
residual <=1e-8 for stored A,b. Report actual solution error only when a reference exists.
Repeat independent comparisons twice, and short controls at least three times if needed.
Retain failures and baseline wins. Baseline options and same stored A,b are shared.

## Go/no-go
Development gains alone do not suffice. Seek >=25% median total-time reduction on the
predeclared hard subset, <=10% overhead on controls, and meaningful gain at the largest
size. Compare with fixed-budget correction to justify adaptivity separately. Novelty
requires a concrete distinction from existing adaptive coarse spaces, level-set/path
cover, deflation, and adaptive eigenvector tolerances; no claim of priority is presumed.
