# Boomer rescue feasibility study, 2026-11-16

This is a new exploratory experiment. No claimed novelty or speedup is presumed.
Existing repository files are not modified. Code, raw JSONL results, and final report live here.

## Questions
1. Can a localized bootstrap correction beat tuned BoomerAMG including setup?
2. Can sparse nonlinear lifting of a coarse-grid, preconditioner-aware slow coordinate
   overcome the fixed-rank global-deflation limitation with inexpensive application?
3. Are improvements beyond stronger existing Boomer options and linear slow-mode deflation?

## Candidate definitions
- Local: restrict coarse-grid slow test vectors to overlapping spatial hats, locally QR,
  optionally smooth the resulting sparse basis, and form its Galerkin matrix.
- Coordinate lift: use one or more coarse-grid slow Ritz vectors as scalar coordinates;
  compose piecewise-linear hats with their values on the fine grid. This expands the
  linear span nonlinearly, but the resulting solve is a fixed linear preconditioner.
- Both use a balanced coarse correction around one fixed symmetric Boomer cycle.
  Matrices and coordinates are available, but the manufactured solution and RHS are
  NEVER used to construct these spaces. Geometry/grid use is reported explicitly.
- These components have close prior art (bootstrap AMG, aggregation, deflation,
  level-set/path-cover methods). Novelty is a question to assess, not a premise.

## Development versus validation
Development: p=[0.5,0.5,0.5], chi=1e4, circ/disk/rot, n=127/255/511.
Candidate choices may be changed only on development cases. All attempts retained.
Before final validation, write a separate LOCK.json fixing candidates and parameters.
Validation: distinct geometry seeds, chi=1e3/1e5, n=511/1023 and, if feasible, 2047.
Include scalar Poisson/high-contrast controls and a second RHS.

## Fairness and stopping
- Single process per measured solve, one BLAS/OpenMP thread, randomized method order.
- Same assembled A and b, zero initial guess, true relative residual <=1e-8.
- Recompute residual at candidate convergence; independent long-double residual and
  manufactured-solution error after solve. Residual replacement restarts CG.
- Include matrix-to-PETSc conversion, setup, probes, basis build, coarse factorization,
  and all solve work in total. Exclude common PDE assembly and independent final audit.
- Record process CPU and wall time, iterations, failures, sparse-basis nnz, coarse size,
  and peak RSS. Use paired repeated timings for final comparisons; no concurrent jobs.
- Existing H48 can be a contextual comparison; new timings use this driver and criteria.
- Screening is not included in per-solve time; report that selection cost separately.
- Targets: median hard-case total-time ratio <=0.75, easy-case overhead <=10%, and
  no disappearance of gain at the largest tested size. Targets are not promised results.

## Verification
On small matrices check symmetry/positivity of the complete balanced preconditioner,
Galerkin identity, nondegenerate basis, and agreement with direct solve. Do not infer
PDE-discretization accuracy from algebraic convergence. Do not infer generality from circ.
