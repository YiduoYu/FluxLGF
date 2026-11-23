# Audit and scope notes

- Before LOCK.json, detected that NumPy longdouble is binary64 on this Apple arm64
  environment. Development results used recomputed binary64 residuals; do not label
  those as higher-precision audits. All locked validation uses chunked double-double
  residuals from the repository's DDResidual implementation. The implementation was
  checked against exact Fraction arithmetic on every row of small test matrices.
- A 0.3-second, 225-unknown symmetry diagnostic for built-in Boomer smoothers was
  launched during the early portion of validate1023. This is an unintended departure
  from strictly isolated timings. The first five records are conservatively flagged
  for clean remeasurement; original records remain preserved.
- Topology stress tests (elliptic closed lines and two-vortex streamfunction) are
  additional tests of the same locked method, not development cases. No parameter
  selection will use them.
- Pilot gating is RHS-dependent, but coordinate/basis construction uses only the
  operator, grid, and rediscretization callback; neither solution values nor RHS enter
  the spectral-coordinate computation. The pilot threshold is an existing-style
  heuristic and is NOT claimed novel.
- A balanced external correction is implemented, not a modification of hypre's
  interpolation hierarchy. Local prototype is local hat/QR enrichment, NOT a full
  GenEO or LS-AMG-DD implementation.
- After observing the first 2047-scale results, predeclare an exploratory mechanism
  check: increase only coarse Ritz budget from 32 to 96, on circ/disk 2047 with two
  repeats each. This is POST-HOC development, not a new independent validation set.
  Store in diagnose_scale.jsonl, excluded from summarize.py and headline statistics.
