# H46 protocol: SNN-predicted deflation for BoomerAMG-PCG on anisotropic families (DRAFT — not locked)

## Question
Can a subspace neural network (SNN, Xu & Sheng arXiv 2404.08223), trained offline on coarse instances of a
parametric PDE family, predict the BoomerAMG slow subspace of *unseen* instances on *finer* grids well enough that
deflated BoomerAMG-PCG beats plain BoomerAMG-PCG in online time?  And does a non-neural, per-instance alternative
(coarse-grid Lanczos, interpolated) do as well?

## Fixed setup
- Discretization: our own 9-point SPD tensor stencil (`anisotropic/ops_aniso.py`), K = chi e e^T + (I - e e^T),
  parametric families in `snn/cases_param.py`: circ (centre), disk (centre, radius), band (angle, offset), rot (angle;
  AMG-easy control).  chi log-uniform in [1e3, 1e5], not a network input.
- Baseline: BoomerAMG HMIS / ext+i / P_max 4 / theta 0.25 / agg_nl 0 / symmetric-SOR/Jacobi (agg_nl 0 chosen in the
  exploration: 3-4x fewer its than the H40 setting on circ), PCG, ||r|| <= 1e-8 ||b||, x0 = 0, single thread.
- Deflated PCG: A-orthonormal W, x0 = W W^T b, z = (I - W (AW)^T) B r.
- Online time = AMG setup + basis construction (network evaluation or coarse Lanczos) + A-orthonormalisation + solve.
  Training (data generation + network training) is offline and reported separately.

## Methods
- base: BoomerAMG-PCG.
- snn: SNN basis Phi(x, y; p) evaluated on the target grid.
- coarse-lz: per-instance Lanczos (K = 100) for B_c A_c on n_c = 127, M Ritz vectors bilinearly interpolated.
- oracle (n = 511 only): M slowest modes of the target itself (Lanczos K = 200), cost not counted.

## Exploration disclosed (see NOTES_explore.md)
(TBD after the smoke tests)
