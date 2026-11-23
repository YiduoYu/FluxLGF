# Novelty update, 2026-11-16

This round retains the first-round novelty limitations; see ../boomer_rescue_20261116/LITERATURE.md.

- Relative Ritz/eigenpair residual stopping is standard. SLEPc explicitly documents
  residual/eigenvalue criteria, cheap Krylov residual estimates and true residual checks:
  https://slepc.upv.es/release/documentation/manual/eps.html.
  Our A-norm criterion applied to B A is an implementation choice, not a new principle.
- Smooth-error level-set paths and adaptive aggregation are existing work:
  Hu, Lin, Zikatanov, https://doi.org/10.1137/18M1194493 and
  https://arxiv.org/abs/1806.07028. Splitting geometric support components should not be
  advertised as the invention of topology-aware coarsening.
- A particularly relevant additional paper is *Two-level preconditioning with
  diffusion-filtered coarse spaces for ill-conditioned finite-element SPD systems*,
  Computers & Mathematics with Applications 222 (15 Nov 2026), 244–267,
  https://doi.org/10.1016/j.camwa.2026.09.024.
  Publisher abstract/highlights and introduction/section excerpts were retrieved,
  not the full article. A direct full-page fetch returned HTTP 403; no full method
  reproduction or associated code was obtained. It already combines
  baseline-aware/preconditioned filtering, adaptive Krylov work, SVD compression and
  balanced two-level coupling. Thus merely adding an adaptive spectral budget around
  a baseline preconditioner is NOT a defensible novelty claim. Its stated emphasis is
  reusable/amortized solves; our tested distinction remains nonlinear sparse value-
  coordinate lifting with O(1) basis entries per node. No performance superiority to
  that paper has been established.

The second-round contribution must be assessed against its own fixed-96-step control.
If adaptive stopping is worse than a fixed budget, the experiment should report that,
not rebrand the fixed-budget construction as an adaptive innovation.
