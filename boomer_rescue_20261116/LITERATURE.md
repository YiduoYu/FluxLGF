# Novelty audit (2026-11-16)

This is a scoped literature search, not proof of priority. A new name is not a contribution.

## Closest work found

1. Hu, Lin, Zikatanov, *An Adaptive Multigrid Method Based on Path Cover*,
   SIAM J. Sci. Comput., DOI 10.1137/18M1194493; preprint https://arxiv.org/abs/1806.07028.
   Read abstract and full-text algorithm discussion. Approximates smooth-error level sets
   by vertex-disjoint paths, aggregates along those paths, and rebuilds a multilevel UA-AMG
   hierarchy as convergence slows. Thus neither “use error level sets” nor “adaptive AMG”
   can be claimed new here. Main stated setting is positively weighted graph Laplacians.

2. Tang and Vuik, *New Variants of Deflation Techniques for Pressure Correction in Bubbly
   Flow Problems*, JNAIAM 2 (2007), 227–249.
   https://jnaiam.org/2024/02/new-variants-of-deflation-techniques-for-pressure-correction-in-bubbly-flow-problems/
   Read publisher abstract (website migration date 2024 is NOT publication date).
   Levelset and levelset-subdomain deflation spaces use bubble locations. Thus “level-set
   deflation” alone is long-established. Full article not inspected in this session.

3. Brandt, Brannick, Kahl, Livshits, *Algebraic distance for anisotropic diffusion problems:
   multilevel results*, ETNA 44 (2015), 472–496.
   https://etna.ricam.oeaw.ac.at/vol.44.2015/pp472-496.dir/pp472-496.pdf
   Uses test-vector least-squares relationships, compatible relaxation, and interpolation
   to handle non-grid-aligned/non-M-matrix anisotropy. Learning hard error is not new.

4. Brezina et al., *Adaptive Smoothed Aggregation (alphaSA)*, SISC 25 (2004), 1896–1920.
   https://www.math.mun.ca/~smaclachlan/research/aSA.pdf
   Localized test vectors and bootstrap enrichment belong to this existing tradition.

5. Southworth, Al Daas, Wimmer, Threlfall, *Algebraic Multigrid with Overlapping Schwarz
   Smoothers and Local Spectral Coarse Grids for Least Squares Problems*, 2026 preprint.
   https://arxiv.org/abs/2601.04112
   Sparse A=G^T G structure permits localized PSD splitting and recursive spectral spaces.
   The paper itself notes suboptimal closed-field-line scaling and emphasizes convergence
   factors/operator complexity rather than a production wall-time implementation.

## Candidate distinction tested here

The candidate is **preconditioner-aware, amplitude-weighted spectral-coordinate lifting**:

1. On a geometrically coarsened rediscretization, use A-inner-product Lanczos on B A,
   where B is an existing symmetric Boomer cycle. Lift one slow Ritz vector v to the fine grid.
2. Normalize v, and construct Z_ij = v_i h_j(v_i) with piecewise-linear hats h_j in
   the *value coordinate*, not in physical coordinates. The hats partition unity, so
   their weighted sum reproduces v. Every row has at most two nonzeros per coordinate.
3. Form C=Z^T A Z and balance its correction around the existing fine-grid Boomer.
   Do not rebuild Boomer's hierarchy. C can have O(n) dimensions on an n-by-n grid,
   without O(N times dim(C)) dense basis storage.
4. Optional pilot is standard Ritz-based gating, inspired by the project's H48;
   it is an engineering safeguard, NOT a novelty claim.

These differences from the specific papers above are real implementation distinctions,
but their combination is only a **candidate incremental contribution**. No assertion
that this is the first such construction is justified by this search. Further search of
nonlinear coarse-space enrichment, spectral aggregation, algebraic coordinates, and
operator compression is required before a paper-level priority claim.

## Necessary ablations and limitations

- Compare raw linear Ritz-vector correction, local hat/QR correction, and scalar-coordinate lifting.
- Replace B A slow coordinate with a low eigenvector of A to test preconditioner awareness.
- Compare amplitude-weighted hats with unweighted hats; do not attribute all gains to weighting.
- Compare tuned Boomer, include setup, verify stored-matrix residual independently.
- First prototype needs grid and coefficient access for rediscretization; it is NOT a
  matrix-only, unstructured, parallel, or 3D solver.
- A scalar coordinate aliases different regions with equal values and may fail at multiple
  vortices/branches. Coordinate quality, pilot errors, and high setup cost can erase gains.
- Fixed rank global deflation and the new coordinate space have different sparsity costs;
  compare total cost, not merely the number of basis functions.

## Mechanism (heuristic, not convergence theorem)

If v is nearly constant along a strongly diffusive field direction, so are smooth functions
v*g(v), since grad(v*g(v)) = [g(v)+v*g'(v)] grad(v). Hat expansion can represent many
transverse variations from a single coordinate. This only works when that coordinate
separates the relevant slow modes. It does not establish mesh-independent convergence,
particularly for non-M matrices, finite-precision coordinates, or topological aliasing.

## Additional recent checks

- Krzysik, Southworth, Al Daas, *Two-level convergence of Algebraic Multigrid with
  Overlapping Smoothers and Spectral Coarse Grids*, June 2026 preprint,
  https://arxiv.org/abs/2606.18404. The arXiv abstract was retrieved through search;
  direct page fetch failed. It develops a two-level bound for LS-AMG-DD using local
  spectral cutoffs and smoother norm comparisons. Full text was not inspected here.
  This further rules out claiming local spectral coarse spaces or their adaptive
  dimension selection as the present contribution.
- Xu and Zikatanov, *Algebraic Multigrid Methods*,
  https://arxiv.org/abs/1611.01917; survey PDF
  https://multigrid.org/xu/paper/xu2017algebraic.pdf. The smoother-dependent notion
  of algebraic smoothness and quasi-optimal coarse spaces is established theory.
- Anisotropy-robust multiscale spectral spaces are also a stronger future comparator:
  *Multiscale approximation and two-grid preconditioner for extremely anisotropic
  heat flow* (2025), https://www.sciencedirect.com/science/article/abs/pii/S002199912500484X.
  Only publisher abstract/search material was checked; no implementation comparison
  with this paper was performed in the present experiments.
