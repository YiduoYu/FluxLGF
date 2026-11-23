# Literature and comparator boundaries, round 3

Primary sources checked on 2026-11-16:

1. Brezina et al., Adaptive Smoothed Aggregation (alpha-SA), SISC 25(6),
   1896–1920, DOI https://doi.org/10.1137/S1064827502418598.
   Author-hosted paper: https://www.math.mun.ca/~smaclachlan/research/aSA.pdf.
   Related SIAM Review article: https://doi.org/10.1137/050626272.
   Establishes adaptive discovery of algebraically smooth candidates and adapting
   interpolation/coarsening. These general ideas are not our novelty.
2. PyAMG adaptive_sa_solver documentation/source:
   https://pyamg.readthedocs.io/en/latest/generated/pyamg.aggregation.html
   https://pyamg.readthedocs.io/en/latest/_modules/pyamg/aggregation/adaptive.html.
   Local installed version is 5.3.0; its actual signature/docstring were inspected.
   We run this implementation, not a hand-written substitute, with protocol-locked
   options. A poor result is specific to this untuned configuration and matrix set.
3. Hu, Lin, Zikatanov, An Adaptive Multigrid Method Based on Path Cover,
   https://doi.org/10.1137/18M1194493; https://arxiv.org/abs/1806.07028.
   Following smooth-error level sets to construct aggregates/rebuild hierarchy is
   already established. Our basis construction differs from path-cover aggregation,
   but a level-set motivation alone is not novel.
4. Two-level preconditioning with diffusion-filtered coarse spaces for
   ill-conditioned finite-element SPD systems, CMwA 222 (2026), 244–267,
   https://doi.org/10.1016/j.camwa.2026.09.024,
   publisher https://www.sciencedirect.com/science/article/abs/pii/S0898122126004153.
   Publisher abstract/highlights/section excerpts were rechecked. The method uses
   diffusion/heat filtering of probes, compression, and balanced correction,
   including a left-preconditioned strategy. Full algorithm and code were not
   obtained; this round is NOT a numerical reproduction of that paper. Direct
   Ritz deflation is a related spectral comparator, not an interchangeable proxy
   for every filtered-space method.
5. Bootstrap AMG, https://doi.org/10.1137/090752973, and Lean Algebraic Multigrid,
   https://doi.org/10.1137/110843563. Publisher abstracts were checked. Bootstrap
   uses test-vector-based weighted least-squares interpolation and eigensolver/
   relaxation improvement; LAMG combines affinity-based aggregation, lean
   interpolation and coarse energy correction. Neither is reproduced here.
   They further limit any broad claim that learning relationships between nodes
   from smooth vectors or constructing sparse interpolation is itself novel.
6. For the separate joint-coordinate pilot, Delaunay-based AMG/preconditioning is
   also not new in itself. Olson (2007), *Algebraic Multigrid Preconditioning of
   High-Order Spectral Elements for Elliptic Problems on a Simplicial Mesh*,
   https://doi.org/10.1137/060663465, author PDF
   https://lukeo.cs.illinois.edu/files/2007_Olson_hoamgtri.pdf, constructs a low-order
   preconditioner using local triangulation of spectral-element interpolation
   nodes. That is physical/reference-element geometry, unlike our two Ritz-value
   coordinates; the distinction does not establish priority for the latter.

The defensible candidate distinction remains the concrete sparse nonlinear map
v_k -> v_k*h_j(v_k), its <=4 basis entries per node for two coordinates, and the
cost/accuracy behavior of a single solve with all setup included. Generic balanced
deflation, low-mode discovery, relative Ritz stopping, and local near-nullspace
enrichment are standard. Keyword searches not finding this exact formula are
not a proof of priority. Comparisons in RESULTS_TABLE.md can support usefulness
of this implementation, not a blanket claim of state of the art.

The geometric POU baseline is deliberately labeled as an elementary enrichment
control. It does not solve local eigenproblems and is not GenEO, nor does it
reproduce alpha-SA's multilevel hierarchy. Direct2/16/32 use the same 96-step
Arnoldi budget, but ranks differ; the lifted space has many more sparse columns.
No equal-rank superiority claim is justified by these tests.
