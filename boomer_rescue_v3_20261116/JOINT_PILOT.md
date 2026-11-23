# Separate exploratory joint-coordinate pilot

Declared during the round-3 timing campaign, before running this prototype.
It is motivated by the algebraic orientation issue in THEORY.md, not by selecting
a successful rotation experiment. It does not change LOCK.json or any timed method.

Construct a 2D point cloud from two coarse Ritz coordinates. Apply one common
normalization (not separate axis normalization), include the origin and every
convex hull vertex, then farthest-point sample to approximately the original lifted
dimension. Delaunay triangulation and barycentric interpolation of the fine-grid
coordinates give at most three entries per row. Drop the origin basis to ensure
the coordinate origin maps to zero; other vertex coordinates then reproduce both
input coordinates. Bilinear fine-grid prolongation of coarse values with zero
boundary lies within the coarse-point-plus-origin convex hull.

If hull vertex count exceeds the target, keep the hull and report the excess
dimension. Do not hide out-of-hull coordinates with nearest-center assignment.
Selecting all hull vertices can become expensive and is a limitation. Tied farthest
points or cocircular points may cause finite-precision orientation differences;
do not claim exact numerical rotation invariance without checking it.

When the coarse grid nodes are included in the fine grid, each non-origin center
is an actual fine-grid coordinate sample. Nodal barycentric interpolation then
gives an identity submatrix and full column rank in exact arithmetic. This does
not guarantee good conditioning. Nearly one-dimensional coordinate clouds can
produce skinny triangles; record normalized triangle area/maximum edge squared.
Such triangles can amplify coordinate errors even if the construction is invariant
under rotation of a fixed, exact two-dimensional subspace.

Use all four existing 255² diagnostic families and the same coarse coordinates,
probe vectors, RHS and four angles (0,22.5,45,90 degrees). Record iterations,
independent DD residuals, energy projection errors, coarse conditioning, dimensions,
sparsity and selected-center identity. These are mechanistic exploratory results,
not independently validated end-to-end speedups. No timing claim from reused B/W.

This combines standard sampling, triangulation and interpolation ideas. A better
result would justify a new validation round; it would not establish literature
priority or superiority to all adaptive aggregation methods.
