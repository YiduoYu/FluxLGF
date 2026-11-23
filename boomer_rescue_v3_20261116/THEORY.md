# What the proposed space actually adds

This is a derivation for the implemented construction, not a new convergence theorem
or a priority claim. Assume A and B symmetric positive definite and full column rank
of Z; finite precision and dropping need separate numerical checks.

## Exact algebraic properties

For one normalized coordinate v, let h_j be the continuous piecewise-linear nodal
hats on its value interval, including all endpoint hats. Let Z_j=v*h_j(v), with
pointwise multiplication. The hats obey partition of unity and linear reproduction:

    sum_j h_j(t)=1;   sum_j t_j h_j(t)=t.

Consequently Z contains BOTH v and the pointwise square v^2:

    sum_j Z_j=v;      sum_j t_j Z_j=v^2.

More generally it represents v*s(v), where s is a linear spline in the VALUE
coordinate. It does not merely reweight the same two-dimensional linear space.
For two coordinates the construction concatenates the spaces; it does not form
a tensor product. Thus it has no general representation guarantee for v1*v2,
or for arbitrary functions of (v1,v2). With L intervals and two coordinates,
dimension <=2(L+1), at most four basis entries per row before numerical dropping.

The actual implementation drops entries with absolute value <=1e-14 and removes
empty/very small columns, so the identities above are exact for the ideal map;
their floating-point implementation is checked numerically on small examples.
Column normalization does not change the mathematical span.

Four entries per row bounds storage of Z, not the total solver complexity. A Z
can have more entries, Z^T A Z can couple many columns, and its sparse factorization
can fill in. The implemented bin rule increases coarse dimension with n; no
linear-complexity or grid-independent convergence theorem follows from row sparsity.

For P_Z=Z(Z^T A Z)^(-1)Z^T A and any w, the energy projection minimizes
||w-Zc||_A. Because span{v1,v2} is included in the lifted space,

    ||(I-P_lift)w||_A <= ||(I-P_direct2)w||_A.

This says nothing by itself about total solve time, comparison with direct16/32,
or grid-independent convergence. The geometric POU comparator also contains
the two coordinates, so this elementary inclusion does not distinguish the two.

For Q=Z(Z^T A Z)^(-1)Z^T and
M=Q+(I-QA)B(I-AQ), symmetry follows by transpose. Positive definiteness follows
from r^T M r=r^T Q r + w^T B w with w=(I-AQ)r. Both terms nonnegative;
if both vanish, Z^T r=0 and thus w=r=0. Also M A Z=Z in exact arithmetic.
These are standard balanced correction properties, not novel contributions.

## Coordinate accuracy: why a Ritz threshold alone is insufficient

Freeze the knot interval and spacing Delta; take |t|<=M0 and define the row map
F(t)=[t*h_j(t)]_j. Away from knots there are only two active hats. By the triangle
inequality and their slopes +/-1/Delta,

    ||F'(t)||_2 <= 1 + sqrt(2)*M0/Delta.

The continuous piecewise-smooth map therefore has this global Lipschitz bound:

    ||Z(v)-Z(w)||_F <= (1+sqrt(2)*M0/Delta)*||v-w||_2.

For multiple coordinates sum the squared bounds. This is deliberately a loose
bound, but exposes amplification as the value bins narrow. For SPD A, a further
energy-weighted bound is ||A^(1/2) E||_2 <= sqrt(lambda_max(A))*||E||_F.
This can be pessimistic on high-contrast matrices; it is not a useful uniform
condition-number theorem. In the actual implementation knots and normalization
depend on the coordinate, so their perturbations need extra terms: the frozen-
knot bound cannot be applied silently to that full algorithm.

If X=A^(1/2)Z has smallest singular value sigma>0 and perturbation D satisfies
||D||_2<sigma, one elementary directional subspace bound is

    ||(I-P_X)P_(X+D)||_2 <= ||D||_2/(sigma-||D||_2).

Indeed, ||(X+D)c|| >= (sigma-||D||)||c|| and (I-P_X)(X+D)c=(I-P_X)Dc.
The bound depends on the chosen basis scaling; normalize columns consistently.
It identifies near-dependence as a second issue beyond coordinate residual.

Small Ritz residuals only certify an approximate eigen-equation. Converting
residual to eigenvector error needs spectral separation, while converting that
error to the lifted-space error also needs value-bin and basis conditioning
control. Accordingly the round-2 eta<=0.1 rule remains empirical, not a proof.

## A concrete vulnerability: orientation of a clustered eigenspace

Direct deflation is invariant under any nonsingular change of coordinates V->VR.
The additive nonlinear space sum_k v_k*s_k(v_k) generally is not invariant under
a rotation mixing v1 and v2. Sign changes or permutation with symmetric handling
are benign, but an arbitrary rotation changes the available nonlinear functions.
When slow eigenvalues cluster, accurately identifying the two-dimensional subspace
may still fail to identify a stable basis orientation. This is a testable failure
mode, and a reason to investigate joint-coordinate constructions if observed.
Such constructions may cost more than four entries per node or require more
complicated interpolation, so they are not automatically better.

A second structural restriction is embedding collisions: if two physical nodes
have the same (v1,v2) values, they have identical rows of the ideal lifted basis.
No choice of coarse coefficients can distinguish those nodes. In particular,
all coordinates zero gives a zero basis row. A joint-coordinate construction can
remove the additive-separability restriction but cannot resolve such collisions
without additional coordinates or spatial information. More bins alone do not
remove either limitation.

The diagnostic script checks weighted/direct projection and rotation sensitivity
on small problems after the timing campaign. It is mechanistic evidence, not
additional independent performance validation or a reason to tune locked results.
