"""Exploratory: do the BoomerAMG slow modes lie in the low eigenspace of A itself?  For each slow mode u_i (A-normalised)
report ||Pi_m u_i||_A^2, Pi_m = A-orthogonal projector onto the m lowest eigenvectors of A.  usage: python overlap.py n chi"""
import sys
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, "../../anisotropic")
from cases_aniso import make_case  # noqa: E402

n, chi = int(sys.argv[1]), float(sys.argv[2])
A = make_case("circ", n, chi).A.tocsc()
U = np.load(f"results/slowmodes_n{n}_chi{chi:g}.npy").astype(float)[:, :32]
t0 = perf_counter(); mu, E = eigsh(A, k=160, sigma=0, which="LM"); print(f"eigsh {perf_counter()-t0:.1f}s, mu[:5] {mu[:5]}, mu[-1] {mu[-1]:.3e}")
o = np.argsort(mu); mu, E = mu[o], E[:, o]
AU = A @ U; nrm = np.einsum("ij,ij->j", U, AU)
for m in (8, 16, 32, 64, 160):
    C = E[:, :m].T @ AU                                   # E is orthonormal eigvecs: ||Pi u||_A^2 = sum mu_j (e_j^T u)^2
    cap = (C ** 2 / mu[:m, None]).sum(0) / nrm
    print(f"m={m:3d} captured A-energy of slow modes 1..32: min {cap[:16].min():.3f} median16 {np.median(cap[:16]):.3f} "
          f"| modes 17-32 median {np.median(cap[16:]):.3f}")
