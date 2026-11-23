"""Multigroup radiation diffusion coupled to a material temperature (H50 test family; not 3T, not SC26_01).

Linearised, backward Euler, cell-centred FV 7-point on m^3 unit cells (symmetry planes Neumann, far faces Marshak-type
Robin for the groups). Unknowns per cell: ng group energies and the material emission variable Theta; symmetric form
(E_g scaled by sqrt(w_g)), so A is a symmetric M-matrix (Stieltjes):
    group g :  (c_r + sigma_g) E_g - div(k_g grad E_g) - sigma_g sqrt(w_g) Theta                         = b_g
    material:  (c_v + sum_g sigma_g w_g) Theta - div(kappa grad Theta) - sum_g sigma_g sqrt(w_g) E_g     = b_T
sigma_g = sigma(x) s_g, s_g log-spaced 1e1 ... 1e-2 over the groups, k_g = 1 / (3 sigma_g + 0.03),
w_g = Planck fractions at fixed group centres. sigma(x) = sbar * (thick ball, x100 inside r < 0.35 m) * lognormal
smooth random field (sd 1 in log, seed). Knobs:
    sbar   opacity scale (optically thin ... thick per cell)
    zeta   material heat capacity c_v = zeta * sbar (x10 inside the ball)
    lam    material conduction kappa = lam * (c_v + sum sigma w) / 6 per cell (0: no conduction)
usage: from mgrd import generate; A, b = generate(m, ng, sbar, zeta, lam, seed)   (blocked [E_1..E_ng, Theta])
"""
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver_challenge"))
from gen3t import diffusion   # noqa: E402

C_R = 1e-2


def planck(ng):
    x = np.geomspace(0.3, 12.0, ng)
    w = x**3 / np.expm1(x) * np.gradient(np.log(x)) * x
    return w / w.sum()


def generate(m, ng, sbar, zeta, lam, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.indices((m, m, m)) / m
    ball = (np.sqrt(x**2 + y**2 + z**2) < 0.35).ravel()
    field = gaussian_filter(rng.standard_normal((m, m, m)), sigma=m / 10, mode="wrap")
    field = (field / field.std()).ravel()
    sigma = sbar * np.where(ball, 100.0, 1.0) * np.exp(field)
    s = np.geomspace(1e1, 1e-2, ng)
    w = planck(ng)
    n = m**3
    far = np.zeros((m, m, m))
    far[:, :, -1] += 1; far[:, -1, :] += 1; far[-1, :, :] += 1
    far = far.ravel()
    cv = zeta * sbar * np.where(ball, 10.0, 1.0)
    blocks = [[None] * (ng + 1) for _ in range(ng + 1)]
    dT = cv.copy()
    for g in range(ng):
        sg = sigma * s[g]
        kg = 1.0 / (3 * sg + 0.03)
        robin = far * 2 * kg / (1 + 4 * kg)                       # Marshak: flux = E / 2 through a half cell
        blocks[g][g] = diffusion(m, kg) + sp.diags(C_R + sg + robin)
        cpl = sp.diags(-sg * np.sqrt(w[g]))
        blocks[g][ng] = cpl
        blocks[ng][g] = cpl
        dT += sg * w[g]
    kappa = lam * dT / 6.0
    blocks[ng][ng] = diffusion(m, kappa) + sp.diags(dT) if lam > 0 else sp.diags(dT)
    A = sp.bmat(blocks).tocsr()
    A.sort_indices()
    old = 1 + 0.5 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    old = old.ravel() * (1 + 0.01 * rng.standard_normal(n))
    b = np.concatenate([C_R * old * np.sqrt(w[g]) for g in range(ng)] + [cv * old])
    return A, b
