"""H56 exploration: frequency-calibrated background (homogenisation-aware Green's function).
Perforated/inclusion-laden media make smooth modes see an effective conductivity below the matrix value, so the
constant-coefficient background G is too stiff at low frequency (Ritz values of A P ~ kappa_eff / kappa). FDM can invert
any function of the eigenvalue, so measure the response of the block preconditioner p on a few background eigenmodes
v_j (DCT-like, diagonal wavenumbers (j, j, j)):  rho_j = v_j^T M p(v_j) / v_j^T v_j, and rescale the symbol of G by
rho(den) (log-log interpolation in the symbol value). Cost: a handful of applications of p at setup.
Related: optimal circulant / cosine preconditioners (T. Chan 1988), reference-medium choice in FFT homogenisation
(Moulinec-Suquet 1998)."""
import numpy as np


def probe_modes(G, js):
    V, m = G.V, G.m
    for j in js:
        v = np.einsum("z,y,x->zyx", V[:, j], V[:, j], V[:, j]).ravel()
        yield np.log(G.den[j, j, j]), v


def calibrate(p, M, G, js=None, rounds=1):
    """Rescale G.den in place so that p (whose background inverse is G) responds ~1 on the probed modes."""
    m = G.m
    if js is None:
        js = sorted({0, 1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, m - 1} & set(range(m)))
    for _ in range(rounds):
        xs, ys = [], []
        for x, v in probe_modes(G, js):
            rho = float(v @ (M @ p(v)) / (v @ v))
            xs.append(x); ys.append(np.log(np.clip(rho, 0.05, 20.0)))
        order = np.argsort(xs)
        xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
        G.den = G.den * np.exp(np.interp(np.log(G.den), xs, ys))
    return np.exp(ys), xs
