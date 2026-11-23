"""H54 exploration: how much of each scalar block is covered by large homogeneous (constant-coefficient) regions?
Cell homogeneous: max/min of its face coefficients <= 1 + rtol. Regions: 6-connected components of homogeneous cells."""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver_challenge"))
import run_real
from bt_elim import schur_ion


def faces(M, m):
    n = m**3
    f = np.full((n, 6), np.nan)
    for k, s in enumerate((1, m, m * m)):
        v = np.zeros(n); v[:n - s] = -M.diagonal(s)
        idx = np.arange(n)
        coord = (idx // s) % m
        plus = coord < m - 1
        f[plus, 2 * k] = v[plus]
        minus = coord > 0
        f[minus, 2 * k + 1] = v[idx[minus] - s]
    return f


def blocks(case):
    A, b = run_real.load(case)
    N = A.shape[0] // 3
    s = [slice(k * N, (k + 1) * N) for k in range(3)]
    d22 = A[s[1], s[1]].diagonal(); a23 = A[s[1], s[2]].diagonal()
    return {"S3": schur_ion(A[s[2], s[2]].tocsr(), a23, a23, d22), "A11": A[s[0], s[0]].tocsr()}, round(N ** (1 / 3))


for case in sys.argv[1:]:
    B, m = blocks(case)
    for name, M in B.items():
        f = faces(M, m)
        ratio = np.nanmax(f, axis=1) / np.nanmin(f, axis=1)
        for rtol in (1e-6, 1e-2, 1.0):
            h = (ratio <= 1 + rtol).reshape(m, m, m)
            lab, n = ndimage.label(h)
            sz = np.sort(np.bincount(lab.ravel())[1:])[::-1]
            big = sz[sz >= max(512, m**3 // 1000)]
            print(f"{case} {name} rtol {rtol:g}: homogeneous {h.mean():.2%}; regions >= threshold: {len(big)} sizes "
                  f"{(big / m**3 * 100).round(2).tolist()[:6]} %; outside big regions {(1 - big.sum() / m**3):.2%}", flush=True)
