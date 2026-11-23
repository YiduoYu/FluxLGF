"""z = 0 plane (through the corner ball) maps of the real SolverChallenge26_01 matrix: log10 of the x-face coefficient of T1, T3, the diagonal of
T2, and the T1–T2 coupling ratio a12/a21, with the FD-LGF feature region F (white contour).   usage: python fig_structure.py A.npz OUT.png"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_dilation

sys.path.insert(0, "..")
import fdlgf_real as F

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
A = sp.load_npz(sys.argv[1]).tocsr()
N, m = A.shape[0] // 3, 128
d = A.diagonal()
face = lambda k: -np.append(A.diagonal(1), 0)[k * N:(k + 1) * N]          # x-face coefficient (0 on the last x-face)
cube = lambda v: v.reshape(m, m, m)[0]                          # z = 0 plane, rows y, cols x
a12, a21 = A.diagonal(N)[:N], A.diagonal(-N)[:N]
bg, e12 = F.fit_background(A, m)                                # feature mask exactly as in FDLGFReal
mats = [F.background_matrix(m, *p)[0] for p in bg]
Abg = sp.bmat([[mats[0], -e12 * sp.identity(N), None], [None, mats[1], None], [None, None, mats[2]]]).tocsr()
bad = np.asarray(abs(A - Abg).max(axis=1).todense()).ravel() > 1e-6 * abs(A).max(axis=1).toarray().ravel()
mask = binary_dilation(bad.reshape(3, N).any(axis=0).reshape(m, m, m), iterations=4)[0]
maps = [("T1 的 x 面系数", face(0)), ("T3 的 x 面系数", face(2)), ("T2 的对角元", d[N:2 * N]),
        ("耦合比 $a_{12}/a_{21}$", a12 / a21)]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, (title, v) in zip(axes, maps):
    s = cube(v)[:, :-1] if "面" in title else cube(v)
    im = ax.imshow(np.log10(np.maximum(np.abs(s), 1e-300)), origin="lower", cmap="viridis")
    ax.contour(mask[:, :s.shape[1]], levels=[0.5], colors="w", linewidths=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("log$_{10}$", fontsize=8)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=170)
for title, v in maps:
    print(title, f"{np.abs(v[v != 0]).min():.2e} .. {np.abs(v).max():.2e}")
