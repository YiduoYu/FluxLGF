"""Radial profiles (distance from the origin corner, in cells) of the real SolverChallenge26_01 coefficients:
x-face diffusion coefficient of T1 (radiation) and T3 (ion), diagonal of T2 (electron), and the couplings.
usage: python fig_radial.py A.npz OUT.png"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
A = sp.load_npz(sys.argv[1]).tocsr()
N, m = A.shape[0] // 3, 128
z, y, x = np.indices((m, m, m)).reshape(3, -1)
rad = np.sqrt(x**2 + y**2 + z**2)
inner = x < m - 1                                               # cells that have an x+ face
face = -np.append(A.diagonal(1), 0)
d = A.diagonal()
series = [("T1 辐射：x 面扩散系数", face[:N], inner), ("T3 离子：x 面扩散系数", face[2 * N:], inner),
          ("T2 电子：对角元", d[N:2 * N], None), ("耦合 $-a_{12}$（T1 行中的 T2）", -A.diagonal(N)[:N], None),
          ("耦合 $-a_{21}$（T2 行中的 T1）", -A.diagonal(-N)[:N], None), ("耦合 $-a_{23}=-a_{32}$（电子-离子）", -A.diagonal(N)[N:2 * N], None)]
bins = np.arange(0, 100)
fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True)
for ax, (title, v, mask) in zip(axes.ravel(), series):
    keep = (mask if mask is not None else np.ones(N, bool)) & (v > 0)
    idx = np.digitize(rad[keep], bins)
    vv = np.log10(v[keep])
    lo = [np.min(vv[idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(bins))]
    md = [np.median(vv[idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(bins))]
    hi = [np.max(vv[idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(bins))]
    ax.fill_between(bins[:-1], lo, hi, color="#3b6ea5", alpha=0.25, linewidth=0, label="最小–最大")
    ax.plot(bins[:-1], md, color="#3b6ea5", linewidth=2, label="中位数")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    print(title, "range log10", np.nanmin(lo), np.nanmax(hi))
for ax in axes[1]:
    ax.set_xlabel("到原点角的距离（格）")
for ax in axes[:, 0]:
    ax.set_ylabel("log$_{10}$")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=160)
