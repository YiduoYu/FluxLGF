"""H56 diagnosis: Ritz values of A P (right-preconditioned, 60 Arnoldi steps from b) for bt-fd / bt-fdd: how many
eigenvalues are far from 1, and does that number grow with the number of inclusions?"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real
from btfd import BTFD
k = 60
for case in sys.argv[2:]:
    A, b = run_real.load(case)
    for mode in sys.argv[1].split(","):
        P = BTFD(A, multi="defl" if mode == "fdd" else False)
        V = np.zeros((A.shape[0], k + 1)); H = np.zeros((k + 1, k))
        V[:, 0] = b / np.linalg.norm(b)
        for j in range(k):
            w = A @ P(V[:, j])
            for _ in range(2):                                     # re-orthogonalised Gram-Schmidt
                h = V[:, :j + 1].T @ w; w -= V[:, :j + 1] @ h; H[:j + 1, j] += h
            H[j + 1, j] = np.linalg.norm(w); V[:, j + 1] = w / H[j + 1, j]
        ev = np.linalg.eigvals(H[:k, :k])
        d = np.abs(ev - 1)
        print(f"{case} {mode}: Ritz |lam-1| > 0.5: {np.sum(d > 0.5)}, > 0.9: {np.sum(d > 0.9)}; smallest |lam| "
              + " ".join(f"{x:.1e}" for x in np.sort(np.abs(ev))[:6]), flush=True)
