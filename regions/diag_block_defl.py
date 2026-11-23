"""H55 diagnosis: per-block GMRES(50) to 1e-10 on SC26_01 blocks (S3, A11) with variants of the block preconditioner."""
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solver_challenge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bt_elim import schur_ion  # noqa: E402
import rsolve  # noqa: E402
case = sys.argv[1] if len(sys.argv) > 1 else "real"
import run_real  # noqa: E402
A, b = run_real.load(case)
N = A.shape[0] // 3; m = round(N ** (1 / 3)); s = [slice(k * N, (k + 1) * N) for k in range(3)]
d22 = A[s[1], s[1]].diagonal(); a23 = A[s[1], s[2]].diagonal(); a32 = A[s[2], s[1]].diagonal()
blocks = {"S3": schur_ion(A[s[2], s[2]].tocsr(), a23, a32, d22), "A11": A[s[0], s[0]].tocsr()}
rhs = {"S3": b[s[2]] - a32 * b[s[1]] / d22, "A11": b[s[0]]}


def make(variant, M):
    if variant == "scalar":
        return rsolve.ScalarFDLGF(M, m)
    if variant == "defl":
        return rsolve.DeflatedFDLGF(M, m)
    if variant == "largest":                       # multi-region code path, but only the largest region in C
        rsolve.MultiFDLGF.ADMIT = "big"; P = rsolve.MultiFDLGF(M, m); rsolve.MultiFDLGF.ADMIT = "open"
        P.regions = [r for r in P.regions if r["largest"]]
        return P
    raise ValueError(variant)


for name, M in blocks.items():
    for variant in sys.argv[2].split(","):
        P = make(variant, M)
        res = []
        x, info = spla.gmres(M, rhs[name], rtol=1e-10, restart=50, maxiter=4, M=spla.LinearOperator(M.shape, P, dtype=float),
                             callback=lambda r: res.append(r), callback_type="pr_norm")
        true = np.linalg.norm(rhs[name] - M @ x) / np.linalg.norm(rhs[name])
        hit = lambda t: next((i + 1 for i, r in enumerate(res) if r < t), None)  # noqa: E731
        print(f"{case} {name} {variant:10s}: F {len(P.F) / m**3:.2%}  its to 1e-4 {hit(1e-4)}, 1e-8 {hit(1e-8)}, 1e-10 {hit(1e-10)}; true {true:.1e}", flush=True)
