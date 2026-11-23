import numpy as np
import scipy.sparse as sp

from fdlgf3t import ConstantGreen, FDLGF3T, block_scaling, cell_blocks
from surrogate3t import build


def test_surrogate_sparsity_matches_challenge_formula():
    for m in (8, 16):
        A, _ = build(m)
        N = m**3
        assert A.shape == (3 * N, 3 * N)
        assert A.nnz == 3 * (7 * N - 6 * m * m) + 4 * N          # 52,133,888 for m = 128
        assert abs(A - A.T).max() == 0


def test_constant_green_is_exact_inverse():
    m, K, r = 8, 2.5, 0.3
    A, _ = build(m)
    e = np.ones(m)
    t = sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1]).tolil()
    t[0, 0] = t[-1, -1] = 3                                        # Dirichlet at faces, cell-centred
    I = sp.identity(m)
    L = (sp.kron(sp.kron(t, I), I) + sp.kron(sp.kron(I, t), I) + sp.kron(sp.kron(I, I), t)) * m**2
    u = np.random.default_rng(0).standard_normal(m**3)
    assert np.abs(ConstantGreen(K, r, m)((K * L + r * sp.identity(m**3)) @ u) - u).max() < 1e-10


def test_block_scaling_gives_unit_cell_blocks():
    m = 8
    A, _ = build(m)
    S = block_scaling(A, m**3)
    B = cell_blocks((S @ A @ S.T).tocsr(), m**3)
    assert np.abs(B - np.eye(3)).max() < 1e-10


def test_preconditioner_symmetric_positive():
    m = 16
    A, b = build(m)
    P = FDLGF3T(A, m, box=4, coarse="ml", lu_size=500, omega=0.6, cycles=2)
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((2, A.shape[0]))
    assert abs(x @ P(y) - y @ P(x)) < 1e-9 * abs(x @ P(y))
    assert all(v @ P(v) > 0 for v in rng.standard_normal((5, A.shape[0])))


def test_vcycle_symmetric_positive_and_lgf_free_variant():
    m = 16
    A, b = build(m)
    rng = np.random.default_rng(2)
    x, y = rng.standard_normal((2, A.shape[0]))
    for kw in (dict(form="vcycle", box=2, omega=0.8, nu=1, cycles=2, green=False),
               dict(box=2, omega=0.5, cycles=1, green=False)):
        P = FDLGF3T(A, m, coarse="ml", lu_size=500, **kw)
        assert abs(x @ P(y) - y @ P(x)) < 1e-9 * abs(x @ P(y))
        assert all(v @ P(v) > 0 for v in rng.standard_normal((5, A.shape[0])))
