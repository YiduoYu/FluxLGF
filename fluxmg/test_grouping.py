"""Checks of Reduction in flux_mg.py.
1. Without the probe it equals multifield/amfr.py exactly: same structure, same preconditioner output, bit for bit.
2. With the probe, the 3T systems keep their groups and a star-coupled multigroup system is joined.
3. The rule is invariant to a scaling of the rows and columns of a field (units).
usage: test_grouping.py
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "generalize"), str(ROOT / "regions"), str(ROOT / "solver_challenge"), str(ROOT / "multifield")]
import amfr  # noqa: E402
import flux_mg  # noqa: E402
from try_one import load  # noqa: E402


def core_solver(dims, N):
    def solver(M):
        M = M.tocsr()
        if M.nnz == M.shape[0] and np.count_nonzero(M.diagonal()) == M.shape[0]:
            d = M.diagonal().copy()
            return lambda r: r / d
        return flux_mg.Core(M, dims, M.shape[0] // N)
    return solver


def test_equals_amfr():
    flux_mg.PROBE_TAU = np.inf
    for case in ("g3t:16:2e-11:2e-4", "g3t:16:1:1e-2", "g3t:16:2e-11:1", "gen:32:3:2:1:0:2e-2:1", "mg:16:4:1:1:0:7", "mg:16:4:1:1:1:7", "mg:16:8:1e2:1e-2:1:7", "mg:16:8:1e-2:1e2:0:7"):
        A, _, dims, F = load(case)
        solver = core_solver(dims, A.shape[0] // F)
        keep = amfr.vcycle
        amfr.vcycle = solver
        try:
            old = amfr.AMFR(A, F, verbose=False)
        finally:
            amfr.vcycle = keep
        new = flux_mg.Reduction(A, dims, F, solver)
        r = np.random.default_rng(1).standard_normal(A.shape[0])
        assert old.structure() == new.structure(), (case, old.structure(), new.structure())
        assert np.array_equal(old(r), new(r)), case
        print("equal to amfr.AMFR:", case, new.structure())


def test_rule():
    flux_mg.PROBE_TAU = 0.7
    for case in ("g3t:32:2e-11:2e-4", "gen:32:3:2:1:0:2e-2:1", "gen:32:10:8:0.5:2:2e-2:3"):
        A, _, dims, F = load(case)
        P = flux_mg.Reduction(A, dims, F, core_solver(dims, A.shape[0] // F))
        assert len(P.groups) == 2 and P.probe < 0.01, (case, P.structure(), P.probe)
        print("3T keeps its groups:", case, P.structure(), f"probe {P.probe:.1e}")
    A, _, dims, F = load("mg:32:8:1:1e-2:0:7")
    P = flux_mg.Reduction(A, dims, F, core_solver(dims, A.shape[0] // F))
    assert len(P.groups) == 1 and P.probe > 0.9, (P.structure(), P.probe)
    print("star coupling joined:", P.structure(), f"probe {P.probe:.3f}")


def test_units():
    """Field scaling 1e-2 ... 1e2. The probe is invariant for a given solve order. The order itself is not (it comes from the
    clipped gains of H50), so the probe value can move a little between units; a clear decision must not."""
    flux_mg.PROBE_TAU = 0.7
    probe = flux_mg.probe_contraction
    for case, joined in (("mg:16:4:1:0.02:0:7", True), ("mg:16:8:1e-2:1e2:0:7", False)):
        A, _, dims, F = load(case)
        N = A.shape[0] // F
        S = sp.diags(np.repeat(np.logspace(-2, 2, F), N))
        seen = []

        def spy(B, order, dims):
            seen.append((B, order))
            return probe(B, order, dims)
        flux_mg.probe_contraction = spy
        a = flux_mg.Reduction(A, dims, F, lambda M: (lambda r: r))
        b = flux_mg.Reduction((S @ A @ S).tocsr(), dims, F, lambda M: (lambda r: r))
        flux_mg.probe_contraction = probe
        same = [probe(seen[0][0], seen[0][1], dims), probe(seen[1][0], seen[0][1], dims)]
        assert abs(same[0] - same[1]) < 1e-6, (case, same)
        assert (len(a.groups) == 1) == joined and (len(b.groups) == 1) == joined, (case, a.structure(), b.structure())
        print("units:", case, a.structure(), b.structure(), f"same order {same[0]:.6f} {same[1]:.6f}; probes {a.probe:.3f} {b.probe:.3f}")


if __name__ == "__main__":
    test_equals_amfr()
    test_rule()
    test_units()
    print("all passed")
