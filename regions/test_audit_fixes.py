"""Regression tests for the code defects found by the 2026-11-05 audit. Small matrices only (seconds).
run: OMP_NUM_THREADS=1 /opt/anaconda3/envs/hypre_env/bin/python test_audit_fixes.py"""
import itertools
import json
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from petsc4py import PETSc

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "solver_challenge")]
from gen3t import generate  # noqa: E402
from ir_fp128 import DDResidual  # noqa: E402
from rsolve import FDMBox  # noqa: E402
from run_ir import inner, solve_ir  # noqa: E402
from run_real import run  # noqa: E402
from lock_manifest import reachable  # noqa: E402


def box_operator(sl, m, kappa, r, beta, dirichlet):
    """Dense operator on the box from its faces, cell by cell (independent of the 1D matrices FDMBox diagonalizes)."""
    cells = list(itertools.product(*(range(s.start, s.stop) for s in sl)))
    index = {c: i for i, c in enumerate(cells)}
    M = np.zeros((len(cells), len(cells)))
    for c, i in index.items():
        M[i, i] += r
        for ax, step in itertools.product(range(3), (-1, 1)):
            nb = c[:ax] + (c[ax] + step,) + c[ax + 1:]
            if nb in index:
                M[i, i] += kappa; M[i, index[nb]] -= kappa
            elif nb[ax] == m:                                    # far face of the cube: Robin
                M[i, i] += beta
            elif nb[ax] >= 0 and dirichlet:                      # box face inside the cube
                M[i, i] += kappa
    return cells, M


BOXES = [
    ((2, 3), (2, 3), (2, 3)),      # interior point
    ((0, 1), (0, 1), (0, 1)),      # corner on the Neumann faces
    ((4, 5), (4, 5), (4, 5)),      # corner on the Robin faces
    ((2, 3), (1, 4), (1, 3)),      # one-cell-thick slab
    ((1, 4), (2, 3), (4, 5)),      # line on a Robin face
    ((0, 2), (3, 5), (1, 4)),      # no thin direction: the n >= 2 path must not have moved
]


def check_fdmbox(box, dirichlet):
    m, kappa, r, beta = 5, 10.0, 1.0, 0.7
    mask = np.zeros((m, m, m), bool)
    sl = tuple(slice(*b) for b in box)
    mask[sl] = True
    G = FDMBox(mask, kappa, r, beta, m, dirichlet=dirichlet)
    cells, M = box_operator(sl, m, kappa, r, beta, dirichlet)
    rhs = np.random.default_rng(0).standard_normal(len(cells))
    f = np.zeros((m, m, m))
    for c, v in zip(cells, rhs):
        f[c] = v
    got = G(f.ravel()).reshape(m, m, m)
    np.testing.assert_allclose([got[c] for c in cells], np.linalg.solve(M, rhs), rtol=1e-12)


def check_fdmbox_interior_point_is_one_over_61():
    mask = np.zeros((5, 5, 5), bool); mask[2, 2, 2] = True
    G = FDMBox(mask, 10.0, 1.0, 0.0, 5, dirichlet=True)
    np.testing.assert_allclose(G(mask.ravel().astype(float))[62], 1 / 61, rtol=1e-14)


def view_text(ksp, path):
    viewer = PETSc.Viewer().createASCII(str(path))
    ksp.view(viewer)
    viewer.destroy()
    return Path(path).read_text()


def check_amg_preset_survives_and_default_applies(tmp_path):
    A, _ = generate(8, 2e-11, 2e-4)
    key = "ir_amg_pc_hypre_boomeramg_strong_threshold"
    opts = PETSc.Options()
    assert "Threshold for strong coupling 0.5" in view_text(inner(A, 1e-6, "amg-3d")[0], tmp_path / "a.txt")
    opts[key] = 0.7
    assert "Threshold for strong coupling 0.7" in view_text(inner(A, 1e-6, "amg-3d")[0], tmp_path / "b.txt")
    opts.delValue(key)


def check_unknown_method_is_an_error():
    A, _ = generate(4, 2e-11, 2e-4)
    for call in (lambda: inner(A, 1e-6, "bt-fdx"), lambda: run(A, np.ones(A.shape[0]), "bt-fdx", 1e-4)):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("unknown method accepted")


def check_ir_saves_a_checkable_result(kind, tmp_path):
    A, b = generate(12, 2e-11, 2e-4)
    rows = solve_ir(A, b, DDResidual(A, b), kind, 1e-6, str(tmp_path / "r"), prep_s=1.5)
    row = rows[1]
    assert row["target"] == 1e-8 and row["steps"] > 0 and row["prep_s"] == 1.5
    np.testing.assert_allclose(row["total_s"], row["setup_s"] + row["solve_s"] + row["resid_s"], rtol=0.05)
    assert all(int(v) > 0 for v in row["inner_reasons"].split())
    x = np.load(tmp_path / f"r_{kind}.npz")
    # Independent recheck of the saved solution: exact rational residual of the stored A, b.
    rat = []
    for i in range(len(b)):
        lo, hi = A.indptr[i], A.indptr[i + 1]
        ax = sum((Fraction(float(v)) * (Fraction(float(x["xh"][j])) + Fraction(float(x["xl"][j])))
                  for j, v in zip(A.indices[lo:hi], A.data[lo:hi])), Fraction(0))
        rat.append(float(Fraction(float(b[i])) - ax))
    exact = np.linalg.norm(rat) / np.linalg.norm(b)
    assert exact < 1e-8
    np.testing.assert_allclose(exact, row["final_rel"], rtol=1e-6)
    print(f"{kind}: steps {row['steps']} inner {row['inner_its']} dd {row['final_rel']:.6e} exact {exact:.6e}")
    meta = json.loads((tmp_path / f"r_{kind}.meta.json").read_text())
    assert meta["A_sha256"] and meta["b_sha256"] and meta["petsc"]
    if kind == "amg-3d":
        assert meta["features"] is None
    else:
        S3 = meta["features"]["S3"]
        assert 0 < S3["core_fraction"] <= S3["dilated_fraction"] <= 1 and S3["factor_nnz"] > 0
    ksp_pc = inner(A, 1e-6, kind)[0].getPC()
    if kind == "bt-fd-chol":
        assert ksp_pc.getPythonContext().p3.ksp.getPC().getType() == "cholesky"
    elif kind == "bt-fd":
        assert ksp_pc.getPythonContext().p3.ksp.getPC().getType() == "lu"


def check_lock_manifest_finds_indirect_dependencies():
    names = {p.name for p in reachable(HERE / "seq_isolated.py")}
    # gen3t via gen_gen, fdlgf_real via rsolve, provenance + bt_elim from inside a function
    assert {"seq_isolated.py", "bt_elim.py", "btfd.py", "rsolve.py", "fdlgf_real.py", "gen_gen.py", "gen3t.py", "provenance.py"} <= names, names
    assert {"ir_fp128.py", "run_ir.py"} <= {p.name for p in reachable(HERE / "run_ir.py")}


if __name__ == "__main__":
    check_lock_manifest_finds_indirect_dependencies()
    print("lock manifest ok")
    for box, dirichlet in itertools.product(BOXES, (False, True)):
        check_fdmbox(box, dirichlet)
    check_fdmbox_interior_point_is_one_over_61()
    print("FDMBox ok")
    check_unknown_method_is_an_error()
    print("unknown method ok")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        check_amg_preset_survives_and_default_applies(tmp)
        print("AMG override ok")
        for kind in ("amg-3d", "bt-fd", "bt-fd-chol"):
            check_ir_saves_a_checkable_result(kind, tmp)
    print("all ok")
