"""Checks of the hypre interface in compare_structured.py: run to high accuracy as stand-alone solvers, each must solve the right
system, with a non-cubic grid and different coefficients per axis, so that a wrong index order or stencil cannot pass.
usage: test_interface.py
"""
import numpy as np

import compare_structured as cs
from try_one import lap7


def field(shape, seed):
    """One matrix with non-uniform, unsymmetric-free faces and a positive reaction."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    faces = [3.0 * np.exp(rng.standard_normal((nz - 1, ny, nx))), np.exp(rng.standard_normal((nz, ny - 1, nx))), 0.3 * np.exp(rng.standard_normal((nz, ny, nx - 1)))]
    return lap7(faces, 0.1 + rng.random(shape), shape).tocsr()


def check(P, A, b, what):
    x = P(b)
    rel = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    assert rel < 1e-7, (what, rel)
    print(f"{what:32s} rel {rel:.1e}")


def main():
    shape = (9, 14, 21)                                 # (nz, ny, nx), all different
    N = int(np.prod(shape))
    A = field(shape, 0)
    b = np.random.default_rng(1).standard_normal(N)
    check(cs.Mono(A, shape, maxit=200, tol=1e-9), A, b, "PFMG, one field")
    check(cs.Mono(A, shape, smg=True, maxit=200, tol=1e-9), A, b, "SMG, one field")
    for rap in (0, 1):
        for relax in (1, 2, 3):
            check(cs.Mono(A, shape, relax=relax, rap=rap, maxit=500, tol=1e-9), A, b, f"PFMG relax {relax} rap {rap}")
    print("all passed")


if __name__ == "__main__":
    main()
