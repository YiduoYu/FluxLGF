# FluxLGF

FluxLGF develops preconditioned solvers for variable-coefficient elliptic equations. The methods combine flux discretizations with structured background inverses and local corrections. The research goal is to reduce total solution time relative to hypre BoomerAMG on reproducible benchmarks at matched residual and accuracy tolerances.

- [FD-LGF method](FD-LGF.md) explains the separable background and fictitious-domain preconditioner.
- [Benchmark protocol](protocol.md) documents the original two-dimensional comparison. [BoomerAMG field-line experiments](boomer_flux_adapter/README.md) cover the anisotropic prototype and its tested cases.

## Run the core tests

Clone this repository with `--recurse-submodules`, or initialize the pinned LGF-CNN dependency after cloning.

```sh
git submodule update --init
python -m pip install numpy scipy pytest
PYTHONPATH="$PWD:$PWD/dependencies/lgf-cnn" python -m pytest -q test_fdlgf.py test_baselines.py three_d/test_3d.py
```

The PETSc and hypre benchmarks also need a `petsc4py` installation with hypre support. Some historical experiments require their original external input datasets or locally built native libraries.
