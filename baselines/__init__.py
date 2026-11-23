"""Baselines for comparison with 方法二: traditional LGF, GMG, hypre BoomerAMG (all through axy_lgf's common PCG loop).
BoomerAMG needs petsc4py (/opt/anaconda3/envs/hypre_env); it is imported lazily so LGF/GMG work without it."""

from .gmg import GeometricVcycle, solve_gmg_pcg
from .lgf import DirichletLGF, solve_lgf_pcg


def __getattr__(name):
    if name in ("BoomerAMG", "solve_boomeramg_pcg"):
        from . import boomeramg
        return getattr(boomeramg, name)
    raise AttributeError(name)
