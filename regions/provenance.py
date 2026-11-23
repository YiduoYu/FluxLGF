"""What a result needs in order to be re-checked later (audit 2026-11-05, items 2 and 9): the effective solver setup
(ksp.view + PETSc options database), library versions, input hashes. Writes STEM.view.txt and STEM.meta.json."""
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from petsc4py import PETSc


def sha256(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(a.data)
    return h.hexdigest()


def feature_stats(P):
    """Per-block feature size and factor size of a bt-fd preconditioner: core (before the ring) and dilated fractions
    are different quantities (audit item 8), so both are kept."""
    out = {}
    for name, s in (("S3", P.p3), ("A11", P.p1)):
        pc = s.ksp.getPC()
        out[name] = dict(core_fraction=s.core_fraction, dilated_fraction=len(s.F) / P.N, factor_type=pc.getType(),
                         factor_nnz=pc.getFactorMatrix().getInfo()["nz_used"])
    return out


def save(stem, ksp, A, b, **extra):
    viewer = PETSc.Viewer().createASCII(f"{stem}.view.txt")
    ksp.view(viewer)
    viewer.destroy()
    meta = dict(petsc=".".join(map(str, PETSc.Sys.getVersion())), numpy=np.__version__, scipy=scipy.__version__,
                options=PETSc.Options().getAll(), A_sha256=sha256(A.indptr, A.indices, A.data), b_sha256=sha256(b), **extra)
    Path(f"{stem}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
