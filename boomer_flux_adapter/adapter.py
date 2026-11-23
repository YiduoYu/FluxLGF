"""Optional closed-flux-region correction for a supplied serial SPD Boomer cycle."""
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage, sparse
from scipy.interpolate import BSpline


def _profiles(coordinate, columns, degree):
    """Use at most columns free coefficients, with a zero trace at coordinate=0."""
    while columns:
        order = min(degree, columns)
        breaks = np.linspace(0., 1., columns - order + 2)
        knots = np.r_[np.zeros(order), breaks, np.ones(order)]
        bank = sparse.csr_matrix(BSpline.design_matrix(coordinate, knots, order))[:, 1:]
        bank.eliminate_zeros()
        # Plateaus and coarsely sampled profiles can give dependent columns.
        gram = (bank.T @ bank).toarray()
        eigenvalues = np.linalg.eigvalsh(gram)
        if eigenvalues[0] > 1e-10 * eigenvalues[-1]:
            return bank
        columns //= 2
    return sparse.csr_matrix((len(coordinate), 0))


def flux_basis(psi, budget=96, degree=3, separate=True):
    """psi includes the Dirichlet boundary. Interior DOFs are in C order (y,x).

    Use connected components strictly outside the boundary flux range. Parts of
    the domain with open contours get no correction. A finite budget prioritizes
    larger components. Same-sign pooling is available for the ablation.
    """
    psi = np.asarray(psi, dtype=float)
    if psi.ndim != 2 or min(psi.shape) < 3 or not np.isfinite(psi).all():
        raise ValueError("expected a finite 2D nodal flux array including its boundary")
    if not isinstance(budget, (int, np.integer)) or budget < 1 or degree not in (1, 3):
        raise ValueError("expected a positive integer budget and degree 1 or 3")
    boundary = np.r_[psi[0], psi[-1], psi[1:-1, 0], psi[1:-1, -1]]
    interior = psi[1:-1, 1:-1]
    deadband = 64 * np.finfo(float).eps * np.ptp(psi)
    regions = []
    for depth in (interior - boundary.max(), boundary.min() - interior):
        mask = depth > deadband
        if separate:
            labels, _ = ndimage.label(mask)
        else:
            labels = mask
        ids = np.flatnonzero(mask)
        if ids.size:
            groups = labels.ravel()[ids]
            order = np.argsort(groups, kind="stable")
            cuts = np.flatnonzero(np.diff(groups[order])) + 1
            for points in np.split(ids[order], cuts):
                values = depth.ravel()[points]
                regions.append((points, values / values.max()))
    regions.sort(key=lambda item: -len(item[0]))
    regions = regions[:budget]
    if not regions:
        return sparse.csr_matrix((interior.size, 0)), 0
    weights = np.sqrt([len(points) for points, _ in regions])
    quota = (budget - len(regions)) * weights / weights.sum()
    counts = 1 + np.floor(quota).astype(int)
    extra = np.argsort(-(quota % 1), kind="stable")[:budget-counts.sum()]
    counts[extra] += 1
    rows, cols, data, offset = [], [], [], 0
    for (points, coordinate), count in zip(regions, counts):
        bank = _profiles(coordinate, count, degree).tocoo()
        rows.append(points[bank.row])
        cols.append(offset + bank.col)
        data.append(bank.data)
        offset += bank.shape[1]
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(interior.size, offset)), len(regions)


class FluxAdapter:
    """Wrap an existing SPD Boomer action. The caller owns and closes the base."""

    def __init__(self, A, base, psi=None, *, budget=96, degree=3, separate=True):
        self._apply = base
        self.regions = self.columns = self.row_cap = 0
        if psi is None:
            return
        shape = np.shape(psi)
        if len(shape) != 2 or A.shape != ((shape[0]-2)*(shape[1]-2),) * 2:
            raise ValueError("matrix dimensions must match the nodal flux interior")
        Z, self.regions = flux_basis(psi, budget, degree, separate)
        self.columns = Z.shape[1]
        if self.columns:
            # Import the project's balanced two-level operator only on activation.
            from experiment import Correction
            self.row_cap = int(np.diff(Z.indptr).max())
            self._apply = Correction(A, base, Z)

    def __call__(self, residual):
        return self._apply(residual)


# The existing Boomer and balanced-correction classes live in a dated study.
_study_dir = str(Path(__file__).resolve().parents[1] / "boomer_rescue_20261116")
if _study_dir not in sys.path:
    sys.path.append(_study_dir)
