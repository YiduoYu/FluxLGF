"""Recover a smooth flux coordinate from a row-major unit-square diffusion stencil."""
import numpy as np
import scipy.linalg as la
from numpy.polynomial import legendre
from scipy.sparse import issparse

try:
    from .adapter import FluxAdapter
except ImportError:  # Direct import from this study directory.
    from adapter import FluxAdapter


def _directions(A, axis, n):
    ys, xs = np.meshgrid(axis, axis, indexing="ij")
    nodes = (ys*n + xs).ravel()
    stencil = A[nodes].tocoo()
    dy = stencil.col // n - ys.ravel()[stencil.row]
    dx = stencil.col % n - xs.ravel()[stencil.row]
    if np.any(((abs(dx)>1) | (abs(dy)>1)) & (stencil.data != 0)):
        raise ValueError("sampled rows do not have a row-major 9-point stencil")
    def moment(weight):
        return np.bincount(stencil.row, weights=-stencil.data*weight, minlength=len(nodes))
    xx, xy, yy = moment(dx*dx), moment(dx*dy), moment(dy*dy)
    gap = np.hypot(xx-yy, 2*xy)
    high = (xx + yy + gap)/2
    low = high-gap
    strong = (high>0) & (low <= high/50) & (low >= -.05*high)
    angle = .5*np.arctan2(2*xy, xx-yy)
    return np.cos(angle), np.sin(angle), float(np.mean(strong))


def _gradients(axis, n, degree):
    grid = 2*(axis+1)/(n+1)-1
    values = legendre.legvander(grid, degree)
    deriv = values[:, :-1] @ legendre.legder(np.eye(degree+1), axis=0)
    def products(first, second):
        # Columns are (y-degree, x-degree). Exclude the exact constant.
        return np.einsum("yi,xj->yxij", first, second).reshape(len(grid)**2,-1)[:,1:]
    return products(values,deriv), products(deriv,values)


def recover_flux(A, n):
    """Return nodal flux or None, and fit diagnostics. No case or RHS input.

    Assumes the repository's uniform unit square with n by n Dirichlet interior
    DOFs (x varies fastest), real SPD diffusion and CSR storage. Sampled stencil
    moments identify the principal axis up to sign. Fit and check grids do not
    share nodes. A global polynomial model deliberately limits this prototype.
    """
    if not isinstance(n,(int,np.integer)) or n < 1 or A.shape != (n*n, n*n):
        raise ValueError("expected n by n interior nodes on the unit square")
    if not issparse(A) or A.format != "csr" or np.iscomplexobj(A.data):
        raise ValueError("expected a real CSR matrix")
    if n < 65:
        return None, {"reason":"small_grid"}
    axis = np.rint(np.linspace(1,n-2,62)).astype(int)
    train, check = axis[::2], axis[1::2]
    ex,ey,strong = _directions(A,train,n)
    hx,hy,check_strong = _directions(A,check,n)
    info = {"strong_fraction":min(strong,check_strong)}
    if info["strong_fraction"] < .9:
        return None, dict(info, reason="weak_or_local_anisotropy")
    for degree in (1,2,4,8,12,16):
        dx,dy = _gradients(train,n,degree)
        along = ex[:,None]*dx + ey[:,None]*dy
        gram = dx.T @ dx + dy.T @ dy
        loss = along.T @ along
        # Break near-ties in favour of lower polynomial order.
        levels = np.arange(degree+1)
        rank = np.add.outer(levels, levels).ravel()[1:]
        loss.flat[::loss.shape[0]+1] += 1e-8 * rank**2 * gram.diagonal()
        _, vectors = la.eigh(loss, gram, subset_by_index=(0,0))
        c = vectors[:,0]
        gx,gy = _gradients(check,n,degree)
        vx,vy = gx@c, gy@c
        defect = la.norm(hx*vx+hy*vy)/np.hypot(la.norm(vx),la.norm(vy))
        info.update(degree=degree, defect=float(defect))
        if defect <= .025:
            grid = legendre.legvander(np.linspace(-1,1,n+2),degree)
            coefficients = np.r_[0.,c].reshape(degree+1,degree+1)
            return grid @ coefficients @ grid.T, dict(info,reason="fit")
    return None, dict(info,reason="direction_fit_failed")


class StencilFluxAdapter(FluxAdapter):
    """Construct a balanced correction using only A, its grid width and base PC."""
    def __init__(self, A, base, n, *, budget=96):
        psi, self.fit = recover_flux(A,n)
        super().__init__(A,base,psi,budget=budget)
