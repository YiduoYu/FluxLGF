"""H56: GCRO-DR (Parks, de Sturler, Mackey, Johnson, Maiti, SISC 2006) for a sequence of systems A_s x = b_s with a
right preconditioner P that is built once and kept (factorisation reuse) -> operator Ahat_s = A_s P.

Recycle space: U (preconditioned variables), C = Ahat U orthonormal, Y = P U (stored so that, when the matrix changes,
C is rebuilt from k sparse products A_s Y instead of k applications of P). Each cycle: m - k Arnoldi steps on
(I - C C^T) Ahat, solution update x += Z y - Y (B y) with Z = P V, then the recycle space is refreshed with the k
harmonic Ritz vectors of smallest magnitude of the augmented cycle (G^T G z = theta G^T W^T Vhat z). First system,
first cycle: plain GMRES cycle + GMRES-DR extraction. k = 0 gives plain restarted GMRES(m) (reference).
Stopping test: the solver's own residual estimate (least-squares residual of the cycle) <= tol * ||b||, as PETSc
GMRES (H40 "solver-reported" criterion). The true residual of these near-singular-core matrices floors at ~1e-4 in
double precision, so it is only reported (scaled by diag(A) in seq.py)."""
import numpy as np
import scipy.linalg as sla


def _real_basis(Z, k):
    """Real orthonormal basis of the span of k (possibly complex) eigenvectors (pairs kept together where possible)."""
    cols = []
    for z in Z.T:
        cols.append(z.real)
        if np.linalg.norm(z.imag) > 1e-12 * np.linalg.norm(z.real):
            cols.append(z.imag)
    Q, _ = np.linalg.qr(np.column_stack(cols))
    return Q[:, :k]


class Recycler:
    def __init__(self, A, P, m=50, k=10):
        self.P, self.m, self.k = P, m, k
        self.U = self.C = self.Y = None
        self.set_matrix(A)

    def set_matrix(self, A):
        self.A = A
        if self.Y is not None:                       # new matrix, same P: C = A Y, re-orthonormalise
            C = np.asarray(A @ self.Y)
            Q, R = np.linalg.qr(C)
            Ri = np.linalg.inv(R)
            self.C, self.U, self.Y = Q, self.U @ Ri, self.Y @ Ri

    def solve(self, b, tol, maxit=500, x0=None):
        A, P, m, k = self.A, self.P, self.m, self.k
        n = b.size
        bn = np.linalg.norm(b)
        x = np.zeros(n) if x0 is None else x0.copy()
        r = b - A @ x
        if self.C is not None:
            c = self.C.T @ r
            x += self.Y @ c
            r -= self.C @ c
        its = 0
        hist = [np.linalg.norm(r) / bn]
        est = hist[0]
        while est > tol and its < maxit:
            kk = 0 if self.C is None else self.C.shape[1]
            if kk:                                    # keep r orthogonal to C (exact in theory, cheap in practice)
                c = self.C.T @ r
                x += self.Y @ c
                r -= self.C @ c
            steps = m - kk
            V = np.zeros((n, steps + 1), order="F"); Z = np.zeros((n, steps), order="F")   # contiguous columns
            H = np.zeros((steps + 1, steps)); B = np.zeros((kk, steps))
            beta = np.linalg.norm(r)
            V[:, 0] = r / beta
            j_done = 0
            for j in range(steps):
                Z[:, j] = P(V[:, j])
                w = A @ Z[:, j]
                if kk:
                    B[:, j] = self.C.T @ w
                    w -= self.C @ B[:, j]
                for _ in range(2):                    # classical Gram-Schmidt, twice
                    h = V[:, :j + 1].T @ w
                    w -= V[:, :j + 1] @ h
                    H[:j + 1, j] += h
                H[j + 1, j] = np.linalg.norm(w)
                V[:, j + 1] = w / H[j + 1, j]
                its += 1; j_done = j + 1
                e1 = np.zeros(j + 2); e1[0] = beta
                y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)
                est = np.linalg.norm(e1 - H[:j + 2, :j + 1] @ y) / bn
                if est <= tol or its >= maxit:
                    break
            J = j_done
            Hb = H[:J + 1, :J]
            x += Z[:, :J] @ y
            if kk:
                x -= self.Y @ (B[:, :J] @ y)
            r = b - A @ x
            hist.append(est)
            if k:
                self._update(V[:, :J + 1], Z[:, :J], Hb, B[:, :J], kk)
        return x, its, hist

    def _update(self, V, Z, Hb, B, kk):
        k = min(self.k, Hb.shape[1])
        J = Hb.shape[1]
        if kk == 0:                                   # GMRES-DR harmonic Ritz from the cycle alone
            Hs = Hb[:J, :J]
            hlast = Hb[J, J - 1]
            em = np.zeros(J); em[-1] = 1.0
            try:
                Mh = Hs + hlast**2 * np.outer(np.linalg.solve(Hs.T, em), em)
            except np.linalg.LinAlgError:
                return
            th, Zv = np.linalg.eig(Mh)
            Pk = _real_basis(Zv[:, np.argsort(np.abs(th))[:k]], k)
            Q, R = np.linalg.qr(Hb @ Pk)
            Ri = np.linalg.inv(R)
            self.C = V @ Q
            self.U = V[:, :J] @ (Pk @ Ri)
            self.Y = Z @ (Pk @ Ri)
            return
        # augmented cycle: Ahat [U~, V_J] = [C, V_{J+1}] G,  G = [[D, B], [0, Hb]],  U~ = U D
        d = 1.0 / np.linalg.norm(self.U, axis=0)
        Ut, Yt = self.U * d, self.Y * d
        G = np.zeros((kk + J + 1, kk + J))
        G[:kk, :kk] = np.diag(d); G[:kk, kk:] = B; G[kk:, kk:] = Hb
        WV = np.zeros((kk + J + 1, kk + J))          # [C, V_{J+1}]^T [U~, V_J]
        WV[:kk, :kk] = self.C.T @ Ut
        WV[kk:, :kk] = V.T @ Ut
        WV[kk:kk + J, kk:] = np.eye(J)
        try:
            th, Zv = sla.eig(G.T @ G, G.T @ WV)
        except (np.linalg.LinAlgError, ValueError):
            return
        ok = np.isfinite(th)
        idx = np.flatnonzero(ok)[np.argsort(np.abs(th[ok]))[:self.k]]
        Pk = _real_basis(Zv[:, idx], len(idx))
        Q, R = np.linalg.qr(G @ Pk)
        Ri = np.linalg.inv(R)
        PR = Pk @ Ri                                  # blockwise products: no n x (k + J) copies (n = 6.3M at m = 128)
        self.C = self.C @ Q[:kk] + V @ Q[kk:]
        self.U = Ut @ PR[:kk] + V[:, :J] @ PR[kk:]
        self.Y = Yt @ PR[:kk] + Z @ PR[kk:]
