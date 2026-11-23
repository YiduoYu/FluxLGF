"""Budgeted P1 spline enrichment in two approximate slow-mode value coordinates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'boomer_rescue_v3_20261116'))
from study import s, np, sp, la, DenseCorrection
import time


def initial_grid(values, intervals):
    lo = min(values.min(), 0.)
    hi = max(values.max(), 0.)
    if lo == hi:
        raise ValueError("Coordinate is identically zero")
    if lo == 0 or hi == 0:
        return np.linspace(lo, hi, intervals + 1)
    negative = np.clip(round(intervals * (-lo) / (hi - lo)), 1, intervals - 1)
    return np.r_[np.linspace(lo, 0, negative + 1), np.linspace(0, hi, intervals - negative + 1)[1:]]


def nodal_basis(values, grids):
    """Piecewise linear interpolation, anchored to zero at the zero knot."""
    blocks = []
    for v, knots in zip(values.T, grids):
        j = np.clip(np.searchsorted(knots, v, side='right') - 1, 0, len(knots) - 2)
        f = np.clip((v - knots[j]) / (knots[j + 1] - knots[j]), 0, 1)
        indices = np.column_stack((j, j + 1)).ravel()
        weights = np.column_stack((1 - f, f)).ravel()
        keep = (knots[indices] != 0) & (weights > 1e-14)
        block = sp.coo_matrix((weights[keep], (np.repeat(np.arange(len(v)), 2)[keep], indices[keep])),
                              shape=(len(v), len(knots))).tocsr()
        blocks.append(block[:, knots != 0])
    return sp.hstack(blocks, format='csr')


def learn_knots(A, values, grids, probes, additions):
    """Greedy exact single-column energy gain, subject to dyadic parent precedence."""
    base = nodal_basis(values, grids)
    candidates, parents, rows, cols, entries = [], [], [], [], []
    for axis, knots in enumerate(grids):
        v = values[:, axis]
        for low, high in zip(knots[:-1], knots[1:]):
            mid = (low + high) / 2
            parent = len(candidates)
            for left, right, required in [(low, high, -1), (low, mid, parent), (mid, high, parent)]:
                center = (left + right) / 2
                ids = np.flatnonzero((v > left) & (v < right))
                rows.append(ids)
                cols.append(np.full(len(ids), len(candidates)))
                entries.append(np.maximum(0,1 - abs(v[ids] - center) / (center - left)))
                candidates.append((axis, center))
                parents.append(required)
    tents = sp.coo_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(values), len(candidates))).tocsr()
    dictionary = sp.hstack((base, tents), format='csr')
    gram = (dictionary.T @ (A @ dictionary)).toarray()
    gram = (gram + gram.T) / 2
    rhs = dictionary.T @ (A @ probes)
    size = base.shape[1]
    inverse = la.pinvh(gram[:size,:size], rtol=1e-12)
    coupling = gram[size:,:size]
    residual_gram = gram[size:,size:] - coupling @ inverse @ coupling.T
    correlations = rhs[size:] - coupling @ inverse @ rhs[:size]
    chosen = np.zeros(len(candidates), dtype=bool)
    required = np.array(parents)
    threshold = max(residual_gram.diagonal().max(),1e-30)*1e-12
    accumulated = []
    output = [list(knots) for knots in grids]
    for _ in range(additions):
        norm = residual_gram.diagonal()
        available = ~chosen & ((required < 0) | chosen[np.maximum(0,required)]) & (norm > threshold)
        if not np.any(available):
            break
        gains = np.full(len(norm), -np.inf)
        gains[available] = np.sum(correlations[available]**2,axis=1)/norm[available]
        idx = int(np.argmax(gains))
        accumulated.append(float(gains[idx]))
        projection = residual_gram[:,idx].copy()/np.sqrt(norm[idx])
        correlations -= np.outer(projection, correlations[idx].copy()/np.sqrt(norm[idx]))
        residual_gram -= np.outer(projection,projection)
        chosen[idx] = True
        axis, center = candidates[idx]
        output[axis].append(center)
    return [np.sort(knots) for knots in output], accumulated


def measure(case, mode, fraction=1, sweeps=0, steps=32):
    n, family, chi, p, rhs, seed = case
    A,b,u = s.problem(family,n,chi,p,rhs,seed)
    started = time.perf_counter()
    cpu = time.process_time()
    B = s.v1.Boomer(A,'p8')
    info = {}
    if mode == 'boomer':
        M = B
    else:
        nc = (n+1)//4-1
        Ac,_,_ = s.problem(family,nc,chi,p,0,seed)
        Bc = s.v1.Boomer(Ac,'p8')
        count = 8 if mode=='greedy' else (16 if mode=='direct16' else 2)
        W,_ = s.coordinates(Ac,Bc,count,fixed_steps=steps)
        Bc.close()
        V = s.v1.prolong(nc,n) @ W[:,:count if mode=='direct16' else 2]
        for _ in range(sweeps):
            for axis in range(V.shape[1]):
                V[:,axis] -= B(A @ V[:,axis])
        bins = max(48,3*(n+1)//16)//fraction
        if mode=='direct16':
            M = DenseCorrection(A,B,V)
        else:
            if mode=='legacy':
                Z = s.v1.lift_basis(V,bins,'hats')
            else:
                V /= np.max(abs(V),axis=0)
                base_count = bins//2 if mode=='greedy' else bins
                grids = [initial_grid(v,base_count) for v in V.T]
                if mode=='greedy':
                    sample = V.reshape(n,n,2)[3::4,3::4].reshape(nc*nc,2)
                    grids, gains = learn_knots(Ac,sample,grids,W[:,2:8],2*(bins-base_count))
                    info.update(selected=len(gains),training_gain=sum(gains))
                info['intervals']=[len(k)-1 for k in grids]
                Z = nodal_basis(V,grids)
            Z = s.v1.smooth_basis(A,Z,0)
            M = s.v1.Correction(A,B,Z)
            info.update(dim=Z.shape[1],row_cap=int(np.diff(Z.indptr).max()))
    setup = time.perf_counter()-started
    x,it,ok,_,_,msg = s.cg(A,M,b)
    total = time.perf_counter()-started
    cpu_s = time.process_time()-cpu
    rr = np.linalg.norm(s.v1.audit_residual(A,b,x))/np.linalg.norm(b)
    B.close()
    return dict(info,case=case,mode=mode,fraction=fraction,sweeps=sweeps,steps=steps,it=it,
                ok=bool(ok and rr<=1e-8),residual=float(rr),message=msg,total=total,setup=setup,cpu=cpu_s)
