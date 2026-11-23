"""Untimed diagnostics on separate 127^2 matrices after all wall-clock measurements."""
import json
from pathlib import Path
from shared import problem,probe
from method import s,np,sp,la,DenseCorrection
from relax import bank


def projection(M,A,Q):
    aq=A@Q
    if isinstance(M,DenseCorrection):
        coeff=M.coarse(M.Z.T@aq)
    else:
        coeff=M.lu.solve(M.ZT@aq)
    R=Q-M.Z@coeff
    return np.sqrt(np.maximum(0,np.sum(R*(A@R),axis=0))/np.sum(Q*aq,axis=0))


def run(family):
    n=127
    params=[.34,.62,.83]
    A,b,_=problem(n,family,1e5,params,3,18031)
    Ac,_,_=problem(31,family,1e5,params,0,18031)
    B=s.v1.Boomer(A,'p8')
    Bc=s.v1.Boomer(Ac,'p8')
    W,_=probe(Ac,Bc,np.inf)
    Bc.close()
    V=s.v1.prolong(31,n)@W
    R=np.column_stack([v-B(A@v) for v in V.T])
    targets,meta=s.coordinates(A,B,12,fixed_steps=96)
    raw=s.v1.lift_basis(V,48,'hats')
    aligned=s.v1.lift_basis(R,48,'hats')
    bounded=bank(A,V,48,2)
    # Exact one V-cycle filter for EVERY column. Diagnostic uses dense arrays
    # at this small size. This is deliberately not a large-scale timing claim.
    D=raw.toarray()
    filtered=np.column_stack([v-B(A@v) for v in D.T])
    # The cycle can annihilate hats exactly on the disk. Select a numerically
    # independent subset before Cholesky, keeping the actual filtered columns.
    _,triangular,order=la.qr(filtered,mode='economic',pivoting=True)
    pivots=np.abs(np.diag(triangular))
    rank=np.count_nonzero(pivots>1.e-12*pivots[0])
    post=filtered[:,order[:rank]]
    variants={'direct2':V,'aligned_direct2':R,'raw':raw,'aligned':aligned,
              'post_all_columns':post,'bounded':bounded,
              'bounded_plus_original':sp.hstack((bounded,sp.csr_matrix(V)),format='csr')}
    results={}
    for key,Z in variants.items():
        if sp.issparse(Z):
            M=s.v1.Correction(A,B,s.v1.smooth_basis(A,Z,0))
        else:
            M=DenseCorrection(A,B,Z)
        errors=projection(M,A,targets)
        x,it,ok,_,_,_=s.cg(A,M,b)
        rr=float(np.linalg.norm(s.v1.audit_residual(A,b,x))/np.linalg.norm(b))
        row_nnz=np.diff(Z.indptr) if sp.issparse(Z) else np.count_nonzero(Z,axis=1)
        results[key]=dict(dim=M.dim,it=it,relative_projection_error=errors.tolist(),
                          max_row_nnz=int(row_nnz.max()),mean_row_nnz=float(row_nnz.mean()),
                          initial_coordinate_errors=projection(M,A,V).tolist(),
                          ok=bool(ok and rr<=1e-8),residual=rr)
    B.close()
    def symbols(v):
        v=v/max(abs(v))
        lo,hi=min(0.,v.min()),max(0.,v.max())
        return np.minimum(np.floor(48*(v-lo)/(hi-lo)).astype(int),47)
    changed=[float(np.mean(symbols(v)!=symbols(w))) for v,w in zip(V.T,R.T)]
    raw_energy=float(raw.multiply(A@raw).sum())
    bounded_energy=float(bounded.multiply(A@bounded).sum())
    return dict(family=family,n=n,target_ritz_residual=meta['ritz_residual'],post_filter_calls=raw.shape[1],
                post_rank=int(rank),post_zero_columns=int(np.sum(np.all(filtered==0,axis=0))),
                raw_energy=raw_energy,bounded_energy=bounded_energy,interval_changed_fraction=changed,
                comparisons=results)
if __name__=='__main__':
    with (Path(__file__).resolve().parent/'mechanism.jsonl').open('w') as out:
        for fam in ['circ','disk','two_vortices','four_cells']:
            row=run(fam)
            out.write(json.dumps(row)+'\n');out.flush()
            print(fam,[(k,v['it']) for k,v in row['comparisons'].items()],flush=True)
