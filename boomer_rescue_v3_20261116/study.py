"""Round 3: frozen-coordinate ablations and an existing adaptive-SA implementation.

Imports rounds 1/2 unchanged. Dense Ritz deflation uses dense BLAS and Cholesky,
not a sparse representation that would unfairly penalize dense vectors.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
os.environ['OMPI_MCA_btl']='self'
import sys,json,time,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'boomer_rescue_v2_20261116'))
import solver as s
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

class DenseCorrection:
    def __init__(self,A,B,V):
        self.B=B
        self.Z=np.asarray(V/np.linalg.norm(V,axis=0),order='F')
        self.AZ=np.asarray(A@self.Z,order='F')
        C=self.Z.T@self.AZ
        self.C=(C+C.T)*.5
        self.factor=la.cho_factor(self.C,lower=True)
        self.dim=V.shape[1]; self.nnz=int(self.Z.size); self.cnnz=int(C.size)
    def coarse(self,r):
        return la.cho_solve(self.factor,r)
    def __call__(self,r):
        c=self.coarse(self.Z.T@r)
        z=self.B(r-self.AZ@c)
        return z+self.Z@(c-self.coarse(self.AZ.T@z))

def local_pou(V,n,target_dim):
    """Geometric overlapping partition of unity times local {1,v1,v2}.

    This is a control inspired by local near-nullspace enrichment, NOT alpha-SA.
    Local pivoted QR changes only coordinates; each patch spans the same vectors.
    k is chosen before solving, solely from the desired coarse dimension.
    """
    rank=V.shape[1]+1
    k=max(2,int(round(np.sqrt(target_dim/rank))))
    centers=np.linspace(0,n-1,k); width=centers[1]-centers[0]
    axis=np.arange(n); rows=[];cols=[];vals=[]; col=0
    for cy in centers:
        ys=axis[np.abs(axis-cy)<width]
        for cx in centers:
            xs=axis[np.abs(axis-cx)<width]
            ids=(ys[:,None]*n+xs[None,:]).ravel()
            w=((1-abs(ys-cy)/width)[:,None]*(1-abs(xs-cx)/width)[None,:]).ravel()
            U=np.column_stack((np.ones(len(ids)),V[ids]))
            U/=np.maximum(np.linalg.norm(U,axis=0),1e-30)
            Q,R,_=la.qr(U,mode='economic',pivoting=True)
            keep=int(np.sum(abs(np.diag(R))>1e-10*abs(R[0,0])))
            for j in range(keep):
                rows.append(ids);cols.append(np.full(len(ids),col));vals.append(w*Q[:,j]);col+=1
    return sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(n*n,col))

def basis(A,V,n,kind,bins):
    if kind=='direct':return V
    if kind in ['weighted','hats']:Z=s.v1.lift_basis(V,bins,kind)
    elif kind=='local':Z=local_pou(V,n,2*(bins+1))
    else:raise ValueError(kind)
    return s.v1.smooth_basis(A,Z,0)

def run(spec):
    args=dict(n=511,fam='circ',chi=1e5,p=[.5]*3,rhs=4,rhs_seed=1171,
              variant='weighted',m=2,steps=96,cfg='p8',maxiter=1500)
    args.update(spec)
    # The frozen round-2 candidate is called directly, without reimplementation.
    if args['variant'] in ['adaptive','base','p8']:
        allowed={k:args[k] for k in ['n','fam','chi','p','rhs','rhs_seed','maxiter']}
        if args['variant']=='adaptive':
            allowed.update(method='adaptive',cfg='p8',m=2,pilot=16,gate='both',gate_steps=40,
                           gate_lambda=.04,rtol=.1,min_steps=24,max_steps=128)
        else:allowed.update(method='amg',cfg=args['variant'])
        return dict(s.run(allowed),variant=args['variant'])
    A,b,u=s.problem(args['fam'],args['n'],args['chi'],args['p'],args['rhs'],args['rhs_seed'])
    t=time.perf_counter(); cpu=time.process_time(); info={}; variant=args['variant']; B=None
    if variant=='asa':
        import pyamg
        np.random.seed(91117)
        ml,work=pyamg.aggregation.adaptive_sa_solver(
            sp.csr_array(A),symmetry='symmetric',num_candidates=2,candidate_iters=5,
            improvement_iters=0,strength='symmetric',aggregate='standard',
            prepostsmoother=('gauss_seidel',{'sweep':'symmetric'}),
            smooth=('jacobi',{}),max_levels=10,max_coarse=10,coarse_solver='pinv')
        M=ml.aspreconditioner(cycle='V').matvec
        info.update(pyamg_version=pyamg.__version__,asa_work=float(work),
                    levels=len(ml.levels),operator_complexity=ml.operator_complexity())
    else:
        B=s.v1.Boomer(A,args['cfg']); info['base_setup_s']=time.perf_counter()-t
        tc=time.perf_counter(); nc=max(15,(args['n']+1)//4-1)
        Ac,_,_=s.problem(args['fam'],nc,args['chi'],args['p'],0,args['rhs_seed'])
        Bc=s.v1.Boomer(Ac,args['cfg'])
        W,ri=s.coordinates(Ac,Bc,m=args['m'],fixed_steps=args['steps'])
        Bc.close(); V=s.v1.prolong(nc,args['n'])@W
        info.update(ri,coordinate_s=time.perf_counter()-tc)
        bins=max(48,3*(args['n']+1)//16)
        Z=basis(A,V,args['n'],variant,bins)
        M=DenseCorrection(A,B,Z) if variant=='direct' else s.v1.Correction(A,B,Z)
        info.update(dim=M.dim,nnz=M.nnz,coarse_nnz=M.cnnz,bins=bins,
                    max_row_nnz=(M.dim if variant=='direct' else int(np.diff(Z.indptr).max())))
    setup=time.perf_counter()-t
    x,it,ok,_,hist,msg=s.cg(A,M,b,maxiter=args['maxiter'])
    total=time.perf_counter()-t; cpu_s=time.process_time()-cpu
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    residual=float(np.linalg.norm(s.v1.audit_residual(A,b,x))/np.linalg.norm(b))
    error=None if u is None else float(np.linalg.norm(x-u)/np.linalg.norm(u))
    if B is not None:B.close()
    return dict(args,**info,ok=bool(ok and residual<=1e-8),iterations=it,residual=residual,
                error=error,setup_s=setup,solve_s=total-setup,total_s=total,cpu_s=cpu_s,
                peak_rss_bytes=peak,history=hist,message=msg,audit='double-double')

if __name__=='__main__':
    spec=json.loads(sys.argv[1])
    try: print(json.dumps(run(spec)),flush=True)
    except Exception as exc:
        import traceback
        print(json.dumps(dict(spec,ok=False,exception=repr(exc),traceback=traceback.format_exc())),flush=True)
        sys.exit(1)
