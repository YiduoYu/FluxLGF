"""Reproducible Boomer rescue prototypes. See PROTOCOL.md; no training or RHS leakage."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ['OMPI_MCA_btl'] = 'self'
import argparse
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'generalize' / 'snn'))
from cases_param import make, field
from ops_aniso import assemble
sys.path.insert(0, str(ROOT / 'solver_challenge'))
from ir_fp128 import DDResidual

BASE = dict(coarsen_type='HMIS', interp_type='ext+i', P_max=4,
            strong_threshold=.25, agg_nl=0, relax_type_all='symmetric-SOR/Jacobi',
            max_iter=1, tol=0.)
CONFIGS = {
    'base': {}, 'theta50': {'strong_threshold': .5}, 'theta70': {'strong_threshold': .7},
    'agg': {'agg_nl': 1}, 'p8': {'P_max': 8},
    'fal': {'coarsen_type': 'Falgout', 'interp_type': 'classical', 'P_max': 0},
    'schwarz': {'smooth_type': 'Schwarz-smoothers', 'smooth_num_levels': 1},
    'schwarz2': {'smooth_type': 'Schwarz-smoothers', 'smooth_num_levels': 2},
    'cheb': {'relax_type_all': 'Chebyshev'},
}

class Boomer:
    count = 0
    def __init__(self, A, cfg='base'):
        Boomer.count += 1
        self.prefix = f'r{Boomer.count}_'
        self.A = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data), comm=PETSc.COMM_SELF)
        self.pc = PETSc.PC().create(comm=PETSc.COMM_SELF)
        self.pc.setOptionsPrefix(self.prefix)
        self.pc.setOperators(self.A)
        self.pc.setType('hypre'); self.pc.setHYPREType('boomeramg')
        self.cfg = dict(BASE, **(CONFIGS[cfg] if isinstance(cfg, str) else cfg))
        opts = PETSc.Options()
        for k, v in self.cfg.items(): opts[self.prefix + 'pc_hypre_boomeramg_' + k] = v
        self.pc.setFromOptions(); self.pc.setUp()
        self.x, self.y = self.A.createVecs()
    def __call__(self, r):
        self.x.setArray(np.ascontiguousarray(r, dtype=np.float64))
        self.pc.apply(self.x, self.y)
        return self.y.getArray().copy()
    def close(self):
        self.x.destroy(); self.y.destroy(); self.pc.destroy(); self.A.destroy()
        opts = PETSc.Options()
        for k in self.cfg: del opts[self.prefix + 'pc_hypre_boomeramg_' + k]

def problem(fam, n, chi, p, rhs=0):
    if fam in ('poisson', 'jump'):
        t = np.linspace(0, 1, n+2); X, Y = np.meshgrid(t, t)
        a = np.ones_like(X) if fam == 'poisson' else np.where((X-.47)**2+(Y-.53)**2 < .04, chi, 1.)
        q = np.ones((n,n)); A = assemble(a, np.zeros_like(a), a, q, 1/(n+1))
        xx, yy = X[1:-1,1:-1].ravel(), Y[1:-1,1:-1].ravel()
    else:
        A, _, xy = make(fam, n, chi, p); xx, yy = xy.T
    u = np.sin(np.pi*xx)*np.sin(np.pi*yy)
    if rhs == 1:
        u = (u + .23*np.sin(3*np.pi*xx)*np.sin(2*np.pi*yy)
             + .17*np.sin(7*np.pi*xx)*np.sin(5*np.pi*yy))
    if rhs == 2: u = np.random.default_rng(7891).standard_normal(n*n)
    return A, A@u, u

def prolong(nc, n):
    x = np.arange(1,n+1)*(nc+1)/(n+1)
    j = np.floor(x).astype(int); f = x-j
    rows = np.repeat(np.arange(n),2); cols = np.column_stack((j-1,j)).ravel()
    vals = np.column_stack((1-f,f)).ravel(); keep=(cols>=0)&(cols<nc)&(vals>0)
    T = sp.csr_matrix((vals[keep],(rows[keep],cols[keep])),shape=(n,nc))
    return sp.kron(T,T,format='csr')

def slow_ritz(A, B, steps=32, seed=91):
    """A-inner-product Lanczos with explicit A reapplication and two-pass reorthogonalization."""
    v=np.random.default_rng(seed).standard_normal(A.shape[0]); av=A@v
    v/=np.sqrt(v@av)
    V=np.empty((A.shape[0],steps+1)); AV=np.empty_like(V)
    V[:,0]=v; AV[:,0]=A@v; H=np.zeros((steps+1,steps))
    k=steps
    for j in range(steps):
        w=B(AV[:,j])
        for _ in range(2):
            h=AV[:,:j+1].T@w; w-=V[:,:j+1]@h; H[:j+1,j]+=h
        aw=A@w; norm=np.sqrt(max(w@aw,0.))
        if norm<1e-13: k=j+1; break
        H[j+1,j]=norm; V[:,j+1]=w/norm; AV[:,j+1]=aw/norm
    mu,Y=la.eigh((H[:k,:k]+H[:k,:k].T)/2)
    return mu,V[:,:k]@Y

def coarse_tests(fam,n,chi,p,cfg,m=4,steps=32,div=4,source='ritz'):
    nc=max(15,(n+1)//div-1)
    Ac,_,_=problem(fam,nc,chi,p)
    if source=='operator':
        mu,V=spla.eigsh(Ac,k=m,sigma=0.,which='LM',v0=np.random.default_rng(91).standard_normal(Ac.shape[0]))
    else:
        Bc=Boomer(Ac,cfg)
        mu,V=slow_ritz(Ac,Bc,steps=steps)
        Bc.close()
    return prolong(nc,n)@V[:,:m], mu[:m]

def smooth_basis(A,Z,steps):
    if steps:
        # Gershgorin bound ensures stable damped Jacobi also for non-M matrices.
        inv=1/np.asarray(abs(A).sum(axis=1)).ravel()
        for _ in range(steps): Z=(Z-sp.diags(.9*inv)@(A@Z)).tocsr()
    norms=np.sqrt(np.asarray(Z.multiply(Z).sum(axis=0)).ravel())
    keep=norms>1e-13
    return (Z[:,keep]@sp.diags(1/norms[keep])).tocsr()

def local_basis(V,n,width=64):
    # Tensor hats overlapping by half their support; local QR preserves their span.
    rows=[]; cols=[]; data=[]; col=0
    coords=np.arange(n)
    for cy in range(0,n+width,width):
        ys=coords[np.abs(coords-cy)<width]
        if not len(ys): continue
        for cx in range(0,n+width,width):
            xs=coords[np.abs(coords-cx)<width]
            if not len(xs): continue
            ids=(ys[:,None]*n+xs[None,:]).ravel()
            w=((1-abs(ys-cy)/width)[:,None]*(1-abs(xs-cx)/width)[None,:]).ravel()
            local=np.column_stack((np.ones(len(ids)),V[ids]))
            norms=np.linalg.norm(local,axis=0); local/=np.maximum(norms,1e-30)
            Q,R,piv=la.qr(local,mode='economic',pivoting=True)
            rank=int(np.sum(abs(np.diag(R))>1e-7*abs(R[0,0])))
            for j in range(rank):
                rows.append(ids); cols.append(np.full(len(ids),col)); data.append(w*Q[:,j]); col+=1
    return sp.csr_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(n*n,col))

def lift_basis(V,bins=48,kind='hats'):
    rows=[]; cols=[]; vals=[]; offset=0
    for v in V.T:
        v=v/np.max(abs(v)); lo=min(0.,v.min()); hi=max(0.,v.max())
        t=(v-lo)/(hi-lo)*bins; j=np.minimum(np.floor(t).astype(int),bins-1); f=t-j
        rr=np.repeat(np.arange(len(v)),2); cc=np.column_stack((j,j+1)).ravel()
        vv=np.column_stack((1-f,f)).ravel()
        # Omit the hat nearest coordinate=0 to respect zero-boundary mode structure,
        # avoid duplicate constants across coordinates; no knowledge of the solution.
        if kind=='weighted':
            vv*=np.repeat(v,2)
            keep=abs(vv)>1e-14
        else:
            zero=int(round(-lo/(hi-lo)*bins)); keep=(cc!=zero)&(abs(vv)>1e-14)
        rows.append(rr[keep]); cols.append(cc[keep]+offset); vals.append(vv[keep]); offset+=bins+1
    Z=sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(V.shape[0],offset))
    keep=np.asarray(Z.getnnz(axis=0)).ravel()>0
    return Z[:,keep].tocsr()

class Correction:
    def __init__(self,A,B,Z):
        self.B=B; self.Z=Z; self.ZT=Z.T.tocsr(); self.AZ=(A@Z).tocsr(); self.AZT=self.AZ.T.tocsr()
        C=(self.ZT@self.AZ).tocsc(); self.C=((C+C.T)*.5).tocsc()
        self.lu=spla.splu(self.C)
        self.nnz=int(Z.nnz); self.dim=Z.shape[1]; self.cnnz=self.C.nnz
    def __call__(self,r):
        c=self.lu.solve(self.ZT@r)
        z=self.B(r-self.AZ@c)
        return z+self.Z@(c-self.lu.solve(self.AZT@z))

def solve(A,B,b,tol=1e-8,maxiter=1500,x0=None,state=None,return_state=False):
    x=np.zeros_like(b) if x0 is None else x0.copy(); normb=np.linalg.norm(b)
    if state is None:
        r=b-A@x; z=B(r); p=z.copy(); rz=r@z
    else:
        r,p,rz=state['r'].copy(),state['p'].copy(),state['rz']
    restarts=0; al=[]; be=[]
    hist=[]
    for k in range(1,maxiter+1):
        ap=A@p; pap=p@ap
        if pap<=0 or rz<=0 or not np.isfinite(rz): return x,k,False,hist,'breakdown'
        alpha=rz/pap; al.append(alpha); x+=alpha*p; r-=alpha*ap
        rel=np.linalg.norm(r)/normb
        if k%10==0: hist.append([k,float(rel)])
        if rel<tol*.8:
            r=b-A@x
            if np.linalg.norm(r)/normb<tol: return x,k,True,hist,''
            restarts+=1; z=B(r); p=z.copy(); rz=r@z
            if restarts>8: return x,k,False,hist,'residual floor'
            continue
        z=B(r); rz2=r@z; beta=rz2/rz; be.append(beta); p=z+beta*p; rz=rz2
    msg=dict(r=r,p=p,rz=rz,al=al,be=be) if return_state else 'iteration limit'
    return x,maxiter,False,hist,msg

def probe_min(state):
    al=np.asarray(state['al']); be=np.asarray(state['be'])
    diag=1/al; diag[1:]+=be[:len(al)-1]/al[:-1]
    off=np.sqrt(np.maximum(be[:len(al)-1],0))/al[:-1]
    return float(la.eigvalsh_tridiagonal(diag,off,select='i',select_range=(0,0))[0])

def audit_residual(A,b,x):
    # np.longdouble is only binary64 on this Apple arm64 host. Use error-free
    # products/sums (repository DDResidual), chunked to bound audit memory.
    rr=np.empty_like(b); zeros=np.zeros_like(x)
    for lo in range(0,len(b),65536):
        hi=min(lo+65536,len(b))
        rr[lo:hi]=DDResidual(A[lo:hi],b[lo:hi])(x,zeros)
    return rr

def run(args):
    p=np.asarray(args.p)
    A,b,u=problem(args.fam,args.n,args.chi,p,args.rhs)
    t=time.perf_counter(); cpu=time.process_time()
    B=Boomer(A,args.cfg); tbase=time.perf_counter()-t
    extra={}; solver=B; pilot_s=0.; pilot_it=0; start=None; state=None; done=False; hist0=[]
    active=args.method!='amg'
    if active and args.pilot:
        tp=time.perf_counter()
        start,pilot_it,done,hist0,state=solve(A,B,b,maxiter=args.pilot,return_state=True)
        pilot_s=time.perf_counter()-tp
        lam=1. if done else probe_min(state)
        active=not done and lam<args.gate
        extra.update(probe_min=lam,triggered=active,pilot_s=pilot_s,pilot_iterations=pilot_it)
    if active:
        V,mu=coarse_tests(args.fam,args.n,args.chi,p,args.cfg,args.m,args.steps,args.div,args.source)
        tv=time.perf_counter()-t-tbase-pilot_s
        if args.method=='local': Z=local_basis(V,args.n,args.width)
        elif args.method=='lift': Z=lift_basis(V,args.bins,args.basis)
        elif args.method=='linear': Z=sp.csr_matrix(V)
        else: raise ValueError(args.method)
        Z=smooth_basis(A,Z,args.smooth)
        solver=Correction(A,B,Z)
        extra.update(test_s=tv,mu=mu.tolist(),dim=solver.dim,nnz=solver.nnz,coarse_nnz=solver.cnnz)
    setup=time.perf_counter()-t-pilot_s
    if done: x,it,ok,hist,msg=start,pilot_it,True,hist0,''
    else:
        x,it,ok,hist,msg=solve(A,solver,b,maxiter=args.maxiter-pilot_it,x0=start,state=None if active else state)
        it+=pilot_it; hist=hist0+[[i+pilot_it,r] for i,r in hist]
    total=time.perf_counter()-t; cpusec=time.process_time()-cpu
    # Independent double-double audit of stored A,b; outside timed solve.
    rss_solver=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rr=audit_residual(A,b,x)
    real=float(np.sqrt(rr@rr)/np.linalg.norm(b)); err=np.linalg.norm(x-u)/np.linalg.norm(u)
    result=dict(**vars(args),base_setup_s=tbase,setup_s=setup,solve_s=total-setup,total_s=total,cpu_s=cpusec,
                iterations=it,ok=bool(ok and real<=1e-8),residual=real,error=float(err),message=msg,
                peak_rss_bytes=rss_solver,history=hist,audit='double-double',
                petsc_version=PETSc.Sys.getVersion(),amg_config=B.cfg,**extra)
    B.close()
    return result

def verify():
    A,b,u=problem('circ',15,1e3,np.array([.5,.5,.5]))
    B=Boomer(A); V,mu=coarse_tests('circ',15,1e3,np.array([.5,.5,.5]),'base',2,20,2)
    for name,Z in [('lift',lift_basis(V[:,:1],8)),('weighted',lift_basis(V[:,:1],8,'weighted')),('local',local_basis(V,15,8)),('linear',sp.csr_matrix(V))]:
        Z=smooth_basis(A,Z,1); M=Correction(A,B,Z)
        dense=np.column_stack([M(v) for v in np.eye(A.shape[0])])
        sym=np.linalg.norm(dense-dense.T)/np.linalg.norm(dense)
        eig=la.eigvalsh((dense+dense.T)/2)[0]
        x,it,ok,_,_=solve(A,M,b,tol=1e-10)
        direct=spla.spsolve(A,b)
        diff=np.linalg.norm(x-direct)/np.linalg.norm(direct)
        assert sym<1e-10 and eig>0 and ok and diff<1e-7,(name,sym,eig,diff)
        print(json.dumps(dict(check=name,symmetry=sym,min_eigenvalue=eig,direct_error=diff,iterations=it)))
        from fractions import Fraction
        dd=audit_residual(A,b,x)
        exact=np.array([float(Fraction(float(b[i]))-sum(Fraction(float(a))*Fraction(float(x[j]))
            for a,j in zip(A.data[A.indptr[i]:A.indptr[i+1]],A.indices[A.indptr[i]:A.indptr[i+1]])))
            for i in range(len(b))])
        assert np.linalg.norm(dd-exact)/np.linalg.norm(b)<1e-25
    x0,k,ok,h,st=solve(A,B,b,maxiter=3,return_state=True)
    x1,k1,ok1,_,_=solve(A,B,b,x0=x0,state=st)
    x2,k2,ok2,_,_=solve(A,B,b)
    assert ok1 and ok2 and k+k1==k2 and np.linalg.norm(x1-x2)<1e-13
    print(json.dumps(dict(check='pilot_continuation',iterations=k2,probe_min=probe_min(st))))
    B.close()

if __name__=='__main__':
    if '--verify' in sys.argv: verify(); sys.exit()
    ap=argparse.ArgumentParser()
    ap.add_argument('--fam',default='circ'); ap.add_argument('--n',type=int,default=255)
    ap.add_argument('--chi',type=float,default=1e4); ap.add_argument('--p',nargs=3,type=float,default=[.5,.5,.5])
    ap.add_argument('--rhs',type=int,default=0); ap.add_argument('--cfg',default='base',choices=CONFIGS)
    ap.add_argument('--method',default='amg',choices=['amg','local','lift','linear'])
    ap.add_argument('--m',type=int,default=1); ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--div',type=int,default=4); ap.add_argument('--width',type=int,default=64)
    ap.add_argument('--source',choices=['ritz','operator'],default='ritz')
    ap.add_argument('--basis',choices=['hats','weighted'],default='hats')
    ap.add_argument('--bins',type=int,default=48); ap.add_argument('--smooth',type=int,default=0)
    ap.add_argument('--maxiter',type=int,default=1500)
    ap.add_argument('--pilot',type=int,default=0); ap.add_argument('--gate',type=float,default=.1)
    args=ap.parse_args()
    try: print(json.dumps(run(args)),flush=True)
    except Exception as exc:
        import traceback
        print(json.dumps(dict(**vars(args),ok=False,exception=repr(exc),traceback=traceback.format_exc())),flush=True)
        sys.exit(1)
