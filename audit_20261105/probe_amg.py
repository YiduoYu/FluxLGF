"""Independent, one-process-per-configuration benchmark; original solvers unchanged."""
import sys,json,time
from pathlib import Path
import numpy as np
from petsc4py import PETSc
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'regions'),str(ROOT/'solver_challenge')]
from gen_gen import generate
from btfd import BTFD
from ir_fp128 import DDResidual,two_sum
m,n,theta,agg,pmax,coarsen,mode=sys.argv[1:]
m,n,agg,pmax=int(m),int(n),int(agg),int(pmax);theta=float(theta)
A,b=generate(m,n,2,1,0,.02,1)
res=DDResidual(A,b); nb=np.linalg.norm(b)
t0=time.perf_counter()
M=PETSc.Mat().createAIJ(size=A.shape,csr=(A.indptr,A.indices,A.data))
k=PETSc.KSP().create();k.setOperators(M); k.setType('gmres');k.setGMRESRestart(50)
k.setPCSide(PETSc.PC.Side.RIGHT);k.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
k.setTolerances(rtol=1e-6,atol=0,max_it=500)
pc=k.getPC();pc.setType('python' if coarsen=='bt-fd' else 'hypre')
if coarsen=='bt-fd': pc.setPythonContext(BTFD(A,tol=.1))
else:
 pc.setHYPREType('boomeramg');k.setOptionsPrefix('audit_')
 for key,v in dict(coarsen_type=coarsen,interp_type='ext+i',P_max=pmax,strong_threshold=theta,agg_nl=agg,max_iter=1,tol=0).items(): PETSc.Options()['audit_pc_hypre_boomeramg_'+key]=v
k.setFromOptions();k.setUp();dv,rv=M.createVecs();setup=time.perf_counter()-t0
xh=np.zeros_like(b);xl=np.zeros_like(b);r=b.copy();hist=[]
for step in range(10):
 rv.setArray(r);k.solve(rv,dv)
 xh,t=two_sum(xh,dv.getArray());xl+=t;xh,xl=two_sum(xh,xl)
 r=res(xh,xl);rel=np.linalg.norm(r)/nb
 hist.append([k.getIterationNumber(),int(k.getConvergedReason()),rel])
 if rel<1e-8 or not np.isfinite(rel):break
total=time.perf_counter()-t0
print(json.dumps(dict(m=m,n=n,theta=theta,agg=agg,pmax=pmax,coarsen=coarsen,setup_s=setup,total_s=total,hist=hist,success=bool(rel<1e-8))))
