"""Production IR factory + exact rational verification of every row, small 3T system."""
import sys,json,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'regions'),str(ROOT/'solver_challenge')]
from gen3t import generate
from gen_gen import generate as gen
from ir_fp128 import DDResidual,two_sum
from run_ir import inner
A,b=generate(12,2e-11,2e-4);res=DDResidual(A,b);nb=np.linalg.norm(b);out={}
for kind in ['amg-3d','bt-fd']:
 k,(dv,rv)=inner(A,1e-6,kind);h=np.zeros_like(b);l=np.zeros_like(b);r=b.copy();hist=[]
 for step in range(10):
  rv.setArray(r);k.solve(rv,dv);h,t=two_sum(h,dv.getArray());l+=t;h,l=two_sum(h,l)
  r=res(h,l);hist.append([k.getIterationNumber(),int(k.getConvergedReason()),float(np.linalg.norm(r)/nb)])
  if np.linalg.norm(r)/nb<1e-8:break
 exact=[]
 for i in range(len(b)):
  exact.append(float(F(float(b[i]))-sum((F(float(v))*(F(float(h[j]))+F(float(l[j]))) for j,v in zip(A.indices[A.indptr[i]:A.indptr[i+1]],A.data[A.indptr[i]:A.indptr[i+1]])),F(0))))
 out[kind]={'history':hist,'exact_relative_residual':float(np.linalg.norm(exact)/nb),'dd_reference_difference_over_b':float(np.linalg.norm(r-exact)/nb),'rounded_solution_relative_residual':float(np.linalg.norm(res(h,np.zeros_like(l)))/nb)}
# Intended conservative diffusion: report the stored core row sums using fsum, without matvec cancellation.
stats=[]
for phase in (0.,1.35):
 A,b=gen(32,1,2,1,0,.1,1,phase=phase);N=len(b)//3;M=A[2*N:,2*N:]
 core=np.flatnonzero(M.diagonal()>1e16)
 rr=np.array([math.fsum(M.data[M.indptr[i]:M.indptr[i+1]]) for i in core])
 stats.append({'phase':phase,'core_rows':len(core),'row_sum_min':float(rr.min()),'row_sum_max':float(rr.max()),'nominal_core_reaction':512.,'max_relative_deviation':float(np.max(abs(rr-512))/512)})
out['stored_reaction_roundoff']=stats
print(json.dumps(out,indent=2))
(Path(__file__).parent/'evidence/ir_exact.json').write_text(json.dumps(out,indent=2)+'\n')
