"""Two new analytic fields to test transfer and the boundary-range limitation."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from bench import np, assemble, s, tensor
from adapter import FluxAdapter
from recover import StencilFluxAdapter


def field(n, name):
    axis = np.linspace(0,np.pi,n+2)
    x,y = np.meshgrid(axis,axis)
    sx,sy = np.sin(x),np.sin(y)
    if name == "interior_island":
        # psi stays within boundary range [0,1]. Local extrema occur inside.
        h = .23*sx**2*sy**2
        psi = y/np.pi + h*np.sin(4*y)
        gx = .23*np.sin(2*x)*sy**2*np.sin(4*y)
        gy = 1/np.pi + .23*sx**2*np.sin(2*y)*np.sin(4*y) + 4*h*np.cos(4*y)
    else:
        h = np.cos(2*x)+.72*np.cos(2*y)+.27*np.sin(2*x+y)-.1
        psi = sx*sy*h
        gx = np.cos(x)*sy*h + sx*sy*(-2*np.sin(2*x)+.54*np.cos(2*x+y))
        gy = sx*np.cos(y)*h + sx*sy*(-1.44*np.sin(2*y)+.27*np.cos(2*x+y))
    return psi,gx,gy


def sample(name, method):
    n = 383
    _,gx,gy = field(n,name)
    A = assemble(*tensor(np.arctan2(gx,-gy),1e5,1.),np.ones((n,n)),1/(n+1))
    b = np.random.default_rng(44231).normal(size=n*n)
    start = time.perf_counter()
    B = s.v1.Boomer(A,"base" if method=="p4" else "p8")
    try:
        M = B
        if method == "recover":
            M = StencilFluxAdapter(A,B,n,budget=96)
        elif method == "flux":
            M = FluxAdapter(A,B,field(n,name)[0],budget=96)
        setup = time.perf_counter()-start
        x,it,ok,_,_,_ = s.cg(A,M,b)
        total = time.perf_counter()-start
        rr = float(np.linalg.norm(s.v1.audit_residual(A,b,x))/np.linalg.norm(b))
        return dict(total=total,setup=setup,solve=total-setup,it=it,residual=rr,
                    ok=bool(ok and rr<=1e-8),fit=getattr(M,"fit",None),
                    regions=getattr(M,"regions",0),columns=getattr(M,"columns",0))
    finally:
        B.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--one",nargs=2)
    group.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.one:
        print(json.dumps(sample(*args.one)))
    else:
        rng=random.Random(25591)
        with args.output.open("x") as out:
            for rep in range(3):
                jobs=[(name,m) for name in ("warped_islands","interior_island")
                      for m in ("p4","p8","flux","recover")]
                rng.shuffle(jobs)
                for name,m in jobs:
                    raw=subprocess.check_output([sys.executable,__file__,"--one",name,m],text=True)
                    row=json.loads(raw.strip().splitlines()[-1])
                    print(json.dumps(dict(case=name,method=m,rep=rep,**row)),file=out,flush=True)
                    if not row["ok"]:
                        raise RuntimeError(f"residual failure in {name}, {m}")
