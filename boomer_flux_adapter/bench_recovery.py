"""Paired post-design study. Each run rebuilds A and the preconditioner."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from bench import CASES, inputs, np, run, s
from recover import StencilFluxAdapter


def recovery(index):
    case = CASES[index]
    A,b,u = inputs(*case)
    start = time.perf_counter()
    B = s.v1.Boomer(A,"p8")
    try:
        M = StencilFluxAdapter(A,B,case[0],budget=(case[0]+1)//4)
        setup = time.perf_counter()-start
        x,it,ok,_,_,_ = s.cg(A,M,b)
        total = time.perf_counter()-start
        rr = float(np.linalg.norm(s.v1.audit_residual(A,b,x))/np.linalg.norm(b))
        return dict(total=total,setup=setup,solve=total-setup,it=it,ok=bool(ok and rr<=1e-8),
                    residual=rr,solution_error=None if u is None else float(np.linalg.norm(u-x)/np.linalg.norm(u)),
                    regions=M.regions,columns=M.columns,row_cap=M.row_cap,fit=M.fit)
    finally:
        B.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--one", nargs=2,metavar=("INDEX","METHOD"))
    group.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.one:
        i,method=args.one
        print(json.dumps(recovery(int(i)) if method=="recover" else run(int(i),method)))
    else:
        rng=random.Random(61293)
        with args.output.open("x") as out:
            for rep in range(3):
                order=list(range(len(CASES)))
                rng.shuffle(order)
                for i in order:
                    variants=["recover","p8","p4","flux","v4"]
                    if i in (3,4,8):
                        variants.append("anchor")
                    rng.shuffle(variants)
                    for method in variants:
                        cmd=[sys.executable,__file__,"--one",str(i),method]
                        raw=subprocess.check_output(cmd,text=True)
                        row=json.loads(raw.strip().splitlines()[-1])
                        print(json.dumps(dict(case=CASES[i],index=i,rep=rep,method=method,**row)),
                              file=out,flush=True)
                        if not row["ok"]:
                            raise RuntimeError(f"residual failure in {i}, {method}")
