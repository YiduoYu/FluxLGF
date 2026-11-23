"""One process per sample. Run with the repository's hypre_env Python."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ["OMPI_MCA_btl"] = "self"
sys.path.insert(0, str(HERE.parent / "boomer_rescue_v4_20261118"))
from pilot import np, problem, s
from ops_aniso import tensor, assemble
from adapter import FluxAdapter
import cluster_probe
import shared

# Values are (grid width, family, anisotropy, geometry, RHS, RNG seed).
CASES = [
    (383, "circ", 1e5, [.57,.28,.43], 3, 29231),
    (383, "ellipse", 5e4, [.19,.61,.72], 4, 29233),
    (383, "two_vortices", 1e5, [.43,.77,.32], 4, 29235),
    (383, "four_cells", 1e5, [.36,.72,.81], 3, 29237),
    (511, "six_cells", 1e5, [.38,-.17,0], 3, 29239),
    (383, "rot", 1e5, [.41,.6,0], 3, 29241),
    (383, "disk", 1e5, [.36,.71,.42], 4, 29243),
    (383, "circ", 1., [.57,.28,.43], 3, 29245),
    (767, "four_cells", 8e4, [.27,.48,.13], 4, 29247),
]


def field_axes(n, family, p):
    axis = np.linspace(0, 1, n+2)
    x, y = np.meshgrid(axis, axis)
    if family == "six_cells":
        ax, ay = p[:2]
        return (x+ax*np.sin(np.pi*x)/np.pi, y+ay*np.sin(np.pi*y)/np.pi,
                1+ax*np.cos(np.pi*x), 1+ay*np.cos(np.pi*y), 3*np.pi, 2*np.pi)
    scale = .7 if family == "four_cells" else .6
    ax, ay = scale*(p[0]-.5), scale*(p[1]-.5)
    return (x+ax*np.sin(2*np.pi*x)/(2*np.pi), y+ay*np.sin(2*np.pi*y)/(2*np.pi),
            1+ax*np.cos(2*np.pi*x), 1+ay*np.cos(2*np.pi*y),
            2*np.pi, (2 if family == "four_cells" else 1)*np.pi)


def flux(n, family, p):
    if family == "disk":
        return None
    if family in ("two_vortices", "four_cells", "six_cells"):
        x,y,_,_,kx,ky = field_axes(n, family, p)
        return np.sin(kx*x)*np.sin(ky*y)
    axis = np.linspace(0,1,n+2)
    x,y = np.meshgrid(axis,axis)
    if family == "circ":
        return (x-.3-.4*p[0])**2 + (y-.3-.4*p[1])**2
    if family == "ellipse":
        return (((x-.53-.12*(p[0]-.5))/(1.25+.4*(p[2]-.5)))**2
                + ((y-.47-.12*(p[1]-.5))/(.65-.2*(p[2]-.5)))**2)
    if family == "rot":
        return -x*np.sin(np.pi*p[0])+y*np.cos(np.pi*p[0])
    raise ValueError(family)


def inputs(n, family, chi, p, rhs, seed):
    if family != "six_cells":
        return problem(n, family, chi, p, rhs, seed)
    x,y,dx,dy,kx,ky = field_axes(n, family, p)
    vx = -ky*np.sin(kx*x)*np.cos(ky*y)*dy
    vy = kx*np.cos(kx*x)*np.sin(ky*y)*dx
    A = assemble(*tensor(np.arctan2(vy,vx),chi,1.), np.ones((n,n)),1/(n+1))
    return A, np.random.default_rng(seed).standard_normal(n*n), None


def run(index, method):
    case = CASES[index]
    n,family,_,p,_,_ = case
    if method in ("v4", "anchor"):
        module = shared if method == "v4" else cluster_probe
        with patch.object(module, "problem", inputs):
            row = shared.solve(case,1,.08) if method == "v4" else cluster_probe.run(case,"anchor")
        return dict(total=row["total"], setup=row["setup"], solve=row["total"]-row["setup"],
                    it=row["it"], residual=row["residual"], ok=row["ok"], columns=row.get("dim",0),
                    row_cap=row.get("row_cap",0))
    A,b,u = inputs(*case)
    start = time.perf_counter()
    base = s.v1.Boomer(A, "base" if method == "p4" else "p8")
    try:
        M = base
        if method in ("flux", "pooled", "linear"):
            M = FluxAdapter(A, base, flux(n,family,p), budget=(n+1)//4,
                            degree=1 if method=="linear" else 3, separate=method!="pooled")
        setup = time.perf_counter()-start
        x,it,ok,_,_,_ = s.cg(A,M,b)
        total = time.perf_counter()-start
        rr = float(np.linalg.norm(s.v1.audit_residual(A,b,x))/np.linalg.norm(b))
        error = None if u is None else float(np.linalg.norm(x-u)/np.linalg.norm(u))
        return dict(total=total,setup=setup,solve=total-setup,it=it,residual=rr,
                    ok=bool(ok and rr<=1e-8),solution_error=error,regions=getattr(M,"regions",0),
                    columns=getattr(M,"columns",0),row_cap=getattr(M,"row_cap",0))
    finally:
        base.close()


def methods(index):
    variants = ["p4","p8","v4","flux"]
    if index in (3,4):
        variants.extend(["anchor","pooled","linear"])
    if index == 8:
        variants.append("anchor")
    return variants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one",nargs=2,metavar=("INDEX","METHOD"))
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.one:
        index,method = args.one
        if method not in methods(int(index)):
            parser.error("method outside case schedule")
        print(json.dumps(run(int(index), method)))
    elif args.output:
        rng = random.Random(29229)
        with args.output.open("x") as out:
            for rep in range(3):
                order = list(range(len(CASES)))
                rng.shuffle(order)
                for index in order:
                    variants = methods(index)
                    rng.shuffle(variants)
                    for method in variants:
                        result = subprocess.run([sys.executable, __file__, "--one", str(index), method],
                                                text=True,capture_output=True)
                        if result.returncode:
                            raise RuntimeError(result.stderr or result.stdout)
                        row = json.loads(result.stdout.strip().splitlines()[-1])
                        print(json.dumps(dict(case=CASES[index],index=index,rep=rep,method=method,**row)),
                              file=out,flush=True)
                        if not row["ok"]:
                            raise RuntimeError(f"residual check failed for {index}, {method}")
    else:
        parser.error("choose --one or --output")
