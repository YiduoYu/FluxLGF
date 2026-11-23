"""Validate complete case/method/repetition identities and emit comparable medians."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
from bench import CASES, methods


def summarize(path, cohort):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if cohort == "topology":
        expected={name:["p4","p8","flux","recover"] for name in ("warped_islands","interior_island")}
    else:
        expected={i:methods(i) if cohort=="analytic" else
                  ["p4","p8","v4","flux","recover"]+(["anchor"] if i in (3,4,8) else [])
                  for i in range(len(CASES))}
    grouped=defaultdict(list)
    for row in rows:
        i=row["case"] if cohort=="topology" else row["index"]
        if cohort!="topology" and row["case"]!=list(CASES[i]):
            raise ValueError("case parameters differ")
        if not row["ok"] or not 0<=row["residual"]<=1e-8:
            raise ValueError("residual failure")
        grouped[i,row["method"]].append(row)
    if set(grouped)!={(i,m) for i,variants in expected.items() for m in variants}:
        raise ValueError("case/method schedule mismatch")
    cases={}
    for i,variants in expected.items():
        stats={}
        for m in variants:
            group=grouped[i,m]
            if sorted(r["rep"] for r in group)!=[0,1,2] or len({r["it"] for r in group})!=1:
                raise ValueError(f"missing repetition or unstable iterations: {i}, {m}")
            stat={k:median(r[k] for r in group) for k in ("setup","solve","total","it")}
            stat.update(time_range=[min(r["total"] for r in group),max(r["total"] for r in group)])
            for k in ("regions","columns","row_cap","fit","solution_error"):
                if k in group[0]:
                    stat[k]=group[0][k]
            stats[m]=stat
        cases[i]=stats
    result=dict(records=len(rows),max_residual=max(r["residual"] for r in rows),
                max_reported_solution_error=max((r["solution_error"] for r in rows
                       if r.get("solution_error") is not None),default=None),cases=cases)
    if cohort!="topology":
        method="flux" if cohort=="analytic" else "recover"
        targets=[0,1,2,3,4,8]
        ratios=[]
        v4=[]
        for i,stats in cases.items():
            ratio=stats[method]["total"]/min(stats[m]["total"] for m in ("p4","p8"))
            if i in targets:
                ratios.append(ratio)
                v4.append(stats[method]["total"]/stats["v4"]["total"])
            else:
                result.setdefault("control_time_ratios",{})[i]=ratio
        result.update(native_median_ratio=median(ratios),native_wins=sum(r<1 for r in ratios),
                      v4_median_ratio=median(v4),v4_wins=sum(r<1 for r in v4))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort",choices=["analytic","recovery","topology"])
    parser.add_argument("path",type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.path,args.cohort),indent=2))
