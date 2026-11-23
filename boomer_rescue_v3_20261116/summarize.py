import json,statistics,sys
from pathlib import Path
from run import hashes
HERE=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (HERE/'results.jsonl').read_text().splitlines()]
lock=json.loads((HERE/'LOCK.json').read_text());assert lock['hashes']==hashes()
assert len({(r['case'],r['tag'],r['rep']) for r in rows})==len(rows)
full='--complete' in sys.argv
if full:assert len(rows)==200
med=statistics.median
table={};summary=[]
for c in lock['cases']:
    name=c['case'];table[name]={}
    expected=3 if c['n']==511 else 2
    for tag in lock['methods']:
        rr=[r for r in rows if r['case']==name and r['tag']==tag]
        if full:assert len(rr)==expected and {r['rep'] for r in rr}==set(range(expected))
        ok=[r for r in rr if r['ok']]
        d=dict(repeats=len(rr),successes=len(ok),failures=len(rr)-len(ok))
        if ok:
            for key in ['total_s','cpu_s','setup_s','solve_s','iterations','residual']:
                d[key]=med([r[key] for r in ok])
            d['wall_range']=[min(r['total_s'] for r in ok),max(r['total_s'] for r in ok)]
            d['max_residual']=max(r['residual'] for r in ok)
            d['iterations_range']=[min(r['iterations'] for r in ok),max(r['iterations'] for r in ok)]
            d['peak_rss_gb']=max(r['peak_rss_bytes'] for r in ok)/1e9
            for key in ['dim','nnz','max_row_nnz','ritz_steps']:
                if key in ok[0]:d[key]=med([r[key] for r in ok])
        table[name][tag]=d
    valid=lambda tag:table[name][tag].get('successes')==expected
    if all(valid(t) for t in ['base','p8','adaptive','weighted','direct2','direct16','direct32','local','hats']):
        d=table[name];bm=min(['base','p8'],key=lambda t:d[t]['total_s'])
        dm=min(['direct2','direct16','direct32'],key=lambda t:d[t]['total_s'])
        summary.append(dict(case=name,baseline=bm,best_direct=dm,
                            adaptive_ratio=d['adaptive']['total_s']/d[bm]['total_s'],
                            adaptive_cpu_ratio=d['adaptive']['cpu_s']/min(d['base']['cpu_s'],d['p8']['cpu_s']),
                            weighted_direct_ratio=d['weighted']['total_s']/d[dm]['total_s'],
                            weighted_direct_cpu_ratio=d['weighted']['cpu_s']/min(d[t]['cpu_s'] for t in ['direct2','direct16','direct32']),
                            weighted_local_ratio=d['weighted']['total_s']/d['local']['total_s'],
                            weighted_hats_ratio=d['weighted']['total_s']/d['hats']['total_s']))
aggregate=dict(records=len(rows),cases=len(table),successes=sum(r['ok'] for r in rows),
               failures=sum(not r['ok'] for r in rows),hashes_match=True,
               timeouts=sum(bool(r.get('timeout')) for r in rows),
               budget_skipped=sum(bool(r.get('budget_skipped')) for r in rows),
               completed_solver_runs=sum('residual' in r for r in rows),
               max_successful_residual=max((r['residual'] for r in rows if r['ok']),default=None),
               max_successful_manufactured_error=max((r['error'] for r in rows if r['ok'] and r.get('error') is not None),default=None),
               complete_comparison_cases=len(summary),per_case=summary)
if summary:
    for key in ['adaptive_ratio','adaptive_cpu_ratio','weighted_direct_ratio','weighted_direct_cpu_ratio',
                'weighted_local_ratio','weighted_hats_ratio']:
        aggregate[key+'_median']=med(r[key] for r in summary)
        aggregate[key+'_wins']=sum(r[key]<1 for r in summary)
    if full:
        aggregate['practical_criterion_pass']=aggregate['adaptive_ratio_wins']>=6 and aggregate['adaptive_ratio_median']<=.8
        aggregate['lifting_vs_direct_criterion_pass']=aggregate['weighted_direct_ratio_wins']>=6
(HERE/'summary.json').write_text(json.dumps(table,indent=2)+'\n')
(HERE/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
tags=list(lock['methods'])
lines=['# Round 3 measurements','',
       'Seconds include setup + solve. Parentheses are outer PCG iterations. Failures are retained.',
       'All medians use 3 repeats at 511², 2 at 1023². A failed group is not ranked as a winner.','',
       '| Case | '+' | '.join(tags)+' |', '|---|'+'---:|'*len(tags)]
for case,d in table.items():
    cells=[]
    for tag in tags:
        v=d[tag]
        if v['failures']:cells.append(f"FAIL {v['failures']}/{v['repeats']}")
        elif 'total_s' in v:cells.append(f"{v['total_s']:.3f} ({v['iterations']:.0f})")
        else:cells.append('pending')
    lines.append('| '+case+' | '+' | '.join(cells)+' |')
lines+=['','## Case comparisons','','| Case | Adaptive / faster Boomer | Weighted96 / best direct96 | Best direct rank | Weighted / local | Weighted / hats |',
        '|---|---:|---:|---|---:|---:|']
for r in summary:
    lines.append(f"| {r['case']} | {r['adaptive_ratio']:.3f} | {r['weighted_direct_ratio']:.3f} | {r['best_direct']} | {r['weighted_local_ratio']:.3f} | {r['weighted_hats_ratio']:.3f} |")
(HERE/'RESULTS_TABLE.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(aggregate,indent=2))
