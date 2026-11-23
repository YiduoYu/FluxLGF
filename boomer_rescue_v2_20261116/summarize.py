"""Audit and summarize all locked validation, retaining failures and baseline wins."""
import json,hashlib,statistics as st
from pathlib import Path
from collections import defaultdict
HERE=Path(__file__).resolve().parent
def read(name):
    p=HERE/name
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def key(r):return (r['fam'],r['n'],r['chi'],tuple(r['p']),r['rhs'],r['rhs_seed'])
def group_name(r):
    if r['fam'] in ['circ','disk']:return 'hard' if r['chi']==1e5 else 'controls'
    return 'topology' if r['fam'] in ['ellipse','two_vortices'] else 'controls'
def collect():return sum([read(name) for name in ['validate511.jsonl','validate1023.jsonl','validate2047.jsonl','controls_extra.jsonl','controls_moderate_extra.jsonl']],[])
def stats(rows):
    grouped=defaultdict(lambda:defaultdict(list))
    for r in rows:grouped[key(r)][r['tag']].append(r)
    ss=[]
    for k,g in sorted(grouped.items()):
        if set(g)!={'base','p8','adaptive','fixed96'}:continue
        r=dict(fam=k[0],n=k[1],chi=k[2],p=list(k[3]),rhs=k[4],rhs_seed=k[5])
        r['group']=group_name(r); r['all_ok']=all(x['ok'] for xx in g.values() for x in xx)
        r['times']={tag:st.median(x['total_s'] for x in xx) if all(x.get('ok') for x in xx) else None for tag,xx in g.items()}
        good=[tag for tag in ['base','p8'] if r['times'][tag] is not None]
        r['reference']=min(good,key=lambda tag:r['times'][tag]) if good else None
        r['baseline_s']=r['times'][r['reference']] if good else None
        for tag in ['adaptive','fixed96']:
            r[tag+'_ratio']=r['times'][tag]/r['baseline_s'] if r['times'][tag] is not None and r['baseline_s'] else None
        r['adaptive_over_fixed']=r['times']['adaptive']/r['times']['fixed96'] if all(r['times'][k] for k in ['adaptive','fixed96']) else None
        r['iterations']={tag:[x.get('iterations') for x in xx] for tag,xx in g.items()}
        r['ritz_steps']={tag:[x.get('ritz_steps',0) for x in xx] for tag,xx in g.items() if tag in ['adaptive','fixed96']}
        r['triggered']=any(x.get('triggered') for x in g['adaptive'])
        r['quality_met']=all(x.get('quality_met',True) for x in g['adaptive'])
        r['repeats']={tag:len(xx) for tag,xx in g.items()}
        r['max_residual']=max(x.get('residual',float('inf')) for xx in g.values() for x in xx)
        r['setup']={tag:st.median(x.get('setup_s',float('inf')) for x in xx) for tag,xx in g.items()}
        r['solve']={tag:st.median(x.get('solve_s',float('inf')) for x in xx) for tag,xx in g.items()}
        r['peak_rss_gb']={tag:max(x.get('peak_rss_bytes',0) for x in xx)/1e9 for tag,xx in g.items()}
        r['cpu']={tag:st.median(x.get('cpu_s',float('inf')) for x in xx) for tag,xx in g.items()}
        r['adaptive_cpu_ratio']=r['cpu']['adaptive']/min(r['cpu'][tag] for tag in ['base','p8'])
        r['time_ranges']={tag:[min(x.get('total_s',float('inf')) for x in xx),max(x.get('total_s',float('inf')) for x in xx)] for tag,xx in g.items()}
        ss.append(r)
    return ss
def median_or_none(xs):return st.median(xs) if xs else None
def tables(ss):
    lines=['| 场景 | 网格 | χ | 几何 | RHS | Boomer 秒 | 自适应秒 | 固定96步秒 | 自适应/Boomer | Ritz步数 |',
           '|---|---:|---:|---|---|---:|---:|---:|---:|---:|']
    ps=json.loads((HERE/'VALIDATION_CASES.json').read_text())['p']
    fmt=lambda x:'FAIL' if x is None else f'{x:.3f}'
    for r in ss:
        p='A' if r['p']==ps[0] else 'B';rhs='随机 forcing' if r['rhs']==3 else '新平滑'
        chi=1 if r['fam']=='poisson' else r['chi']
        steps='/'.join(map(str,sorted(set(r['ritz_steps']['adaptive']))))
        lines.append(f"| {r['fam']} | {r['n']}² | {chi:.0e} | {p} | {rhs} | {fmt(r['baseline_s'])} | {fmt(r['times']['adaptive'])} | {fmt(r['times']['fixed96'])} | {fmt(r['adaptive_ratio'])} | {steps} |")
    return '\n'.join(lines)+'\n'
def plots(ss):
    import os,tempfile
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'boomer_rescue_mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    rr=[r for r in ss if r['adaptive_ratio'] is not None]
    fig,ax=plt.subplots(figsize=(11, max(6,len(rr)*.29)),layout='constrained')
    ys=np.arange(len(rr)); vals=[r['adaptive_ratio'] for r in rr]
    ps=json.loads((HERE/'VALIDATION_CASES.json').read_text())['p']
    labels=[f"{r['fam']} {r['n']}² χ={(1 if r['fam']=='poisson' else r['chi']):.0e} {'A' if r['p']==ps[0] else 'B'} {'random' if r['rhs']==3 else 'smooth'}" for r in rr]
    ax.barh(ys,vals,color=['#278775' if v<1 else '#ba6b46' for v in vals])
    ax.set_yticks(ys,labels,fontsize=8);ax.invert_yaxis();ax.axvline(1,ls='--',c='#30343b')
    for i,v in enumerate(vals):ax.text(v+.015,i,f'{v:.2f}',va='center',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',pad=.2))
    ax.set_xlim(0,max(vals)*1.15);ax.set_xlabel('Adaptive total time / faster Boomer baseline (lower is better)')
    ax.set_title('Round 2: all locked independent validation cases')
    fig.savefig(HERE/'all_cases.png',dpi=170);plt.close(fig)
    big=[r for r in ss if r['n']==2047]
    if big:
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        xx=np.arange(len(big));w=.25
        for j,tag in enumerate(['reference','fixed96','adaptive']):
            vals=[r['baseline_s'] if tag=='reference' else r['times'][tag] for r in big]
            ax.bar(xx+(j-1)*w,vals,w,label={'reference':'Faster Boomer','fixed96':'Fixed 96 (same gate)','adaptive':'Adaptive Ritz budget'}[tag],color=['#818991','#ddaa66','#278775'][j])
            for x,v in zip(xx+(j-1)*w,vals):ax.text(x,v+.8,f'{v:.1f}',ha='center',fontsize=9)
        ax.set_xticks(xx,[r['fam'] for r in big]);ax.set_ylabel('Setup + solve (seconds)')
        ax.set_title('Independent 2047² cases | χ=100,000 | new smooth RHS');ax.legend(fontsize=8)
        fig.savefig(HERE/'largest.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    rows=collect();ss=stats(rows);lock=json.loads((HERE/'LOCK.json').read_text())
    assert len({key(r)+(r['tag'],r['rep']) for r in rows})==len(rows),'Duplicate raw measurement identities'
    hashes={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in lock['source_hashes']}
    assert hashes==lock['source_hashes'],'Locked source/dependency changed'
    for r in rows:
        for k,v in lock['methods'][r['tag']].items():assert r[k]==v,(r,k)
    if '--complete' in __import__('sys').argv:
        assert len(rows)==304 and len(ss)==30,(len(rows),len(ss))
        for r in ss:
            expected=2 if r['group']=='hard' else 3
            assert all(v==expected for v in r['repeats'].values())
    original=stats([r for r in rows if r['rep']<2])
    agg=dict(cases=len(ss),measurements=len(rows),failures=sum(not r['ok'] for r in rows),
             max_residual=max(r.get('residual',float('inf')) for r in rows),
             max_manufactured_error=max(r['error'] for r in rows if r.get('error') is not None),
             overall_median_ratio=median_or_none([r['adaptive_ratio'] for r in ss if r['adaptive_ratio'] is not None]),
             overall_median_cpu_ratio=median_or_none([r['adaptive_cpu_ratio'] for r in ss]),
             quality_not_met_cases=sum(not r['quality_met'] for r in ss),
             groups={g:dict(cases=sum(r['group']==g for r in ss),median_ratio=median_or_none([r['adaptive_ratio'] for r in ss if r['group']==g and r['adaptive_ratio'] is not None]),
                            max_ratio=max([r['adaptive_ratio'] for r in ss if r['group']==g and r['adaptive_ratio'] is not None],default=None)) for g in ['hard','controls','topology']},
             triggered_adaptive_vs_fixed_median=median_or_none([r['adaptive_over_fixed'] for r in ss if r['triggered'] and r['adaptive_over_fixed'] is not None]),
             locked_hashes_passed=True)
    agg['original_two_repeat_controls']=dict(
        median_ratio=median_or_none([r['adaptive_ratio'] for r in original if r['group']=='controls' and r['adaptive_ratio'] is not None]),
        max_ratio=max([r['adaptive_ratio'] for r in original if r['group']=='controls' and r['adaptive_ratio'] is not None],default=None))
    largest=[r for r in ss if r['n']==2047]
    agg['criteria']=dict(
        accuracy=agg['failures']==0,
        hard_median_25pct=agg['groups']['hard']['median_ratio'] is not None and agg['groups']['hard']['median_ratio']<=.75,
        every_control_overhead_10pct=agg['groups']['controls']['max_ratio'] is not None and agg['groups']['controls']['max_ratio']<=1.10,
        both_largest_25pct=len(largest)==2 and all(r['adaptive_ratio'] is not None and r['adaptive_ratio']<=.75 for r in largest))
    agg['all_criteria_point_estimates_pass']=all(agg['criteria'].values())
    (HERE/'summary.json').write_text(json.dumps(ss,indent=2)+'\n')
    (HERE/'aggregate.json').write_text(json.dumps(agg,indent=2)+'\n')
    (HERE/'RESULTS_TABLE.md').write_text(tables(ss))
    print(json.dumps(agg,indent=2));print(tables(ss))
    if '--plots' in __import__('sys').argv:plots(ss)
