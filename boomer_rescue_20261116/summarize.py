"""Generate auditable tables and scientific plots from all locked validation records."""
import json
from pathlib import Path
from collections import defaultdict
import statistics as st

HERE=Path(__file__).resolve().parent

def read(name):
    p=HERE/name
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

def casekey(r): return r['fam'],r['n'],r['chi'],tuple(r['p']),r['rhs']
def identity(r): return casekey(r)+(r['tag'],r['rep'])

def collect():
    rows=read('validate511.jsonl')+read('validate1023.jsonl')+read('validate2047.jsonl')
    replacements={identity(r):r for r in read('remeasure1023.jsonl')}
    rows=[replacements.get(identity(r),r) for r in rows]
    rows+=read('controls_repeats.jsonl')+read('topology.jsonl')
    return rows

def stats(rows):
    grouped=defaultdict(lambda:defaultdict(list))
    for r in rows: grouped[casekey(r)][r['tag']].append(r)
    result=[]
    for key,g in sorted(grouped.items()):
        if set(g)!={'base','p8','rescue'}: continue
        med={tag:st.median(r['total_s'] for r in rr) for tag,rr in g.items()}
        winner=min(['base','p8'],key=lambda tag:med[tag]); b=g[winner]; c=g['rescue']
        row=dict(fam=key[0],n=key[1],chi=key[2],p=list(key[3]),rhs=key[4],reference=winner,
                 baseline_s=med[winner],rescue_s=med['rescue'],ratio=med['rescue']/med[winner],
                 speedup=med[winner]/med['rescue'],times=med,
                 baseline_iterations=st.median(r['iterations'] for r in b),
                 rescue_iterations=st.median(r['iterations'] for r in c),
                 base_setup=st.median(r['setup_s'] for r in b),rescue_setup=st.median(r['setup_s'] for r in c),
                 base_solve=st.median(r['solve_s'] for r in b),rescue_solve=st.median(r['solve_s'] for r in c),
                 baseline_rss_gb=max(r['peak_rss_bytes'] for r in b)/1e9,
                 rescue_rss_gb=max(r['peak_rss_bytes'] for r in c)/1e9,
                 cpu_ratio=st.median(r['cpu_s'] for r in c)/st.median(r['cpu_s'] for r in b),
                 triggered=any(r.get('triggered',False) for r in c),dim=max(r.get('dim',0) for r in c),
                 max_residual=max(r['residual'] for rr in g.values() for r in rr),
                 max_error=max(r['error'] for rr in g.values() for r in rr),
                 repetitions={tag:len(rr) for tag,rr in g.items()},
                 all_ok=all(r['ok'] for rr in g.values() for r in rr))
        result.append(row)
    return result

def table(rows):
    lines=['| 场景 | 网格 | 各向异性/对比度 | RHS | Boomer 秒 | 候选秒 | 时间比 | 迭代数 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        chi='1' if r['fam']=='poisson' else f"{r['chi']:.0e}"
        lines.append(f"| {r['fam']} | {r['n']}² | {chi} | {r['rhs']} | {r['baseline_s']:.3f} | {r['rescue_s']:.3f} | {r['ratio']:.3f} | {r['baseline_iterations']:g} → {r['rescue_iterations']:g} |")
    return '\n'.join(lines)

def plots(ss):
    import os,tempfile
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'boomer_rescue_mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for ax,fam in zip(axs,['circ','disk']):
        rr=sorted([r for r in ss if r['fam']==fam and r['chi']==1e5 and r['rhs']==1],key=lambda r:r['n'])
        x=np.arange(len(rr)); w=.34
        ax.bar(x-w/2,[r['baseline_s'] for r in rr],w,label='Faster Boomer baseline',color='#808891')
        ax.bar(x+w/2,[r['rescue_s'] for r in rr],w,label='Spectral-coordinate rescue',color='#2b8c79')
        ax.set_yscale('log'); ax.set_xticks(x,[str(r['n'])+'²' for r in rr]); ax.set_xlabel('Grid')
        ax.set_ylabel('Setup + solve (seconds, log scale)'); ax.set_title(fam+' | anisotropy = 100,000')
        for i,r in enumerate(rr):ax.text(i+w/2,r['rescue_s']*1.12,f"{r['speedup']:.2f}×",ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    axs[0].legend(fontsize=8,loc='upper left')
    fig.savefig(HERE/'scaling.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10, max(5,len(ss)*.31)),layout='constrained')
    y=np.arange(len(ss)); vals=[r['ratio'] for r in ss]
    labels=[f"{r['fam']}  {r['n']}²  χ={(1 if r['fam']=='poisson' else r['chi']):.0e}  RHS{r['rhs']}" for r in ss]
    ax.barh(y,vals,color=['#2b8c79' if v<1 else '#bd6a46' for v in vals])
    ax.set_yticks(y,labels,fontsize=8); ax.invert_yaxis(); ax.axvline(1,color='#30343b',linestyle='--')
    ax.set_xlabel('Total time / faster of two Boomer baselines (lower is better)')
    for yi,v in zip(y,vals):ax.text(v+.015,yi,f'{v:.2f}',va='center',fontsize=8)
    ax.set_xlim(0,max(vals)*1.15); ax.set_title('All locked and topology tests; negative results retained')
    fig.savefig(HERE/'all_cases.png',dpi=170); plt.close(fig)

if __name__=='__main__':
    rows=collect(); ss=stats(rows)
    (HERE/'summary.json').write_text(json.dumps(ss,indent=2)+'\n')
    (HERE/'RESULTS_TABLE.md').write_text(table(ss)+'\n')
    main=[r for r in ss if r['fam'] in ['circ','disk']]
    high=[r for r in main if r['chi']==1e5]
    aggregate=dict(unique_cases=len(ss),measurements=len(rows),failures=sum(not r['ok'] for r in rows),
                   all_case_median_ratio=st.median(r['ratio'] for r in ss),
                   circ_disk_median_ratio=st.median(r['ratio'] for r in main),
                   high_anisotropy_median_ratio=st.median(r['ratio'] for r in high),
                   max_residual=max(r['residual'] for r in rows),max_error=max(r['error'] for r in rows),
                   worst_ratio=max(r['ratio'] for r in ss))
    (HERE/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2)); print(table(ss))
    if '--plots' in __import__('sys').argv: plots(ss)
