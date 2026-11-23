import os
os.environ['MPLCONFIGDIR']='/tmp/fd_lgf_v3_mpl'
os.environ['XDG_CACHE_HOME']='/tmp/fd_lgf_v3_cache'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'summary.json').read_text())
tags=['adaptive','weighted','direct2','direct16','direct32','hats','local','asa']
labels=['Adaptive\nlift','Fixed 96\nlift','Direct\n2 modes','Direct\n16 modes',
        'Direct\n32 modes','Unweighted\nhats','Geometric\nPOU','PyAMG\nalpha-SA']
cases=list(data);values=np.full((len(cases),len(tags)),np.nan)
for i,c in enumerate(cases):
    d=data[c];base=min(d[t]['total_s'] for t in ['base','p8'] if not d[t]['failures'])
    for j,t in enumerate(tags):
        if not d[t]['failures']:values[i,j]=d[t]['total_s']/base
fig,ax=plt.subplots(figsize=(12,6.5),layout='constrained')
cmap=plt.get_cmap('RdYlGn_r').copy();cmap.set_bad('#c8c8c8')
im=ax.imshow(np.log2(values),cmap=cmap,norm=TwoSlopeNorm(vmin=-1.5,vcenter=0,vmax=2.0),aspect='auto')
for i in range(len(cases)):
    for j in range(len(tags)):
        val=values[i,j];ax.text(j,i,'FAIL' if not np.isfinite(val) else f'{val:.2f}',ha='center',va='center',fontsize=11)
ax.set_xticks(range(len(tags)),labels);ax.set_yticks(range(len(cases)),[c.replace('_1023',' 1023²').replace('_511',' 511²').replace('_',' ') for c in cases])
ax.set_title('Round 3: total time / faster Boomer baseline\nAll setup included; lower is better; failures receive no speedup',pad=16)
fig.supxlabel('Direct = prolonged coarse-grid Ritz vectors; alpha-SA = untuned PyAMG configuration',fontsize=10)
cb=fig.colorbar(im,ax=ax,ticks=[-1,0,1,2],shrink=.7);cb.ax.set_yticklabels(['0.5×','1×','2×','4×+'])
fig.savefig(HERE/'comparison.png',dpi=170);plt.close(fig)

cp=HERE/'CONFIRMATION_SUMMARY.json'
if cp.exists():
    conf=json.loads(cp.read_text())
    if conf['triggered']:
        fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
        for i,r in enumerate(conf['per_case']):
            ax.scatter(r['pair_ratios'],[i-.08,i+.08],color='#969da4',s=30,zorder=3)
            ax.scatter([r['ratio']],[i],marker='D',color='#228774' if r['ratio']<1 else '#b96948',s=55,zorder=4)
        ax.axvline(1,color='#454c53',ls='--',lw=1.5)
        ax.set_yticks(range(8),[r['case'].replace('_1023',' 1023²').replace('_511',' 511²').replace('_',' ') for r in conf['per_case']]);ax.invert_yaxis()
        ax.grid(axis='x',alpha=.2);ax.set_xlabel('Fixed96 sparse lift / selected prolonged coarse-Ritz correction time')
        ax.set_title('Within-session paired timing check\nGray: two individual pairs; diamond: ratio of medians; lower is better')
        fig.savefig(HERE/'confirmation.png',dpi=170);plt.close(fig)

diagpath=HERE/'diagnostics.jsonl'
if diagpath.exists():
    diag=[json.loads(l) for l in diagpath.read_text().splitlines()]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharex=True,sharey=True)
    for ax,d in zip(axes.ravel(),diag):
        for tag,color in [('direct2','#747c84'),('direct16','#5479ac'),('direct32','#9e77b1'),('local','#d59d58'),('weighted','#228774')]:
            row=next(r for r in d['comparisons'] if r['tag']==tag)
            ax.plot(range(1,17),row['projection_error'],label=tag,color=color,marker='.',lw=1.5)
        ax.set_title(d['fam']);ax.set_ylim(0,1.05);ax.grid(alpha=.18)
    axes[1,0].set_xlabel('Fine-grid Ritz probe index');axes[1,1].set_xlabel('Fine-grid Ritz probe index')
    axes[0,0].set_ylabel('Relative energy projection error');axes[1,0].set_ylabel('Relative energy projection error')
    axes[0,0].legend(ncols=2,fontsize=8)
    fig.suptitle('Mechanism diagnostic at 255²; approximate Ritz probes, not exact eigenvectors')
    fig.savefig(HERE/'projection.png',dpi=170);plt.close(fig)
