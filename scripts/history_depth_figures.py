# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Render the history-depth figure from saved, reconciled subgroup results."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(RESULTS/'matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


def main():
    source = RESULTS/'history_depth/summary.json'
    result = json.loads(source.read_text())
    assert result['validation']['all_sample_scores_and_intervals_match_existing_ledger']
    groups = ['0', '1-9', '10-49', '50+']
    periods = [('primary', 'Historical evaluation', '#397da8', 'o', .13),
               ('october', 'Holdout', '#c67b32', 's', -.13)]
    contrasts = [('Overall vs pair', 'A. Overall records vs archetype-pair'),
                 ('Player-Skill vs overall', 'B. Archetype-specific records\nvs overall records')]
    plt.rcParams.update({'font.family':'DejaVu Sans', 'axes.spines.top':False,
                         'axes.spines.right':False, 'font.size':10})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    plotted = []
    for ax, (contrast, title) in zip(axes, contrasts):
        endpoints = [0.]
        for period, _, color, marker, offset in periods:
            rows = {r['group']:r for r in result['periods'][period]['groups'] if r['kind']=='depth'}
            for i, group in enumerate(groups):
                c = rows[group]['comparisons'][contrast]
                assert c['zero_support_replicates']==0 and c['ci'] is not None
                point, (lo, hi) = c['gain'], c['ci']
                y = 3-i+offset
                ax.hlines(y,lo,hi,color=color,linewidth=1.3)
                ax.plot([lo,hi],[y,y],linestyle='none',marker='|',color=color,markersize=6)
                ax.plot(point,y,marker=marker,color=color,markersize=5)
                endpoints += [lo,hi,point]
                plotted.append({'period':period,'group':group,'comparison':contrast,
                                'gain':point,'ci':[lo,hi],'matches':rows[group]['matches']})
        width = max(endpoints)-min(endpoints)
        ax.set_xlim(min(endpoints)-width*.1,max(endpoints)+width*.1)
        ax.set_ylim(-.5,3.5)
        ax.set_yticks([3,2,1,0],['0 (one or both unseen)','1-9','10-49','50+'])
        ax.set_title(title,fontsize=12,pad=14)
        ax.axvline(0,color='#555555',linewidth=.8,linestyle='--')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.tick_params(axis='both',labelsize=10)
        ax.tick_params(axis='y',length=0)
        ax.spines['left'].set_visible(False)
    axes[0].set_ylabel('Smaller prior-match count of the two players',fontsize=10)
    axes[1].set_yticklabels(['0','1-9','10-49','50+'])
    fig.legend(handles=[Line2D([0],[0],color=c,marker=m,label=l,linewidth=1.2)
                        for _,l,c,m,_ in periods],loc='upper center',ncol=2,
               bbox_to_anchor=(.58,.995),frameon=False,fontsize=9)
    fig.text(.58,.055,'Log-loss reduction (nats; positive favors the first model)',ha='center',fontsize=10)
    fig.text(.58,.015,'Panels use different horizontal scales.',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.235,right=.985,bottom=.19,top=.79,wspace=.34)
    out=FIGURES/'journal_history_depth.png';out.parent.mkdir(exist_ok=True)
    fig.savefig(out,dpi=300);plt.close(fig)
    audit={'summary_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
           'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'figure_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'plotted':plotted}
    (source.parent/'figure_sources.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
