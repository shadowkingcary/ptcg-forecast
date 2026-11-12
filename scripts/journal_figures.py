# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Draw journal figures from the protocol, saved predictions, and result ledger.
The five-model figure retains the original reliability-plot text sizes in a
larger panel layout. Counts, scores, intervals, and source hashes are checked.
"""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,os,math,sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.journal_comparisons import holdout_comparisons
os.environ.setdefault("MPLCONFIGDIR",str(RESULTS/"matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
ROOT=Path(__file__).resolve().parents[1]
OUT=FIGURES
P=RESULTS
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':8,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})

def design():
 fig=plt.figure(figsize=(7.1,4.1))
 ax=fig.add_axes([.03,.57,.94,.39]);ax.axis('off');ax.set_xlim(-.01,1.01)
 ax.text(0,.95,'A. Five models, one scoring framework',fontsize=9,va='bottom')
 labels=['Archetype information\n1. Logistic baseline\n2. Archetype-pair estimate','Player records\n3. Overall records\n4. Archetype-specific records','Joint estimation\n5. Joint model\nDeck and player effects']
 for k,label in enumerate(labels):
  x=.005+k*.345
  ax.add_patch(FancyBboxPatch((x,.39),.297,.49,boxstyle='round,pad=.008',facecolor='#eff4f8',edgecolor='#60788b',linewidth=.8))
  ax.text(x+.1485,.635,label,ha='center',va='center',linespacing=1.6,fontsize=8.2)
  if k<2:ax.annotate('',xy=(x+.337,.635),xytext=(x+.306,.635),arrowprops={'arrowstyle':'->','lw':1,'color':'#455a64'})
 ax.text(.5,.17,'Same matches, monthly cutoffs, and tournament-weighted log loss',ha='center',va='center',fontsize=8.1)
 ax.text(.5,.015,'Additional checks: calibration, holdout, and update frequency',ha='center',va='center',linespacing=1.4,fontsize=8,color='#455a64')
 ax=fig.add_axes([.03,.14,.94,.32]);ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(-.5,2.1)
 ax.text(0,2.02,'B. Chronological evaluation windows (schematic)',fontsize=9,va='bottom')
 colors={'fit':'#e2e5e8','tune':'#f3d9b6','test':'#c9dfed','holdout':'#c9e3d6'}
 blocks=[(0,1.75,'fit','Initial fitting\nDec 2023–Mar 2024'),(1.75,1.65,'tune','Tuning validation\nApr–Jun 2024'),(3.4,5.2,'test','Historical evaluation\nJul 2024–Sep 2026: 27 months'),(8.6,1.4,'holdout','Holdout\nOct 2026')]
 for x,w,c,label in blocks:
  ax.add_patch(plt.Rectangle((x,1.05),w-.035,.75,fc=colors[c],ec='white'))
  ax.text(x+w/2,1.425,label,ha='center',va='center',fontsize=7.5,linespacing=1.5)
 ax.text(0,.3,'Recalibration',fontsize=8,va='center')
 for x,w,c,label in [(3.4,1.9,'tune','Fit calibration\nJul–Dec 2024'),(5.3,3.3,'test','Later evaluation\nJan 2025–Sep 2026: 21 months'),(8.6,1.4,'holdout','Same\nholdout')]:
  ax.add_patch(plt.Rectangle((x,-.08),w-.035,.75,fc=colors[c],ec='white'))
  ax.text(x+w/2,.295,label,ha='center',va='center',fontsize=7.3,linespacing=1.4)
 fig.legend(handles=[Patch(fc=colors[k],label=v) for k,v in [('fit','Initial training'),('tune','Selection / calibration'),('test','Evaluation'),('holdout','Holdout')]],loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.015),fontsize=7.5)
 fig.savefig(OUT/'journal_experimental_design.png',dpi=300)
 plt.close(fig)


def reliability():
    ledger_path=P/'full_october_2026_run/paper_results.json'
    ledger=json.loads(ledger_path.read_text())
    audit={"ledger_sha256":hashlib.sha256(ledger_path.read_bytes()).hexdigest(),"sources":{},"panels":{}}
    specs=[('M1','Logistic baseline','FINAL_hier_full_predictions.csv','m1_prob'),
           ('Card-only','Archetype-pair','FINAL_cardonly_vs_playerskill_predictions.csv','cardonly_prob'),
           ('Overall','Overall records','FINAL_playerskill_generalonly_predictions.csv','prob'),
           ('Player-Skill','Archetype-specific records','FINAL_playerskill_pooled_predictions.csv','pooled_prob'),
           ('Joint','Joint model','FINAL_joint_predictions.csv','prob')]
    colors=['#397da8','#708894','#438c78','#cc853e','#8b65a4']
    fig,axes=plt.subplots(2,3,figsize=(11,7.2))
    identity={};sharpness=[]
    for k,(ax,(name,label,filename,col),color) in enumerate(zip(axes.flat,specs,colors)):
        counts,sums,wins=defaultdict(int),defaultdict(float),defaultdict(float)
        months=set();seen=set();total=0;absolute_distance=0.
        path=P/filename;digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==ledger['sources'][filename],f'Stale result ledger: {filename}'
        with path.open() as stream:
            for row in csv.DictReader(stream):
                mid=row['match_id'];y=int(row['observed_y']);month=row['month']
                assert mid not in seen
                seen.add(mid)
                if k==0:identity[mid]=(y,month)
                else:assert identity.get(mid)==(y,month),f'Mismatched match/outcome/month: {filename}'
                p=float(row[col]);assert 0<=p<=1
                b=min(int(p*10),9)
                counts[b]+=1;sums[b]+=p;wins[b]+=y;total+=1
                absolute_distance+=abs(p-.5);months.add(month)
        assert total==len(identity)==926811 and min(months)=='2024-07' and max(months)=='2026-09' and len(months)==27
        order=sorted(counts);mad=absolute_distance/total
        rmse=math.sqrt(sum((sums[b]-wins[b])**2/counts[b] for b in order)/total)
        summary=ledger['primary'][name]
        assert math.isclose(mad,summary['mad'],abs_tol=1e-12)
        assert math.isclose(rmse,summary['calibration_rmse'],abs_tol=1e-12)
        assert [(b,counts[b]) for b in order]==[(b['bin'],b['n']) for b in summary['bins']]
        ax.plot([0,1],[0,1],'k--',linewidth=1,alpha=.6)
        ax.scatter([sums[b]/counts[b] for b in order],[wins[b]/counts[b] for b in order],
                   s=[max(20,2000*counts[b]/total) for b in order],alpha=.75,color=color)
        ax.set_xlim(0,1);ax.set_ylim(-.02,1.02)
        ax.set_xticks([0,.2,.4,.6,.8,1]);ax.set_yticks([0,.2,.4,.6,.8,1])
        ax.set_xlabel('Mean predicted probability',fontsize=10)
        ax.set_ylabel('Observed win frequency',fontsize=10)
        ax.tick_params(labelsize=10)
        ax.set_title(f'{chr(65+k)}. {label}',fontsize=12)
        ax.text(.035,.96,f'Calibration RMSE: {rmse:.4f}',transform=ax.transAxes,va='top',fontsize=8)
        if name=='M1':
            for b,offset in [(1,(7,7)),(8,(7,-12))]:
                if counts[b]:ax.annotate(f'n={counts[b]}',(sums[b]/counts[b],wins[b]/counts[b]),xytext=offset,textcoords='offset points',fontsize=8)
        sharpness.append(mad*100)
        audit['sources'][filename]=digest
        audit['panels'][label]={'matches':total,'mad':mad,'calibration_rmse':rmse,
                              'bins':[{'bin':b,'n':counts[b],'mean_pred':sums[b]/counts[b],'observed':wins[b]/counts[b]} for b in order]}
    ax=axes.flat[5]
    ax.barh(range(5),sharpness,color=colors,height=.57)
    ax.set_yticks(range(5),['Logistic','Pair','Overall','Archetype-\nspecific','Joint'],fontsize=10)
    ax.invert_yaxis();ax.set_xlim(0,12.8);ax.set_xticks([0,3,6,9,12]);ax.tick_params(labelsize=10)
    for i,v in enumerate(sharpness):ax.text(v+.15,i,f'{v:.2f}',va='center',fontsize=8)
    ax.set_xlabel('Mean distance from 50%\n(percentage points)',fontsize=10)
    ax.set_title('F. Sharpness summary',fontsize=12)
    fig.subplots_adjust(left=.065,right=.985,bottom=.105,top=.94,hspace=.48,wspace=.38)
    fig.savefig(OUT/'journal_sharpness_calibration.png',dpi=300);plt.close(fig)
    audit['identical_match_outcome_month_sets']=True
    (P/'full_october_2026_run/figure_sources.json').write_text(json.dumps(audit,indent=2)+'\n')


def gains():
    path=P/'full_october_2026_run/paper_results.json';r=json.loads(path.read_text())
    holdout=holdout_comparisons()
    contrasts=[('Adding player records\nOverall records\nvs archetype-pair','Overall vs Card-only','Overall-record control vs Card-only'),
               ('Adding archetype-specific detail\nArchetype-specific records\nvs overall records','Player-Skill vs Overall','Player-Skill vs Overall-record control'),
               ('Joint estimation\nJoint model vs\narchetype-specific records','Joint vs Player-Skill','Joint model vs Player-Skill')]
    fig=plt.figure(figsize=(11,4.6))
    axes=[fig.add_axes([.285,.20,.325,.59]),fig.add_axes([.775,.20,.205,.59])]
    periods=[('Historical evaluation','#397da8','o',.13),('Holdout','#c67b32','s',-.13)]
    audit={'ledger_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'contrasts':[],
           'sources':{artifact_label(p):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [P/'FINAL_october_predictions.csv',P/'history_depth/summary.json',
                                ROOT/'scripts/journal_comparisons.py',Path(__file__)]}}
    for i,(label,historical,october) in enumerate(contrasts):
        h=r['comparisons']['primary'][historical]['gain_ci'];o=holdout[october]
        audit['contrasts'].append({'comparison':label,'historical_gain_ci':h,'october_gain_ci':o})
        for values,(_,color,marker,offset) in zip([h,o],periods):
            point,lo,hi=values;assert lo<=point<=hi
            for j,ax in enumerate(axes):
                if j==1 and i==0:continue
                ax.errorbar(point,2-i+offset,xerr=[[point-lo],[hi-point]],fmt=marker,color=color,
                            capsize=3,markersize=5,elinewidth=1.2)
    axes[0].set_yticks([2,1,0],[x[0] for x in contrasts],fontsize=10)
    axes[0].set_ylim(-.5,2.5);axes[0].set_xlim(-.003,.023)
    axes[0].set_xticks([0,.005,.010,.015,.020]);axes[0].set_title('A. Main comparisons',fontsize=12,pad=14)
    axes[1].set_yticks([1,0],['Archetype\ndetail','Joint\nestimation'],fontsize=10)
    axes[1].set_ylim(-.5,1.5);axes[1].set_xlim(-.002,.0065)
    axes[1].set_xticks([-.002,0,.002,.004,.006]);axes[1].set_title('B. Smaller gains\n(expanded scale)',fontsize=12,pad=9)
    for ax in axes:
        ax.axvline(0,color='#555555',linewidth=.8,linestyle='--')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'));ax.tick_params(axis='x',labelsize=10)
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.legend(handles=[Line2D([0],[0],color=color,marker=marker,label=label,linewidth=1.2) for label,color,marker,_ in periods],
               loc='upper center',bbox_to_anchor=(.60,.99),ncol=2,frameon=False,fontsize=9)
    fig.text(.62,.055,'Log-loss reduction (nats; positive favors the first model)',ha='center',fontsize=10)
    fig.savefig(OUT/'journal_main_gains.png',dpi=300);plt.close(fig)
    (P/'full_october_2026_run/gains_figure_sources.json').write_text(json.dumps(audit,indent=2)+'\n')

if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    design();reliability();gains()
    print('Updated design, five-model sharpness/calibration, and main-gains figures')
