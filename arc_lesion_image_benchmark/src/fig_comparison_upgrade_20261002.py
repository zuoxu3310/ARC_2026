#!/usr/bin/env python
"""Plot audited training/scoring contrasts and generate the MRRI summary table.

main() does both; plot_training_scoring() alone writes only the figure files and its plot-data record.
"""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/comparison-upgrade-2026-10-02'
PAPER=ROOT/'paper'
DATA=ROOT/'arc_lesion_image_benchmark/results'
ARC=DATA/'fixed_test_training_2026-10-02'
MRRI=DATA/'mrri_replication_2026-10-02'
PRIMARY=['has_dwi','has_rsfmri','has_taskfmri','has_flair','multimodal_complete','chronic_365','chronic_180']
LABELS=['Diffusion MRI','Resting-state fMRI','Task fMRI','FLAIR','Multimodal complete','≥365 days','≥180 days']
# Colour and marker encode the model (as in the other figures); fill and line style encode the contrast.
INK='#172D3B';MUTED='#59626A';RF='#246B8E';XGB='#AA5B2A'
MINUS=lambda t:t.replace('-','−')

def panel_labels(fig,axes,letters,gap_pt=5.0):
    """Centre (a), (b) under each axes, a fixed gap below the lowest axis text."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    y=(min(ax.get_tightbbox(renderer).y0 for ax in axes)-gap_pt*fig.dpi/72)/fig.bbox.height
    for ax,letter in zip(axes,letters):
        box=ax.get_position()
        fig.text((box.x0+box.x1)/2,y,f'({letter})',ha='center',va='top',fontsize=8.2,fontfamily='Times New Roman')

def plot_training_scoring(images=PAPER/'images',metadata=OUT/'training-scoring-figure-data.json'):
    for kind in ['arc','mrri']:
        check=OUT/f'{kind}-independent-verification.json'
        if check.exists():  # verification records are kept with the original run, not the package
            assert json.loads(check.read_text())['passed']
    d=pd.read_csv(ARC/'factorial_contrasts.csv')
    # Sizes are set for printing at 97.6% (7.16 in figure on a 6.99 in text width).
    plt.rcParams.update({'font.family':'Arial','font.size':9.5,'axes.labelsize':9.5,'axes.titlesize':10,
        'xtick.labelsize':9.4,'ytick.labelsize':9.5,
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
        'axes.edgecolor':'#8A9BA6','axes.linewidth':.65,'xtick.major.width':.65,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(7.16,4.75),sharey=True)
    fig.subplots_adjust(left=.225,right=.975,top=.865,bottom=.185,wspace=.16)
    # Training restriction: filled marker, solid interval. Scoring restriction: open marker, dashed interval.
    specs=[('RandomForest','training_at_full_scoring','o',RF,-.27,True),
           ('XGBoost','training_at_full_scoring','s',XGB,-.09,True),
           ('RandomForest','scoring_at_full_training','o',RF,.09,False),
           ('XGBoost','scoring_at_full_training','s',XGB,.27,False)]
    shown=[]
    for ax,task,title,factor in zip(axes,['regression','classification'],['Correlation','Balanced accuracy'],[1,100]):
        for j in range(len(PRIMARY)-1):
            ax.axhline(j+.5,color='#E3E7EA',lw=.55,zorder=0)
        ax.axvline(0,color='#647B89',lw=.75,zorder=1)
        for model,contrast,marker,color,offset,filled in specs:
            for j,subset in enumerate(PRIMARY):
                r=d.query('task==@task and model==@model and subset==@subset and metric=="score" and contrast==@contrast').iloc[0]
                ax.plot([r.ci_low*factor,r.ci_high*factor],[j+offset,j+offset],color=color,lw=1.05,
                        ls='-' if filled else (0,(3,1.6)),zorder=2)
                ax.scatter([r.delta*factor],[j+offset],s=19 if filled else 17,marker=marker,
                           color=color if filled else 'white',edgecolor='white' if filled else color,
                           linewidth=.55 if filled else 1.0,zorder=3)
                shown.append(r.to_dict())
        ax.set_title(title,loc='left',pad=8,fontweight='bold')
        ax.set_yticks(range(len(PRIMARY)),LABELS)
        ax.tick_params(axis='y',length=0,pad=7)
        ax.set_ylim(len(PRIMARY)-.52,-.52)
        if factor==1:
            ax.set_xlim(-.17,.065);ax.set_xticks([-.15,-.10,-.05,0,.05])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:'0' if abs(x)<1e-9 else MINUS(f'{x:+.2f}')))
            ax.set_xlabel('Change in mean Pearson r',labelpad=6)
        else:
            ax.set_xlim(-12,6);ax.set_xticks([-10,-5,0,5])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:'0' if abs(x)<1e-9 else MINUS(f'{x:+.0f}')))
            ax.set_xlabel('Change in balanced accuracy\n(percentage points)',labelpad=6)
        ax.tick_params(axis='x',length=3)
    for t in axes[0].get_yticklabels():
        if t.get_text()=='FLAIR':t.set_fontweight('bold')
    handles=[Line2D([],[],marker='o',color=RF,lw=1.05,markersize=4.6,markeredgecolor='white',markeredgewidth=.55,label='Random forest'),
             Line2D([],[],marker='s',color=XGB,lw=1.05,markersize=4.4,markeredgecolor='white',markeredgewidth=.55,label='XGBoost'),
             Line2D([],[],marker='o',color=MUTED,lw=1.05,markersize=4.6,markeredgecolor='white',markeredgewidth=.55,label='Training restriction'),
             Line2D([],[],marker='o',color=MUTED,lw=1.05,ls=(0,(3,1.6)),markersize=4.2,markerfacecolor='white',
                    markeredgewidth=1.0,label='Scoring restriction')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.6,.995),ncol=4,frameon=False,
               fontsize=9.4,columnspacing=1.3,handlelength=2.2,handletextpad=.5)
    panel_labels(fig,axes,'ab')
    for ext in ['pdf','svg','png']:
        fig.savefig(Path(images)/f'training_scoring_controls.{ext}',dpi=300)
    plt.close(fig)
    metadata_out={'data_sha256':hashlib.sha256((ARC/'factorial_contrasts.csv').read_bytes()).hexdigest(),
        'plot_rows':shown,'width_inches':7.16,'height_inches':4.75,'vector_pdf':True,'bootstrap':'pointwise conditional paired 95%'}
    Path(metadata).write_text(json.dumps(metadata_out,indent=2)+'\n')

def write_mrri_table():
    d=pd.read_csv(MRRI/'factorial_contrasts.csv');manifest=pd.read_csv(MRRI/'subset_manifest.csv').set_index('subset')
    names={'chronic_6m':'At least six months','chronic_12m':'At least twelve months','has_synonymy':'Synonymy assessment',
       'has_ppvt':'Peabody vocabulary test','has_cct':'Camel and Cactus test','has_prt':'Philadelphia repetition test',
       'has_nwr':'Nonword repetition test','has_isr':'Immediate serial recall','has_ppt':'Pyramids and Palm Trees'}
    lines=[r'\begin{table*}[tp]',r'\caption{Independent MRRI behavioral extension: every eligibility rule. Differences are selected-minus-full scores with predictions fixed. Brackets contain pointwise paired conditional 95\% intervals. Classification values are percentage points. BA denotes balanced accuracy. The full pool contains 183 patients, including 25 severe cases. Assessment rules require an observed value, without adding that value as a predictor.}',
           r'\label{tab:mrri-scoring}',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
           r'Eligibility rule & $N$ & Severe & RF: $\Delta r$ & XGBoost: $\Delta r$ & RF: $\Delta$ BA & XGBoost: $\Delta$ BA\\',r'\midrule']
    for sub,name in names.items():
        vals=[]
        for task,model in [('regression','RandomForest'),('regression','XGBoost'),('classification','RandomForest'),('classification','XGBoost')]:
            r=d.query('subset==@sub and task==@task and model==@model and metric=="score" and contrast=="scoring_at_full_training"').iloc[0]
            fac,dp=(1,3) if task=='regression' else (100,2)
            vals.append(f'${r.delta*fac:+.{dp}f}$ [{r.ci_low*fac:.{dp}f}, {r.ci_high*fac:.{dp}f}]')
        row=manifest.loc[sub]
        lines.append(f'{name} & {int(row.n)} & {int(row.n_severe)} & '+' & '.join(vals)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}'])
    (PAPER/'tables/mrri_scoring.tex').write_text('\n'.join(lines)+'\n')

def main():
    plot_training_scoring()
    write_mrri_table()
    print('Training/scoring figure and MRRI table generated from verified summaries.')

if __name__=='__main__':main()
