"""Render the complete matched-policy table and paired-contrast figure from saved results.

Running the script does both; plot_decision_controls() alone writes only the figure files.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parents[3]
TASK=Path(__file__).resolve().parents[1]
p=TASK/'evidence/image-controls'
means=pd.read_csv(p/'mean-metrics.csv');contrasts=pd.read_csv(p/'paired-contrasts.csv')
methods={'CNN':'CNN','RF_inverse':'RF, inverse frequency','RF_cnnratio':'RF, CNN cost ratio',
         'XGB_inverse':'XGB, inverse frequency','XGB_cnnratio':'XGB, CNN cost ratio','XGB_developed':'Developed XGB, CNN cost ratio'}
# Figure row names spell out the learners as in the other figures.
figure_labels={'RF_inverse':'Random forest, inverse frequency','RF_cnnratio':'Random forest, CNN cost ratio',
               'XGB_inverse':'XGBoost, inverse frequency','XGB_cnnratio':'XGBoost, CNN cost ratio',
               'XGB_developed':'Developed XGBoost, CNN cost ratio'}
RF_C,XGB_C,MUTED='#246B8E','#AA5B2A','#59626A'

def write_table():
 lines=[r'\begin{table*}[!t]',r'\caption{Image comparisons with aligned fitting, validation, and test patients. Every predeclared reference and decision rule is shown. Values average 20 repeated six-fold evaluations. The last column is CNN minus the corresponding reference under the same threshold rule, with a paired conditional 95\% interval. The two classical weight schemes have equal mean weight; CNN cost ratio matches the positive-to-negative cost ratio.}',r'\label{tab:matched-image}',r'\centering\small',r'\setlength{\tabcolsep}{5pt}',r'\begin{tabular}{@{}llrrrrl@{}}',r'\toprule',r'Procedure & Threshold rule & F1 & BA & AUC & FN / FP & CNN minus reference F1 \\',r'\midrule']
 for method,label in methods.items():
  for decision,name in [('fixed_0.5','0.5'),('validation_F1','Validation F1')]:
   r=means[(means.method==method)&(means.decision==decision)].iloc[0]
   gap='---'
   if method!='CNN':
    c=contrasts[(contrasts.left_method=='CNN')&(contrasts.right_method==method)&(contrasts.left_decision==decision)&(contrasts.right_decision==decision)&(contrasts.metric=='F1')].iloc[0]
    gap=f'${c.delta:.3f}$ [$ {c.ci_low:.3f}, {c.ci_high:.3f} $]'
   lines.append(f'{label if decision=="fixed_0.5" else ""} & {name} & {r.F1:.3f} & {r.balanced_accuracy:.3f} & {r.AUC:.3f} & {r.FN:.1f} / {r.FP:.1f} & {gap}'+r' \\')
 lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
 (ROOT/'paper/tables/matched_image_controls.tex').write_text('\n'.join(lines)+'\n')

def plot_decision_controls(images=ROOT/'paper/images'):
 # Arial as in the other data figures; sizes print at >= 9 pt after scaling 7.16 in to the 6.99 in text width.
 plt.rcParams.update({'font.family':'Arial','font.size':9.5,'axes.labelsize':9.5,'xtick.labelsize':9.4,
                      'ytick.labelsize':9.5,'legend.fontsize':9.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
 fig,ax=plt.subplots(figsize=(7.16,3.2));fig.subplots_adjust(left=.35,right=.975,bottom=.17,top=.865)
 references=list(methods)[1:]
 # Colour gives the reference learner (RF blue, XGBoost rust); shape gives the threshold rule.
 for decision,offset,marker in [('fixed_0.5',-.14,'s'),('validation_F1',.14,'o')]:
  for i,m in enumerate(references):
   c=contrasts[(contrasts.left_method=='CNN')&(contrasts.right_method==m)&(contrasts.left_decision==decision)&(contrasts.right_decision==decision)&(contrasts.metric=='F1')].iloc[0]
   ax.errorbar(c.delta,i+offset,xerr=[[c.delta-c.ci_low],[c.ci_high-c.delta]],fmt=marker,color=RF_C if m.startswith('RF') else XGB_C,
               markersize=5.5,capsize=3,lw=1.3)
 ax.axvline(0,color='#747474',ls='--',lw=.8)
 ax.set_yticks(range(len(references)),[figure_labels[m] for m in references]);ax.invert_yaxis()
 ax.set_xlabel('CNN F1 minus classical-reference F1')
 ax.set_xlim(-.035,.36);ax.set_xticks([0,.1,.2,.3]);ax.grid(axis='x',color='#e5e5e5',lw=.7)
 ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:'0' if abs(x)<1e-9 else f'{x:.1f}'))
 ax.set_axisbelow(True);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=8)
 handles=[Line2D([],[],marker='s',color=MUTED,markersize=5.5,lw=1.3,label='Threshold 0.5'),
          Line2D([],[],marker='o',color=MUTED,markersize=5.5,lw=1.3,label='Common validation-F1 rule')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.6625,.995),ncol=2,frameon=False,columnspacing=2.5)
 for ext in ['pdf','png','svg']:fig.savefig(Path(images)/f'matched_decision_controls.{ext}',dpi=300)
 plt.close(fig)

if __name__=='__main__':
 write_table()
 plot_decision_controls()
