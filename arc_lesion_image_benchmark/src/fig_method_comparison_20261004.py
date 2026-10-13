#!/usr/bin/env python3
"""Generate image-case figure and complete table from saved aggregate CSVs.

Running the script does both; plot_method_comparison() alone writes only the figure files.
"""
from pathlib import Path
import json, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/hourglass-revision-2026-10-04/method-figure'
OUT.mkdir(parents=True,exist_ok=True)
a=ROOT/'outputs/cnn-same-input-2026-10-03/summary.csv'
b=ROOT/'outputs/xgboost-optuna-2026-10-04/summary.csv'
initial=pd.read_csv(a).set_index('method'); tuned=pd.read_csv(b).set_index('method')
allrows=pd.concat([initial,tuned.loc[['XGB_Optuna','XGB_Optuna_threshold','XGB_fixed_threshold']]])
labels={'CNN':'CNN','RandomForest':'Random forest','XGBoost':'XGBoost: fixed','XGB_fixed_threshold':'XGBoost: threshold','XGB_Optuna':'XGBoost: tuned',
'LightGBM':'LightGBM','LogReg':'Logistic regression','MLP':'Multilayer perceptron','Majority':'Majority reference','RandomForest_CNNfit':'RF: CNN fit patients',
'SVM_linear':'Linear SVM','SVM_rbf':'Radial SVM','SVM_selected':'Inner-selected SVM','SVM_linear_CNNfit':'Linear SVM: CNN fit patients','SVM_rbf_CNNfit':'Radial SVM: CNN fit patients','XGB_Optuna_threshold':'XGBoost: tuned + threshold'}
order=['CNN','RandomForest','XGBoost','XGB_fixed_threshold','XGB_Optuna']
# Colour and marker follow the learner encoding of the other figures (CNN teal, RF blue,
# XGBoost rust squares); an open square marks the selected threshold and a dashed
# segment the tuned parameters.
CNN_C,RF_C,XGB_C='#24786E','#246B8E','#AA5B2A'
style={'CNN':('D',CNN_C,CNN_C,'-'),'RandomForest':('o',RF_C,RF_C,'-'),'XGBoost':('s',XGB_C,XGB_C,'-'),
       'XGB_fixed_threshold':('s',XGB_C,'white','-'),'XGB_Optuna':('s',XGB_C,XGB_C,(0,(3,1.6)))}

def panel_labels(fig,axes,letters,gap_pt=5.0):
    """Centre (a), (b), (c) under each axes, a fixed gap below the lowest axis text."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    y=(min(ax.get_tightbbox(renderer).y0 for ax in axes)-gap_pt*fig.dpi/72)/fig.bbox.height
    for ax,letter in zip(axes,letters):
        box=ax.get_position()
        fig.text((box.x0+box.x1)/2,y,f'({letter})',ha='center',va='top',fontsize=8.2,fontfamily='Times New Roman')

def plot_method_comparison(images=ROOT/'paper/images'):
    # Sizes are set for printing at 97.6% (7.16 in figure on a 6.99 in text width).
    plt.rcParams.update({'font.family':'Arial','font.size':9.5,'xtick.labelsize':9.4,'ytick.labelsize':9.5,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.45),sharey=True)
    fig.subplots_adjust(left=.225,right=.985,top=.885,bottom=.215,wspace=.22)
    for ax,metric,title in zip(axes,['F1','BA','AUC'],['Severe-class F1','Balanced accuracy','AUC']):
        for y,key in enumerate(order):
            row=allrows.loc[key];marker,col,face,ls=style[key]
            bars=ax.errorbar(row[metric+'_mean'],y,xerr=row[metric+'_sd'],fmt=marker,color=col,markerfacecolor=face,
                             markeredgewidth=1.1,markersize=4.6 if marker=='D' else 5,capsize=3,elinewidth=1.2)[2]
            bars[0].set_linestyle(ls)
        ax.set_title(title,fontsize=9.5,fontweight='bold',pad=8)
        ax.set_yticks(range(len(order)));ax.set_yticklabels([labels[k] for k in order]);ax.invert_yaxis() if not ax.yaxis_inverted() else None
        ax.grid(axis='x',color='#e3e9ed',lw=.6);ax.set_axisbelow(True)
        ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#afbcc4');ax.tick_params(axis='y',length=0)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_ylim(4.55,-.55)
    axes[0].set_xlim(.49,.72);axes[0].set_xticks([.50,.60,.70])
    axes[1].set_xlim(.64,.82);axes[1].set_xticks([.65,.70,.75,.80])
    axes[2].set_xlim(.77,.88);axes[2].set_xticks([.78,.82,.86])
    panel_labels(fig,axes,'abc')
    for ext in ['pdf','svg','png']:
        fig.savefig(Path(images)/f'method_comparison.{ext}',dpi=300)
    plt.close(fig)

def write_complete_table():
    orderfull=['CNN','RandomForest','SVM_linear','SVM_rbf','SVM_selected','XGBoost','LightGBM','LogReg','MLP','RandomForest_CNNfit','SVM_linear_CNNfit','SVM_rbf_CNNfit','Majority','XGB_Optuna','XGB_Optuna_threshold','XGB_fixed_threshold']
    tex=r'''\begin{table*}[!t]
\caption{Complete same-input classification results on 173 author-released records.
Values are mean $\pm$ SD across twenty six-fold repeats. All methods share image
information and outer test patients; fitting and selection procedures are specified
in Appendix~\ref{app:cnn}. The three final rows are exploratory additions.}
\label{tab:cnn-complete}
\centering\small
\begin{tabular}{lccc}
\toprule
Method & Severe-class F1 & Balanced accuracy & AUC\\
\midrule
'''
    for i,key in enumerate(orderfull):
        if i in [9,13]:tex+='\\midrule\n'
        row=allrows.loc[key]; vals=[f'${row[m+"_mean"]:.3f} \\pm {row[m+"_sd"]:.3f}$' for m in ['F1','BA','AUC']]
        tex+=labels[key]+' & '+' & '.join(vals)+r'\\'+'\n'
    tex+='\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
    (ROOT/'paper/tables/cnn_complete.tex').write_text(tex)
    (OUT/'provenance.json').write_text(json.dumps({'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a,b]},'table_rows':len(orderfull),'figure_rows':order,'uncertainty':'standard deviation over repeated outer partitions, not a confidence interval'},indent=2)+'\n')

if __name__=='__main__':
    plot_method_comparison()
    write_complete_table()
    print('Generated 3 figure formats and 16-row full table from two saved CSV files.')
