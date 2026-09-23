from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/tables'
f = pd.read_csv(ROOT / 'arc_lesion_image_benchmark/results/frozen_prediction_control_2026-09-22/frozen_prediction_summary.csv')
labels = {'full':'Full pool', 'has_dwi':'Diffusion MRI', 'has_rsfmri':'Resting-state fMRI', 'has_taskfmri':'Task fMRI', 'has_flair':'FLAIR', 'multimodal_complete':'Multimodal complete', 'chronic_365':r'$\ge365$ days', 'chronic_180':r'$\ge180$ days', 'aq_le90':r'AQ $\le90$ (secondary)', 'teghipco_idlist':'Published-list intersection (secondary)'}
rowend = r' \\'
lines = [r'\begin{table*}[!t]', r'\caption{Complete fixed-prediction scores and selected-minus-full differences. Brackets give conditional 95\% patient-bootstrap intervals for the difference. Classification scores and differences are expressed as proportions here; Fig.~\ref{fig:frozen} uses percentage points for differences.}', r'\label{tab:frozen-complete}', r'\centering\footnotesize', r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrr}',r'\toprule',r'\textbf{Scoring set} & \textbf{RF score} & \textbf{RF difference [95\% interval]} & \textbf{XGB score} & \textbf{XGB difference [95\% interval]}'+rowend,r'\midrule']
for task in ['regression','classification']:
    lines += [r'\multicolumn{5}{l}{\emph{'+('Regression: Pearson $r$' if task=='regression' else 'Classification: balanced accuracy')+'}}'+rowend]
    for sub, label in labels.items():
        cells=[label]
        for model in ['RandomForest','XGBoost']:
            r=f.query('task==@task and model==@model and subset==@sub and metric=="score"').iloc[0]
            cells += [f'{r.observed:.3f}', '--' if sub=='full' else f'${r.delta_from_full:+.3f}\ [{r.delta_ci_low:+.3f},{r.delta_ci_high:+.3f}]$']
        lines += [' & '.join(cells)+rowend]
    lines += [r'\midrule']
lines[-1] = r'\bottomrule'
lines += [r'\end{tabular}',r'\end{table*}']
(OUT/'frozen_complete.tex').write_text('\n'.join(lines)+'\n')
lines = [r'\begin{table*}[!t]',r'\caption{All primary sampling comparisons: Holm-adjusted two-sided tail ranks across seven rules within each task and model. Matching uses either patient count alone or count plus severity-bin membership.}',r'\label{tab:matched-complete}',r'\centering\footnotesize',r'\begin{tabular}{lrrrr}',r'\toprule',r' & \multicolumn{2}{c}{\textbf{RF}} & \multicolumn{2}{c}{\textbf{XGBoost}}'+rowend,r'\textbf{Eligibility rule} & \textbf{Count} & \textbf{Count + severity} & \textbf{Count} & \textbf{Count + severity}'+rowend,r'\midrule']
for task in ['regression','classification']:
    lines += [r'\multicolumn{5}{l}{\emph{'+task.title()+'}}'+rowend]
    for sub,label in list(labels.items())[1:8]:
        cells=[label]
        for model in ['RandomForest','XGBoost']:
            r=f.query('task==@task and model==@model and subset==@sub and metric=="score"').iloc[0]
            cells += [f'{r.size_holm_tail_rank:.3f}',f'{r.aqbin_holm_tail_rank:.3f}']
        lines += [' & '.join(cells)+rowend]
    lines += [r'\midrule']
lines[-1]=r'\bottomrule'
lines += [r'\end{tabular}',r'\end{table*}']
(OUT/'matched_complete.tex').write_text('\n'.join(lines)+'\n')
print('Generated two manuscript tables from frozen_prediction_summary.csv')
