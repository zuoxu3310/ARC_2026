"""Exploratory audit sensitivity using existing scores; no model refitting."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'arc_lesion_image_benchmark/src'))
import s19_variance_decomp as a
OUT=ROOT/'outputs/model_sensitivity'
OUT.mkdir(parents=True,exist_ok=True)
rows=[]; rankings=[]; provenance={}
for task in ['regression','classification']:
 d=a.load_grid(task)
 primary=d[~d.category.isin(['target_defined','exact_persons'])]
 means=d.query('subset=="full"').groupby('model').score.mean().sort_values(ascending=False)
 rankings.extend({'task':task,'model':m,'full_score':float(v),'gap_to_best':float(means.max()-v)} for m,v in means.items())
 for gap in [.03,.05,.075,.10]:
  chosen=means[means >= means.max()-gap-1e-12].index.tolist()
  sub=primary[primary.model.isin(chosen)]
  assert sub.groupby(['subset','model']).size().eq(20).all()
  e=a.eta_sq(sub)
  # Independent closed-form sum-of-squares check for this balanced additive grid.
  y=sub.analysis_y
  grand=y.mean();total=float(((y-grand)**2).sum())
  for factor in ['subset','model','seed']:
   groups=sub.groupby(factor).analysis_y.agg(['mean','size'])
   expected=float((groups['size']*(groups['mean']-grand)**2).sum()/total)
   assert abs(e[f'eta2_{factor}']-expected)<1e-10
  rows.append({'task':task,'full_pool_gap_threshold':gap,'n_models':len(chosen),'models':'; '.join(chosen),**e,'cohort_gt_model':e['eta2_subset']>e['eta2_model']})
 for kind in ['fast','TabPFN']:
  p=ROOT/f'arc_lesion_image_benchmark/results/runs/grid_{task}_{kind}.csv'
  if p.exists():
   provenance[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
pd.DataFrame(rows).to_csv(OUT/'model_set_sensitivity.csv',index=False)
pd.DataFrame(rankings).to_csv(OUT/'full_pool_model_ranking.csv',index=False)
(OUT/'model_set_sensitivity_provenance.json').write_text(json.dumps({'status':'Exploratory analysis specified during manuscript review, 2026-09-22','selection':'All tested models within threshold of best mean full-pool score, including boundary','tasks':['regression','classification'],'thresholds':[.03,.05,.075,.10],'scope':'Full pool plus all seven primary rules','metric_for_selection':'raw r or balanced accuracy, mean of 20 OOF-repeat scores','ANOVA':'Fisher z for regression; raw balanced accuracy for classification','independent_check':'balanced-grid closed-form SS matched Type-II ANOVA within 1e-10','source_sha256':provenance},indent=2)+'\n')
print(pd.DataFrame(rows).round(5).to_string(index=False))
print(pd.DataFrame(rankings).round(5).to_string(index=False))
