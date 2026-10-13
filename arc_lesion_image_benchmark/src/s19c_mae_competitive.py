# One-off: variance decomposition of mean absolute error (MAE, insensitive to AQ range),
# strong-model set (with / without TabPFN), regression.
import pandas as pd, numpy as np, os
import statsmodels.formula.api as smf
from pathlib import Path
PKG = Path(__file__).resolve().parents[1]
RUNS = PKG / "results/runs"
SUMM = PKG / "results/summary"
frames=[pd.read_csv(f"{RUNS}/grid_regression_{t}.csv") for t in ["fast","TabPFN"] if os.path.exists(f"{RUNS}/grid_regression_{t}.csv")]
df=pd.concat(frames,ignore_index=True)
PRIMARY=["full","has_dwi","has_rsfmri","has_taskfmri","has_flair","multimodal_complete","chronic_365","chronic_180"]
df=df[df["subset"].isin(PRIMARY)]
def eta2(d,y="mae"):
    m=smf.ols(f"{y} ~ C(subset)+C(model)+C(seed)",data=d).fit()
    # Type-I SS via anova
    import statsmodels.api as sm
    aov=sm.stats.anova_lm(m,typ=1)
    ss=aov["sum_sq"]; tot=ss.sum()
    return {k.replace("C(","").replace(")",""):round(ss[k]/tot,4) for k in ss.index}
COMP=["RandomForest","XGBoost","LightGBM"]
print("=== Regression | MAE variance decomposition | strong-model set ===")
print("Without TabPFN (RF/XGB/LGBM):", eta2(df[df.model.isin(COMP)]))
print("With TabPFN (RF/XGB/LGBM/TabPFN):", eta2(df[df.model.isin(COMP+["TabPFN"])]))
print("\nReference: variance decomposition of r on the same set (raw r, not Fisher-z):")
print("Without TabPFN:", eta2(df[df.model.isin(COMP)],"r"))
print("With TabPFN:", eta2(df[df.model.isin(COMP+["TabPFN"])],"r"))
