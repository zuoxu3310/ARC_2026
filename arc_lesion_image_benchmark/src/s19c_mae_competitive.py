# 一次性: 用"平均误差(MAE,不吃量程)"做方差分解,强模型口径(含/不含 TabPFN),回归。
import pandas as pd, numpy as np, os
import statsmodels.formula.api as smf
RUNS="results/runs"; SUMM="results/summary"
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
print("=== 回归 · 平均误差(MAE)方差分解 · 强模型口径 ===")
print("不含 TabPFN (RF/XGB/LGBM):", eta2(df[df.model.isin(COMP)]))
print("含 TabPFN (RF/XGB/LGBM/TabPFN):", eta2(df[df.model.isin(COMP+["TabPFN"])]))
print("\n参考: 同口径 r(原始,未Fisher-z) 方差分解:")
print("不含 TabPFN:", eta2(df[df.model.isin(COMP)],"r"))
print("含 TabPFN:", eta2(df[df.model.isin(COMP+["TabPFN"])],"r"))
