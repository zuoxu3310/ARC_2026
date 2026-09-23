"""Inspect Teghipco figshare .mat to nail down severe-vs-nonsevere label semantics,
and cross-check against our table's wab_aq<50 on the matchable subset.

Run: PYENV_VERSION=data-analysis python teghipco_inspect_labels.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "datasets/arc_ds004884/teghipco_figshare/DLAphasiaSeverityARCSubset.mat"
COARSE = ROOT / "arc_lesion_image_benchmark/data/ArterialAtlas156_features_merged.tsv"

m = loadmat(str(MAT), squeeze_me=True)
print("=== .mat top-level keys ===")
for k, v in m.items():
    if k.startswith("__"):
        continue
    arr = np.asarray(v)
    print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")

subs = np.asarray(m["subsOnly"]).astype(str)
wc2 = np.asarray(m["wabClass2i"]).astype(int)
wc = np.asarray(m["wabClassi"]).astype(int)

print("\n=== subsOnly sample (first 8) ===", subs[:8].tolist())
print("n subs:", len(subs))

import collections
print("\n=== wabClass2i value counts ===", dict(sorted(collections.Counter(wc2.tolist()).items())))
print("=== wabClassi value counts ===", dict(sorted(collections.Counter(wc.tolist()).items())))

# Cross-tab the two label fields
print("\n=== crosstab wabClassi (rows) x wabClass2i (cols) ===")
print(pd.crosstab(pd.Series(wc, name="wabClassi"), pd.Series(wc2, name="wabClass2i")))

# Build subject id without 'sub-' prefix issue. subsOnly are like 'M2002' per task brief.
mat_ids = [s if s.startswith("sub-") else f"sub-{s}" for s in subs]
mat_df = pd.DataFrame({"participant_id": mat_ids, "subsOnly_raw": subs,
                       "wabClass2i": wc2, "wabClassi": wc})

# Load our coarse table (has wab_aq, age, lesion_volume)
tab = pd.read_csv(COARSE, sep="\t")
tab_small = tab[["participant_id", "wab_aq", "age_at_stroke", "lesion_volume"]].copy()

merged = mat_df.merge(tab_small, on="participant_id", how="inner")
unmatched = sorted(set(mat_df["participant_id"]) - set(tab_small["participant_id"]))
print(f"\n=== matching: {len(merged)} of {len(mat_df)} mat subjects matched to our table ===")
print("unmatched mat ids:", unmatched)

# Derive our severe from wab_aq<50
merged["our_severe"] = (merged["wab_aq"] < 50).astype(int)
print("\nour_severe (wab_aq<50) counts on matched:", dict(sorted(collections.Counter(merged['our_severe'].tolist()).items())))

# Now test which mat field best maps to severe.
# Hypothesis A: wabClass2i is the 4-bin WAB severity (1 mild ... 4 most severe) -> need to find which bins = severe.
# Hypothesis B: wabClassi is already the binary severe label.
print("\n=== mean wab_aq per wabClass2i bin (on matched) ===")
print(merged.groupby("wabClass2i")["wab_aq"].agg(["count", "mean", "min", "max"]))
print("\n=== mean wab_aq per wabClassi value (on matched) ===")
print(merged.groupby("wabClassi")["wab_aq"].agg(["count", "mean", "min", "max"]))

# For wabClass2i: test the dichotomization that best matches wab_aq<50.
# If wabClass2i is ordinal 1..4 with higher=more severe, severe might be {3,4} or {4}. Check agreement.
for severe_bins in [{4}, {3, 4}, {2, 3, 4}]:
    pred = merged["wabClass2i"].isin(severe_bins).astype(int)
    agree = (pred == merged["our_severe"]).mean()
    print(f"\nwabClass2i severe={sorted(severe_bins)}: agreement with wab_aq<50 = {agree:.3f}")
    print(pd.crosstab(merged["our_severe"], pred, rownames=["wab_aq<50"], colnames=[f"class2i in {sorted(severe_bins)}"]))

# Also test wabClassi binary direct (assume 1=severe or 2=severe).
for severe_val in sorted(set(wc.tolist())):
    pred = (merged["wabClassi"] == severe_val).astype(int)
    agree = (pred == merged["our_severe"]).mean()
    print(f"\nwabClassi=={severe_val} as severe: agreement with wab_aq<50 = {agree:.3f}")

merged.to_csv(ROOT / "arc_lesion_image_benchmark/results/summary/teghipco_label_crosscheck.csv", index=False)
print("\nwrote teghipco_label_crosscheck.csv with", len(merged), "rows")
