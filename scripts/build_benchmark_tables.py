"""Build full-model manuscript tables from saved scores, without retraining.

The primary grid requires 8 regression / 7 classification models, 10 sets,
and 20 repeats. Optional reference tables retain the earlier LOO regression
and 10-repeat classification protocols separately.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

REG = ["RandomForest", "TabPFN", "XGBoost", "LightGBM", "ElasticNet", "SVR_rbf", "Ridge", "MLP"]
CLF = ["RandomForest", "XGBoost", "LightGBM", "LogReg", "SVM_rbf", "MLP", "TabPFN"]
SETS = {
    "full": ("Full pool", 226),
    "has_dwi": ("Diffusion MRI", 214),
    "has_rsfmri": ("Resting-state fMRI", 192),
    "has_taskfmri": ("Task fMRI", 189),
    "has_flair": ("FLAIR", 135),
    "multimodal_complete": ("Multimodal complete", 162),
    "chronic_365": (r"$\ge365$ days", 183),
    "chronic_180": (r"$\ge180$ days", 225),
    "aq_le90": (r"AQ $\le90$", 170),
    "teghipco_idlist": ("Published-list intersection", 172),
}
LABELS = {"RandomForest": "RF", "TabPFN": "TabPFN", "XGBoost": "XGBoost",
          "LightGBM": "LightGBM", "ElasticNet": "Elastic net", "SVR_rbf": "RBF SVR",
          "SVM_rbf": "RBF SVM", "Ridge": "Ridge", "MLP": "MLP", "LogReg": "Logistic regression",
          "Dummy": "Majority-class baseline"}
END = r" \\"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_grid(root, task, models, hashes):
    files = [root / f"arc_lesion_image_benchmark/results/runs/grid_{task}_{suffix}.csv"
             for suffix in ("fast", "TabPFN")]
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Full manuscript tables require the complete grid: {path}. Run grid --tabpfn.")
        hashes[str(path.relative_to(root))] = digest(path)
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    keys = ["model", "subset", "seed"]
    assert not df.duplicated(keys).any(), f"Duplicate {task} cells"
    actual = set(df[keys].itertuples(index=False, name=None))
    expected = set(itertools.product(models, SETS, range(1000, 1020)))
    assert actual == expected, f"Incomplete {task} Cartesian grid"
    assert df.task.eq(task).all()
    assert all(df.loc[df.subset.eq(sub), "n"].eq(n).all() for sub, (_, n) in SETS.items())
    metrics = ["r", "mae", "rmse", "r2"] if task == "regression" else ["balanced_acc", "auc", "f1_severe", "mcc"]
    assert np.isfinite(df[metrics].to_numpy()).all()
    assert np.allclose(df.score, df[metrics[0]], rtol=0, atol=1e-14)
    summary = df.groupby(["model", "subset"])[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.insert(0, "task", task)
    return df, summary


def table_start(caption, label, columns, small=True, placement="!t"):
    return [r"\begin{table*}[" + placement + "]", r"\caption{" + caption + "}",
            r"\label{" + label + "}", r"\centering" + (r"\footnotesize" if small else r"\small"),
            r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.12}",
            r"\begin{tabular}{" + columns + "}", r"\toprule"]


def save_table(path, lines):
    path.write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]) + "\n")


def check_majority_reference(root, report, hashes):
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    table = root / "arc_lesion_image_benchmark/data/jhu156_features_merged.tsv"
    hashes[str(table.relative_to(root))] = digest(table)
    labels = pd.read_csv(table, sep="\t", index_col="participant_id").wab_aq.dropna()
    rows = []
    for sub, (_, n) in SETS.items():
        path = root / f"arc_lesion_image_benchmark/data/subsets/{sub}_ids.txt"
        hashes[str(path.relative_to(root))] = digest(path)
        ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        # Match s18's stored subset ordering before generating its partitions.
        ids = [pid for pid in ids if pid in labels.index]
        y = (labels.loc[ids].to_numpy() <= 50).astype(int)
        assert len(y) == n
        X = np.zeros((n, 1))  # DummyClassifier uses training labels only.
        for seed in range(1000, 1020):
            pred, prob = np.empty(n), np.empty(n)
            for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                model = DummyClassifier(strategy="most_frequent").fit(X[train], y[train])
                pred[test] = model.predict(X[test])
                prob[test] = model.predict_proba(X[test])[:, 1]
            ba, auc = balanced_accuracy_score(y, pred), roc_auc_score(y, prob)
            assert ba == auc == .5
            rows.append({"subset": sub, "seed": seed, "n": n, "balanced_acc": ba, "auc": auc})
    pd.DataFrame(rows).to_csv(report / "majority_reference_by_repeat.csv", index=False)
    return len(rows)


def full_pool_table(tables, summaries):
    reg, clf = [x.loc[x.subset.eq("full")].set_index("model") for x in summaries]
    lines = table_start(
        r"All-model benchmark on the same 226 patients and 71 JHU-based inputs: eight regression learners, seven classification learners, and a separate majority-class reference. Scores are means $\pm$ sample SD across 20 five-fold OOF repeats; SD describes partition variability, not a confidence interval. Each task shares folds across models. MAE is in WAB-AQ points; AUC denotes area under the receiver operating characteristic curve. Dashes identify task-inapplicable entries. The reference uses training-fold class counts without SMOTE and is excluded from the learned-model grid.",
        "tab:full-pool-benchmark", "lrrrr", small=False)
    lines += [r" & \multicolumn{2}{c}{\textbf{Regression}} & \multicolumn{2}{c}{\textbf{Classification}}" + END,
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
              r"\textbf{Model} & $r$ & \textbf{MAE} & \textbf{Balanced accuracy} & \textbf{AUC}" + END, r"\midrule"]
    for model in REG + ["LogReg"]:
        cm = "SVM_rbf" if model == "SVR_rbf" else model
        cells = ["RBF support vector" if model == "SVR_rbf" else LABELS[model]]
        for frame, key, metric, digits in [(reg, model, "r", 3), (reg, model, "mae", 2),
                                          (clf, cm, "balanced_acc", 3), (clf, cm, "auc", 3)]:
            cells.append("---" if key not in frame.index else
                         f"${frame.loc[key, metric+'_mean']:.{digits}f} \\pm {frame.loc[key, metric+'_std']:.{digits}f}$")
        lines.append(" & ".join(cells) + END)
    lines += [r"\midrule", r"Majority-class reference & --- & --- & $0.500 \pm 0.000$ & $0.500 \pm 0.000$" + END]
    save_table(tables / "benchmark_full_pool.tex", lines)


def cohort_tables(tables, summaries):
    for task, summary, models, metric in zip(("regression", "classification"), summaries, (REG, CLF), ("r", "balanced_acc")):
        caption = (r"Complete refitted " + task + r" grid: " +
                   (r"Pearson $r$" if task == "regression" else "balanced accuracy") +
                   r" averaged across 20 five-fold OOF repeats in each cell. Features and configurations are held fixed; models are refitted within each cohort. The full pool and seven primary sets enter the variance decomposition; secondary sets are reported separately.")
        lines = table_start(caption, "tab:grid-" + task, "lr" + "r" * len(models), placement="p")
        short = {"RandomForest": "RF", "LightGBM": "LGBM", "XGBoost": "XGB", "ElasticNet": "EN",
                 "SVR_rbf": "SVR", "SVM_rbf": "SVM", "LogReg": "LR"}
        lines += [" & ".join([r"\textbf{Cohort}", "$N$"] + [short.get(m, m) for m in models]) + END, r"\midrule"]
        values = summary.set_index(["model", "subset"])
        for sub, (label, n) in SETS.items():
            if sub == "aq_le90":
                lines += [r"\midrule", r"\multicolumn{" + str(2 + len(models)) + r"}{l}{\emph{Secondary sets}}" + END]
            lines.append(" & ".join([label, str(n)] + [f"{values.loc[(m, sub), metric+'_mean']:.3f}" for m in models]) + END)
        save_table(tables / f"benchmark_grid_{task}.tex", lines)


def reference_table(root, tables, report, hashes):
    sources = [root / "arc_wab_aq_benchmark/results/summary/benchmark_loo.csv",
               root / "arc_lesion_image_benchmark/results/summary/benchmark_loo.csv",
               root / "arc_lesion_image_benchmark/results/summary/classification_arterial_coarse.csv",
               root / "arc_lesion_image_benchmark/results/summary/classification_JHU_fine.csv"]
    for path in sources:
        hashes[str(path.relative_to(root))] = digest(path)
    arterial, jhu, ac, jc = [pd.read_csv(p).set_index("model") for p in sources]
    assert set(arterial.index) == set(REG)
    assert set(REG) - {"MLP"} <= set(jhu.index) <= set(REG)
    assert set(ac.index) == set(jc.index) == set(CLF) | {"Dummy"}
    # Recalculate every retained LOO point estimate from its patient predictions.
    checks = []
    for benchmark, frame in [("arc_wab_aq_benchmark", arterial), ("arc_lesion_image_benchmark", jhu)]:
        for model, row in frame.iterrows():
            path = root / benchmark / f"results/runs/{model}_loo_predictions.csv"
            pred = pd.read_csv(path)
            assert len(pred) == 226 and pred.participant_id.nunique() == 226
            r = float(np.corrcoef(pred.y_true, pred.y_pred)[0, 1])
            mae = float(np.abs(pred.y_true - pred.y_pred).mean())
            assert abs(r - row.pearson_r) < .00051 and abs(mae - row.mae) < .0051, (benchmark, model)
            hashes[str(path.relative_to(root))] = digest(path)
            checks.append({"benchmark": benchmark, "model": model, "r": r, "mae": mae})
    (report / "reference_loo_recomputed.json").write_text(json.dumps(checks, indent=2) + "\n")
    lines = table_start(
        r"Earlier fixed-pipeline atlas benchmarks on 226 patients. Regression uses leave-one-out OOF predictions; classification uses the mean score across ten five-fold OOF repeats. Arterial and JHU features use 15 and 71 inputs, respectively. These reference protocols are distinct from the 20-repeat main grid. " +
        ("A dash marks the JHU MLP LOO run for which no result was retained. " if "MLP" not in jhu.index else "") +
        "Classification values are reported as means here.",
        "tab:reference-benchmarks", "lrrrr", placement="p")
    lines += [r" & \multicolumn{2}{c}{\textbf{Arterial}} & \multicolumn{2}{c}{\textbf{JHU}}" + END,
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
              r"\textbf{Regression model} & $r$ & \textbf{MAE} & $r$ & \textbf{MAE}" + END, r"\midrule"]
    for model in REG:
        cells = [LABELS[model]]
        for frame in (arterial, jhu):
            cells += [f"{frame.loc[model, 'pearson_r']:.3f}", f"{frame.loc[model, 'mae']:.2f}"] if model in frame.index else ["---", "---"]
        lines.append(" & ".join(cells) + END)
    lines += [r"\midrule", r"\textbf{Classification model} & \textbf{Balanced accuracy} & \textbf{AUC} & \textbf{Balanced accuracy} & \textbf{AUC}" + END, r"\midrule"]
    for model in CLF + ["Dummy"]:
        cells = [LABELS[model]] + [str(frame.loc[model, metric]).split("±")[0] for frame in (ac, jc) for metric in ("balanced_acc", "auc")]
        lines.append(" & ".join(cells) + END)
    save_table(tables / "benchmark_reference.tex", lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--include-reference", action="store_true", help="also check/build the separately scored earlier benchmarks")
    args = parser.parse_args()
    root = args.root.resolve()
    tables = root / "paper/tables"
    report = args.report_dir or root / "outputs/benchmark_tables"
    tables.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    hashes, grids, summaries = {}, [], []
    for task, models in [("regression", REG), ("classification", CLF)]:
        grid, summary = load_grid(root, task, models, hashes)
        summary.to_csv(report / f"{task}_all_model_cohort_metrics.csv", index=False)
        grids.append(grid)
        summaries.append(summary)
    reference_cells = check_majority_reference(root, report, hashes)
    full_pool_table(tables, summaries)
    cohort_tables(tables, summaries)
    if args.include_reference:
        reference_table(root, tables, report, hashes)
    result = {"evaluations": sum(map(len, grids)), "regression_evaluations": len(grids[0]),
              "classification_evaluations": len(grids[1]), "regression_models": REG,
              "classification_models": CLF, "cohorts": list(SETS), "seeds": list(range(1000, 1020)),
              "complete_cartesian_grid": True, "duplicate_cells": 0, "finite_metrics": True,
              "sd_ddof": 1, "reference_table_included": args.include_reference, "retrained_learned_models": 0,
              "majority_reference_evaluations_separate_from_grid": reference_cells,
              "input_sha256": hashes}
    (report / "benchmark_table_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {result['evaluations']} evaluations; generated 3 primary tables" + (" and the reference table" if args.include_reference else ""))


if __name__ == "__main__":
    main()
