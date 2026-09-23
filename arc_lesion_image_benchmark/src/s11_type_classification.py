#!/usr/bin/env python
"""
s11_type_classification.py

把"失语类型"做成与 s3(severe 二分类)同协议的基准。三种目标(--task):
  fluent    : 非流畅 {Broca,Global,TranscorticalMotor,TranscorticalMixed}
              vs 流畅 {Wernicke,Conduction,Anomic,TranscorticalSensory}  (1=非流畅)
  broca     : Broca vs 其它失语类型(去 None/NaN)                        (1=Broca)
  multiclass: 6 类失语类型(Broca/Anomic/Conduction/Global/Wernicke/TCM,去 None/NaN)
              —— 探索性, 少数类(Wernicke/TCM 个位数)指标不可靠, 不作门面

协议完全对齐 s3: 折内 StandardScaler -> SMOTE -> 分类器(imblearn Pipeline 防泄露);
分层重复 K 折(5x10), 每 repeat 汇总折外预测算指标, 跨 repeat 取均值±标准差。
multiclass 的 SMOTE 用自适应 k_neighbors, 训练折某类样本太少则该折跳过 SMOTE(计数上报)。

二分类指标: balanced_acc / AUC / F1(pos) / MCC / pos 精确率召回率(同 s3)。
multiclass 指标: balanced_acc(=macro recall) / macro_F1 / accuracy / macro_AUC(OvR)。

副产物(供 s12/s14 等下游):
  results/runs/type_{task}_{feature}_{model}_oof.csv      每人池化折外预测(真值/预测/概率)
  results/runs/type_{task}_{feature}_{model}_perrepeat.csv 逐 repeat 指标向量(供配对检验)

运行(非 TabPFN):
  PYENV_VERSION=data-analysis python src/s11_type_classification.py \
      --table data/jhu156_features_merged.tsv --name JHU_fine --task fluent
TabPFN 单独进程 + 硬超时(见 CLAUDE.md):
  perl -e 'alarm 1800; exec @ARGV' PYENV_VERSION=data-analysis python src/s11_type_classification.py \
      --table data/ArterialAtlas156_features_merged.tsv --name arterial_coarse --task fluent --models TabPFN
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
# 少数类(TCM=3)< n_splits 的分层警告无害(分层仍成立, 个别折该类为 0), 静音保持日志干净
warnings.filterwarnings("ignore", message="The least populated class in y has only")
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             accuracy_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

PKG = Path(__file__).resolve().parents[1]
PARTICIPANTS = PKG.parent / "datasets" / "arc_ds004884" / "participants.tsv"
SEED = 42
SPARSE_DROP_FRAC = 0.10
N_SPLITS, N_REPEATS = 5, 10
ALL_MODELS = ["Dummy", "LogReg", "SVM_rbf", "RandomForest", "XGBoost", "LightGBM", "MLP", "TabPFN"]

NONFLUENT = {"Broca", "Global", "TranscorticalMotor", "TranscorticalMixed"}
FLUENT = {"Wernicke", "Conduction", "Anomic", "TranscorticalSensory"}
NON_TYPE = {"None", "nan", "NaN", ""}   # 非失语 / 缺失, 类型任务里全部剔除


def estimator(name, multiclass=False):
    if name == "Dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "LogReg":
        return LogisticRegression(max_iter=5000, random_state=SEED)
    if name == "SVM_rbf":
        return SVC(kernel="rbf", probability=True, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             n_jobs=-1, verbosity=0,
                             eval_metric=("mlogloss" if multiclass else "logloss"))
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                              n_jobs=-1, verbose=-1)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=SEED)
    if name == "TabPFN":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device="cpu")
    raise ValueError(name)


def smote_for(y_train):
    """自适应 SMOTE: k_neighbors = min(5, 最小类样本-1); 最小类<2 则返回 None(不重采样)。"""
    counts = np.bincount(y_train)
    nz = counts[counts > 0]
    smallest = nz.min()
    if smallest < 2:
        return None
    k = int(min(5, smallest - 1))
    return SMOTE(random_state=SEED, k_neighbors=k)


def build_pipe(name, y_train, multiclass=False):
    if name == "Dummy":
        return ImbPipeline([("clf", estimator(name, multiclass))])
    sm = smote_for(y_train)
    steps = [("scaler", StandardScaler())]
    if sm is not None:
        steps.append(("smote", sm))
    steps.append(("clf", estimator(name, multiclass)))
    return ImbPipeline(steps), (sm is None)


def load_xy(table_path, task):
    df = pd.read_csv(table_path, sep="\t", index_col="participant_id")
    pp = pd.read_csv(PARTICIPANTS, sep="\t", index_col="participant_id")["wab_type"]
    df = df.join(pp, how="left")
    df["wab_type"] = df["wab_type"].astype(str)

    if task == "fluent":
        keep_mask = df["wab_type"].isin(NONFLUENT | FLUENT)
        sub = df[keep_mask].copy()
        y = sub["wab_type"].isin(NONFLUENT).astype(int).values     # 1 = 非流畅
        classes = ["fluent", "non-fluent"]
    elif task == "broca":
        keep_mask = ~df["wab_type"].isin(NON_TYPE) & df["wab_type"].notna()
        sub = df[keep_mask].copy()
        y = (sub["wab_type"] == "Broca").astype(int).values        # 1 = Broca
        classes = ["other", "Broca"]
    elif task == "multiclass":
        keep_mask = ~df["wab_type"].isin(NON_TYPE) & df["wab_type"].notna()
        sub = df[keep_mask].copy()
        cats = sorted(sub["wab_type"].unique())
        code = {c: i for i, c in enumerate(cats)}
        y = sub["wab_type"].map(code).values
        classes = cats
    else:
        raise ValueError(task)

    feat = sub.drop(columns=[c for c in ["wab_aq", "wab_type"] if c in sub.columns])
    feat = feat.fillna(0)
    n = len(feat)
    keepc = [c for c in feat.columns if (feat[c] != 0).sum() > SPARSE_DROP_FRAC * n]
    feat = feat[keepc]
    return feat.values, y, list(feat.columns), classes, list(sub.index)


def metrics_binary(y_true, y_pred, y_prob):
    return {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "f1_pos": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "prec_pos": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "rec_pos": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def metrics_multi(y_true, y_pred, prob, n_classes):
    out = {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    try:
        out["macro_auc"] = roc_auc_score(y_true, prob, multi_class="ovr",
                                         average="macro", labels=list(range(n_classes)))
    except Exception:
        out["macro_auc"] = np.nan
    return out


def run_model(name, X, y, classes, task):
    multiclass = (task == "multiclass")
    n_classes = len(classes)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    mkeys = (["balanced_acc", "macro_f1", "accuracy", "macro_auc"] if multiclass
             else ["balanced_acc", "auc", "f1_pos", "mcc", "prec_pos", "rec_pos"])
    per_repeat = {k: [] for k in mkeys}

    oof_pred = np.full(len(y), -1)
    oof_prob = (np.zeros((len(y), n_classes)) if multiclass else np.zeros(len(y)))
    # 池化: 跨 repeat 累加每人概率, 末尾取均值 -> 供下游混淆矩阵/校准
    pool_prob = (np.zeros((len(y), n_classes)) if multiclass else np.zeros(len(y)))
    pool_cnt = np.zeros(len(y))
    smote_skips = 0

    for i, (tr, te) in enumerate(rskf.split(X, y)):
        if name == "Dummy":
            pipe = build_pipe(name, y[tr], multiclass); skipped = False
        else:
            pipe, skipped = build_pipe(name, y[tr], multiclass)
        smote_skips += int(skipped)
        try:
            pipe.fit(X[tr], y[tr])
        except ValueError:
            # SMOTE 仍报错(极端类不平衡) -> 退回无重采样
            steps = [("scaler", StandardScaler()), ("clf", estimator(name, multiclass))]
            if name == "Dummy":
                steps = [("clf", estimator(name, multiclass))]
            pipe = ImbPipeline(steps); pipe.fit(X[tr], y[tr]); smote_skips += 1
        oof_pred[te] = pipe.predict(X[te])
        if multiclass:
            try:
                p = pipe.predict_proba(X[te])
                cls = pipe.named_steps["clf"].classes_
                full = np.zeros((len(te), n_classes))
                for j, c in enumerate(cls):
                    full[:, int(c)] = p[:, j]
                oof_prob[te] = full
                pool_prob[te] += full
            except Exception:
                pass
        else:
            try:
                oof_prob[te] = pipe.predict_proba(X[te])[:, 1]
                pool_prob[te] += oof_prob[te]
            except Exception:
                oof_prob[te] = oof_pred[te]; pool_prob[te] += oof_pred[te]
        pool_cnt[te] += 1

        if (i + 1) % 5 == 0:
            print(f"    [{name}] fold {i+1}/{N_SPLITS*N_REPEATS}", flush=True)
        if (i + 1) % N_SPLITS == 0:
            if multiclass:
                m = metrics_multi(y, oof_pred, oof_prob, n_classes)
            else:
                m = metrics_binary(y, oof_pred, oof_prob)
            for k in per_repeat:
                per_repeat[k].append(m[k])
            oof_pred = np.full(len(y), -1)
            oof_prob = (np.zeros((len(y), n_classes)) if multiclass else np.zeros(len(y)))

    pool_cnt[pool_cnt == 0] = 1
    mean_prob = pool_prob / (pool_cnt[:, None] if multiclass else pool_cnt)
    summary = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in per_repeat.items()}
    return summary, per_repeat, mean_prob, smote_skips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--name", required=True, help="特征集标签: JHU_fine / arterial_coarse")
    ap.add_argument("--task", required=True, choices=["fluent", "broca", "multiclass"])
    ap.add_argument("--models", nargs="*", default=ALL_MODELS)
    args = ap.parse_args()

    tp = PKG / args.table if not Path(args.table).is_absolute() else Path(args.table)
    X, y, feat, classes, ids = load_xy(tp, args.task)
    dist = pd.Series(y).value_counts().sort_index()
    lab = {i: c for i, c in enumerate(classes)} if args.task == "multiclass" else \
          ({0: classes[0], 1: classes[1]})
    distr = ", ".join(f"{lab[int(k)]}={int(v)}" for k, v in dist.items())
    print(f"[{args.name}/{args.task}] n={len(y)}  类别: {distr}  features={X.shape[1]}")

    runs = PKG / "results" / "runs"; runs.mkdir(parents=True, exist_ok=True)
    rows = []
    import time
    for name in args.models:
        t0 = time.time()
        summ, per_repeat, mean_prob, skips = run_model(name, X, y, classes, args.task)
        dt = time.time() - t0
        rows.append({"task": args.task, "dataset": args.name, "model": name,
                     **{k: f"{m:.3f}±{s:.3f}" for k, (m, s) in summ.items()},
                     "smote_skips": skips, "seconds": round(dt, 1)})
        ba = summ["balanced_acc"][0]
        extra = (f"macroF1={summ['macro_f1'][0]:.3f}" if args.task == "multiclass"
                 else f"AUC={summ['auc'][0]:.3f} F1={summ['f1_pos'][0]:.3f}")
        print(f"  {name:14s} bAcc={ba:.3f}  {extra}  ({dt:.1f}s)" +
              (f"  [SMOTE跳过{skips}折]" if skips else ""))

        # dump 逐 repeat 指标(配对检验用)
        pr = pd.DataFrame(per_repeat); pr.insert(0, "repeat", range(1, len(pr) + 1))
        pr.to_csv(runs / f"type_{args.task}_{args.name}_{name}_perrepeat.csv", index=False)
        # dump 每人池化折外(混淆矩阵/校准用)
        if args.task == "multiclass":
            od = pd.DataFrame(mean_prob, columns=[f"prob_{c}" for c in classes])
            od.insert(0, "participant_id", ids); od.insert(1, "y_true", y)
            od["y_pred"] = mean_prob.argmax(1)
        else:
            od = pd.DataFrame({"participant_id": ids, "y_true": y,
                               "prob_pos": mean_prob, "y_pred": (mean_prob >= 0.5).astype(int)})
        od.to_csv(runs / f"type_{args.task}_{args.name}_{name}_oof.csv", index=False)

    new = pd.DataFrame(rows)
    summ_path = PKG / "results" / "summary" / f"type_{args.task}_{args.name}.csv"
    if summ_path.exists():
        old = pd.read_csv(summ_path)
        old = old[~old["model"].isin(new["model"])]
        new = pd.concat([old, new], ignore_index=True)
    new["_s"] = new["balanced_acc"].astype(str).str.split("±").str[0].astype(float)
    new = new.sort_values("_s", ascending=False).drop(columns=["_s"])
    new.to_csv(summ_path, index=False)
    print(f"\nsummary -> {summ_path.relative_to(PKG)}")


if __name__ == "__main__":
    raise SystemExit(main())
