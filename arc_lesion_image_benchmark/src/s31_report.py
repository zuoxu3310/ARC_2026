#!/usr/bin/env python
"""Generate the human-readable experiment report directly from saved summaries."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/frozen_prediction_control_2026-09-22"
s = pd.read_csv(OUT / "frozen_prediction_summary.csv")
old = pd.read_csv(OUT / "existing_grid_all_cohort_differences.csv")
names = {"full": "全部患者", "has_dwi": "有 DWI 扫描", "has_rsfmri": "有静息态 fMRI",
         "has_taskfmri": "有任务态 fMRI", "has_flair": "有 FLAIR 扫描",
         "multimodal_complete": "三类模态齐全", "chronic_365": "病程至少 365 天",
         "chronic_180": "病程至少 180 天", "aq_le90": "去掉 AQ>90 的患者",
         "teghipco_idlist": "已发表的 172 人名单"}


def get(task, model, subset, metric="score"):
    rows = s.query("task == @task and model == @model and subset == @subset and metric == @metric")
    assert len(rows) == 1
    return rows.iloc[0]


def delta(row, scale=1, digits=3):
    return (f"{row.delta_from_full*scale:+.{digits}f} "
            f"[{row.delta_ci_low*scale:+.{digits}f}, {row.delta_ci_high*scale:+.{digits}f}]")


rf_full = get("regression", "RandomForest", "full")
rf_flair = get("regression", "RandomForest", "has_flair")
rf_list = get("regression", "RandomForest", "teghipco_idlist")
xg_list = get("regression", "XGBoost", "teghipco_idlist")
lines = [
"# 固定预测对照：选人能让分数改变多少",
"",
"2026-09-22｜环境：`data-analysis`（Python 3.11.8）｜新增二次分析；未对外发布。",
"",
"## 结论和文章主线",
"",
f"**不改变任何一个人的预测，仅改变哪些患者被计入评分，仍然可以产生明显的成绩变化。**"
f"随机森林的相关系数从全体 226 人的 {rf_full.observed:.3f}，变成已发表名单对应 172 人的 "
f"{rf_list.observed:.3f}；只评估有 FLAIR 扫描的 135 人时，则变成 {rf_flair.observed:.3f}。"
"所有选择规则、两个模型和两个任务都完整保留，见下表。",
"",
"> 我们要追问的是：ARC 论文里的分数提高，究竟提供了多少方法进步的证据？"
"我们先把算法和流程固定，发现改变纳入患者就能让成绩明显波动；"
"现在进一步把逐人预测也固定，仍然能得到分数升降。"
"因此，跨研究的分数优势需要先排除选人带来的解释，才足以支持新方法的增量价值。",
"",
"这次没有更换研究主线。旧实验中，患者名单一变，训练人群和评分人群会一起变；"
"新实验去掉了重新训练这个环节，直接展示评分人群本身足以改变成绩。"
"它给原有质疑增加了一个更直接的对照。",
"",
"## 实际完成了什么",
"",
"- 重整旧网格全部 3,000 个单元：每种模型、每条筛选规则、20 个种子；输出相对全体患者的变化及分开标记的跨度。",
"- 新跑 Random Forest、XGBoost 的回归和分类，共 20 次五折划分、400 次折内训练；保存 18,080 条逐人留出预测。",
"- 80 个全体患者分数与旧 s18 网格逐项吻合，最大误差 2.22×10⁻¹⁶；训练与分析均正常退出。",
"- 完成 2,000 次患者重抽样，以及每个子集各 2,000 次等人数和 2,000 次严重度分箱匹配抽样；输出三张可编辑图。",
"",
"全程使用原来同一套病灶特征。有 DWI、fMRI 或 FLAIR 在这里是纳入条件，"
"没有把这些扫描作为额外模型输入。每折只用训练患者拟合标准化和分类过采样；"
"先保存所有留出预测，再应用筛选名单。",
"",
"## 全部主要筛选规则：固定预测后，相对全体患者改变多少",
"",
"回归表中是相关系数变化。分类表中是平衡准确率变化，单位为**百分点**。"
"方括号是固定这批预测条件下的患者重抽样 95% 区间；它没有覆盖重新训练整个学习流程的不确定性。",
"",
"| 筛选规则 | N | Random Forest：Δr [95% 区间] | XGBoost：Δr [95% 区间] |",
"|---|---:|---:|---:|",
]
order = s.query("task=='regression' and model=='RandomForest' and metric=='score' and scope=='primary' and subset!='full'").subset.tolist()
for sub in order:
    rf, xg = get("regression", "RandomForest", sub), get("regression", "XGBoost", sub)
    lines.append(f"| {names[sub]} | {rf.n} | {delta(rf)} | {delta(xg)} |")
lines += ["", f"全体患者参照分数：Random Forest r={rf_full.observed:.4f}；XGBoost r={get('regression','XGBoost','full').observed:.4f}。", "",
          "| 筛选规则 | N | Random Forest：变化百分点 [95% 区间] | XGBoost：变化百分点 [95% 区间] |",
          "|---|---:|---:|---:|"]
for sub in order:
    rf, xg = get("classification", "RandomForest", sub), get("classification", "XGBoost", sub)
    lines.append(f"| {names[sub]} | {rf.n} | {delta(rf, 100, 2)} | {delta(xg, 100, 2)} |")
lines += ["", f"全体患者参照平衡准确率：Random Forest {100*get('classification','RandomForest','full').observed:.2f}%；XGBoost {100*get('classification','XGBoost','full').observed:.2f}%。", "",
          "![固定预测主要结果](figures/frozen_prediction_primary.png)", "",
          "[可编辑 SVG](figures/frozen_prediction_primary.svg) · [矢量 PDF](figures/frozen_prediction_primary.pdf)", "",
          "## 升分实例与两项单列分析", "",
          "这两项在运行前就单列：一项直接依据严重度去掉接近满分的患者；"
          "另一项使用已有论文公开的患者名单。它们没有混入上面的主要筛选规则或主要跨度。", "",
          "| 单列规则 | 模型 | 全体 r → 子集 r | Δr [95% 区间] | 分类变化（百分点） |",
          "|---|---|---:|---:|---:|"]
for sub in ["aq_le90", "teghipco_idlist"]:
    for model in ["RandomForest", "XGBoost"]:
        r = get("regression", model, sub)
        c = get("classification", model, sub)
        lines.append(f"| {names[sub]} | {model} | {r.full_reference:.3f} → {r.observed:.3f} | {delta(r)} | {delta(c,100,2)} |")
lines += ["", f"172 人名单上的回归升幅，Random Forest 为 {rf_list.delta_from_full:+.4f}，"
          f"XGBoost 为 {xg_list.delta_from_full:+.4f}。两者的条件区间都在零以上。"
          "这展示了选人可以产生正向分数优势；该名单在这里仅作为选人规则，"
          "不是复现原论文模型或替原论文计算方法贡献比例。分类对应变化的区间跨零，两个任务的证据强度不同。", "",
          "![两项单列结果](figures/frozen_prediction_secondary.png)", "",
          "## 随机挑同样人数，能否解释这些变化", "",
          "等人数对照从同一个 226 人池中随机选人，沿用同样的固定预测。"
          "下表的尾秩表示观察分数在这些随机集合中的极端程度；主要规则的数值已在每个模型、任务内对七条规则做 Holm 校正。"
          "它是有限患者池的抽样参照，不是随机分配纳入条件的因果检验。", "",
          "| 主要规则与任务 | Random Forest：校正尾秩 | XGBoost：校正尾秩 |",
          "|---|---:|---:|"]
for task, sub in [("regression", "has_flair"), ("regression", "has_taskfmri"), ("classification", "has_flair")]:
    a, b = get(task,"RandomForest",sub), get(task,"XGBoost",sub)
    lines.append(f"| {'回归' if task=='regression' else '分类'}：{names[sub]} | {a.size_holm_tail_rank:.4f} | {b.size_holm_tail_rank:.4f} |")
lines += ["", "在主要规则中，FLAIR 与任务态 fMRI 的回归差异，两个模型都达到校正尾秩低于 0.05；"
          "分类的 FLAIR 差异只有 XGBoost 达到该门槛。其余主要规则没有达到。"
          "因此，不能写成所有纳入条件都会造成超出随机抽样的差异。", "",
          "把严重度的粗分布也匹配后，主要规则的尾秩均未达到校正后 0.05。"
          "这与严重度构成参与分数变化的解释相容，但不证明它已经解释了全部差异。"
          "对于 AQ≤90，一旦严格匹配这些分箱人数，抽样只能选回原来那 170 人；"
          "这个参照没有任何随机自由度，不提供额外的机制证据。", "",
          "## 与原实验相比，证据具体增加在哪里", "",
          "| Random Forest 的 FLAIR 子集 | 原实验：重新训练并评分 | 新实验：仅筛选评分患者 |",
          "|---|---:|---:|"]
for task in ["regression","classification"]:
    r = get(task,"RandomForest","has_flair")
    o = old.query("task==@task and model=='RandomForest' and subset=='has_flair'").iloc[0]
    scale = 1 if task=="regression" else 100
    lines.append(f"| {'相关系数变化' if task=='regression' else '平衡准确率变化（百分点）'} | {o.delta_from_full*scale:+.4f} | {r.delta_from_full*scale:+.4f} |")
lines += ["", "回归的新对照保留了与旧实验接近的下降幅度；分类的新对照仍有下降，但幅度较小。"
          "这说明仅改变评分人群已足以制造一部分明显的分数差，训练条件也可能进一步影响分类结果。"
          "两套设计相减不是训练影响的因果估计，不能据此分配百分比。", "",
          "![新旧设计并列](figures/original_and_frozen_primary.png)", "",
          "还有一个值得写进解释的结果：相关系数变化不等于平均预测误差同步变化。"
          f"Random Forest 在 FLAIR 子集上的 r 从 {rf_full.observed:.3f} 降至 {rf_flair.observed:.3f}，"
          f"而平均绝对误差只从 {get('regression','RandomForest','full','mae').observed:.2f} 变成 "
          f"{get('regression','RandomForest','has_flair','mae').observed:.2f} AQ 分；误差变化的区间跨零。"
          "文章应把‘排行榜分数变化’与‘每位患者的预测误差改善’分开讲。", "",
          "## 写进文章时的论证顺序", "",
          "1. 先提出核心问题：在不同患者集合上得到更高分，方法增量究竟有多少证据？",
          "2. 用现有完整网格说明：固定方法、改变纳入规则，确实产生明显分数变化。主分析和两项单列分析分开报告，跨度不冒充相对全体的涨幅。",
          "3. 紧接新对照：连逐人预测都固定后，选人仍会改变分数。这是本次增加的关键证据。",
          "4. 再用等人数、严重度匹配和原来的机制分析解释变化来源；粒度、简单特征和成本实验作为补充支撑。",
          "",
          "主文这轮尚未改写。新实验已经能够支撑上述论证顺序；文献调研与 IEEE Access 整体稿件要求仍需按此前评估继续处理，"
          "不能把完成这一项对照等同于整篇论文已经达到投稿状态。",
          "",
          "## 数据、复核和复现", "",
          "[分析前冻结的方案](ANALYSIS_PLAN.md) · [完整新结果 CSV](frozen_prediction_summary.csv) · "
          "[新旧并列表](original_vs_frozen.csv) · [旧网格全部差异](existing_grid_all_cohort_differences.csv) · "
          "[80 项锚点核对](anchor_reconciliation.csv)", "",
          "独立验收见 [复核报告](review/results-review.md) 和 [机器核对记录](review/verification_result.json)。"
          "本次 60 个汇总、228 个抽样分布的独立重算全部通过，最大数值误差 1.71×10⁻¹⁴。"
          "预测、折名单、抽样权重及完整抽样分布均已保存。"
          "`NUMERICAL_TIE_CORRECTION.json` 记录相同样本的浮点并列值修正："
          "分数和区间不变，仅修正包容并列值的尾秩及其校正值；原表另存于审计目录。", "",
          "这是已探索数据上的二次分析。模型固定的是每一折的模型和它生成的逐人留出预测，不是一个对所有人都在样本内预测的模型。"
          "划分重复不作为独立患者。未新调参，未使用外部验证集，未重新拟合不可获得的深度网络或脑区模拟方法。", "",
          "从项目根目录运行：", "", "```bash",
          "PYENV_VERSION=data-analysis python arc_lesion_image_benchmark/src/s31_watchdog.py train",
          "PYENV_VERSION=data-analysis python arc_lesion_image_benchmark/src/s31_watchdog.py analyze",
          "PYENV_VERSION=data-analysis python arc_lesion_image_benchmark/src/s31_resampling_tie_fix.py",
          "MPLCONFIGDIR=/private/tmp/arc-mpl-cache PYENV_VERSION=data-analysis python arc_lesion_image_benchmark/src/fig_frozen_prediction_control.py",
          "PYENV_VERSION=data-analysis python arc_lesion_image_benchmark/src/s31_report.py", "```", "",
          "训练命令会核对冻结的代码、输入和环境指纹，并复用已经保存的完整单元。"
          "分析与尾秩修正按上述顺序运行。三个图同时提供 SVG、PDF、PNG。", ""]
(OUT / "REPORT.zh-CN.md").write_text("\n".join(lines))
print(f"Saved {OUT / 'REPORT.zh-CN.md'}")
