Image-input case: analysis scripts and requirements

These scripts implement the 173-patient same-input image comparison, the
exploratory Optuna XGBoost extension and the matched development controls
(paper: "Same-Input Image-Based Method Comparison" and the appendix
"Image-Input Comparison and Reference Strengthening"). They are separate from
benchmark.py's 71-input ARC reference workflow. Keep the ARC_2026 directory
structure; the scripts use fixed relative paths.

External inputs (not redistributed here):
- DLAphasiaSeverityARCSubset.mat from https://doi.org/10.6084/m9.figshare.23579943.v1
  at datasets/arc_ds004884/teghipco_figshare/DLAphasiaSeverityARCSubset.mat
  (downloaded and checksum-verified by "python reproduce.py prepare").
- The authors' vgg2.py from https://github.com/alexteghipco/volDNN at commit
  bdfefe165dcef18729054a8b142949cd5836ec4e. Save its unmodified contents as
  outputs/targeted-replication-feasibility-2026-10-03/raw/deep-vgg2.py.txt.
  Expected SHA256: 6088e711ae2486f327d5fcbe7d6e703c5dd46792c0932847eefada29d4a59623.
  Obtain the authors' code under its original repository terms.
- A PDF at paper/main.pdf: the preparation step records its hash as a provenance
  marker; the PDF contents do not enter training or scoring.

The frozen protocols (PLAN.txt in each folder) and source-alignment.tsv specify
the exact settings. The CNN trainer requires Apple Metal (PyTorch MPS);
cross-platform training has not been validated. NumPy, SciPy, pandas,
scikit-learn, joblib, threadpoolctl, matplotlib, PyTorch, LightGBM, XGBoost and
Optuna are required. The package versions of the original runs are recorded in
the result metadata of the supplementary results.

In a fresh checkout with the external inputs in place:
python outputs/cnn-same-input-2026-10-03/study.py prepare
python outputs/cnn-same-input-2026-10-03/study.py cnn
python outputs/cnn-same-input-2026-10-03/study.py classical
python outputs/cnn-same-input-2026-10-03/study.py aggregate
python outputs/cnn-same-input-2026-10-03/verify.py
python outputs/xgboost-optuna-2026-10-04/run.py prepare
python outputs/xgboost-optuna-2026-10-04/run.py run

Preparation regenerates the deterministic partitions and writes new run
manifests; do not place a new run on top of a completed one. The partitions and
frozen manifests in the supplementary results are provenance records of the
original runs; their source hashes refer to the original working directory
(including the manuscript PDF at that time) and are not expected to match a
fresh checkout. Checkpoint-reload verification, and the matched development
controls in outputs/supplementary-controls/scripts/image_controls.py, require
the CNN checkpoints produced by the corresponding training run (not shipped).

The supplementary results include exported predictions, mean/SD summaries,
paired intervals, selected XGBoost parameters and verification records. They
exclude raw MRI, source clinical tables and trained weights. The original study,
this public-subset comparison and the 226-patient reference benchmark use
different populations and must not be scored against one another.
