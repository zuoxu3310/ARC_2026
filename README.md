# ARC cohort selection

Code for *Higher Scores, Unchanged Predictions: Cohort Selection in Lesion-Based Aphasia Severity Prediction*. The experiments compare cohort and model effects, then hold individual predictions fixed to measure how changing the scoring cohort changes reported performance.

Use Python 3.11 and run from the repository root:

```bash
pip install -r requirements-optional.txt
python reproduce.py prepare --extra-atlases
python reproduce.py grid --tabpfn
python reproduce.py fixed
python reproduce.py statistics
python reproduce.py figures
```

Inputs are downloaded from [ARC / OpenNeuro](https://openneuro.org/datasets/ds004884/versions/1.0.2), the [ARC demo](https://github.com/neurolabusc/AphasiaRecoveryCohortDemo), [Figshare](https://doi.org/10.6084/m9.figshare.23579943.v1), and [Neuroparc](https://github.com/neurodata/neuroparc); data and generated outputs are not included in this repository. `prepare_data.py` pins source versions and checks downloads.

The full grid contains 3,000 model/cohort/repeat cells and requires TabPFN model access. For a run without TabPFN, install `requirements.txt` and omit `--tabpfn`; this produces a smaller grid. Re-running `grid` replaces its score files; `fixed` resumes completed training cells and checks their input/code versions.

Analysis and appendix scripts are in `arc_lesion_image_benchmark/src/`; `arc_wab_aq_benchmark/src/` supplies the leave-one-out benchmark. Results are written under the benchmark directories, with figures/tables in `paper/` and model-set sensitivity in `outputs/`. References cited in the manuscript are in [bibliography.bib](bibliography.bib).
