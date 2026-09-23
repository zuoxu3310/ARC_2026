# Data sources and transformations

## ARC clinical metadata

- Dataset: [OpenNeuro ds004884, version 1.0.2](https://doi.org/10.18112/openneuro.ds004884.v1.0.2).
- Dataset publication: [Gibson et al. (2024)](https://doi.org/10.1038/s41597-024-03819-7).
- License: CC0, recorded in the included `datasets/arc_ds004884/dataset_description.json`.
- The supplied `participants.tsv` contains the public participant identifiers
  and clinical metadata. The modelable pool is defined by the 226-row feature
  and outcome intersection, with ID alignment checked before scoring.
- `arc_modality_availability.tsv` is the frozen participant-level availability
  table used to construct the imaging rules. The supplied flags, clinical
  metadata, and feature IDs are sufficient to regenerate every cohort list
  using `s17_freeze_pool_subsets.py`.

The original dataset documentation states that the source studies had
University of South Carolina Institutional Review Board approval and that the
released dataset is fully anonymized. This analysis uses the public release.

## Normalized lesion masks and JHU/arterial features

Normalized masks and the original two atlases come from
[neurolabusc/AphasiaRecoveryCohortDemo](https://github.com/neurolabusc/AphasiaRecoveryCohortDemo/tree/1d24ad4049f4dba536d51f5f80efdb0900393ac0),
pinned at commit `1d24ad4049f4dba536d51f5f80efdb0900393ac0`.
The repository's BSD-2-Clause notice is preserved in `ARC_DEMO_LICENSE.txt`.
The download list and recorded file checksums are in
`arc_lesion_image_benchmark/data/raw/`.

The 228 binary masks are in a common 157×189×156, 1 mm grid; the 226 masks with
aligned WAB-AQ labels define the modeling pool. Regional lesion burden is the
fraction of each atlas region overlapped by the lesion. `s2_extract_atlas_features.py`
implements this operation and joins age and WAB-AQ using participant IDs.
The source label table is `merged_artery_participants.tsv`.

The JHU representation has 189 nonzero region labels. The arterial image has
32 nonzero labels; some historical output names contain `arterial_33` because
the background was counted in those names. The manuscript uses 32 regions.

## Additional atlas representations

AAL, Brodmann, and AICHA source images come from
[neurodata/neuroparc](https://github.com/neurodata/neuroparc/tree/5a5e7469671e65cb58087c47b69d0edb71dc2966/atlases/label/Human).
The optional download manifest pins that commit and checks each image against
the SHA-256 of the image used for this study. `s23_resample_atlases.py` uses
nearest-neighbor resampling and int16 labels. Feature tables used in the
archived analysis are supplied without requiring these downloads.

Source publications:
[arterial territories](https://doi.org/10.1038/s41597-022-01923-0),
[Neuroparc](https://doi.org/10.1038/s41597-021-00849-3),
[AAL](https://doi.org/10.1006/nimg.2001.0978), and
[AICHA](https://doi.org/10.1016/j.jneumeth.2015.07.013).
Atlas definitions and source files retain their upstream attribution and terms.

## Secondary published-list intersection

The participant identities come from `DLAphasiaSeverityARCSubset.mat` in
[Teghipco's Figshare release, version 1](https://doi.org/10.6084/m9.figshare.23579943.v1).
The Figshare API identifies this release as CC0. The optional downloader checks
the source file's SHA-256; its Figshare MD5 is
`badd46f8d6098a123049f36c40e6fd6c`.

The source release contains 173 participants; 172 intersect the present
226-person pool. `teghipco_inspect_labels.py` creates the ID/label cross-check,
and `s17_freeze_pool_subsets.py` freezes the intersection. The manuscript uses
this set as a secondary scoring-membership control.

## Saved analysis artifacts

Prediction CSVs and fold JSONs use the public ARC identifiers. They contain
held-out predictions and the actual training/test memberships. Sampling arrays
contain patient multiplicities or inclusion indicators, allowing the bootstrap
and matched-reference calculations to be rerun exactly. Hashes and the original
runtime record are retained; the portable wrapper records new runs separately.
