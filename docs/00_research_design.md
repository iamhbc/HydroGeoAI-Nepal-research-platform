# 0. Research design

## Central question
Can multimodal geospatial-temporal representation learning improve the detection and characterization of
hydroclimatic extremes across spatially heterogeneous and data-sparse mountainous regions?

## Research questions → experiments → evidence

| RQ | Question | Experiment in this repository | Evidence produced |
|---|---|---|---|
| RQ1 Representation | Does self-supervised learning extract useful representations without complete labels? | Hydroclimatic Encoder pretrained on training windows only; ablation **E vs B**; held-out masked reconstruction error | `metrics/ssl_pretraining.json`, `tables/ablation_A_to_F` |
| RQ2 Spatial transfer | Does a model trained on some stations generalise to unseen stations/regions? | `spatial` (held-out basins, same period) and `spatiotemporal` (unseen basins + future) splits | `tables/transfer_auprc_by_split`, per-station maps |
| RQ3 Nonstationarity | Is performance stable under changing regimes? | chronological splits with embargo; per-year test metrics + Mann-Kendall; val→test drop | `tables/temporal_degradation_by_year`, H4 row |
| RQ4 Multimodality | Do topography/geospatial data add value over meteorology? | ablation **A→B→C→D→F**, also with HistGBM (second model family) | `tables/ablation_A_to_F`, H1/H3 rows |
| RQ5 Extremes | Are rare extremes detected better than with conventional ML/DL? | benchmark vs climatology, persistence, GLM, RF, GBM, LSTM/GRU/TCN/Transformer | `tables/benchmark_main` |
| RQ6 Uncertainty | Can the system flag insufficient evidence? | ensembles, MC dropout, conformal, failure quadrants, error clustering | `metrics/uncertainty.json`, `tables/failure_*` |

## Hypotheses and decision rules
Verdicts are computed automatically in `experiments/runner.py::assess_hypotheses` and are deliberately conservative.

| H | Statement | Test | "Supported" only if |
|---|---|---|---|
| H1 | Multimodal > meteorology-only | paired station-block bootstrap of ΔAUPRC (F − A; HGB D − HGB A), per split | 95% CI of Δ > 0 |
| H2 | SSL helps when labels are scarce | E − B at 5%, 20%, 100% of training labels (temporal split) | CI > 0 at low label fractions |
| H3 | Geospatial representations improve transfer | C − B on spatial and spatio-temporal splits | CI > 0 on unseen stations |
| H4 | Performance deteriorates under temporal shift | val→test drop; MK trend of yearly test AUPRC | significant negative trend or a clear drop |
| H5 | Uncertainty identifies unreliable predictions | error rate high vs low uncertainty; AUROC(epistemic → error); conformal coverage on events | higher error at high uncertainty and AUROC > 0.6 |
| H6 | Previously uncharacterised regime transitions | whiplash engine + regime head; exploratory | never automatic; needs expert review on real data |

"Contradicted" means the CI excludes zero in the opposite direction; otherwise the verdict is "inconclusive".
A null result is a result: the ablation is designed to show **when** multimodal learning adds value, not to
confirm that it always does.

## Prediction task (primary)
Binary: does an `extreme_wet_day` (> station P95 of wet days, reference period) occur on day t+1, given the
previous 60 days and static context? Rare (~2% of days), so metrics are chosen for imbalance (see
[05_evaluation.md](05_evaluation.md)). The taxonomy supports other targets (`configs/events/taxonomy.yaml →
prediction_target`).

## Threats to validity (and mitigations)
* **Leakage**: random day splits are not offered; thresholds/normalisation/SSL use training data only; embargo
  between periods; `check_leakage` asserts station/period disjointness.
* **Inhomogeneous records**: breaks are detected and flagged; analyses can exclude flagged series.
* **Label definition circularity**: labels are station-relative; model inputs are *not* station-climatology
  anomalies, so geospatial context has to explain local climate (a fair test of RQ4).
* **Multiple comparisons**: BH-FDR across stations; bootstrap CIs on all headline metrics.
* **Synthetic data**: the generator has known structure; real data are required for any scientific claim.
* **Small number of stations**: station-block bootstrap acknowledges that stations, not days, are the
  independent units; CIs are wide by design.
