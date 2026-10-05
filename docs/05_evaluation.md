# 5. Evaluation

## Splits (`experiments/splits.py`)
| Mode | Train | Validation | Test | Isolates |
|---|---|---|---|---|
| temporal | all stations, 1991–2010 | all stations, 2011–2016 | all stations, 2017–2024 | nonstationarity |
| spatial | training-basin stations, 1991–2010 | held-out stations from training basins, 1991–2010 | **held-out basins** (Koshi, Mahakali), 1991–2010 | spatial transfer |
| spatiotemporal | training-basin stations, 1991–2010 | val stations, 2011–2016 | held-out basins, 2017–2024 | both (most realistic) |

A 60-day embargo is applied at the start of the validation and test periods, and sample dates stop `lead` days
before each period ends, so targets never cross boundaries. `check_leakage` raises if stations or periods overlap.
Random day-level splits are not offered.

## Metrics: why these
* **AUPRC** (primary): events occur on about 2% of days; AUPRC measures how well positives are ranked, and it is
  always reported with the base rate (`auprc_lift` = AUPRC / base rate).
* **AUROC**: reported, but optimistic under imbalance.
* **Brier score and Brier Skill Score** vs training-period climatology: probabilistic accuracy.
* **ECE + reliability curves**: calibration, needed if probabilities inform decisions.
* **Precision / recall / F1**: at the threshold that maximises F1 on **validation**, never chosen on test.
* **CRPS**: for ensemble/probabilistic amount forecasts (it equals Brier for binary outcomes).
* **Regression** (amounts): MAE, RMSE, R², bias.
* **Uncertainty of metrics**: station-block bootstrap (stations are the independent units). Model comparisons use
  paired bootstrap of differences.

## Benchmark: the HydroGeoAI Benchmark
For each split, the runner evaluates classification/extreme-event detection, spatial transfer, temporal transfer,
calibration, uncertainty (coverage, error detection) and missing-data robustness (0–50% of input days removed).
Tables: `benchmark_main`, `ablation_A_to_F`, `transfer_auprc_by_split`, `missing_data_robustness`,
`temporal_degradation_by_year`, `label_scarcity`, `hypotheses_H1_H6`.

## Three kinds of validation
1. **Statistical**: metrics with CIs on held-out data.
2. **Spatial**: performance on unseen basins and its relation to distance from training stations
   (`failure_clustering_* → station_density`).
3. **Scientific**: do patterns make hydrological sense? Seasonality of predicted risk (monsoon peak), attribution
   to antecedent precipitation rather than spurious static features, elevation dependence and consistency with
   orographic expectations. A high F1 alone does not establish scientific validity, and the paper must discuss
   these checks qualitatively.

## Failure analysis (`evaluation/failure.py`)
Quadrants: confident-correct, confident-wrong, uncertain-correct, uncertain-wrong ("confident" = low uncertainty
category). Error, miss and false-alarm rates by elevation band, season, basin, distance to the nearest training
station and window missingness, with χ² tests of independence. Confident-wrong cases are the most informative
for the paper.
