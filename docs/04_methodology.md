# 4. Methodology

## 4.1 Quality control (`qc/checks.py`)
| Check | Severity | Action |
|---|---|---|
| exact duplicates | – | dropped (counted) |
| conflicting duplicates (same station-date, different values) | error | value set NaN in the kept row, audited |
| irregular intervals | info | counted before calendar regularisation |
| negative precipitation, precip > 500 mm/day, temperature outside [−45, 48] °C, RH outside [0,100], Tmax < Tmin | error | excluded from `*_qc` |
| station-year total > 4× station median (unit error) | error | excluded |
| precip > 2.5× station P99.9, day-to-day ΔT > 15 °C, robust z of deseasonalised anomaly > 6, ≥ 7 identical values | suspicious | kept and flagged |
| coordinates outside bbox/outline, elevation metadata vs DEM > 300 m, near-duplicate locations, documented history | station issue | reported |
Completeness tables are produced per station, year, season (DJF, MAM, JJAS, ON) and month.

## 4.2 Homogeneity (`qc/homogeneity.py`)
Annual series (precipitation totals; Tmax/Tmin means; years ≥ 80% complete) are tested with Pettitt, SNHT and
Buishand (Monte-Carlo p-values). A break is accepted when ≥ 2 of 3 tests agree. Each station is tested twice:
**raw** and **relative to a reference** (correlation-weighted composite of up to 5 neighbours within 250 km, chosen
on first-difference correlation; ratio series for precipitation). Classification:

* break in the relative series + documented change within ±2 years → *artifact supported by metadata*
* break in the relative series, undocumented → *possible artifact* (expert review)
* break only in the raw series → *regional signal* (shared with neighbours, consistent with climate)
* relative shift below practical relevance (0.3 °C, 8% precipitation) → *minor break*

Reference selection is **iterative**: stations flagged as artifacts are removed from the reference pool and all
stations are re-tested, so one inhomogeneous neighbour does not create spurious breaks elsewhere. On the synthetic
data all 4 injected breaks are recovered at the correct year, with magnitude errors under 10%. No series is
adjusted automatically.

Trends: Mann-Kendall with trend-free pre-whitening, Sen's slope, BH-FDR across stations.

## 4.3 Features
* **Dynamic model inputs** (60-day window): log1p(precip), Tmax, Tmin (globally standardised with training
  statistics), observation masks, day-of-year sin/cos, monsoon flag. Inputs are not station-climatology anomalies,
  so the model has to use geospatial context to interpret raw values at new locations.
* **Tabular baseline features**: current values, lags 1/3/7/15/30, rolling mean/std/max/min (3/7/15/30 days),
  wet/dry/warm spell lengths, calendar.
* **Static groups**: `gis` (lat, lon, elevation, slope, northness, eastness, relief, river distance, climate zone,
  land cover), `rs` (NDVI mean/amplitude, snow cover, LST), `meta` (instrument, reference completeness). Basin
  identity is intentionally excluded because it cannot transfer to held-out basins.

## 4.4 Baselines (`models/baselines.py`, `models/deep.py`)
Climatology (day-of-year ±15 d, pooled, so it applies to unseen stations), probabilistic persistence, logistic
regression, Tweedie GLM (amount regression), Random Forest, HistGradientBoosting, XGBoost/LightGBM (if installed),
LSTM, GRU, causal TCN (receptive field 63 d), Transformer. Deep baselines use the same inputs as model D.

## 4.5 Hydroclimatic Encoder (`models/encoder.py`)
Transformer encoder (d=64, 3 layers, 4 heads, learned positions). Self-supervised objectives:
1. **masked-span reconstruction**: geometric spans (mean 5 days) covering 30% of positions; masked values and
   masks are zeroed, so masked days look missing and the encoder learns gap-robust representations; MSE only
   on masked *and* observed positions;
2. **temporal contrastive** (NT-Xent, τ = 0.1) between two corrupted views of the same window;
3. **multi-scale targets**: 7- and 30-day means at the end of the window from the pooled representation.
Pretraining uses only the training stations and period of the split being evaluated. Spatial-consistency
objectives are disabled by default; if enabled they may use training-station neighbours only.

## 4.6 Multimodal model (`models/multimodal.py`)
Encoders produce one token per modality (meteorology from the Hydroclimatic Encoder; GIS/RS/metadata from MLPs).
The tokens get modality-type embeddings and pass through one cross-modal Transformer layer, then **gated
attention pooling**, then heads for the extreme-event logit and the SPI-30 regime at t+1 (auxiliary, weight 0.2).
**Modality dropout** (p = 0.1) trains robustness to missing modalities and enables inference-time modality
ablation. Fine-tuning uses a 0.3× learning rate on the pretrained encoder.

## 4.7 Uncertainty (`uncertainty/`)
Deep ensemble (M members, different seeds) and MC dropout. Predictive entropy is decomposed into aleatoric
(expected member entropy) and epistemic (mutual information) parts. Temperature scaling is fitted on validation
data. **Class-conditional (Mondrian) split conformal prediction** guarantees ≥ 1−α coverage separately for events
and non-events. Outputs: probability, confidence (1 − normalised entropy), epistemic, conformal set and a
low/moderate/high category based on validation quantiles of epistemic uncertainty. The final method is chosen
by comparing ECE, Brier, coverage and error detection (AUROC of epistemic → error).

## 4.8 Explainability (`explain/`)
Integrated gradients (zero baseline = training mean with all observations missing) per channel, lag and static
feature; channel permutation importance (ΔAUPRC); inference-time modality ablation; gate weights; counterfactual
rescaling of antecedent precipitation. **Every output carries the disclaimer that attribution is not causation.**
Physical interpretation needs process reasoning (e.g. orographic uplift, moisture transport) and independent
evidence.
