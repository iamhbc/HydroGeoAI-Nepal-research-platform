# 3. Event taxonomy and hydroclimatic whiplash

All thresholds are **station-specific** and fitted on the **reference period only** (`configs/data/default.yaml →
reference_period`, by default 1991–2010, inside the training period). Labels are `NaN` where the underlying data
are missing; a missing observation is never treated as "no event".

| Group | Event | Definition | Rationale |
|---|---|---|---|
| Precipitation | extreme_wet_day | P > P95 of wet days (≥1 mm) | R95p-style; comparable across climates |
| | very_extreme_wet_day | P > P99 of wet days | R99p-style; flood-relevant tail |
| | extreme_rainfall_event | 3-day sum > P99 of 3-day sums | multi-day storms drive floods/landslides |
| | consecutive_wet_days | wet run ≥ 5 days | CWD-style persistence |
| | rainfall_persistence | ≥ 8 wet days in 10 | saturation / landslide preconditioning |
| Dry | dry_spell | dry run ≥ 15 days | CDD-style |
| | precipitation_deficit | SPI-30 ≤ −1 | short-term (agricultural) deficit |
| | meteorological_drought | SPI-90 ≤ −1 | seasonal meteorological drought |
| Thermal | hot_day | Tmax > calendar-day P90 (±7-day window) | TX90p (ETCCDI) |
| | warm_spell | ≥ 3 consecutive hot days | relaxed WSDI for daily labels |
| | cold_night | Tmin < calendar-day P10 | TN10p |
| Compound | hot_dry | hot day and SPI-30 ≤ −1 | heat–drought co-occurrence (fire, crops) |
| | wet_hot | extreme wet day and hot day | rare; convective extremes |
| | rain_after_dryness | extreme wet day after ≥ 20 dry days | runoff on dry/crusted soils, debris flows |

SPI uses a gamma distribution with a point mass at zero (Thom estimator) fitted **per calendar month**, so
seasonality (dry pre-monsoon, wet monsoon) is removed. Why SPI/SPEI/ETCCDI are used, and their limits, is
documented in `features/indices.py`.

## Hydroclimatic whiplash
**Definition.** A *dry-to-wet whiplash* occurs when a dry episode (SPI-30 ≤ −1) is followed within 30 days by a wet
episode (SPI-30 ≥ +1). *Wet-to-dry* is the mirror case. Because SPI is month-standardised, the normal monsoon onset
does **not** count as whiplash, only anomalous transitions do.

**Metrics per event.** Duration, transition time, intensity (SPI swing between the extremes), speed (swing per day
between the extremes), spatial extent (fraction of reporting stations with a same-direction event within ±15 days).

**Trend analysis without presupposition.** Frequencies are normalised per adequately observed station-year.
Periods (1990–2000, 2001–2010, 2011–2020, 2021–present) are compared with Kruskal–Wallis (frequency) and a
permutation test (intensity). Annual series use Mann-Kendall + Sen's slope, and station-level trends use
Benjamini–Hochberg FDR. Results are reported whether significant or not. Unequal period lengths and coverage are
stated in the output.
