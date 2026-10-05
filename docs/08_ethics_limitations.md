# 8. Ethics, limitations, attribution vs causation

## Attribution is not causation
Integrated gradients, permutation importance, gate weights and modality ablation describe **what the statistical
model relies on**. They do not show physical mechanisms. A high attribution to elevation, for example, may reflect
correlated station-network sampling and not orographic processes. This statement is attached to every
explanation in the API, Space, web and reports (`hydrogeoai.ATTRIBUTION_DISCLAIMER`). Physical interpretation
needs process reasoning and independent evidence (reanalysis diagnostics, case studies, literature).

## Climate signal vs observational artifact
Homogeneity results are probabilistic flags for expert review, not proof. Station metadata in Nepal can be
incomplete; undocumented breaks must not be "corrected" automatically.

## Operational use
Predictions are research outputs. They are not warnings, and official early warning in Nepal is the mandate of DHM.
The interfaces state model and dataset versions and the synthetic-data caveat. Do not present outputs as official
forecasts or use them for insurance or financial decisions.

## Data ethics and licensing
The data describe physical climate, not people. Respect provider licences (DHM data policy, Copernicus, CHIRPS,
MODIS). Cite sources, and do not redistribute restricted station data. Credit local institutions and consider
co-authorship or acknowledgement for data providers.

## Known limitations
* Sparse high-elevation stations; snowfall and gauge undercatch are not corrected.
* Station-relative thresholds make events comparable across climates but do not encode absolute impact.
* Small number of independent spatial units means wide confidence intervals.
* The synthetic generator encodes the developer's assumptions; skill on it says nothing about real predictability.
* Point gauges vs gridded products differ in scale.

## Reproducibility and openness
Code, configs, seeds and environment are versioned. Every result is traceable to dataset and code versions.
Negative and inconclusive results are reported.
