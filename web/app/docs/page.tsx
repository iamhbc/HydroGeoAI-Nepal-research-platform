import { API } from "@/lib/api";

export default function Docs() {
  return (
    <>
      <h1>Documentation</h1>
      <p className="sub">Full documentation lives in <code>docs/</code> of the repository. API reference: <a href={`${API}/docs`} target="_blank">{API}/docs</a>.</p>
      <div className="panel"><h2>Central research question</h2>
        <p>Can multimodal geospatial-temporal representation learning improve the detection and characterization of hydroclimatic
          extremes across spatially heterogeneous and data-sparse mountainous regions?</p>
        <ul>
          <li><b>RQ1 Representation</b> — does self-supervised learning extract useful representations without complete labels?</li>
          <li><b>RQ2 Spatial transfer</b> — does a model generalise to unseen stations/basins?</li>
          <li><b>RQ3 Nonstationarity</b> — is performance stable under changing regimes?</li>
          <li><b>RQ4 Multimodality</b> — does topography/geospatial information add value over meteorology alone?</li>
          <li><b>RQ5 Extremes</b> — does the model identify rare extremes better than conventional ML/DL?</li>
          <li><b>RQ6 Uncertainty</b> — can it tell confident predictions from insufficient evidence?</li>
        </ul></div>
      <div className="panel"><h2>Hypotheses</h2>
        <ol><li>H1 Multimodal models outperform meteorology-only models.</li>
          <li>H2 Self-supervised pretraining helps when labelled extremes are scarce.</li>
          <li>H3 Geospatial representations improve transfer to unseen stations.</li>
          <li>H4 Performance deteriorates under temporal distribution shift.</li>
          <li>H5 Uncertainty identifies regions/periods where predictions are unreliable.</li>
          <li>H6 The model can reveal previously uncharacterised regime transitions (exploratory).</li></ol></div>
      <div className="panel"><h2>Methodological safeguards</h2>
        <ul>
          <li>No random day-level splits: spatial (held-out basins), temporal (chronological with embargo) and spatio-temporal splits.</li>
          <li>Thresholds, normalisation and SSL pretraining use training stations/periods only.</li>
          <li>Metrics justified for rare events: AUPRC with base rate, Brier skill, ECE; thresholds chosen on validation only; station-block bootstrap CIs.</li>
          <li>Homogeneity: breaks are classified as regional signal vs possible artifact using neighbour-relative series and metadata; never auto-adjusted.</li>
          <li><b>Attribution ≠ causation:</b> integrated gradients, permutation importance and modality ablation describe model reliance, not physical mechanisms.</li>
        </ul></div>
      <div className="panel"><h2>Key API endpoints</h2>
        <pre className="md">{`GET  /stations, /stations/{id}
GET  /datasets, /datasets/{version}
GET  /climate?variable=&station_id|basin|province=&agg=
GET  /extremes?event=&...      GET /whiplash
POST /predict  {basin|province|station_ids, start, end, model_id?}
POST /explain  {station_id, issued}
GET  /models, /models/{id}     GET /experiments, /experiments/{id}
POST /experiments {profile: quick|main}
GET  /maps, /maps/stations?metric=, /maps/{basins|rivers|outline}
GET  /health                   /admin/* (X-API-Key)`}</pre></div>
    </>
  );
}
