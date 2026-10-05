"use client";
import { useState } from "react";
import { HBars, LineChart } from "@/components/charts";
import NepalMap, { Legend } from "@/components/NepalMap";
import { Caveat, Region, RegionPicker, Stat, Status, Table, useApi } from "@/components/ui";
import { api, fmt } from "@/lib/api";

export default function Run() {
  const models = useApi<any>("/models");
  const [region, setRegion] = useState<Region>({ province: "Karnali" });
  const [start, setStart] = useState("2022-06-01");
  const [end, setEnd] = useState("2022-09-30");
  const [modelId, setModelId] = useState("");
  const [res, setRes] = useState<any>(null);
  const [exp, setExp] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function go() {
    setBusy(true); setErr(null); setExp(null);
    try {
      const body = { station_ids: region.station_id ? [region.station_id] : null, basin: region.basin, province: region.province,
        start, end, phenomenon: "extreme_wet_day", model_id: modelId || null };
      const r = await api("/predict", { method: "POST", body: JSON.stringify(body) });
      setRes(r);
      const top = [...r.predictions].sort((a: any, b: any) => b.probability - a.probability)[0];
      if (top) setExp(await api("/explain", { method: "POST", body: JSON.stringify({ station_id: top.station_id, issued: top.issued, model_id: modelId || null }) }));
    } catch (e: any) { setErr(e.message); setRes(null); }
    setBusy(false);
  }

  const top = res ? [...res.predictions].sort((a: any, b: any) => b.probability - a.probability)[0] : null;
  return (
    <>
      <h1>Run Model</h1>
      <p className="sub">Phenomenon: extreme wet day on the next day (station-specific &gt;P95 of wet days). Output = prediction + confidence + uncertainty + explanation.</p>
      <div className="panel">
        <RegionPicker value={region} onChange={setRegion} />
        <div className="field-row">
          <label>Start (issue date)<input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></label>
          <label>End<input type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></label>
          <label>Phenomenon<select disabled><option>Extreme precipitation (extreme wet day, t+1)</option></select></label>
          <label>Model<select value={modelId} onChange={(e) => setModelId(e.target.value)}>
            <option value="">Deployed ({models.data?.deployed?.id ?? "none"})</option>
            {models.data?.models.map((m: any) => <option key={m.id} value={m.id}>{m.id} [{m.status}]</option>)}</select></label>
          <button onClick={go} disabled={busy}>{busy ? "Running…" : "Run analysis"}</button>
        </div>
      </div>
      <Status error={err} />
      {res && <>
        <div className="stats">
          <Stat label="Predictions" value={res.summary.n_predictions} hint={`${res.summary.n_stations} station(s)`} />
          <Stat label="Mean probability" value={fmt(res.summary.mean_probability)} />
          <Stat label="High-uncertainty share" value={`${(res.summary.frac_high_uncertainty * 100).toFixed(1)}%`} />
          <Stat label="Retrospective AUPRC" value={fmt(res.evaluation?.auprc)} hint={res.evaluation ? `${res.evaluation.events} events; base rate ${(res.evaluation.events / res.evaluation.n).toFixed(3)}` : "no observed events"} />
          <Stat label="Model version" value={<span className="mono small">{res.model_version}</span>} />
          <Stat label="Dataset version" value={<span className="mono">{res.dataset_version}</span>} />
        </div>
        {top && <div className="panel"><h2>Highest-probability day</h2>
          <div className="grid3">
            <div><div className="small muted">Extreme event probability</div><div style={{ fontSize: "1.8rem", fontWeight: 700 }}>{top.probability.toFixed(3)}</div>
              <div className="small">{top.station_id} · target {top.target_date} · observed: {top.observed_event === null ? "unknown" : top.observed_event ? "event" : "no event"}</div></div>
            <div><div className="small muted">Prediction confidence</div><div style={{ fontSize: "1.8rem", fontWeight: 700 }}>{top.confidence.toFixed(2)}</div>
              <div className="small">1 − normalised predictive entropy</div></div>
            <div><div className="small muted">Uncertainty</div><div style={{ fontSize: "1.4rem", fontWeight: 700 }}><span className={`pill ${top.uncertainty}`}>{top.uncertainty}</span></div>
              <div className="small">epistemic {top.epistemic.toExponential(2)} · ensemble σ {top.ensemble_std.toFixed(3)} · conformal set {top.conformal_set}</div></div>
          </div></div>}
        <div className="grid2">
          <div className="panel"><h2>Probability through time (regional mean, 10–90% across stations)</h2>
            <LineChart xIsDate yLabel="probability" hline={{ y: res.decision_threshold, label: "decision threshold" }}
              series={[{ name: "probability", points: res.timeseries.map((r: any) => ({ x: r.target_date, y: r.probability })),
                band: res.timeseries.map((r: any) => ({ lo: r.p_low, hi: r.p_high })) }]}
              markers={res.timeseries.filter((r: any) => r.observed > 0).map((r: any) => ({ x: r.target_date, y: r.probability }))} />
            <p className="muted small">Red markers: days with ≥1 observed event in the selection.</p></div>
          <div className="panel"><h2>Spatial map — mean probability</h2>
            <NepalMap height={330} points={{ data: res.station_summary, valueKey: "mean_probability", label: "mean probability", ramp: "risk" }} />
            <Legend label="mean probability" ramp="risk" values={res.station_summary.features.map((f: any) => f.properties.mean_probability)} /></div>
        </div>
        <div className="grid2">
          <div className="panel"><h2>Modality attention (mean gate weight)</h2>
            <HBars data={Object.entries(res.modality_weights).map(([k, v]: any) => ({ label: k, value: v }))} />
            {exp && <><h3>Important factors for {exp.station_id} on {exp.issued} (integrated gradients)</h3>
              <HBars data={[...Object.entries(exp.channel_attribution), ...Object.entries(exp.top_static_attribution)]
                .map(([k, v]: any) => ({ label: k, value: v })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 12)} />
              <h3>Change in probability without each modality</h3>
              <HBars data={Object.entries(exp.modality_ablation_delta_p).map(([k, v]: any) => ({ label: `without ${k}`, value: -v }))} />
              <Caveat>{exp.disclaimer}</Caveat></>}
          </div>
          <div className="panel"><h2>Reproducibility</h2>
            <Table rows={[{ key: "prediction_id", value: res.prediction_id }, { key: "timestamp", value: res.timestamp },
              { key: "model_version", value: res.model_version }, { key: "dataset_version", value: res.dataset_version },
              { key: "code_version", value: res.code_version }, { key: "training_experiment", value: res.training_experiment },
              { key: "training_split", value: res.training_split }, { key: "decision_threshold", value: res.decision_threshold },
              { key: "parameters", value: JSON.stringify(res.parameters) }]} />
            <ul className="small muted">{res.notes.map((n: string) => <li key={n}>{n}</li>)}</ul>
            {res.evaluation && <Caveat>{res.evaluation.note}</Caveat>}
          </div>
        </div>
        <div className="panel"><h2>All predictions</h2><Table rows={res.predictions} max={300} /></div>
      </>}
    </>
  );
}
