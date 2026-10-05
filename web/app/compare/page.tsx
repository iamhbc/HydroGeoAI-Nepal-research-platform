"use client";
import { useEffect, useMemo, useState } from "react";
import { HBars } from "@/components/charts";
import { Caveat, Status, Table, useApi } from "@/components/ui";
import { api } from "@/lib/api";

export default function Compare() {
  const exps = useApi<any[]>("/experiments");
  const [eid, setEid] = useState("");
  const [mode, setMode] = useState("");
  const [metric, setMetric] = useState("auprc");
  const [bench, setBench] = useState<any>(null);
  const [hyp, setHyp] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const done = (exps.data ?? []).filter((e) => e.status === "completed");
  useEffect(() => { if (!eid && done.length) setEid(done[0].id); }, [done, eid]);
  useEffect(() => {
    if (!eid) return;
    setErr(null);
    Promise.all([api(`/experiments/${eid}/tables/benchmark_main`), api(`/experiments/${eid}/tables/hypotheses_H1_H6`)])
      .then(([b, h]) => { setBench(b); setHyp(h); const ms = [...new Set(b.rows.map((r: any) => r.mode))] as string[]; setMode(ms[0]); })
      .catch((e) => setErr(e.message));
  }, [eid]);
  const modes = useMemo(() => [...new Set((bench?.rows ?? []).map((r: any) => r.mode))] as string[], [bench]);
  const rows = (bench?.rows ?? []).filter((r: any) => r.mode === mode);
  const pivot = useMemo(() => {
    const models = [...new Set((bench?.rows ?? []).map((r: any) => r.model))] as string[];
    return models.map((m) => Object.fromEntries([["model", m], ...modes.map((md) => [md, bench.rows.find((r: any) => r.model === m && r.mode === md)?.[metric] ?? null])]));
  }, [bench, modes, metric]);
  return (
    <>
      <h1>Compare Models</h1>
      <p className="sub">Baselines → classical ML → deep sequence models → HydroGeoAI ablation (A–F) on identical splits. CIs: station-block bootstrap.</p>
      <Status loading={exps.loading} error={exps.error || err} />
      {!done.length && !exps.loading && <Caveat>No completed experiments yet. Run <code>make reproduce-quick</code> or launch one from Research Experiments.</Caveat>}
      <div className="panel field-row">
        <label>Experiment<select value={eid} onChange={(e) => setEid(e.target.value)}>{done.map((e) => <option key={e.id}>{e.id}</option>)}</select></label>
        <label>Split<select value={mode} onChange={(e) => setMode(e.target.value)}>{modes.map((m) => <option key={m}>{m}</option>)}</select></label>
        <label>Metric<select value={metric} onChange={(e) => setMetric(e.target.value)}>
          {["auprc", "auroc", "brier_skill_vs_climatology", "ece", "f1", "brier"].map((m) => <option key={m}>{m}</option>)}</select></label>
      </div>
      {bench && <div className="grid2">
        <div className="panel"><h2>{metric} — {mode} split</h2>
          <HBars data={rows.map((r: any) => ({ label: r.model, value: r[metric] })).sort((a: any, b: any) => b.value - a.value)} />
          {metric === "auprc" && rows[0] && <p className="small muted">Base rate (no-skill AUPRC): {rows[0].base_rate.toFixed(4)}</p>}
        </div>
        <div className="panel"><h2>Spatial vs temporal transfer ({metric})</h2><Table rows={pivot} /></div>
      </div>}
      {bench && <div className="panel"><h2>Full benchmark ({mode})</h2>
        <Table rows={rows} columns={["model", "family", "auprc", "auprc_ci_low", "auprc_ci_high", "auroc", "brier_skill_vs_climatology", "ece", "precision", "recall", "f1"]} /></div>}
      {hyp && <div className="panel"><h2>Hypotheses H1–H6</h2>
        <Table rows={hyp.rows} columns={["hypothesis", "mode", "comparison", "diff", "ci_low", "ci_high", "verdict"]} />
        <Caveat>Verdicts are automatic: “supported” only when the bootstrap CI excludes zero in the hypothesised direction. Synthetic-data verdicts say nothing about Nepal.</Caveat></div>}
    </>
  );
}
