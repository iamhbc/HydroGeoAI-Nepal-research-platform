"use client";
import { use, useEffect, useState } from "react";
import { Status, Table, useApi } from "@/components/ui";
import { API, api } from "@/lib/api";

export default function ExperimentDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const e = useApi<any>(`/experiments/${id}`);
  const [table, setTable] = useState("");
  const [rows, setRows] = useState<any>(null);
  useEffect(() => { if (e.data?.tables?.length && !table) setTable(e.data.tables.includes("benchmark_main") ? "benchmark_main" : e.data.tables[0]); }, [e.data, table]);
  useEffect(() => { if (table) api(`/experiments/${id}/tables/${table}`).then(setRows).catch(() => setRows(null)); }, [id, table]);
  const runs = e.data?.runs ?? [];
  return (
    <>
      <h1>Experiment</h1>
      <p className="sub mono">{id}</p>
      <Status loading={e.loading} error={e.error} />
      {e.data && <>
        <div className="panel small">
          <b>Status</b> {e.data.status} · <b>dataset</b> <span className="mono">{e.data.dataset_version}</span> · <b>code</b> <span className="mono">{e.data.code_version}</span> ·
          <b> seed</b> {e.data.seed} · <b>runs</b> {runs.length} · <b>created</b> {e.data.created} · <b>finished</b> {e.data.finished ?? "–"}
        </div>
        <div className="panel"><h2>Tables</h2>
          <div className="field-row"><label>Table<select value={table} onChange={(ev) => setTable(ev.target.value)}>
            {e.data.tables.map((t: string) => <option key={t}>{t}</option>)}</select></label>
            {table && <a href={`${API}/experiments/${id}/tables/${table}`} target="_blank">JSON</a>}</div>
          {rows && <Table rows={rows.rows} columns={rows.columns} max={400} />}
        </div>
        <div className="panel"><h2>Figures</h2>
          <div className="figs">{e.data.figures.map((f: string) => <figure key={f} style={{ margin: 0 }}>
            <img src={`${API}/experiments/${id}/figures/${f}`} alt={f} /><figcaption className="small muted">{f}</figcaption></figure>)}</div>
        </div>
        {e.data.summary_md && <div className="panel"><h2>SUMMARY.md</h2><pre className="md">{e.data.summary_md}</pre></div>}
        <div className="panel"><h2>Runs (registry)</h2>
          <Table rows={runs.map((r: any) => ({ name: r.name, model: r.model, split: r.split_mode, train: r.train_period?.join(" → "),
            test: r.test_period?.join(" → "), auprc: r.metrics?.auprc, auroc: r.metrics?.auroc, ece: r.metrics?.ece }))} /></div>
        <div className="panel"><h2>Configuration</h2><pre className="md">{JSON.stringify(e.data.config, null, 2)}</pre></div>
      </>}
    </>
  );
}
