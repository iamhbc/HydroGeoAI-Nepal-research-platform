"use client";
import { useEffect, useState } from "react";
import { Caveat, Stat, Table } from "@/components/ui";
import { api, API } from "@/lib/api";

export default function Admin() {
  const [key, setKey] = useState("");
  const [st, setSt] = useState<any>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => { try { setKey(localStorage.getItem("hgai-admin-key") ?? ""); } catch {} }, []);

  async function load() {
    setMsg(null);
    try {
      setSt(await api("/admin/status", { adminKey: key }));
      setLogs(await api("/admin/logs?n=80", { adminKey: key }));
      try { localStorage.setItem("hgai-admin-key", key); } catch {}
    } catch (e: any) { setMsg(e.message); setSt(null); }
  }
  async function act(path: string, label: string) {
    try { const r = await api(path, { method: "POST", adminKey: key }); setMsg(`${label}: ${JSON.stringify(r).slice(0, 200)}`); load(); }
    catch (e: any) { setMsg(`${label} failed: ${e.message}`); }
  }
  async function upload(f: File) {
    const fd = new FormData(); fd.append("file", f);
    const r = await fetch(`${API}/admin/datasets/upload`, { method: "POST", body: fd, headers: { "X-API-Key": key } });
    setMsg(`Upload: ${JSON.stringify(await r.json()).slice(0, 400)}`);
  }

  return (
    <>
      <h1>Admin / Researcher panel</h1>
      <p className="sub">Dataset lifecycle (upload → QC → approve → version → archive), model lifecycle (register → evaluate → deploy → rollback), experiments and monitoring.</p>
      <div className="panel field-row">
        <label>Admin API key<input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder="X-API-Key" /></label>
        <button onClick={load}>Connect</button>
      </div>
      {msg && <Caveat>{msg}</Caveat>}
      {st && <>
        <div className="stats">
          <Stat label="Active dataset" value={<span className="mono">{st.dataset_version}</span>} />
          <Stat label="Deployed model" value={<span className="mono small">{st.deployed_model?.id ?? "none"}</span>} />
          <Stat label="Jobs (failed)" value={`${st.jobs.length} (${st.failed_jobs.length})`} />
          <Stat label="Storage data / results" value={`${st.storage_mb.data} / ${st.storage_mb.results} MB`} />
        </div>
        <div className="panel"><h2>Datasets</h2>
          <div className="field-row">
            <button onClick={() => act("/admin/datasets/build", "Build dataset")}>Rebuild dataset (background)</button>
            <label>Upload observations (CSV/Parquet)<input type="file" accept=".csv,.parquet" onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} /></label>
            <button className="ghost" onClick={() => act("/admin/reload", "Reload")}>Reload API data</button>
          </div>
          <div className="table-wrap"><table><thead><tr><th>Key</th><th>Status</th><th>Updated</th><th>Actions</th></tr></thead>
            <tbody>{st.datasets.map((d: any) => <tr key={d.key}><td className="mono">{d.key}</td><td>{d.status}</td><td>{d.updated}</td>
              <td><button className="ghost" onClick={() => act(`/admin/datasets/${d.key}/qc`, "QC")}>Run QC</button>{" "}
                <button className="ghost" onClick={() => act(`/admin/datasets/${d.key}/approve`, "Approve")}>Approve</button>{" "}
                <button className="ghost" onClick={() => act(`/admin/datasets/${d.key}/archive`, "Archive")}>Archive</button></td></tr>)}</tbody></table></div>
        </div>
        <div className="panel"><h2>Models</h2>
          <div className="table-wrap"><table><thead><tr><th>Model</th><th>Status</th><th>AUPRC</th><th>Dataset</th><th>Actions</th></tr></thead>
            <tbody>{st.models.map((m: any) => <tr key={m.id}><td className="mono">{m.id}</td><td>{m.status}</td><td>{m.metrics?.auprc?.toFixed(4)}</td>
              <td className="mono">{m.dataset_version}</td>
              <td><button className="ghost" onClick={() => act(`/admin/models/${m.id}/deploy`, "Deploy")}>Deploy</button>{" "}
                <button className="ghost" onClick={() => act(`/admin/models/${m.id}/archive`, "Archive")}>Archive</button></td></tr>)}</tbody></table></div>
          <p><button className="ghost" onClick={() => act(`/admin/models/hydrogeoai-nepal-model/rollback`, "Rollback")}>Roll back deployed model</button></p>
        </div>
        <div className="grid2">
          <div className="panel"><h2>Jobs</h2><Table rows={st.jobs.map((j: any) => ({ id: j.id, kind: j.kind, status: j.status, started: j.started, error: j.error }))} /></div>
          <div className="panel"><h2>Recent experiments</h2><Table rows={st.experiments} /></div>
        </div>
        <div className="panel"><h2>Logs</h2><pre className="md">{logs.join("\n") || "(no log lines yet)"}</pre></div>
      </>}
    </>
  );
}
