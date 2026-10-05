"use client";
import Link from "next/link";
import { useState } from "react";
import { Caveat, Status, useApi } from "@/components/ui";
import { api } from "@/lib/api";

export default function Experiments() {
  const [tick, setTick] = useState(0);
  const exps = useApi<any[]>("/experiments", [tick]);
  const [msg, setMsg] = useState<string | null>(null);
  async function launch(profile: string) {
    try {
      const r = await api("/experiments", { method: "POST", body: JSON.stringify({ profile, modes: profile === "quick" ? ["temporal"] : null }) });
      setMsg(`Launched job ${r.job_id}. Refresh in a few minutes.`);
      setTimeout(() => setTick((t) => t + 1), 4000);
    } catch (e: any) { setMsg(e.message); }
  }
  return (
    <>
      <h1>Research Experiments</h1>
      <p className="sub">Each experiment stores dataset version, code version, configuration, periods, geographic split, seed, metrics and artifacts.</p>
      <div className="panel field-row">
        <button onClick={() => launch("quick")}>Launch quick experiment (temporal split)</button>
        <button className="ghost" onClick={() => launch("main")}>Launch main experiment (all splits, ~2–4 h)</button>
        <button className="ghost" onClick={() => setTick((t) => t + 1)}>Refresh</button>
      </div>
      {msg && <Caveat>{msg}</Caveat>}
      <Status loading={exps.loading} error={exps.error} />
      <div className="panel"><div className="table-wrap"><table>
        <thead><tr><th>Experiment</th><th>Status</th><th>Created</th><th>Dataset</th><th>Code</th><th>Seed</th></tr></thead>
        <tbody>{(exps.data ?? []).map((e) => <tr key={e.id}>
          <td><Link href={`/experiments/${e.id}`}>{e.id}</Link></td><td><span className={`pill ${e.status === "failed" ? "high" : e.status === "running" ? "moderate" : ""}`}>{e.status}</span></td>
          <td>{e.created?.slice(0, 19)}</td><td className="mono">{e.dataset_version}</td><td className="mono">{e.code_version}</td><td>{e.seed}</td></tr>)}</tbody>
      </table></div></div>
    </>
  );
}
