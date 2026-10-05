"use client";
import { useEffect, useState } from "react";
import { api, Station } from "@/lib/api";

export function useApi<T = any>(path: string | null, deps: any[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!path) return;
    let alive = true;
    setLoading(true); setError(null);
    api<T>(path).then((d) => alive && setData(d)).catch((e) => alive && setError(e.message)).finally(() => alive && setLoading(false));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);
  return { data, error, loading };
}

export function Status({ loading, error }: { loading?: boolean; error?: string | null }) {
  if (error) return <div className="alert">⚠ {error}{error.includes("fetch") ? " — is the API running? (make api)" : ""}</div>;
  if (loading) return <div className="loading">Loading…</div>;
  return null;
}

export type Region = { station_id?: string; basin?: string; province?: string };

export function RegionPicker({ value, onChange, allowAll = false }: { value: Region; onChange: (r: Region) => void; allowAll?: boolean }) {
  const { data: stations } = useApi<Station[]>("/stations");
  const basins = [...new Set((stations ?? []).map((s) => s.basin))].sort();
  const provinces = [...new Set((stations ?? []).map((s) => s.province))].sort();
  const kind = value.station_id ? "station" : value.basin ? "basin" : value.province ? "province" : "all";
  return (
    <div className="field-row">
      <label>Region
        <select value={kind} onChange={(e) => {
          const k = e.target.value;
          if (k === "station") onChange({ station_id: stations?.[0]?.station_id });
          else if (k === "basin") onChange({ basin: basins[0] });
          else if (k === "province") onChange({ province: provinces[0] });
          else onChange({});
        }}>
          <option value="station">Station</option><option value="basin">Basin</option><option value="province">Province</option>
          {allowAll && <option value="all">All Nepal</option>}
        </select>
      </label>
      {kind === "station" && <label>Station
        <select value={value.station_id} onChange={(e) => onChange({ station_id: e.target.value })}>
          {(stations ?? []).map((s) => <option key={s.station_id} value={s.station_id}>{s.station_id} · {s.name} ({s.elevation_m} m)</option>)}
        </select></label>}
      {kind === "basin" && <label>Basin
        <select value={value.basin} onChange={(e) => onChange({ basin: e.target.value })}>
          {basins.map((b) => <option key={b}>{b}</option>)}</select></label>}
      {kind === "province" && <label>Province
        <select value={value.province} onChange={(e) => onChange({ province: e.target.value })}>
          {provinces.map((b) => <option key={b}>{b}</option>)}</select></label>}
    </div>
  );
}

export function Table({ rows, columns, max = 200, digits = 3 }: { rows: any[]; columns?: string[]; max?: number; digits?: number }) {
  if (!rows?.length) return <div className="empty">No rows.</div>;
  const cols = columns ?? Object.keys(rows[0]);
  return (
    <div className="table-wrap"><table>
      <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>{rows.slice(0, max).map((r, i) => <tr key={i}>{cols.map((c) => <td key={c}>{cell(r[c], digits)}</td>)}</tr>)}</tbody>
    </table>{rows.length > max && <div className="muted small">Showing {max} of {rows.length} rows.</div>}</div>
  );
}

function cell(v: any, d: number) {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") return Number.isInteger(v) ? v : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(d);
  if (typeof v === "string" && /^\d{4}-\d\d-\d\dT/.test(v)) return v.slice(0, 10);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export function Stat({ label, value, hint }: { label: string; value: any; hint?: string }) {
  return <div className="stat"><div className="stat-v">{value ?? "–"}</div><div className="stat-l">{label}</div>{hint && <div className="stat-h">{hint}</div>}</div>;
}

export function Caveat({ children }: { children: React.ReactNode }) {
  return <p className="caveat">{children}</p>;
}
