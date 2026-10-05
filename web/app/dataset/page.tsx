"use client";
import { Caveat, Stat, Status, Table, useApi } from "@/components/ui";
import { fmt } from "@/lib/api";

export default function Dataset() {
  const ds = useApi<any>("/datasets");
  const v = ds.data?.active_version;
  const d = useApi<any>(v ? `/datasets/${v}` : null);
  const qc = useApi<any>("/qc/summary");
  const meta = d.data?.metadata;
  const s = qc.data?.summary;
  return (
    <>
      <h1>Data quality</h1>
      <p className="sub">Data are never silently modified: every step is recorded with parameters and content hashes; QC adds flags and *_qc columns.</p>
      <Status loading={ds.loading || d.loading} error={ds.error || d.error} />
      {meta && <>
        {meta.notes && <Caveat>{meta.notes}: values are simulated for pipeline development and are not observations.</Caveat>}
        <div className="stats">
          <Stat label="Version" value={<span className="mono">{meta.version}</span>} />
          <Stat label="Source" value={meta.source} />
          <Stat label="Coverage" value={`${meta.temporal_coverage[0]} → ${meta.temporal_coverage[1]}`} />
          <Stat label="Stations / records" value={`${meta.n_stations} / ${meta.n_records.toLocaleString()}`} />
          <Stat label="CRS" value={meta.crs} />
          <Stat label="QC status" value={d.data.status} />
        </div>
        <div className="grid2">
          <div className="panel"><h2>Variables</h2>
            <Table rows={Object.entries(meta.variables).map(([k, x]: any) => ({ variable: k, units: x.units, description: x.long_name, missing_pct: meta.missing_pct[k] }))} /></div>
          <div className="panel"><h2>QC summary</h2>
            {s && <Table rows={[
              { check: "exact duplicates dropped", n: s.exact_duplicates_dropped }, { check: "conflicting duplicates", n: s.conflicting_duplicates },
              { check: "irregular intervals", n: s.irregular_intervals }, { check: "errors (excluded)", n: s.n_errors },
              { check: "suspicious (kept, flagged)", n: s.n_suspicious }, { check: "station-years < 80% complete", n: s.station_years_below_completeness },
              ...Object.entries(s.flags_by_type).map(([k, n]) => ({ check: `flag: ${k}`, n }))]} />}
          </div>
        </div>
        <div className="panel"><h2>Station metadata issues</h2><Table rows={qc.data?.station_issues ?? []} /></div>
        <div className="panel"><h2>Homogeneity (raw vs neighbour-relative series)</h2>
          <Table rows={(qc.data?.homogeneity ?? []).filter((h: any) => h.classification !== "homogeneous")}
            columns={["station_id", "variable", "classification", "raw_break_year", "rel_break_year", "shift_estimate", "reference_stations", "documented_events", "interpretation"]} />
          <p className="small muted">Homogeneous series are hidden. Breaks are flagged, not adjusted.</p></div>
        <div className="panel"><h2>Provenance ({d.data.provenance?.steps?.length} steps)</h2>
          <Table rows={(d.data.provenance?.steps ?? []).map((p: any) => ({ step: p.step, description: p.description, n_affected: p.n_affected,
            output_hash: p.output_hash?.slice(0, 12), timestamp: p.timestamp }))} /></div>
        <div className="panel"><h2>Catalog</h2><Table rows={ds.data.datasets} /></div>
      </>}
      {meta && <p className="small muted">Reference period: {fmt(meta.reference_period?.join(" → "))}</p>}
    </>
  );
}
