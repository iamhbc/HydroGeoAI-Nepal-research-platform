"use client";
import { useState } from "react";
import { BarChart, LineChart } from "@/components/charts";
import { Region, RegionPicker, Status, useApi } from "@/components/ui";
import { qs } from "@/lib/api";

const VARS = [["precip", "Precipitation"], ["tmax", "Max temperature"], ["tmin", "Min temperature"], ["rh", "Relative humidity"],
  ["spi30", "SPI-30"], ["spi90", "SPI-90"]];
const MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"];

export default function Climate() {
  const [region, setRegion] = useState<Region>({ basin: "Gandaki" });
  const [variable, setVariable] = useState("precip");
  const [agg, setAgg] = useState("annual");
  const [start, setStart] = useState("1990-01-01");
  const [end, setEnd] = useState("2024-12-31");
  const q = qs({ variable, agg, start, end, ...region });
  const ts = useApi<any>(`/climate${q}`);
  const clim = useApi<any>(`/climate${qs({ variable, agg: "climatology", start, end, ...region })}`);
  const station = useApi<any>(region.station_id ? `/stations/${region.station_id}` : null);
  return (
    <>
      <h1>Explore Climate</h1>
      <p className="sub">QC&apos;d series (values flagged as errors are excluded; originals are preserved in the dataset).</p>
      <div className="panel">
        <RegionPicker value={region} onChange={setRegion} />
        <div className="field-row">
          <label>Variable<select value={variable} onChange={(e) => setVariable(e.target.value)}>
            {VARS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
          <label>Aggregation<select value={agg} onChange={(e) => setAgg(e.target.value)}>
            <option value="daily">Daily</option><option value="monthly">Monthly</option><option value="annual">Annual</option></select></label>
          <label>Start<input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></label>
          <label>End<input type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></label>
        </div>
      </div>
      <Status loading={ts.loading} error={ts.error} />
      {ts.data && <div className="panel">
        <h2>{VARS.find((v) => v[0] === variable)?.[1]} — {agg} ({ts.data.units})</h2>
        <LineChart xIsDate yLabel={ts.data.units}
          series={[{ name: variable, points: ts.data.series.map((r: any) => ({ x: r.date, y: r.value })) }]} />
        <p className="muted small">Mean over {ts.data.stations.length} station(s). Incomplete months (&lt;20 d) / years (&lt;292 d) are omitted.
          Dataset <span className="mono">{ts.data.dataset_version}</span>.</p>
      </div>}
      <div className="grid2">
        <div className="panel"><h2>Monthly climatology</h2>
          {clim.data && <BarChart yLabel={clim.data.units} data={clim.data.series.map((r: any) => ({ label: MONTHS[r.month - 1], value: r.value }))} />}
        </div>
        {station.data && <div className="panel"><h2>Station {station.data.station.name}</h2>
          <p className="small">Elevation {station.data.station.elevation_m} m · {station.data.station.basin} · record from {station.data.record.first_valid}</p>
          <p className="small">Completeness: {Object.entries(station.data.record.completeness).map(([k, v]: any) => `${k} ${(v * 100).toFixed(1)}%`).join(" · ")}</p>
          <h3>Homogeneity</h3>
          {station.data.homogeneity.map((h: any) => <div key={h.variable} className="small"><b>{h.variable}</b>: <span className="pill">{h.classification}</span> {h.rel_break_year ? `break ${h.rel_break_year}` : ""}</div>)}
          <h3>Trends (Sen slope / yr, MK p)</h3>
          {station.data.trends.map((t: any) => <div key={t.variable} className="small">{t.variable}: {t.sens_slope_per_year.toFixed(3)} (p={t.p_value.toFixed(3)}{t.significant_fdr ? ", FDR-significant" : ""})</div>)}
        </div>}
      </div>
    </>
  );
}
