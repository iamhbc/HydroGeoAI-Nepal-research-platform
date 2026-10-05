"use client";
import { useState } from "react";
import Link from "next/link";
import NepalMap, { Legend } from "@/components/NepalMap";
import { Status, useApi } from "@/components/ui";
import { EVENT_LABELS, qs } from "@/lib/api";

const METRICS: [string, string, "blue" | "heat" | "risk"][] = [
  ["", "Stations only", "blue"], ["precip_annual", "Mean annual precipitation (mm)", "blue"],
  ["tmax", "Mean Tmax (°C)", "heat"], ["tmin", "Mean Tmin (°C)", "heat"],
  ["whiplash", "Whiplash events / year", "risk"],
  ...Object.keys(EVENT_LABELS).map((k) => [k, `${EVENT_LABELS[k]} (days / yr)`, "risk"] as [string, string, "risk"]),
];

export default function MapPage() {
  const [metric, setMetric] = useState("precip_annual");
  const [raster, setRaster] = useState<"dem" | "slope" | null>(null);
  const [basins, setBasins] = useState(true);
  const [rivers, setRivers] = useState(true);
  const [basemap, setBasemap] = useState(true);
  const [start, setStart] = useState("1991-01-01");
  const [end, setEnd] = useState("2024-12-31");
  const [sel, setSel] = useState<string | null>(null);
  const pts = useApi<any>(`/maps/stations${qs({ metric, start, end })}`);
  const station = useApi<any>(sel ? `/stations/${sel}` : null);
  const m = METRICS.find((x) => x[0] === metric)!;
  return (
    <>
      <h1>Map</h1>
      <p className="sub">Select what to show on the map. Click a station to see its details.</p>
      <div className="panel">
        <div className="mapbar">
          <label>Station layer<select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {METRICS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
          <label>Raster<select value={raster ?? ""} onChange={(e) => setRaster((e.target.value || null) as any)}>
            <option value="">None</option><option value="dem">Elevation (DEM)</option><option value="slope">Slope</option></select></label>
          <label>From<input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></label>
          <label>To<input type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></label>
          <label className="chk"><input type="checkbox" checked={basins} onChange={(e) => setBasins(e.target.checked)} />Basins</label>
          <label className="chk"><input type="checkbox" checked={rivers} onChange={(e) => setRivers(e.target.checked)} />Rivers</label>
          <label className="chk"><input type="checkbox" checked={basemap} onChange={(e) => setBasemap(e.target.checked)} />Basemap</label>
        </div>
        <Status loading={pts.loading} error={pts.error} />
        <NepalMap points={pts.data ? { data: pts.data, valueKey: metric ? "value" : undefined, label: m[1], ramp: m[2] } : undefined}
          raster={raster} showBasins={basins} showRivers={rivers} basemap={basemap} onStationClick={setSel} />
        {pts.data && metric && <Legend label={m[1]} ramp={m[2]} values={pts.data.features.map((f: any) => f.properties.value)} />}
      </div>
      {station.data && <div className="panel">
        <h2>{station.data.station.name} <span className="muted small">({station.data.station.station_id})</span></h2>
        <div className="grid3 small">
          <div>Elevation <b>{station.data.station.elevation_m} m</b> (DEM {Math.round(station.data.station.dem_elevation_m)} m)<br />
            Slope {station.data.station.slope_deg?.toFixed(1)}° · relief {station.data.station.relief_m?.toFixed(0)} m<br />
            River distance {station.data.station.dist_river_km?.toFixed(1)} km<br />Land cover {station.data.station.landcover}</div>
          <div>{Object.entries(station.data.event_rates_pct).slice(0, 7).map(([k, v]: any) => <div key={k}>{EVENT_LABELS[k] ?? k}: {v}%</div>)}</div>
          <div>{station.data.issues.map((i: any, j: number) => <div key={j}><span className="pill moderate">{i.issue}</span> {i.detail}</div>)}
            <Link href={`/climate`}>Open in Explore Climate →</Link></div>
        </div>
      </div>}
    </>
  );
}
