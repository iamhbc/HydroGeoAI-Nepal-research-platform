"use client";
import { useState } from "react";
import { BarChart, LineChart } from "@/components/charts";
import { Caveat, Region, RegionPicker, Status, Table, useApi } from "@/components/ui";
import { EVENT_LABELS, qs } from "@/lib/api";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export default function Extremes() {
  const [region, setRegion] = useState<Region>({ province: "Karnali" });
  const [event, setEvent] = useState("extreme_wet_day");
  const tax = useApi<any>("/extremes/taxonomy");
  const ex = useApi<any>(`/extremes${qs({ event, ...region })}`);
  const wl = useApi<any>(`/whiplash${qs({ station_id: region.station_id, basin: region.basin })}`);
  return (
    <>
      <h1>Explore Extremes</h1>
      <p className="sub">Station-specific thresholds fitted on the reference period only; labels are missing (not &quot;no event&quot;) when data are missing.</p>
      <div className="panel">
        <RegionPicker value={region} onChange={setRegion} />
        <label>Event type
          <select value={event} onChange={(e) => setEvent(e.target.value)}>
            {tax.data && Object.entries(tax.data.groups).map(([g, evs]: any) => (
              <optgroup key={g} label={g}>{evs.map((e: string) => <option key={e} value={e}>{EVENT_LABELS[e] ?? e}</option>)}</optgroup>))}
          </select></label>
      </div>
      <Status loading={ex.loading} error={ex.error} />
      {ex.data && <div className="grid2">
        <div className="panel"><h2>Annual frequency (days / year)</h2>
          <LineChart yLabel="days / year" series={[{ name: event, points: ex.data.annual_frequency.map((r: any) => ({ x: r.year, y: r.days_per_year })) }]} />
          <p className="muted small">{ex.data.note}</p></div>
        <div className="panel"><h2>Seasonality (% of days)</h2>
          <BarChart color="var(--c2)" data={ex.data.seasonality.map((r: any) => ({ label: MONTHS[r.month - 1], value: r.pct_days }))} /></div>
      </div>}
      {ex.data && <div className="panel"><h2>Most recent events</h2><Table rows={ex.data.recent_events} max={50} digits={1} /></div>}

      <h1 style={{ marginTop: 28 }}>Hydroclimatic whiplash</h1>
      <p className="sub">Anomaly-based transitions between SPI-30 ≤ −1 and SPI-30 ≥ +1 within 30 days (climatological monsoon onset is not counted).</p>
      <Status loading={wl.loading} error={wl.error} />
      {wl.data && <>
        <div className="panel"><h2>Comparison across periods ({wl.data.n_events} events)</h2>
          <Table rows={wl.data.decadal} />
          <p className="small">Kruskal–Wallis on station frequencies: p = {wl.data.tests.kruskal_frequency?.p_value?.toFixed(3) ?? "–"} ·
            permutation test on intensity (first vs last period): p = {wl.data.tests.intensity_first_vs_last_permutation_p?.toFixed(3) ?? "–"}</p>
          <Caveat>No trend is assumed: periods are compared with non-parametric tests and results are reported whether or not significant. {wl.data.tests.note}</Caveat>
        </div>
        {wl.data.annual && <div className="panel"><h2>Network frequency per station-year</h2>
          <LineChart yLabel="events / station-year" series={[{ name: "frequency", points: wl.data.annual.map((r: any) => ({ x: r.year, y: r.frequency_per_station_year })) }]} /></div>}
        <div className="panel"><h2>Events</h2>
          <Table rows={wl.data.events} columns={["station_id", "direction", "start", "transition_date", "end", "transition_days", "intensity_spi_swing", "speed_spi_per_day", "spatial_extent"]} max={100} /></div>
      </>}
    </>
  );
}
