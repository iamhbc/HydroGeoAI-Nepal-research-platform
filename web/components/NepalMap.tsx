"use client";
import { useEffect, useRef } from "react";
import type * as ML from "maplibre-gl";
import { API } from "@/lib/api";

export type PointLayer = { data: GeoJSON.FeatureCollection; valueKey?: string; label?: string; ramp?: "blue" | "heat" | "risk" };
type Props = {
  points?: PointLayer; showBasins?: boolean; showRivers?: boolean; raster?: "dem" | "slope" | null;
  basemap?: boolean; onStationClick?: (id: string) => void; height?: number;
};

const RAMPS: Record<string, string[]> = {
  blue: ["#deebf7", "#9ecae1", "#4292c6", "#08519c", "#08306b"],
  heat: ["#fff5eb", "#fdbe85", "#fd8d3c", "#d94701", "#7f2704"],
  risk: ["#edf8e9", "#bae4b3", "#fdae61", "#f46d43", "#a50026"],
};

export default function NepalMap({ points, showBasins = true, showRivers = true, raster = null, basemap = true,
  onStationClick, height = 560 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const map = useRef<ML.Map | null>(null);
  const ready = useRef(false);
  const latest = useRef<Props>({});
  latest.current = { points, showBasins, showRivers, raster, basemap, onStationClick };

  useEffect(() => {
    if (!ref.current) return;
    let m: ML.Map | null = null;
    let cancelled = false;
    import("maplibre-gl").then((maplibregl) => {
    if (cancelled || !ref.current) return;
    maplibregl.setWorkerUrl(`${window.location.origin}/maplibre/maplibre-gl-worker.mjs`);
    m = new maplibregl.Map({
      container: ref.current,
      style: {
        version: 8,
        sources: { osm: { type: "raster", tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256,
          attribution: "© OpenStreetMap contributors" } },
        layers: [{ id: "bg", type: "background", paint: { "background-color": "#e9eef2" } },
                 { id: "osm", type: "raster", source: "osm", paint: { "raster-opacity": 0.75 } }],
      },
      bounds: [[79.9, 26.2], [88.3, 30.6]], fitBoundsOptions: { padding: 20 },
    });
    if (!m) return;
    const mm: ML.Map = m;
    mm.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    mm.addControl(new maplibregl.ScaleControl({ unit: "metric" }));
    map.current = m;
    mm.once("style.load", async () => {   // not "load": that waits for every basemap tile
      const [basins, rivers, outline] = await Promise.all(["basins", "rivers", "outline"].map((l) =>
        fetch(`${API}/maps/${l}`).then((r) => (r.ok ? r.json() : null)).catch(() => null)));
      for (const r of ["dem", "slope"] as const) {
        try {
          const meta = await fetch(`${API}/maps/raster/${r}/meta`).then((x) => x.json());
          mm.addSource(`r-${r}`, { type: "image", url: `${API}/maps/raster/${r}/image.png`, coordinates: meta.coordinates });
          mm.addLayer({ id: `r-${r}`, type: "raster", source: `r-${r}`, layout: { visibility: "none" }, paint: { "raster-opacity": 0.8 } });
        } catch {}
      }
      if (outline) { mm.addSource("outline", { type: "geojson", data: outline });
        mm.addLayer({ id: "outline", type: "line", source: "outline", paint: { "line-color": "#334155", "line-width": 1.6 } }); }
      if (basins) { mm.addSource("basins", { type: "geojson", data: basins });
        mm.addLayer({ id: "basins-fill", type: "fill", source: "basins", paint: { "fill-color": "#1f6f8b", "fill-opacity": 0.05 } });
        mm.addLayer({ id: "basins", type: "line", source: "basins", paint: { "line-color": "#1f6f8b", "line-width": 1, "line-dasharray": [2, 2] } });
        mm.addLayer({ id: "basin-labels", type: "symbol", source: "basins", layout: { "text-field": ["get", "basin"], "text-size": 11 },
          paint: { "text-color": "#1f4f63", "text-halo-color": "#fff", "text-halo-width": 1.2 } }); }
      if (rivers) { mm.addSource("rivers", { type: "geojson", data: rivers });
        mm.addLayer({ id: "rivers", type: "line", source: "rivers", paint: { "line-color": "#2b8cbe", "line-width": 2 } }); }
      mm.addSource("points", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      mm.addLayer({ id: "points", type: "circle", source: "points",
        paint: { "circle-radius": 7, "circle-color": "#2a6f97", "circle-stroke-color": "#fff", "circle-stroke-width": 1.5 } });
      mm.on("click", "points", (e) => {
        const f = e.features?.[0];
        if (!f) return;
        const p = f.properties as any;
        const lbl = latest.current.points?.label;
        new maplibregl.Popup().setLngLat((f.geometry as any).coordinates)
          .setHTML(`<b>${p.name ?? p.station_id}</b> (${p.station_id})<br/>${p.basin ?? ""}${p.elevation_m ? ` · ${p.elevation_m} m` : ""}` +
            (latest.current.points?.valueKey && p[latest.current.points.valueKey] !== undefined
              ? `<br/>${lbl ?? latest.current.points.valueKey}: <b>${(+p[latest.current.points.valueKey]).toFixed(3)}</b>` : ""))
          .addTo(mm);
        latest.current.onStationClick?.(p.station_id);
      });
      mm.on("mouseenter", "points", () => (mm.getCanvas().style.cursor = "pointer"));
      mm.on("mouseleave", "points", () => (mm.getCanvas().style.cursor = ""));
      ready.current = true;
      apply();
    });
    });
    return () => { cancelled = true; m?.remove(); map.current = null; ready.current = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function apply() {
    const m = map.current;
    if (!m || !ready.current) return;
    const p = latest.current;
    const vis = (id: string, on: boolean) => m.getLayer(id) && m.setLayoutProperty(id, "visibility", on ? "visible" : "none");
    vis("basins", !!p.showBasins); vis("basins-fill", !!p.showBasins); vis("basin-labels", !!p.showBasins);
    vis("rivers", !!p.showRivers); vis("osm", p.basemap !== false);
    vis("r-dem", p.raster === "dem"); vis("r-slope", p.raster === "slope");
    const src = m.getSource("points") as ML.GeoJSONSource | undefined;
    if (src && p.points) {
      src.setData(p.points.data);
      const k = p.points.valueKey;
      const vals = k ? p.points.data.features.map((f: any) => f.properties?.[k]).filter((v: any) => typeof v === "number") : [];
      if (k && vals.length) {
        const lo = Math.min(...vals), hi = Math.max(...vals);
        const ramp = RAMPS[p.points.ramp ?? "blue"];
        const stops: any[] = [];
        ramp.forEach((c, i) => stops.push(lo + ((hi - lo || 1) * i) / (ramp.length - 1), c));
        m.setPaintProperty("points", "circle-color", ["case", ["==", ["typeof", ["get", k]], "number"],
          ["interpolate", ["linear"], ["get", k], ...stops], "#9ca3af"]);
        m.setPaintProperty("points", "circle-radius", 8);
      } else {
        m.setPaintProperty("points", "circle-color", "#2a6f97");
      }
    }
  }

  useEffect(apply);  // re-apply on every prop change
  return <div ref={ref} className="map" style={{ height }} />;
}

export function Legend({ values, ramp = "blue", label }: { values: number[]; ramp?: "blue" | "heat" | "risk"; label: string }) {
  const v = values.filter((x) => typeof x === "number" && isFinite(x));
  if (!v.length) return null;
  const lo = Math.min(...v), hi = Math.max(...v);
  return (
    <div className="small muted" style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
      <span>{label}</span><span>{lo.toFixed(2)}</span>
      <span style={{ width: 160, height: 10, borderRadius: 5, background: `linear-gradient(90deg, ${RAMPS[ramp].join(",")})` }} />
      <span>{hi.toFixed(2)}</span>
    </div>
  );
}
