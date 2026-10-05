export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T = any>(path: string, init?: RequestInit & { adminKey?: string }): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (init?.adminKey) headers["X-API-Key"] = init.adminKey;
  const res = await fetch(`${API}${path}`, { ...init, headers: { ...headers, ...(init?.headers as any) }, cache: "no-store" });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {}
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export const qs = (o: Record<string, string | number | undefined | null>) =>
  "?" + Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join("&");

export type Station = {
  station_id: string; name: string; lat: number; lon: number; elevation_m: number;
  basin: string; province: string; [k: string]: any;
};

export const fmt = (v: any, d = 3) =>
  v === null || v === undefined || (typeof v === "number" && !isFinite(v)) ? "–"
    : typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(d)) : String(v);

export const EVENT_LABELS: Record<string, string> = {
  extreme_wet_day: "Extreme wet day (>P95)", very_extreme_wet_day: "Very extreme wet day (>P99)",
  extreme_rainfall_event: "Extreme 3-day rainfall (>P99)", consecutive_wet_days: "Consecutive wet days (≥5)",
  rainfall_persistence: "Rainfall persistence", dry_spell: "Dry spell (≥15 d)",
  precipitation_deficit: "Precipitation deficit (SPI-30 ≤ −1)", meteorological_drought: "Meteorological drought (SPI-90 ≤ −1)",
  hot_day: "Hot day (TX90p)", warm_spell: "Warm spell (≥3 hot days)", cold_night: "Cold night (TN10p)",
  hot_dry: "Compound hot + dry", wet_hot: "Compound wet + hot", rain_after_dryness: "Extreme rain after dryness",
};
