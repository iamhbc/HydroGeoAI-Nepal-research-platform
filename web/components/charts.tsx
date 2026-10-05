"use client";
// Lightweight dependency-free SVG charts (line with optional band, bars, horizontal bars).
import { useMemo, useState } from "react";

const PAL = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"];

type Series = { name: string; points: { x: number | string; y: number | null }[]; band?: { lo: number; hi: number }[] };

function ticks(min: number, max: number, n = 5) {
  if (!isFinite(min) || !isFinite(max)) return [];
  if (min === max) return [min];
  const step = Math.pow(10, Math.floor(Math.log10((max - min) / n)));
  const err = ((max - min) / n) / step;
  const s = step * (err >= 7.5 ? 10 : err >= 3 ? 5 : err >= 1.5 ? 2 : 1);
  const out = [];
  for (let v = Math.ceil(min / s) * s; v <= max + 1e-9; v += s) out.push(+v.toFixed(10));
  return out;
}

const toNum = (x: number | string) => (typeof x === "number" ? x : new Date(x).getTime());

export function LineChart({ series, height = 260, yLabel, xIsDate = false, markers, hline }: {
  series: Series[]; height?: number; yLabel?: string; xIsDate?: boolean;
  markers?: { x: number | string; y: number }[]; hline?: { y: number; label: string };
}) {
  const [hover, setHover] = useState<{ x: number; y: number; label: string } | null>(null);
  const W = 760, H = height, m = { l: 56, r: 16, t: 12, b: 34 };
  const all = series.flatMap((s) => s.points.filter((p) => p.y !== null && isFinite(p.y as number)));
  const xs = all.map((p) => toNum(p.x));
  const ys = [...all.map((p) => p.y as number), ...series.flatMap((s) => s.band?.flatMap((b) => [b.lo, b.hi]) ?? []),
              ...(hline ? [hline.y] : [])];
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (y0 > 0 && y0 / (y1 || 1) < 0.35) y0 = 0;
  const pad = (y1 - y0) * 0.05 || 1;
  y1 += pad; if (y0 !== 0) y0 -= pad;
  const sx = (x: number) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const sy = (y: number) => H - m.b - ((y - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const xt = useMemo(() => {
    if (!xIsDate) return ticks(x0, x1, 6).map((v) => ({ v, l: String(v) }));
    const n = 6, out = [];
    for (let i = 0; i <= n; i++) {
      const v = x0 + ((x1 - x0) * i) / n;
      const d = new Date(v);
      out.push({ v, l: (x1 - x0) > 3 * 365 * 864e5 ? String(d.getUTCFullYear()) : d.toISOString().slice(0, 10) });
    }
    return out;
  }, [x0, x1, xIsDate]);
  if (!all.length) return <div className="empty">No data for this selection.</div>;
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={yLabel}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const px = ((e.clientX - r.left) / r.width) * W;
          const xv = x0 + ((px - m.l) / (W - m.l - m.r)) * (x1 - x0);
          const s = series[0];
          let best = s.points[0], bd = Infinity;
          for (const p of s.points) { const d = Math.abs(toNum(p.x) - xv); if (d < bd && p.y !== null) { bd = d; best = p; } }
          if (best && best.y !== null) setHover({ x: sx(toNum(best.x)), y: sy(best.y), label: `${xIsDate ? String(best.x).slice(0, 10) : best.x}: ${(+best.y).toFixed(3)}` });
        }}>
        {ticks(y0, y1).map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={m.l - 6} y={sy(t) + 4} textAnchor="end" className="tick">{Math.abs(t) >= 1000 ? t.toFixed(0) : +t.toFixed(3)}</text>
          </g>
        ))}
        {xt.map((t, i) => <text key={i} x={sx(t.v)} y={H - m.b + 16} textAnchor="middle" className="tick">{t.l}</text>)}
        {yLabel && <text x={14} y={m.t + (H - m.t - m.b) / 2} transform={`rotate(-90 14 ${m.t + (H - m.t - m.b) / 2})`} textAnchor="middle" className="axis">{yLabel}</text>}
        {series.map((s, i) => s.band && (
          <path key={`b${i}`} fill={PAL[i % 6]} opacity={0.18}
            d={"M" + s.points.map((p, j) => `${sx(toNum(p.x))},${sy(s.band![j].hi)}`).join("L") + "L" +
              [...s.points].reverse().map((p, j) => `${sx(toNum(p.x))},${sy(s.band![s.points.length - 1 - j].lo)}`).join("L") + "Z"} />
        ))}
        {hline && <g><line x1={m.l} x2={W - m.r} y1={sy(hline.y)} y2={sy(hline.y)} className="hline" />
          <text x={W - m.r - 4} y={sy(hline.y) - 4} textAnchor="end" className="tick">{hline.label}</text></g>}
        {series.map((s, i) => {
          let d = "", pen = false;
          for (const p of s.points) {
            if (p.y === null || !isFinite(p.y)) { pen = false; continue; }
            d += `${pen ? "L" : "M"}${sx(toNum(p.x))},${sy(p.y)}`; pen = true;
          }
          return <path key={i} d={d} fill="none" stroke={PAL[i % 6]} strokeWidth={1.8} />;
        })}
        {markers?.map((p, i) => <circle key={i} cx={sx(toNum(p.x))} cy={sy(p.y)} r={3.5} className="marker" />)}
        {hover && <g><circle cx={hover.x} cy={hover.y} r={4} fill="var(--fg)" />
          <text x={Math.min(hover.x + 8, W - 150)} y={Math.max(hover.y - 8, 14)} className="hover">{hover.label}</text></g>}
      </svg>
      {series.length > 1 && <div className="legend">{series.map((s, i) => <span key={i}><i style={{ background: PAL[i % 6] }} />{s.name}</span>)}</div>}
    </div>
  );
}

export function BarChart({ data, height = 220, yLabel, color = "var(--c1)" }: {
  data: { label: string; value: number | null }[]; height?: number; yLabel?: string; color?: string;
}) {
  const W = 760, H = height, m = { l: 56, r: 12, t: 12, b: 40 };
  const vals = data.map((d) => d.value ?? 0);
  const y1 = Math.max(...vals, 0) * 1.08 || 1, y0 = Math.min(0, ...vals);
  const bw = (W - m.l - m.r) / Math.max(data.length, 1);
  const sy = (y: number) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
  if (!data.length) return <div className="empty">No data.</div>;
  return (
    <div className="chart"><svg viewBox={`0 0 ${W} ${H}`}>
      {ticks(y0, y1).map((t) => <g key={t}><line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} className="grid" />
        <text x={m.l - 6} y={sy(t) + 4} textAnchor="end" className="tick">{+t.toFixed(3)}</text></g>)}
      {data.map((d, i) => (
        <g key={i}>
          <rect x={m.l + i * bw + bw * 0.15} width={bw * 0.7} y={sy(Math.max(d.value ?? 0, 0))}
            height={Math.abs(sy(d.value ?? 0) - sy(0))} fill={color} rx={2}><title>{`${d.label}: ${fmtv(d.value)}`}</title></rect>
          {data.length <= 24 && <text x={m.l + i * bw + bw / 2} y={H - m.b + 14} textAnchor="middle" className="tick">{d.label}</text>}
        </g>))}
      {yLabel && <text x={14} y={H / 2} transform={`rotate(-90 14 ${H / 2})`} textAnchor="middle" className="axis">{yLabel}</text>}
    </svg></div>
  );
}

export function HBars({ data, unit = "" }: { data: { label: string; value: number }[]; unit?: string }) {
  const max = Math.max(...data.map((d) => Math.abs(d.value)), 1e-9);
  return (
    <div className="hbars">
      {data.map((d) => (
        <div key={d.label} className="hbar">
          <span className="hl">{d.label}</span>
          <span className="ht"><i style={{ width: `${(Math.abs(d.value) / max) * 100}%`, background: d.value < 0 ? "var(--c4)" : "var(--c1)" }} /></span>
          <span className="hv">{fmtv(d.value)}{unit}</span>
        </div>))}
    </div>
  );
}

const fmtv = (v: number | null) => (v === null ? "–" : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(2) : (+v).toFixed(3));
