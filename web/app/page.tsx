// Home page. Server-rendered from one cached API call (/overview), so it shows content at first paint.
import Link from "next/link";
import { API } from "@/lib/api";
import { GLOSSARY, TERMS } from "@/lib/plain";

export const revalidate = 60;

type Overview = {
  dataset_version: string; synthetic: boolean; period: [string, string]; n_stations: number; n_basins: number;
  elevation_range_m: [number, number];
  last_12_months: { extreme_wet_days: number; hot_days: number; dry_spell_days: number; whiplash_events: number };
  data_quality: { errors_removed: number; missing_rain_pct: number };
  model: { id: string; trained_on: string; auprc: number } | null;
};

async function getOverview(): Promise<Overview | null> {
  try {
    const r = await fetch(`${API}/overview`, { next: { revalidate: 60 }, signal: AbortSignal.timeout(8000) });
    return r.ok ? r.json() : null;
  } catch {
    return null;
  }
}

const Icon = ({ d }: { d: string }) => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={d} /></svg>
);

const TASKS = [
  { href: "/map", title: "See the map", text: "See all weather stations, rivers and river basins on a map of Nepal.",
    icon: "M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2zM9 4v14M15 6v14" },
  { href: "/climate", title: "Check rain and temperature", text: "See how rain and temperature change over time at one place.",
    icon: "M3 17l5-6 4 4 8-9M3 21h18" },
  { href: "/extremes", title: "Find extreme events", text: "Find days with very heavy rain, high heat or no rain.",
    icon: "M13 2 4 14h7l-1 8 9-12h-7z" },
  { href: "/run", title: "Get a forecast", text: "Get the chance of an extreme rain day for the next day.",
    icon: "M12 3v3M5.6 5.6l2.1 2.1M3 12h3M18 12h3M16.3 7.7l2.1-2.1M7 17a5 5 0 1 1 10 0z" },
];

export default async function Home() {
  const o = await getOverview();
  const n = (v: number) => v.toLocaleString("en-US");
  return (
    <div className="home">
      <section className="hero">
        <p className="eyebrow">Weather extremes in the mountains of Nepal</p>
        <h1>Find and understand extreme weather in Nepal.</h1>
        <p className="lead">This system shows rain, heat and dry periods at weather stations in Nepal.
          It also tells you how sure the model is about each forecast.</p>
        <div className="actions">
          <Link href="/map" className="btn btn-primary">Open the map</Link>
          <Link href="/run" className="btn btn-secondary">Get a forecast</Link>
        </div>
      </section>

      {!o && (
        <div className="notice notice-error" role="alert">
          <strong>The data service does not respond.</strong> Start it in a terminal with <code>make api</code>. Then refresh this page.
        </div>
      )}
      {o?.synthetic && (
        <div className="notice" role="note">
          <strong>Test data.</strong> The data on this site is simulated. Use it to learn the system. Do not use it to make decisions.
        </div>
      )}

      <section aria-labelledby="tasks-h">
        <h2 id="tasks-h">What do you want to do?</h2>
        <div className="tasks">
          {TASKS.map((t) => (
            <Link key={t.href} href={t.href} className="task">
              <span className="task-icon"><Icon d={t.icon} /></span>
              <span className="task-title">{t.title}</span>
              <span className="task-text">{t.text}</span>
              <span className="task-go" aria-hidden="true">Start →</span>
            </Link>
          ))}
        </div>
      </section>

      {o && (
        <section aria-labelledby="glance-h">
          <h2 id="glance-h">The last 12 months</h2>
          <p className="section-sub">Totals for all {o.n_stations} stations, up to {o.period[1]}.</p>
          <div className="glance">
            <Tile value={n(o.last_12_months.extreme_wet_days)} label={TERMS.extremeRain + "s"} href="/extremes" />
            <Tile value={n(o.last_12_months.hot_days)} label={TERMS.hotDay + "s"} href="/extremes" />
            <Tile value={n(o.last_12_months.dry_spell_days)} label="Days in a dry period" href="/extremes" />
            <Tile value={n(o.last_12_months.whiplash_events)} label={TERMS.suddenChange + "s"} href="/extremes" />
          </div>
        </section>
      )}

      <section aria-labelledby="how-h">
        <h2 id="how-h">How it works</h2>
        <ol className="steps">
          <li><b>Collect.</b> We collect daily rain and temperature data from weather stations.</li>
          <li><b>Check.</b> We find and remove bad values. We keep a record of each change.</li>
          <li><b>Predict.</b> The model gives the chance of an event. It also tells you how sure it is.</li>
        </ol>
      </section>

      <section aria-labelledby="read-h" className="split">
        <div>
          <h2 id="read-h">How to read a forecast</h2>
          <div className="example" aria-label="Example forecast">
            <div><span className="ex-k">{TERMS.chance}</span><span className="ex-v">12%</span></div>
            <div><span className="ex-k">{TERMS.confidence}</span><span className="ex-v">0.72</span></div>
            <div><span className="ex-k">{TERMS.uncertainty}</span><span className="pill moderate">moderate</span></div>
          </div>
          <p className="small muted">Low uncertainty: you can use the forecast. High uncertainty: check other sources too.</p>
        </div>
        <div>
          <h2>Words on this site</h2>
          <dl className="glossary">
            {GLOSSARY.map((g) => <div key={g.term}><dt>{g.term}</dt><dd>{g.text}</dd></div>)}
          </dl>
        </div>
      </section>

      <section aria-labelledby="res-h">
        <h2 id="res-h">For researchers</h2>
        <div className="reslinks">
          <Link href="/compare">Compare models</Link>
          <Link href="/experiments">Experiments</Link>
          <Link href="/dataset">Data quality</Link>
          <Link href="/docs">Help and methods</Link>
          <Link href="/admin">Admin</Link>
        </div>
      </section>

      <footer className="status" aria-label="System status">
        <span className={`dot ${o ? "ok" : "bad"}`} aria-hidden="true" />
        {o ? (
          <span>System ready · {o.n_stations} stations · {o.n_basins} river basins · {o.elevation_range_m[0]}–{o.elevation_range_m[1]} m ·
            {" "}{o.period[0].slice(0, 4)}–{o.period[1].slice(0, 4)} · data <code>{o.dataset_version}</code>
            {o.model && <> · model <code>{o.model.id.split("@")[1]}</code></>}</span>
        ) : <span>Data service not available</span>}
      </footer>
    </div>
  );
}

function Tile({ value, label, href }: { value: string; label: string; href: string }) {
  return (
    <Link href={href} className="tile">
      <span className="tile-v">{value}</span>
      <span className="tile-l">{label}</span>
    </Link>
  );
}
