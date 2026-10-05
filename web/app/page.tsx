"use client";
import Link from "next/link";
import { Caveat, Stat, Status, useApi } from "@/components/ui";
import { fmt } from "@/lib/api";

const CARDS = [
  ["/climate", "Explore Climate", "Station & regional precipitation / temperature series, climatologies, SPI"],
  ["/extremes", "Explore Extremes", "Event taxonomy, frequencies, seasonality, hydroclimatic whiplash"],
  ["/map", "Explore Map", "Stations, basins, rivers, elevation, slope, climate & extreme layers"],
  ["/run", "Run Model", "Probability + confidence + uncertainty + explanation, fully versioned"],
  ["/compare", "Compare Models", "Baselines vs deep vs multimodal across spatial/temporal splits"],
  ["/experiments", "Research Experiments", "Registry with ablation, transfer, uncertainty, failure analysis"],
  ["/dataset", "Dataset", "Provenance, QC flags, homogeneity, versioning"],
  ["/docs", "Documentation", "Research questions, hypotheses, methods and API"],
];

export default function Home() {
  const h = useApi<any>("/health");
  const st = useApi<any[]>("/stations");
  const qc = useApi<any>("/qc/summary");
  const ex = useApi<any[]>("/experiments");
  const s = qc.data?.summary;
  return (
    <>
      <h1>HydroGeoAI-Nepal</h1>
      <p className="sub">Can multimodal geospatial-temporal representation learning improve detection of hydroclimatic extremes
        across heterogeneous, data-sparse mountain regions?</p>
      <Status loading={h.loading} error={h.error} />
      <div className="stats">
        <Stat label="Stations" value={st.data?.length} />
        <Stat label="Dataset version" value={<span className="mono">{h.data?.dataset_version}</span>} />
        <Stat label="Deployed model" value={<span className="mono small">{h.data?.deployed_model ?? "none"}</span>} />
        <Stat label="QC errors / suspicious" value={s ? `${s.n_errors} / ${s.n_suspicious}` : undefined} />
        <Stat label="Missing precip (after QC)" value={s ? `${fmt(s.missing_pct?.precip, 1)} %` : undefined} />
        <Stat label="Experiments" value={ex.data?.length} />
      </div>
      {h.data?.admin_key_is_default && <Caveat>Admin key is the default (<code>change-me</code>). Set HYDROGEOAI_ADMIN_KEY before sharing the API.</Caveat>}
      <Caveat>The default dataset is <b>synthetic</b> (generated for pipeline development). Results shown here validate the
        software and methodology; they are not findings about Nepal&apos;s climate until real observations are ingested.</Caveat>
      <div className="cards">
        {CARDS.map(([href, t, d]) => <Link key={href} href={href} className="card"><b>{t}</b><span>{d}</span></Link>)}
      </div>
    </>
  );
}
