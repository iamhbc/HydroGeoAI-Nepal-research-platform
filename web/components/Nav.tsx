"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  ["/", "Dashboard"], ["/climate", "Explore Climate"], ["/extremes", "Explore Extremes"], ["/map", "Explore Map"],
  ["/run", "Run Model"], ["/compare", "Compare Models"], ["/experiments", "Research Experiments"],
  ["/dataset", "Dataset & QC"], ["/docs", "Documentation"],
];

export default function Nav() {
  const p = usePathname();
  const active = (h: string) => (h === "/" ? p === "/" : p.startsWith(h));
  return (
    <aside className="side">
      <div className="brand">HydroGeoAI-Nepal<small>Hydroclimatic extremes · GeoAI research platform</small></div>
      <nav className="nav">
        {LINKS.map(([h, l]) => <Link key={h} href={h} className={active(h) ? "active" : ""}>{l}</Link>)}
        <div className="sep" />
        <Link href="/admin" className={active("/admin") ? "active" : ""}>Admin / Researcher</Link>
      </nav>
    </aside>
  );
}
