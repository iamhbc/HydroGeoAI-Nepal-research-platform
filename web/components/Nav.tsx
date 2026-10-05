"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV } from "@/lib/plain";

export default function Nav() {
  const p = usePathname();
  const active = (h: string) => (h === "/" ? p === "/" : p.startsWith(h));
  return (
    <aside className="side">
      <Link href="/" className="brand" aria-label="HydroGeoAI-Nepal home">
        <svg width="26" height="26" viewBox="0 0 24 24" aria-hidden="true"><path d="M2 20 9 7l4 7 3-4 6 10z" fill="var(--accent)" /><circle cx="17" cy="5" r="2.2" fill="var(--c2)" /></svg>
        <span>HydroGeoAI<small>Nepal</small></span>
      </Link>
      <nav className="nav" aria-label="Main">
        {NAV.map((g, i) => (
          <div key={i} className="nav-group">
            {g.section && <div className="nav-h">{g.section}</div>}
            {g.links.map(([h, l]) => (
              <Link key={h} href={h} className={active(h) ? "active" : ""} aria-current={active(h) ? "page" : undefined}>{l}</Link>
            ))}
          </div>
        ))}
      </nav>
    </aside>
  );
}
