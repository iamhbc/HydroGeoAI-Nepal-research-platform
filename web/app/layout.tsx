import type { Metadata } from "next";
import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "HydroGeoAI-Nepal",
  description: "Multimodal geospatial-temporal AI for hydroclimatic extremes in Nepal",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <Nav />
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
