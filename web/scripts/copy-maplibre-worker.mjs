// MapLibre v6 loads its web worker from a separate ES module that bundlers do not emit.
// Copy the worker + shared chunk into public/ so it is served at /maplibre/.
import { copyFileSync, mkdirSync } from "node:fs";
const src = "node_modules/maplibre-gl/dist";
mkdirSync("public/maplibre", { recursive: true });
for (const f of ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"]) copyFileSync(`${src}/${f}`, `public/maplibre/${f}`);
console.log("maplibre worker copied to public/maplibre/");
