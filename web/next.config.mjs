import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  turbopack: { root: dirname(fileURLToPath(import.meta.url)) },
};
export default nextConfig;
