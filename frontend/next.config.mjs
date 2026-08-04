import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  // There is a stray package-lock.json in the home directory, so Next walks
  // up and guesses the workspace root is ~ instead of this folder. Pin it.
  outputFileTracingRoot: dirname(fileURLToPath(import.meta.url)),

  // Proxy the API through Next so the browser only ever talks to one origin.
  // No CORS, no mixed ports, and the same build works behind any reverse proxy.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
