/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Built into the Python package (committed), served by FastAPI at /app/.
export default defineConfig({
  plugins: [react()],
  base: "/app/",
  build: {
    outDir: "../static/app",
    emptyOutDir: true,
    chunkSizeWarningLimit: 4000,
  },
  server: {
    proxy: { "/api": "http://127.0.0.1:8765", "/static": "http://127.0.0.1:8765" },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
  },
});
