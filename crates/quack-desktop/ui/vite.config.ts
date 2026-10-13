import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The app serves the built page from the bundle (`frontendDist`); in
// `cargo tauri dev` Vite serves it on the port `tauri.conf.json` names.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: { port: 5173, strictPort: true },
  build: { outDir: "dist", target: "safari16", sourcemap: false },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
