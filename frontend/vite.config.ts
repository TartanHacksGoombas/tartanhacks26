import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    esbuildOptions: {
      // Use native class fields — prevents __publicField helper which breaks MapLibre v5
      target: "es2022"
    }
  },
  build: {
    target: "es2022"
  }
});
