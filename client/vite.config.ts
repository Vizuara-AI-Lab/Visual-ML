import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // Separate vendor chunks for better caching
          "react-vendor": ["react", "react-dom"],
          "router-vendor": ["react-router"],
          "query-vendor": ["@tanstack/react-query"],
          "flow-vendor": ["@xyflow/react"],
          "motion-vendor": ["framer-motion"],
        },
      },
    },
    chunkSizeWarningLimit: 1000, // Increase limit for large chunks
    minify: "esbuild", // Faster minification
    sourcemap: false, // Disable sourcemaps in production for smaller size
  },
  optimizeDeps: {
    include: [
      "react",
      "react-dom",
      "@tanstack/react-query",
      "@xyflow/react",
      "axios",
    ],
  },
});
