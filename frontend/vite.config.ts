import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

/**
 * The dev server proxies /api to the FastAPI backend so that the browser sees a
 * single origin during development.  The production build is emitted to
 * frontend/dist, which siliconstat.api.main mounts as a SPA when it exists.
 */
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': '/src',
    },
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    // plotly.js-cartesian is a ~3 MB vendor bundle; the warning is expected.
    chunkSizeWarningLimit: 4096,
  },
})
