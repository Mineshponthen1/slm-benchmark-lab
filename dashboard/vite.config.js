import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // Served from https://mineshponthen1.github.io/slm-benchmark-lab/ on GitHub Pages
  base: process.env.GITHUB_PAGES ? '/slm-benchmark-lab/' : '/',
  plugins: [react()],
})
