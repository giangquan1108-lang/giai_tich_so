import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ['.ngrok-free.dev'],
    proxy: {
      '/root-finding': 'http://localhost:8001',
      '/nonlinear-system': 'http://localhost:8001',
      '/linear-system': 'http://localhost:8001',
      '/interpolation': 'http://localhost:8001',
      '/integration': 'http://localhost:8001',
      '/comparison': 'http://localhost:8001',
    },
  },
})