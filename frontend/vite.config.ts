import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ['.ngrok-free.dev'],
    proxy: {
      '/root-finding': 'http://localhost:8002',
      '/nonlinear-system': 'http://localhost:8002',
      '/linear-system': 'http://localhost:8002',
      '/integration': 'http://localhost:8002',
      '/matrix-inverse': 'http://localhost:8002',
      '/matrix-decomposition': 'http://localhost:8002',
      '/eigenvalues': 'http://localhost:8002',
    },
  },
})