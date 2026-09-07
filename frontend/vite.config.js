import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  appType: 'spa',
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:5050',
      '/stream': {
        target: 'ws://localhost:5050',
        ws: true,
      },
    },
  },
});
