import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const isMockMode = env.VITE_MOCK_API === 'true'

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: 3000,
      proxy: isMockMode ? {} : {
        '/api': {
          target: 'http://localhost:9000',
          changeOrigin: true,
        },
        '/ws': {
          target: 'ws://localhost:9000',
          ws: true,
        },
      },
    },
    build: {
      outDir: '../tensorcast/global_store/webui_backend/build',
      emptyOutDir: true,
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom', 'react-router-dom'],
            ui: ['antd', '@ant-design/icons'],
            charts: ['recharts'],
          },
        },
      },
    },
  }
})