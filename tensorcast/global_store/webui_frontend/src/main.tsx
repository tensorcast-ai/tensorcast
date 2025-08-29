/// <reference types="vite/client" />
import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ConfigProvider, theme } from 'antd'
import App from './App'
import { WebSocketProvider } from '@/contexts/WebSocketContext'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000, // 5 seconds
      refetchInterval: 5000, // Auto-refresh every 5 seconds
    },
  },
})

const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
const savedTheme = localStorage.getItem('theme') || systemTheme

async function prepareMocks() {
  if (import.meta.env.DEV && import.meta.env.VITE_MOCK_API === 'true') {
    const { worker } = await import('./mocks/browser')
    await worker.start({ onUnhandledRequest: 'bypass' })
    // eslint-disable-next-line no-console
    console.info('%c[Mock API] MSW enabled', 'color:#4caf50')

    // Load test utilities in development
    await import('./mocks/test-mock-api')
    console.info('%c[Mock API] Test utilities loaded. Run testMockAPI() in console to test endpoints', 'color:#2196f3')
  }
}

function renderApp() {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <WebSocketProvider>
        <QueryClientProvider client={queryClient}>
          <ConfigProvider
            theme={{
              algorithm: savedTheme === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
            }}
          >
            <App />
          </ConfigProvider>
        </QueryClientProvider>
      </WebSocketProvider>
    </React.StrictMode>
  )
}

// Initialise mocks first (if enabled), then render react tree
prepareMocks().then(renderApp)