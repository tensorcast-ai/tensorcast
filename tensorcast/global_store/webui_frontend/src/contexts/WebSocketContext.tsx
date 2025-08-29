import { createContext, useContext, useEffect, useRef, useState, ReactNode } from 'react'

export interface WebSocketMessage {
  type: string
  timestamp: string
  updates: Array<{
    topic: string
    payload: Record<string, any>
    timestamp: string
  }>
}

type MsgHandler = (msg: WebSocketMessage) => void

interface WSContextValue {
  /** Send raw data to backend, will JSON.stringify automatically */
  send: (data: any) => void
  /** Register a message handler */
  addHandler: (fn: MsgHandler) => void
  /** Unregister message handler */
  removeHandler: (fn: MsgHandler) => void
  /** Current connection state */
  connectionState: 'connecting' | 'open' | 'closed'
}

const WebSocketContext = createContext<WSContextValue | null>(null)

export const WebSocketProvider = ({ children }: { children: ReactNode }) => {
  const wsRef = useRef<WebSocket | null>(null)
  const handlersRef = useRef(new Set<MsgHandler>())
  const [state, setState] = useState<'connecting' | 'open' | 'closed'>('connecting')

  const isMockMode = import.meta.env.VITE_MOCK_API === 'true'

  /** Establish or re-establish websocket connection */
  const connect = () => {
    if (isMockMode) {
      // In mock mode do not connect – but keep state as open for consumer convenience
      setState('open')
      return
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${protocol}//${window.location.host}/ws/stream`
    const ws = new WebSocket(url)

    wsRef.current = ws
    setState('connecting')

    ws.onopen = () => {
      setState('open')
      // Default subscription – keep in sync with server side topics
      ws.send(
        JSON.stringify({
          action: 'subscribe',
          topics: ['heartbeat', 'replica_update', 'transport'],
        })
      )
    }

    ws.onmessage = (e) => {
      let msg: WebSocketMessage | null = null
      try {
        msg = JSON.parse(e.data)
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('[WebSocket] failed to parse message', err)
      }
      if (!msg) return
      handlersRef.current.forEach((fn) => {
        try {
          fn(msg!)
        } catch (err) {
          // eslint-disable-next-line no-console
          console.error('[WebSocket] handler error', err)
        }
      })
    }

    ws.onclose = () => {
      setState('closed')
      // Retry connection after 5s
      setTimeout(connect, 5000)
    }

    ws.onerror = (event) => {
      // Provide detailed logger information for troubleshooting
      // eslint-disable-next-line no-console
      console.error('[WebSocket] connection error', {
        event,
        readyState: ws.readyState,
        url: ws.url,
      })
      ws.close()
    }
  }

  useEffect(() => {
    connect()

    return () => {
      wsRef.current?.close()
    }
    // Intentionally run once – eslint complain suppressed
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const value: WSContextValue = {
    send: (data) => {
      if (isMockMode) return
      try {
        wsRef.current?.send(typeof data === 'string' ? data : JSON.stringify(data))
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('[WebSocket] send error', err)
      }
    },
    addHandler: (fn) => {
      handlersRef.current.add(fn)
    },
    removeHandler: (fn) => {
      handlersRef.current.delete(fn)
    },
    connectionState: state,
  }

  return <WebSocketContext.Provider value={value}>{children}</WebSocketContext.Provider>
}

export const useWebSocketContext = () => {
  const ctx = useContext(WebSocketContext)
  if (!ctx) {
    throw new Error('useWebSocketContext must be used inside WebSocketProvider')
  }
  return ctx
}