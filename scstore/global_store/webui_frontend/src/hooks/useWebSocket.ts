import { useEffect, useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useWebSocketContext, WebSocketMessage } from '@/contexts/WebSocketContext'

/**
 * Hook that registers WebSocket message listeners and optionally a custom handler.
 * It keeps the previous behaviour of automatically invalidating react-query caches
 * on certain topic updates so that existing pages that call `useWebSocket()`
 * without arguments continue to work.
 */
export const useWebSocket = (onMessage?: (msg: WebSocketMessage) => void) => {
  const { addHandler, removeHandler } = useWebSocketContext()
  const queryClient = useQueryClient()

  // Default handler that mirrors the behaviour of the previous implementation.
  const defaultHandler = useCallback(
    (message: WebSocketMessage) => {
      if (message.type !== 'updates') return

      message.updates.forEach((update) => {
        switch (update.topic) {
          case 'heartbeat':
            queryClient.invalidateQueries({ queryKey: ['workers'] })
            break
          case 'replica_update':
            queryClient.invalidateQueries({ queryKey: ['replicas'] })
            queryClient.invalidateQueries({ queryKey: ['models'] })
            break
          case 'transport':
            queryClient.invalidateQueries({ queryKey: ['transports'] })
            queryClient.invalidateQueries({ queryKey: ['summary'] })
            break
          default:
            break
        }
      })
    },
    [queryClient]
  )

  useEffect(() => {
    // Always register default handler so cache invalidation keeps working.
    addHandler(defaultHandler)

    // Register optional user handler.
    if (onMessage) {
      addHandler(onMessage)
    }

    // Cleanup on unmount.
    return () => {
      removeHandler(defaultHandler)
      if (onMessage) {
        removeHandler(onMessage)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [addHandler, removeHandler, defaultHandler, onMessage])
}