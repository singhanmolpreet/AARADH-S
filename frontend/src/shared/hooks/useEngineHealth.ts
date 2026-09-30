// ─── useEngineHealth ──────────────────────────────────────────────────────────
// Custom hook that subscribes to engine health messages and manages
// the selected-component state for the side panel.

import { useState, useEffect, useCallback } from 'react'
import { subscribeToEngine } from '../data/engineDataSource'
import type { HealthIndexMessage, ComponentHealth } from '../types/healthIndex'

interface UseEngineHealthReturn {
  /** The latest Health Index message, or null before first message arrives */
  message: HealthIndexMessage | null
  /** The component currently selected (for the side panel), or null */
  selectedComponent: ComponentHealth | null
  /** Call with a component to open the side panel for it */
  setSelectedComponent: (c: ComponentHealth | null) => void
  /** True if the data source is currently mocking data */
  isMock: boolean
}

export function useEngineHealth(engineId: string): UseEngineHealthReturn {
  const [message, setMessage] = useState<HealthIndexMessage | null>(null)
  const [selectedComponent, setSelectedComponent] = useState<ComponentHealth | null>(null)
  const [isMock, setIsMock] = useState<boolean>(false)

  useEffect(() => {
    // subscribeToEngine returns an unsubscribe function — call it on cleanup
    const unsubscribe = subscribeToEngine(engineId, (msg, mockFlag = false) => {
      setMessage(msg)
      setIsMock(mockFlag)

      // If a component is selected, keep the panel data fresh with the new message
      setSelectedComponent((prev) => {
        if (prev === null) return null
        const updated = msg.components.find((c) => c.component === prev.component)
        return updated ?? null
      })
    })

    return unsubscribe
  }, [engineId])

  const handleSetSelected = useCallback((c: ComponentHealth | null) => {
    setSelectedComponent(c)
  }, [])

  return { message, selectedComponent, setSelectedComponent: handleSetSelected, isMock }
}
