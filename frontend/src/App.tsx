// ─── App ──────────────────────────────────────────────────────────────────────
// Root component. Wires together:
//   useEngineHealth → EngineScene + ComponentPanel + StatusBar
//
// Keyboard shortcut: Esc closes the side panel.

import { useEffect, useCallback } from 'react'
import { useEngineHealth } from './shared/hooks/useEngineHealth'
import { EngineScene } from './3d/EngineScene'
import { ComponentPanel } from './dashboard/ComponentPanel'
import { healthScoreToHex, healthScoreToStatus } from './shared/utils/healthColor'

const ENGINE_ID = 'ENG-01'

const STATUS_DOT_COLOR: Record<string, string> = {
  green:  '#22c55e',
  yellow: '#eab308',
  orange: '#f97316',
  red:    '#ef4444',
}

export default function App() {
  const { message, selectedComponent, setSelectedComponent, isMock } = useEngineHealth(ENGINE_ID)

  // Close panel on Escape
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelectedComponent(null)
    },
    [setSelectedComponent]
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  const overallScore = message?.overall_health_score ?? null
  const overallStatus = overallScore !== null ? healthScoreToStatus(overallScore) : 'green'
  const overallHex = overallScore !== null ? healthScoreToHex(overallScore) : '#22c55e'
  const ts = message?.timestamp ? new Date(message.timestamp).toLocaleTimeString() : '—'

  return (
    <div className="app">
      {/* ── 3D Canvas ─────────────────────────────────────────────────────── */}
      <div className="canvas-wrapper">

        {/* ── Status overlay (top-left) ─────────────────────────────────── */}
        <div className="status-bar">
          <span className="status-bar__engine-id">{ENGINE_ID}</span>
          <div className="status-bar__row">
            <span
              className="status-bar__score"
              style={{ color: overallHex }}
            >
              {overallScore ?? '—'}
            </span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <span
                className="status-bar__badge"
                style={{
                  background: `${overallHex}22`,
                  color: overallHex,
                  border: `1px solid ${overallHex}55`,
                }}
              >
                <span
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: '50%',
                    background: STATUS_DOT_COLOR[overallStatus],
                    display: 'inline-block',
                  }}
                />
                {overallStatus.toUpperCase()}
              </span>
              <span className="status-bar__label">Overall Health</span>
            </div>
          </div>
          {isMock && (
            <div style={{ marginTop: 8 }}>
              <span className="status-bar__badge" style={{ background: '#ef444422', color: '#ef4444', border: '1px solid #ef444455' }}>
                MOCK DATA
              </span>
            </div>
          )}
          {message?.components?.find(c => c.component === 'sensor_drift' && c.status !== 'green') && (
            <div style={{ marginTop: 8 }}>
              <span className="status-bar__badge" style={{ background: '#f9731622', color: '#f97316', border: '1px solid #f9731655' }}>
                SENSOR DRIFT
              </span>
            </div>
          )}
          <span className="status-bar__timestamp">Updated {ts}</span>
        </div>

        {/* ── Hint bar (bottom-center) ──────────────────────────────────── */}
        <div className="hint">
          Click a component to inspect · Drag to orbit · Scroll to zoom
        </div>

        <EngineScene
          message={message}
          onComponentClick={setSelectedComponent}
        />
      </div>

      {/* ── Side panel (right edge) ────────────────────────────────────────── */}
      <ComponentPanel
        component={selectedComponent}
        message={message}
        onClose={() => setSelectedComponent(null)}
      />
    </div>
  )
}
