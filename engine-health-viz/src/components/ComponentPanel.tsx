// ─── ComponentPanel ───────────────────────────────────────────────────────────
// Slide-in side panel showing per-component health details.
// Animates open/closed via CSS transform (no animation library needed).

import type { ComponentHealth, HealthIndexMessage } from '../types/healthIndex'
import { healthScoreToHex, healthScoreToStatus } from '../utils/healthColor'

interface ComponentPanelProps {
  component: ComponentHealth | null
  message: HealthIndexMessage | null
  onClose: () => void
}

const STATUS_LABELS: Record<string, string> = {
  green:  'Healthy',
  yellow: 'Degraded',
  orange: 'Warning',
  red:    'Critical',
}

export function ComponentPanel({ component, message, onClose }: ComponentPanelProps) {
  const isOpen = component !== null
  const score = component?.health_score ?? 0
  const status = healthScoreToStatus(score)
  const hex = healthScoreToHex(score)

  return (
    <div
      className={`component-panel${isOpen ? ' open' : ''}`}
      role="complementary"
      aria-label="Component health details"
      aria-hidden={!isOpen}
    >
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="panel__header">
        <span className="panel__title">Component Inspector</span>
        <button
          className="panel__close"
          onClick={onClose}
          aria-label="Close panel"
          title="Close (Esc)"
        >
          ✕
        </button>
      </div>

      {component && (
        <div className="panel__body">

          {/* ── Component name ─────────────────────────────────────────── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
              Component
            </span>
            <span style={{ fontSize: 20, fontWeight: 700, fontFamily: "'SF Mono','Fira Code',monospace", color: 'var(--text-primary)' }}>
              {component.component.replace(/_/g, ' ').toUpperCase()}
            </span>
          </div>

          {/* ── Score card ─────────────────────────────────────────────── */}
          <div className="score-card">
            <div
              className="score-card__circle"
              style={{ color: hex, borderColor: hex }}
            >
              {score}
            </div>
            <div className="score-card__info">
              <span className="score-card__status" style={{ color: hex }}>
                {STATUS_LABELS[status] ?? status}
              </span>
              <span className="score-card__sub">Health Score</span>
              <span className="score-card__sub" style={{ color: hex, fontWeight: 600 }}>
                {component.status.toUpperCase()}
              </span>
            </div>
          </div>

          {/* ── Health bar ─────────────────────────────────────────────── */}
          <div className="health-bar-container">
            <span className="health-bar-label">Score Gauge</span>
            <div className="health-bar-track">
              <div
                className="health-bar-fill"
                style={{ width: `${score}%`, backgroundColor: hex }}
              />
            </div>
          </div>

          {/* ── RUL ────────────────────────────────────────────────────── */}
          <div className="metric-row">
            <span className="metric-row__label">Remaining Useful Life</span>
            <span className="metric-row__value">
              {component.rul_hours.toFixed(1)}
              <span className="metric-row__unit">hrs</span>
            </span>
          </div>

          {/* ── Top features ───────────────────────────────────────────── */}
          <div className="feature-tags">
            <span className="feature-tags__label">Top Contributing Features</span>
            <div className="feature-tags__list">
              {component.top_features.map((f) => (
                <span key={f} className="feature-tag">{f}</span>
              ))}
            </div>
          </div>

          {/* ── Active faults (engine-wide) ────────────────────────────── */}
          {message && (
            <div className="faults-section">
              <span className="faults-section__label">
                Active Faults — {message.engine_id}
              </span>
              {message.active_faults.length === 0 ? (
                <span className="no-faults">✓ No active faults</span>
              ) : (
                message.active_faults.map((f) => (
                  <div key={f} className="fault-item">⚠ {f.replace(/_/g, ' ')}</div>
                ))
              )}
            </div>
          )}

        </div>
      )}
    </div>
  )
}
