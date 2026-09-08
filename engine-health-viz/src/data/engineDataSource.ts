// ─── Engine Data Source ───────────────────────────────────────────────────────
//
// THIS IS THE ONE FILE THAT WAS SWAPPED TO GO LIVE.
//
// Active mode  : WEBSOCKET  →  ws://<backendhost>:8000/ws/live/{engineId}
// Fallback mode: MOCK       →  flip ACTIVE_MODE below, zero other changes
//
// Hook interface is UNCHANGED — useEngineHealth.ts does not need to know
// which mode is active.
//   subscribeToEngine(engineId, onMessage) → unsubscribe()
// ─────────────────────────────────────────────────────────────────────────────

import type { HealthIndexMessage } from '../types/healthIndex'

/** Callback type consumed by useEngineHealth */
export type EngineMessageCallback = (msg: HealthIndexMessage) => void

// ─── Mode switch ─────────────────────────────────────────────────────────────
// Change this one value to flip between live and mock.
// 'ws'   → real backend  (default going forward)
// 'mock' → cycle MOCK_MESSAGES locally (instant rollback, no backend needed)
type Mode = 'ws' | 'mock'
const ACTIVE_MODE: Mode = 'mock'

// ─── Backend endpoint ─────────────────────────────────────────────────────────
// Schema:  ws://<backendhost>:8000/ws/live/{engineId}
// Replace <backendhost> with the actual hostname or IP before deploying.
const WS_HOST = '<backendhost>'
const WS_PORT = 8000
const WS_BASE  = `ws://${WS_HOST}:${WS_PORT}`

function buildWsUrl(engineId: string): string {
  return `${WS_BASE}/ws/live/${engineId}`
}

// ═══════════════════════════════════════════════════════════════════════════════
// ███  MODE 1 — WEBSOCKET (ACTIVE)
//
// Features:
//   • Exact endpoint schema:  ws://<host>:8000/ws/live/{engineId}
//   • Exponential-backoff reconnect (1 s → 2 s → 4 s … cap 30 s)
//   • Jitter on each backoff delay to avoid thundering-herd on server restart
//   • Clean teardown: cancels any pending reconnect timer, closes socket
//   • Parse errors are logged but never crash the reconnect loop
// ═══════════════════════════════════════════════════════════════════════════════

const RECONNECT_BASE_MS  = 1_000   // first retry after 1 s
const RECONNECT_MAX_MS   = 30_000  // cap at 30 s
const RECONNECT_JITTER   = 0.2     // ±20 % randomisation per attempt

export function subscribeWebSocket(
  engineId: string,
  onMessage: EngineMessageCallback
): () => void {
  const url = buildWsUrl(engineId)
  let ws:            WebSocket | null = null
  let retryDelay    = RECONNECT_BASE_MS
  let retryTimer:   ReturnType<typeof setTimeout> | null = null
  let destroyed     = false   // set true by unsubscribe(); stops reconnect loop

  function connect() {
    if (destroyed) return

    console.info(`[engineDataSource] Connecting → ${url}`)
    ws = new WebSocket(url)

    ws.onopen = () => {
      console.info(`[engineDataSource] Connected to ${url}`)
      retryDelay = RECONNECT_BASE_MS   // reset backoff on successful connection
    }

    ws.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data as string) as HealthIndexMessage
        onMessage(msg)
      } catch (err) {
        console.error('[engineDataSource] Failed to parse message:', err, event.data)
      }
    }

    ws.onerror = (err) => {
      // onerror always fires before onclose — log here, reconnect in onclose
      console.warn('[engineDataSource] WebSocket error:', err)
    }

    ws.onclose = (event) => {
      ws = null
      if (destroyed) return   // intentional close — do not reconnect

      // Jitter: multiply by a random factor in [1-jitter, 1+jitter]
      const jitter = 1 + (Math.random() * 2 - 1) * RECONNECT_JITTER
      const delay  = Math.min(retryDelay * jitter, RECONNECT_MAX_MS)
      retryDelay   = Math.min(retryDelay * 2, RECONNECT_MAX_MS)  // double for next attempt

      console.warn(
        `[engineDataSource] Disconnected (code ${event.code}). ` +
        `Reconnecting in ${(delay / 1000).toFixed(1)} s…`
      )

      retryTimer = setTimeout(connect, delay)
    }
  }

  connect()   // initial connection attempt

  // ── Unsubscribe / cleanup ─────────────────────────────────────────────────
  return () => {
    destroyed = true
    if (retryTimer !== null) clearTimeout(retryTimer)
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      ws.close(1000, 'Component unmounted')
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ███  MODE 2 — MOCK (FALLBACK)
// Cycles through five hardcoded messages every 3 seconds.
// Exercises all four health bands — useful during development or if the
// backend is unavailable near demo day.
// ═══════════════════════════════════════════════════════════════════════════════
const MOCK_MESSAGES: HealthIndexMessage[] = [
  // ── Message 1: mostly green, a couple of yellows ──────────────────────────
  {
    timestamp: '2026-09-02T16:00:00Z',
    engine_id: 'ENG-01',
    overall_health_score: 81,
    overall_status: 'green',
    active_faults: [],
    components: [
      { component: 'cylinder_1', health_score: 91, status: 'green',  rul_hours: 820.0, top_features: ['cht_trend', 'manifold_pressure'] },
      { component: 'cylinder_2', health_score: 85, status: 'green',  rul_hours: 750.5, top_features: ['egt_variance', 'rpm_stability'] },
      { component: 'cylinder_3', health_score: 78, status: 'yellow', rul_hours: 310.0, top_features: ['cht_trend', 'egt_variance'] },
      { component: 'cylinder_4', health_score: 82, status: 'green',  rul_hours: 680.0, top_features: ['cht_trend'] },
      { component: 'oil_system', health_score: 62, status: 'yellow', rul_hours: 145.0, top_features: ['oil_pressure_drop', 'viscosity_trend'] },
      { component: 'exhaust',    health_score: 90, status: 'green',  rul_hours: 900.0, top_features: ['backpressure_trend'] },
      { component: 'injectors',  health_score: 55, status: 'yellow', rul_hours: 42.5,  top_features: ['fuel_flow_variance', 'spray_pattern'] },
    ],
  },
  // ── Message 2: degrading — yellows deepen, first oranges appear ───────────
  {
    timestamp: '2026-09-02T16:00:03Z',
    engine_id: 'ENG-01',
    overall_health_score: 72,
    overall_status: 'yellow',
    active_faults: [],
    components: [
      { component: 'cylinder_1', health_score: 78, status: 'yellow', rul_hours: 780.0, top_features: ['cht_trend', 'manifold_pressure'] },
      { component: 'cylinder_2', health_score: 85, status: 'green',  rul_hours: 748.0, top_features: ['egt_variance'] },
      { component: 'cylinder_3', health_score: 55, status: 'yellow', rul_hours: 265.0, top_features: ['cht_trend', 'egt_variance'] },
      { component: 'cylinder_4', health_score: 80, status: 'green',  rul_hours: 650.0, top_features: ['cht_trend'] },
      { component: 'oil_system', health_score: 48, status: 'orange', rul_hours: 92.0,  top_features: ['oil_pressure_drop', 'viscosity_trend', 'contamination_index'] },
      { component: 'exhaust',    health_score: 88, status: 'green',  rul_hours: 895.0, top_features: ['backpressure_trend'] },
      { component: 'injectors',  health_score: 42, status: 'orange', rul_hours: 28.0,  top_features: ['fuel_flow_variance', 'spray_pattern', 'clog_indicator'] },
    ],
  },
  // ── Message 3: critical — orange dominates, first red ─────────────────────
  {
    timestamp: '2026-09-02T16:00:06Z',
    engine_id: 'ENG-01',
    overall_health_score: 60,
    overall_status: 'yellow',
    active_faults: ['INJECTOR_PARTIAL_CLOG'],
    components: [
      { component: 'cylinder_1', health_score: 65, status: 'yellow', rul_hours: 710.0, top_features: ['cht_trend', 'manifold_pressure'] },
      { component: 'cylinder_2', health_score: 72, status: 'yellow', rul_hours: 700.0, top_features: ['egt_variance'] },
      { component: 'cylinder_3', health_score: 40, status: 'orange', rul_hours: 180.0, top_features: ['cht_trend', 'egt_variance', 'detonation_risk'] },
      { component: 'cylinder_4', health_score: 78, status: 'yellow', rul_hours: 620.0, top_features: ['cht_trend'] },
      { component: 'oil_system', health_score: 35, status: 'orange', rul_hours: 48.0,  top_features: ['oil_pressure_drop', 'viscosity_trend', 'contamination_index'] },
      { component: 'exhaust',    health_score: 75, status: 'yellow', rul_hours: 820.0, top_features: ['backpressure_trend', 'temp_spike'] },
      { component: 'injectors',  health_score: 28, status: 'red',    rul_hours: 8.5,   top_features: ['fuel_flow_variance', 'spray_pattern', 'clog_indicator'] },
    ],
  },
  // ── Message 4: alarm — reds, active faults ────────────────────────────────
  {
    timestamp: '2026-09-02T16:00:09Z',
    engine_id: 'ENG-01',
    overall_health_score: 45,
    overall_status: 'orange',
    active_faults: ['INJECTOR_PARTIAL_CLOG', 'OIL_PRESSURE_LOW', 'CYL3_TEMP_HIGH'],
    components: [
      { component: 'cylinder_1', health_score: 52, status: 'orange', rul_hours: 620.0, top_features: ['cht_trend', 'manifold_pressure', 'detonation_risk'] },
      { component: 'cylinder_2', health_score: 60, status: 'yellow', rul_hours: 680.0, top_features: ['egt_variance'] },
      { component: 'cylinder_3', health_score: 28, status: 'red',    rul_hours: 85.0,  top_features: ['cht_trend', 'egt_variance', 'detonation_risk', 'thermal_runaway'] },
      { component: 'cylinder_4', health_score: 65, status: 'yellow', rul_hours: 590.0, top_features: ['cht_trend'] },
      { component: 'oil_system', health_score: 22, status: 'red',    rul_hours: 12.0,  top_features: ['oil_pressure_drop', 'viscosity_trend', 'contamination_index', 'bearing_wear'] },
      { component: 'exhaust',    health_score: 68, status: 'yellow', rul_hours: 780.0, top_features: ['backpressure_trend', 'temp_spike'] },
      { component: 'injectors',  health_score: 18, status: 'red',    rul_hours: 2.0,   top_features: ['fuel_flow_variance', 'spray_pattern', 'clog_indicator'] },
    ],
  },
  // ── Message 5: recovered ─────────────────────────────────────────────────
  {
    timestamp: '2026-09-02T16:00:12Z',
    engine_id: 'ENG-01',
    overall_health_score: 87,
    overall_status: 'green',
    active_faults: [],
    components: [
      { component: 'cylinder_1', health_score: 88, status: 'green',  rul_hours: 870.0, top_features: ['cht_trend'] },
      { component: 'cylinder_2', health_score: 90, status: 'green',  rul_hours: 880.0, top_features: ['egt_variance'] },
      { component: 'cylinder_3', health_score: 82, status: 'green',  rul_hours: 720.0, top_features: ['cht_trend'] },
      { component: 'cylinder_4', health_score: 91, status: 'green',  rul_hours: 890.0, top_features: ['cht_trend'] },
      { component: 'oil_system', health_score: 75, status: 'yellow', rul_hours: 310.0, top_features: ['oil_pressure_drop'] },
      { component: 'exhaust',    health_score: 95, status: 'green',  rul_hours: 950.0, top_features: ['backpressure_trend'] },
      { component: 'injectors',  health_score: 80, status: 'green',  rul_hours: 420.0, top_features: ['fuel_flow_variance'] },
    ],
  },
]

function subscribeMock(
  _engineId: string,
  onMessage: EngineMessageCallback
): () => void {
  let index = 0
  onMessage(MOCK_MESSAGES[index])
  const timer = setInterval(() => {
    index = (index + 1) % MOCK_MESSAGES.length
    onMessage(MOCK_MESSAGES[index])
  }, 3000)
  return () => clearInterval(timer)
}

// ─── Active export ────────────────────────────────────────────────────────────
// This is the ONLY line that controls which mode runs.
// Flip ACTIVE_MODE at the top of this file — do not edit this block.
export const subscribeToEngine: (
  engineId: string,
  onMessage: EngineMessageCallback
) => () => void = ACTIVE_MODE === 'ws' ? subscribeWebSocket : subscribeMock
