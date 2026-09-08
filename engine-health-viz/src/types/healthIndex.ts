// ─── Health Index Schema ─────────────────────────────────────────────────────
// Exact TypeScript mirror of the Health Index JSON message format.
// Nothing invented — all fields come from the agreed schema.

export interface ComponentHealth {
  /** Named mesh identifier — must match GLTF mesh name exactly */
  component: string
  /** 0–100 numeric health score */
  health_score: number
  /** Derived status string from health_score bands */
  status: 'green' | 'yellow' | 'orange' | 'red'
  /** Remaining Useful Life in hours */
  rul_hours: number
  /** Top contributing sensor/feature names */
  top_features: string[]
}

export interface HealthIndexMessage {
  /** ISO 8601 timestamp */
  timestamp: string
  /** Engine identifier e.g. "ENG-01" */
  engine_id: string
  /** Per-component health entries */
  components: ComponentHealth[]
  /** Aggregate engine health score 0–100 */
  overall_health_score: number
  /** Aggregate status derived from overall_health_score */
  overall_status: 'green' | 'yellow' | 'orange' | 'red'
  /** List of active fault codes (empty array when none) */
  active_faults: string[]
}

/** The seven named meshes in the GLTF — kept here for reference / validation */
export const KNOWN_MESH_NAMES = [
  'cylinder_1',
  'cylinder_2',
  'cylinder_3',
  'cylinder_4',
  'oil_system',
  'exhaust',
  'injectors',
] as const

export type MeshName = typeof KNOWN_MESH_NAMES[number]
