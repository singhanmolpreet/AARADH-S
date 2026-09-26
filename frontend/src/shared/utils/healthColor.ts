import * as THREE from 'three'

// ─── Health Score → Color Mapping ────────────────────────────────────────────
// Bands are defined by the agreed schema:
//   green  80–100   #22c55e
//   yellow 55–79    #eab308
//   orange 30–54    #f97316
//   red    <30      #ef4444

const COLOR_GREEN  = new THREE.Color('#22c55e')
const COLOR_YELLOW = new THREE.Color('#eab308')
const COLOR_ORANGE = new THREE.Color('#f97316')
const COLOR_RED    = new THREE.Color('#ef4444')

/**
 * Maps a health_score (0–100) to a THREE.Color.
 * Returns a *new* Color instance each call — safe to mutate for lerp.
 */
export function healthScoreToColor(score: number): THREE.Color {
  if (score >= 80) return COLOR_GREEN.clone()
  if (score >= 55) return COLOR_YELLOW.clone()
  if (score >= 30) return COLOR_ORANGE.clone()
  return COLOR_RED.clone()
}

/**
 * Maps a health_score to the CSS variable name so the UI can mirror the 3D colors.
 */
export function healthScoreToCssClass(score: number): string {
  if (score >= 80) return 'color-green'
  if (score >= 55) return 'color-yellow'
  if (score >= 30) return 'color-orange'
  return 'color-red'
}

/**
 * Maps a health_score to a plain CSS hex color string (for inline styles).
 */
export function healthScoreToHex(score: number): string {
  if (score >= 80) return '#22c55e'
  if (score >= 55) return '#eab308'
  if (score >= 30) return '#f97316'
  return '#ef4444'
}

/**
 * Maps a health_score to its status label string.
 */
export function healthScoreToStatus(score: number): 'green' | 'yellow' | 'orange' | 'red' {
  if (score >= 80) return 'green'
  if (score >= 55) return 'yellow'
  if (score >= 30) return 'orange'
  return 'red'
}
