// ─── Engine Health GLSL Shader ───────────────────────────────────────────────
//
// Renders a smooth vertical temperature gradient across each component mesh,
// driven by a live health_score uniform.  A pulsing emissive glow is added
// when the component's status is 'red'.
//
// Temperature mapping (derived from health_score, no schema change needed):
//   temperature = 1.0 − (health_score / 100)
//   0.0 = cool / healthy   →  green palette base
//   1.0 = hot  / critical  →  red palette
//
// The gradient runs bottom-to-top on the mesh (+Y direction via vUv.y),
// so the top of a cylinder always reads a touch hotter than its base —
// visually matching real CHT behaviour.
// ─────────────────────────────────────────────────────────────────────────────

export const vertexShader = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vNormal;

  void main() {
    vUv    = uv;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

export const fragmentShader = /* glsl */ `
  precision mediump float;

  // ── Uniforms updated every frame by useFrame ──────────────────────────────
  uniform float uHealthScore;   // 0.0 – 100.0  (raw value from schema)
  uniform float uTime;          // clock.elapsedTime  (seconds)
  uniform int   uIsRed;         // 1 when component status === 'red'

  varying vec2  vUv;
  varying vec3  vNormal;

  // ── Palette: four colour stops matching the agreed health bands ───────────
  //    These intentionally match healthColor.ts so the 2-D UI and 3-D mesh
  //    always show the same semantic colours.
  const vec3 C_GREEN  = vec3(0.133, 0.769, 0.369);  // #22c55e
  const vec3 C_YELLOW = vec3(0.918, 0.702, 0.031);  // #eab308
  const vec3 C_ORANGE = vec3(0.976, 0.451, 0.086);  // #f97316
  const vec3 C_RED    = vec3(0.937, 0.267, 0.267);  // #ef4444

  // Four-stop gradient: t = 0 (healthy/cool) → 1 (critical/hot)
  vec3 healthPalette(float t) {
    t = clamp(t, 0.0, 1.0);
    if (t < 0.333) return mix(C_GREEN,  C_YELLOW, t * 3.0);
    if (t < 0.667) return mix(C_YELLOW, C_ORANGE, (t - 0.333) * 3.0);
    return              mix(C_ORANGE, C_RED,    (t - 0.667) * 3.0);
  }

  void main() {

    // ── Temperature from health score ────────────────────────────────────────
    float temp = 1.0 - clamp(uHealthScore / 100.0, 0.0, 1.0);

    // ── Vertical gradient (vUv.y: 0 = bottom, 1 = top) ──────────────────────
    // Offset of ±0.15 adds a 30 % swing across the mesh height,
    // so a healthy cylinder still shows a subtle warm-tip gradient
    // and a critical one shows a stark cool-base / red-top split.
    float gradientT = clamp(temp + 0.15 * (vUv.y * 2.0 - 1.0), 0.0, 1.0);

    vec3 color = healthPalette(gradientT);

    // ── Rim highlight — fakes metallic specular without PBR cost ─────────────
    // vNormal is in view space; (0,0,1) is straight toward the camera.
    float rim = 1.0 - abs(dot(normalize(vNormal), vec3(0.0, 0.0, 1.0)));
    rim = pow(rim, 3.5) * 0.20;
    color += vec3(rim);

    // ── Red-status pulsing glow ───────────────────────────────────────────────
    // Period ≈ 2 s  (3.14159 rad/s → 0.5 Hz)
    // Two effects layered:
    //   1. Hue shift toward saturated red  (mix)
    //   2. Additive emissive bloom         (+ red * pulse)
    if (uIsRed == 1) {
      float pulse = 0.5 + 0.5 * sin(uTime * 3.14159);

      // Shift entire surface toward red as pulse peaks
      color = mix(color, C_RED, pulse * 0.45);

      // Additive emissive — brightens highlights at peak
      color += C_RED * 0.18 * pulse;
    }

    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
  }
`

// ─── Uniform factory ─────────────────────────────────────────────────────────
// Creates a fresh uniforms object for a new THREE.ShaderMaterial.
// Initialised to a healthy default so meshes render correctly before
// the first Health Index message arrives.
export function makeHealthUniforms() {
  return {
    uHealthScore: { value: 100.0 },
    uTime:        { value: 0.0 },
    uIsRed:       { value: 0 },
  }
}

// Shape type — used by callers that hold a ref to the uniforms object
export type HealthUniforms = ReturnType<typeof makeHealthUniforms>
