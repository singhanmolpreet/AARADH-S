// ─── Engine Health GLSL Shader ───────────────────────────────────────────────
//
// Renders a smooth vertical temperature gradient across each component mesh,
// driven by a live health_score uniform.  A pulsing emissive glow is added
// when the component's status is 'red'.
//
// V2 — gradient uses LOCAL Y POSITION (vLocalY) instead of UV coordinates.
//   CAD-exported GLTF models have arbitrary UV unwraps that don't map to Y,
//   so vUv.y produced flat / wrong-coloured patches.  vLocalY is always
//   available and always means "up" regardless of UV layout.
//
// uYMin / uYMax uniforms are set from JS once the model's bounding box is
// known so the gradient spans the full height of the engine.
// ─────────────────────────────────────────────────────────────────────────────

export const vertexShader = /* glsl */ `
  varying float vLocalY;   // raw local-space Y — used for the heat gradient
  varying vec3  vNormal;   // view-space normal — used for rim lighting

  void main() {
    vLocalY = position.y;
    vNormal  = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

export const fragmentShader = /* glsl */ `
  precision mediump float;

  // ── Uniforms updated every frame ─────────────────────────────────────────
  uniform float uHealthScore;   // 0.0 – 100.0
  uniform float uTime;          // clock.elapsedTime (seconds)
  uniform int   uIsRed;         // 1 when status === 'red'

  // ── Gradient normalisation — set once from bounding box ──────────────────
  // After auto-scaling, the model's Y extent fits within [uYMin, uYMax].
  // These are the same for every material in a given session.
  uniform float uYMin;          // world min Y after centering (negative)
  uniform float uYMax;          // world max Y after centering (positive)

  varying float vLocalY;
  varying vec3  vNormal;

  // ── Palette: four stops matching the health bands in healthColor.ts ───────
  const vec3 C_GREEN  = vec3(0.133, 0.769, 0.369);   // #22c55e
  const vec3 C_YELLOW = vec3(0.918, 0.702, 0.031);   // #eab308
  const vec3 C_ORANGE = vec3(0.976, 0.451, 0.086);   // #f97316
  const vec3 C_RED    = vec3(0.937, 0.267, 0.267);   // #ef4444

  vec3 healthPalette(float t) {
    t = clamp(t, 0.0, 1.0);
    if (t < 0.333) return mix(C_GREEN,  C_YELLOW, t * 3.0);
    if (t < 0.667) return mix(C_YELLOW, C_ORANGE, (t - 0.333) * 3.0);
    return              mix(C_ORANGE, C_RED,    (t - 0.667) * 3.0);
  }

  void main() {

    // ── Temperature from health score ─────────────────────────────────────
    float temp = 1.0 - clamp(uHealthScore / 100.0, 0.0, 1.0);

    // ── Gradient via local Y — normalized to [0, 1] over engine height ────
    float yRange  = max(uYMax - uYMin, 0.001);
    float yNorm   = clamp((vLocalY - uYMin) / yRange, 0.0, 1.0);

    // Offset ±0.15 adds a 30 % swing across the mesh height regardless of
    // overall health — so even a healthy cylinder shows a cool-base/warm-tip.
    float gradientT = clamp(temp + 0.15 * (yNorm * 2.0 - 1.0), 0.0, 1.0);

    vec3 color = healthPalette(gradientT);

    // ── Rim highlight — cheap metallic-surface cue ─────────────────────────
    float rim = 1.0 - abs(dot(normalize(vNormal), vec3(0.0, 0.0, 1.0)));
    rim = pow(rim, 3.0) * 0.22;
    color += vec3(rim);

    // ── Red-status pulsing glow (~2 s period) ─────────────────────────────
    if (uIsRed == 1) {
      float pulse = 0.5 + 0.5 * sin(uTime * 3.14159);
      color  = mix(color, C_RED, pulse * 0.45);
      color += C_RED * 0.18 * pulse;
    }

    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
  }
`

// ─── Uniform factory ─────────────────────────────────────────────────────────
// uYMin / uYMax default to a ±1 range; overwritten once the GLTF bounding
// box is known (GltfModelShader) or from the placeholder box heights.
export function makeHealthUniforms() {
  return {
    uHealthScore: { value: 100.0 },
    uTime:        { value: 0.0 },
    uIsRed:       { value: 0 },
    uYMin:        { value: -1.0 },
    uYMax:        { value:  1.0 },
  }
}

export type HealthUniforms = ReturnType<typeof makeHealthUniforms>
