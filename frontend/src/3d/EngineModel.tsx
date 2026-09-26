// ─── EngineModel ──────────────────────────────────────────────────────────────
// Renders the engine either from a GLTF file (when available) or as colored
// placeholder boxes (so the project runs immediately without a .glb file).
//
// Two rendering modes controlled by USE_SHADER below:
//   true  → custom GLSL gradient shader (EngineModelShader.tsx)
//   false → flat MeshStandardMaterial with lerp (original, battle-tested)
//
// Color lerp: each mesh's MeshStandardMaterial color is smoothly interpolated
// toward the target health color every frame (~1 second transition at the
// lerp factor used below).

// ─── Demo-day safety toggle ──────────────────────────────────────────────────
// Flip to false in <30 seconds if the shader has any issue near demo day.
// Every other file stays untouched — only this line changes.
const USE_SHADER = true

import { useRef, useMemo, useCallback } from 'react'
import { useFrame } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
import * as THREE from 'three'
import type { HealthIndexMessage, ComponentHealth } from '../shared/types/healthIndex'
import { healthScoreToColor } from '../shared/utils/healthColor'
import { KNOWN_MESH_NAMES } from '../shared/types/healthIndex'
import { PlaceholderShader, GltfModelShader } from './EngineModelShader'

// ─── GLTF path — the Rotax 914 model served from public/ ─────────────────────
// Set back to null to fall through to the procedural placeholder.
const GLTF_PATH: string | null = '/rotax_914.glb'

// ─── Lerp speed: 1 means instant, ~0.03 gives ~1 second smooth transition ───
const LERP_ALPHA = 0.03

// ─── Layout for placeholder boxes when no GLTF is available ─────────────────
// Each entry: [meshName, [x, y, z] position, [w, h, d] size]
type BoxDef = [string, [number, number, number], [number, number, number]]
const PLACEHOLDER_LAYOUT: BoxDef[] = [
  ['cylinder_1', [-2.4,  0.8, 0], [0.7, 1.6, 0.7]],
  ['cylinder_2', [-0.8,  0.8, 0], [0.7, 1.6, 0.7]],
  ['cylinder_3', [ 0.8,  0.8, 0], [0.7, 1.6, 0.7]],
  ['cylinder_4', [ 2.4,  0.8, 0], [0.7, 1.6, 0.7]],
  ['oil_system', [ 0.0, -0.6, 0], [1.4, 0.8, 0.9]],
  ['exhaust',    [ 0.0, -1.8, 0], [3.6, 0.4, 0.5]],
  ['injectors',  [ 0.0,  2.2, 0], [3.6, 0.3, 0.3]],
]

// ─── Props ───────────────────────────────────────────────────────────────────
interface EngineModelProps {
  message: HealthIndexMessage | null
  onComponentClick: (component: ComponentHealth) => void
}

// ─── Lookup: component name → health score ───────────────────────────────────
function buildScoreMap(message: HealthIndexMessage | null): Map<string, number> {
  const map = new Map<string, number>()
  if (!message) return map
  for (const c of message.components) {
    map.set(c.component, c.health_score)
  }
  return map
}

// ─── Placeholder mesh ref store ───────────────────────────────────────────────
// We keep one current color (THREE.Color) per named mesh for the lerp.
function useMeshColorRefs() {
  return useRef<Map<string, { current: THREE.Color; material: THREE.MeshStandardMaterial | null }>>(
    new Map(
      KNOWN_MESH_NAMES.map((name) => [
        name,
        { current: new THREE.Color('#22c55e'), material: null },
      ])
    )
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// PLACEHOLDER MODEL (no GLTF)
// ─────────────────────────────────────────────────────────────────────────────
function PlaceholderModel({ message, onComponentClick }: EngineModelProps) {
  const colorRefs = useMeshColorRefs()
  const scoreMap = useMemo(() => buildScoreMap(message), [message])

  // Lerp all materials toward their target color every frame
  useFrame(() => {
    for (const [name, ref] of colorRefs.current.entries()) {
      if (!ref.material) continue
      const score = scoreMap.get(name) ?? 100
      const target = healthScoreToColor(score)
      ref.current.lerp(target, LERP_ALPHA)
      ref.material.color.copy(ref.current)
    }
  })

  const handleClick = useCallback(
    (meshName: string, e: THREE.Event) => {
      // THREE.js event — stop propagation so OrbitControls doesn't interfere
      ;(e as unknown as { stopPropagation: () => void }).stopPropagation()
      if (!message) return
      const comp = message.components.find((c) => c.component === meshName)
      if (comp) onComponentClick(comp)
    },
    [message, onComponentClick]
  )

  return (
    <group>
      {PLACEHOLDER_LAYOUT.map(([name, pos, size]) => {
        const colorRef = colorRefs.current.get(name)!
        return (
          <mesh
            key={name}
            position={pos}
            onClick={(e) => handleClick(name, e as unknown as THREE.Event)}
          >
            <boxGeometry args={size} />
            <meshStandardMaterial
              ref={(mat) => {
                if (mat) {
                  colorRef.material = mat
                  // Initialise material color to current lerp color
                  mat.color.copy(colorRef.current)
                }
              }}
              roughness={0.35}
              metalness={0.65}
            />
          </mesh>
        )
      })}
      {/* Simple wireframe engine block outline */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[5.2, 3.2, 1.2]} />
        <meshBasicMaterial color="#2e3250" wireframe />
      </mesh>
    </group>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// GLTF MODEL
// ─────────────────────────────────────────────────────────────────────────────
function GltfModel({ path, message, onComponentClick }: EngineModelProps & { path: string }) {
  const { scene } = useGLTF(path)
  const colorRefs = useMeshColorRefs()
  const scoreMap = useMemo(() => buildScoreMap(message), [message])

  // Clone scene to avoid mutating the GLTF cache
  const clonedScene = useMemo(() => scene.clone(true), [scene])

  // Wire up material refs on the cloned scene meshes
  useMemo(() => {
    clonedScene.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return
      const name = obj.name
      if (!colorRefs.current.has(name)) return

      // Ensure each named mesh has its own MeshStandardMaterial
      if (!(obj.material instanceof THREE.MeshStandardMaterial)) {
        obj.material = new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.65 })
      }
      colorRefs.current.get(name)!.material = obj.material as THREE.MeshStandardMaterial
    })
  }, [clonedScene]) // eslint-disable-line react-hooks/exhaustive-deps

  // Lerp materials every frame
  useFrame(() => {
    for (const [name, ref] of colorRefs.current.entries()) {
      if (!ref.material) continue
      const score = scoreMap.get(name) ?? 100
      const target = healthScoreToColor(score)
      ref.current.lerp(target, LERP_ALPHA)
      ref.material.color.copy(ref.current)
    }
  })

  const handleClick = useCallback(
    (e: { stopPropagation: () => void; object: THREE.Object3D }) => {
      e.stopPropagation()
      if (!message) return
      // Walk up the hierarchy to find a named component mesh
      let obj: THREE.Object3D | null = e.object
      while (obj) {
        const comp = message.components.find((c) => c.component === obj!.name)
        if (comp) { onComponentClick(comp); return }
        obj = obj.parent
      }
    },
    [message, onComponentClick]
  )

  return <primitive object={clonedScene} onClick={handleClick} />
}

// ─────────────────────────────────────────────────────────────────────────────
// Public component — chooses rendering mode then GLTF vs placeholder
// ─────────────────────────────────────────────────────────────────────────────
export function EngineModel({ message, onComponentClick }: EngineModelProps) {
  // ── Shader path (USE_SHADER = true) ───────────────────────────────────────
  if (USE_SHADER) {
    if (GLTF_PATH) {
      return (
        <GltfModelShader
          path={GLTF_PATH}
          message={message}
          onComponentClick={onComponentClick}
        />
      )
    }
    return (
      <PlaceholderShader
        message={message}
        onComponentClick={onComponentClick}
      />
    )
  }

  // ── Flat-color fallback (USE_SHADER = false) ───────────────────────────────
  // Original MeshStandardMaterial + lerp — unchanged since initial build.
  if (GLTF_PATH) {
    return <GltfModel path={GLTF_PATH} message={message} onComponentClick={onComponentClick} />
  }
  return <PlaceholderModel message={message} onComponentClick={onComponentClick} />
}

