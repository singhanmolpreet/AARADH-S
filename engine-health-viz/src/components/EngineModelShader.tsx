// ─── EngineModelShader ────────────────────────────────────────────────────────
// Shader-based drop-in replacement for EngineModel's flat-color approach.
// Exposes the same props interface so EngineModel.tsx can switch between
// them with a single USE_SHADER flag.
//
// Two internal variants mirror EngineModel.tsx:
//   PlaceholderShader  — BoxGeometry meshes + ShaderMaterial (no .glb needed)
//   GltfModelShader    — traverses cloned GLTF scene, swaps named mesh materials
//
// Each mesh gets its own THREE.ShaderMaterial (created once on mount, never
// recreated).  Per-frame, useFrame writes uHealthScore / uIsRed / uTime into
// the material's uniforms — no React re-renders, no GC pressure.
// ─────────────────────────────────────────────────────────────────────────────

import { useRef, useMemo, useCallback, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
import * as THREE from 'three'

import type { HealthIndexMessage, ComponentHealth } from '../types/healthIndex'
import { KNOWN_MESH_NAMES } from '../types/healthIndex'
import {
  vertexShader,
  fragmentShader,
  makeHealthUniforms,
} from '../shaders/engineHealthShader'

// ─── Shared layout (mirrors EngineModel.tsx — kept in sync manually) ─────────
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

// ─── Props (identical to EngineModel) ────────────────────────────────────────
export interface EngineModelShaderProps {
  message: HealthIndexMessage | null
  onComponentClick: (component: ComponentHealth) => void
}

// ─── Helper: build component-name → ComponentHealth lookup ───────────────────
function buildCompMap(msg: HealthIndexMessage | null): Map<string, ComponentHealth> {
  const m = new Map<string, ComponentHealth>()
  if (!msg) return m
  for (const c of msg.components) m.set(c.component, c)
  return m
}

// ─── Shared: create one ShaderMaterial per named mesh ────────────────────────
function useHealthMaterials(names: readonly string[]) {
  const materials = useMemo(() => {
    const map = new Map<string, THREE.ShaderMaterial>()
    for (const name of names) {
      map.set(
        name,
        new THREE.ShaderMaterial({
          vertexShader,
          fragmentShader,
          uniforms: makeHealthUniforms(),
          side: THREE.FrontSide,
        })
      )
    }
    return map
  }, [names]) // names is a stable const ref, so this runs once

  // Dispose on unmount to free GPU memory
  useEffect(() => {
    return () => {
      for (const mat of materials.values()) mat.dispose()
    }
  }, [materials])

  return materials
}

// ─────────────────────────────────────────────────────────────────────────────
// PLACEHOLDER SHADER  (BoxGeometry + ShaderMaterial, no .glb required)
// ─────────────────────────────────────────────────────────────────────────────
export function PlaceholderShader({ message, onComponentClick }: EngineModelShaderProps) {
  const names = useMemo(() => PLACEHOLDER_LAYOUT.map(([n]) => n), [])
  const materials = useHealthMaterials(names)
  const msgRef = useRef(message)
  msgRef.current = message   // always-current without recreating callbacks

  // ── Per-frame uniform updates (no React state, no re-render) ──────────────
  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    const compMap = buildCompMap(msgRef.current)

    for (const [name, mat] of materials.entries()) {
      const comp = compMap.get(name)
      mat.uniforms.uHealthScore.value = comp?.health_score ?? 100
      mat.uniforms.uIsRed.value       = comp?.status === 'red' ? 1 : 0
      mat.uniforms.uTime.value        = t
    }
  })

  const handleClick = useCallback(
    (meshName: string, e: { stopPropagation: () => void }) => {
      e.stopPropagation()
      const comp = msgRef.current?.components.find((c) => c.component === meshName)
      if (comp) onComponentClick(comp)
    },
    [onComponentClick]
  )

  return (
    <group>
      {PLACEHOLDER_LAYOUT.map(([name, pos, size]) => (
        <mesh
          key={name}
          position={pos}
          onClick={(e) => handleClick(name, e)}
        >
          <boxGeometry args={size} />
          {/* Attach the pre-created ShaderMaterial imperatively */}
          <primitive object={materials.get(name)!} attach="material" />
        </mesh>
      ))}

      {/* Wireframe outline — uses MeshBasicMaterial, unaffected by shader toggle */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[5.2, 3.2, 1.2]} />
        <meshBasicMaterial color="#2e3250" wireframe />
      </mesh>
    </group>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// GLTF SHADER  (traverses the GLTF scene, replaces named mesh materials)
// ─────────────────────────────────────────────────────────────────────────────
export function GltfModelShader({
  path,
  message,
  onComponentClick,
}: EngineModelShaderProps & { path: string }) {
  const { scene } = useGLTF(path)
  const materials = useHealthMaterials(KNOWN_MESH_NAMES)
  const msgRef    = useRef(message)
  msgRef.current  = message

  // Clone scene once to avoid mutating the shared GLTF cache
  const clonedScene = useMemo(() => scene.clone(true), [scene])

  // Wire our ShaderMaterials onto the matching named meshes (once per clone)
  useMemo(() => {
    clonedScene.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return
      const mat = materials.get(obj.name)
      if (mat) obj.material = mat
    })
  }, [clonedScene, materials])

  // ── Per-frame uniform updates ─────────────────────────────────────────────
  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    const compMap = buildCompMap(msgRef.current)

    for (const [name, mat] of materials.entries()) {
      const comp = compMap.get(name)
      mat.uniforms.uHealthScore.value = comp?.health_score ?? 100
      mat.uniforms.uIsRed.value       = comp?.status === 'red' ? 1 : 0
      mat.uniforms.uTime.value        = t
    }
  })

  const handleClick = useCallback(
    (e: { stopPropagation: () => void; object: THREE.Object3D }) => {
      e.stopPropagation()
      // Walk hierarchy to find a named component
      let obj: THREE.Object3D | null = e.object
      while (obj) {
        const comp = msgRef.current?.components.find((c) => c.component === obj!.name)
        if (comp) { onComponentClick(comp); return }
        obj = obj.parent
      }
    },
    [onComponentClick]
  )

  return <primitive object={clonedScene} onClick={handleClick} />
}
