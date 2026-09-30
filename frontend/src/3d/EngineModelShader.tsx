// ─── EngineModelShader.tsx ────────────────────────────────────────────────────
// Shader-based Rotax 912-style procedural engine model.
//
// Geometry is built entirely from Three.js primitives — no .glb required.
// The engine shape matches the Rotax 912 / 912ULS layout:
//   • 4-cylinder horizontally-opposed (boxer) configuration
//   • Cylinders jut out left and right with visible cooling fins
//   • Large reduction gearbox disc at the front (prop drive side)
//   • Curved exhaust pipes from each cylinder head
//   • Twin carburetors / injector rail on top
//   • Oil cooler below the crankcase
//
// Health ShaderMaterial is applied per-component (cylinder_1–4, oil_system,
// exhaust, injectors). Structural parts (crankcase, gearbox) use a static
// MeshStandardMaterial.
//
// Exports:
//   PlaceholderShader  — full Rotax procedural model (no GLTF)
//   GltfModelShader    — attaches ShaderMaterials to named GLTF meshes
// ─────────────────────────────────────────────────────────────────────────────

import { useRef, useMemo, useCallback, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
import * as THREE from 'three'

import type { HealthIndexMessage, ComponentHealth } from '../shared/types/healthIndex'
import { KNOWN_MESH_NAMES } from '../shared/types/healthIndex'
import {
  vertexShader,
  fragmentShader,
  makeHealthUniforms,
} from './shaders/engineHealthShader'

// ─── Props ────────────────────────────────────────────────────────────────────
export interface EngineModelShaderProps {
  message: HealthIndexMessage | null
  onComponentClick: (component: ComponentHealth) => void
}

// ─── Structural (non-health) materials ───────────────────────────────────────
// Created once at module level — not tied to any component lifecycle.
const CRANKCASE_MAT = new THREE.MeshStandardMaterial({
  color: '#4a5068', roughness: 0.35, metalness: 0.80,
})
const DARK_MAT = new THREE.MeshStandardMaterial({
  color: '#2a2d3a', roughness: 0.50, metalness: 0.70,
})
const RUBBER_MAT = new THREE.MeshStandardMaterial({
  color: '#1a1a22', roughness: 0.90, metalness: 0.05,
})

// ─── Engine geometry constants (Rotax 912 proportions) ───────────────────────
const BARREL_R       = 0.112   // cylinder barrel radius
const BARREL_L       = 0.50    // barrel length (crankcase wall to fin tip area)
const FIN_R          = 0.155   // cooling fin outer radius
const FIN_COUNT      = 7       // fins per cylinder
const HEAD_W         = 0.135   // cylinder head depth (X)
const HEAD_H         = 0.220   // cylinder head height (Y)
const HEAD_D         = 0.230   // cylinder head depth (Z)
const CRANK_ATTACH_X = 0.21    // X distance from engine center to barrel root

// ─── Helper: build a component health map ────────────────────────────────────
function buildCompMap(msg: HealthIndexMessage | null): Map<string, ComponentHealth> {
  const m = new Map<string, ComponentHealth>()
  if (!msg) return m
  for (const c of msg.components) m.set(c.component, c)
  return m
}

// ─── Helper: create one ShaderMaterial per named component ───────────────────
function useHealthMaterials(names: readonly string[]) {
  const materials = useMemo(() => {
    const map = new Map<string, THREE.ShaderMaterial>()
    for (const name of names) {
      map.set(name, new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: makeHealthUniforms(),
        side: THREE.FrontSide,
      }))
    }
    return map
  // names is a module-level const — effectively runs once
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => () => { for (const m of materials.values()) m.dispose() }, [materials])
  return materials
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-GEOMETRY BUILDERS
// These are plain functions (not React components) that return JSX.
// They are called directly inside render — no hooks, just element trees.
// ─────────────────────────────────────────────────────────────────────────────

// ─── Single cylinder assembly: barrel + fins + head + spark-plug bosses ──────
// side: -1 = left bank, +1 = right bank
// zPos: fore (+) or aft (-) offset of this cylinder on the engine
function cylinderAssembly(
  name: string,
  side: -1 | 1,
  zPos: number,
  mat: THREE.ShaderMaterial,
  onClick: (e: { stopPropagation: () => void }) => void
) {
  const dir = side
  // X position of the outboard face of the cylinder head
  const headOuterX = dir * (CRANK_ATTACH_X + BARREL_L + HEAD_W)
  const barrelCtrX = dir * (CRANK_ATTACH_X + BARREL_L / 2)
  const finSpacing = BARREL_L / (FIN_COUNT + 1)

  return (
    <group key={name} position={[0, 0.04, zPos]} onClick={onClick}>
      {/* Barrel — horizontal cylinder rotated 90° around Z */}
      <mesh position={[barrelCtrX, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[BARREL_R, BARREL_R + 0.006, BARREL_L, 18]} />
        <primitive object={mat} attach="material" />
      </mesh>

      {/* Cooling fins — characteristic stacked rings */}
      {Array.from({ length: FIN_COUNT }, (_, i) => {
        const fx = dir * (CRANK_ATTACH_X + finSpacing * (i + 1))
        return (
          <mesh key={i} position={[fx, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[FIN_R, FIN_R, 0.012, 18]} />
            <primitive object={mat} attach="material" />
          </mesh>
        )
      })}

      {/* Cylinder head */}
      <mesh position={[headOuterX - dir * HEAD_W / 2, 0, 0]}>
        <boxGeometry args={[HEAD_W, HEAD_H, HEAD_D]} />
        <primitive object={mat} attach="material" />
      </mesh>

      {/* Spark plug boss — top */}
      <mesh
        position={[headOuterX - dir * HEAD_W * 0.6, 0.115, 0.04]}
        rotation={[0.2, 0, dir * 0.35]}
      >
        <cylinderGeometry args={[0.017, 0.017, 0.060, 8]} />
        <primitive object={mat} attach="material" />
      </mesh>

      {/* Spark plug boss — bottom (dual ignition) */}
      <mesh
        position={[headOuterX - dir * HEAD_W * 0.6, -0.105, -0.04]}
        rotation={[-0.2, 0, dir * 0.35]}
      >
        <cylinderGeometry args={[0.017, 0.017, 0.060, 8]} />
        <primitive object={mat} attach="material" />
      </mesh>

      {/* Intake port stub (top of head) */}
      <mesh position={[headOuterX - dir * HEAD_W * 0.3, 0.14, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.030, 0.030, 0.045, 10]} />
        <primitive object={mat} attach="material" />
      </mesh>

      {/* Exhaust port stub (bottom of head) */}
      <mesh position={[headOuterX - dir * HEAD_W * 0.3, -0.135, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.025, 0.030, 0.040, 10]} />
        <primitive object={mat} attach="material" />
      </mesh>

      {/* Ignition coil bracket on top of head */}
      <mesh position={[headOuterX - dir * HEAD_W * 0.5, 0.145, 0]}>
        <boxGeometry args={[0.06, 0.04, 0.08]} />
        <primitive object={DARK_MAT} attach="material" />
      </mesh>
    </group>
  )
}

// ─── Exhaust pipe (TubeGeometry along a CatmullRom curve) ────────────────────
// Starts at the exhaust port of a cylinder head, curves down and aft, 
// collecting toward the centerline.
function exhaustPipe(
  key: string,
  startX: number,
  startZ: number,
  side: -1 | 1,
  mat: THREE.ShaderMaterial
) {
  const sx = startX
  const sy = -0.09
  const sz = startZ
  const d  = side

  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(sx,             sy,        sz),
    new THREE.Vector3(sx + d * 0.08,  sy - 0.08, sz - 0.05),
    new THREE.Vector3(sx + d * 0.14,  sy - 0.20, sz - 0.20),
    new THREE.Vector3(sx + d * 0.10,  sy - 0.30, sz - 0.42),
    new THREE.Vector3(d * 0.12,       sy - 0.33, sz - 0.60),
    new THREE.Vector3(0,              sy - 0.30, -0.72),
  ])

  const tubeGeo = new THREE.TubeGeometry(curve, 20, 0.022, 8, false)

  return (
    <mesh key={key}>
      <primitive object={tubeGeo} attach="geometry" />
      <primitive object={mat} attach="material" />
    </mesh>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// PLACEHOLDER SHADER — full Rotax 912 procedural model
// ─────────────────────────────────────────────────────────────────────────────
export function PlaceholderShader({ message, onComponentClick }: EngineModelShaderProps) {
  const materials = useHealthMaterials(KNOWN_MESH_NAMES)
  const msgRef    = useRef(message)
  msgRef.current  = message

  // ── Set Y bounds per component (box half-heights) ────────────────────────
  // BoxGeometry local Y runs from -h/2 to +h/2.  Use the tallest box as the
  // global range so the gradient is consistent across all components.
  useMemo(() => {
    // Tallest component box height is 1.6 (cylinders)
    const globalHalfH = 0.8
    for (const mat of materials.values()) {
      mat.uniforms.uYMin.value = -globalHalfH
      mat.uniforms.uYMax.value =  globalHalfH
    }
  }, [materials])

  // ── Per-frame: update all shader uniforms ────────────────────────────────
  useFrame(({ clock }) => {
    const t       = clock.elapsedTime
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
      const comp = msgRef.current?.components.find(c => c.component === meshName)
      if (comp) onComponentClick(comp)
    },
    [onComponentClick]
  )

  // Convenience: get a material by component name
  const m = (name: string) => materials.get(name)!

  // Head outer X coords for exhaust pipe starts (left bank: negative, right: positive)
  const headX_left  = -(CRANK_ATTACH_X + BARREL_L + HEAD_W / 2)
  const headX_right =  (CRANK_ATTACH_X + BARREL_L + HEAD_W / 2)

  return (
    <group>

      {/* ══════════════════════════════════════════════════════════════════════
          STRUCTURAL PARTS — crankcase, gearbox, prop shaft, intake manifold
          Not health-colored. Not clickable.
         ══════════════════════════════════════════════════════════════════════ */}

      {/* ── Main crankcase body (horizontal cylinder, Z-axis) ─────────────── */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.20, 0.22, 1.20, 24]} />
        <primitive object={CRANKCASE_MAT} attach="material" />
      </mesh>

      {/* ── Crankcase side cover plates ───────────────────────────────────── */}
      <mesh position={[-0.18, 0, 0]}>
        <boxGeometry args={[0.06, 0.32, 1.05]} />
        <primitive object={CRANKCASE_MAT} attach="material" />
      </mesh>
      <mesh position={[0.18, 0, 0]}>
        <boxGeometry args={[0.06, 0.32, 1.05]} />
        <primitive object={CRANKCASE_MAT} attach="material" />
      </mesh>

      {/* ── Top crankcase cover ───────────────────────────────────────────── */}
      <mesh position={[0, 0.19, 0]}>
        <boxGeometry args={[0.42, 0.06, 1.10]} />
        <primitive object={CRANKCASE_MAT} attach="material" />
      </mesh>

      {/* ── Bottom sump ───────────────────────────────────────────────────── */}
      <mesh position={[0, -0.24, 0]}>
        <boxGeometry args={[0.36, 0.08, 0.90]} />
        <primitive object={DARK_MAT} attach="material" />
      </mesh>

      {/* ── Gearbox housing (large disc, front of engine) ─────────────────── */}
      <mesh position={[0, 0, 0.72]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.300, 0.310, 0.130, 32]} />
        <primitive object={CRANKCASE_MAT} attach="material" />
      </mesh>

      {/* Gearbox inner recess ring */}
      <mesh position={[0, 0, 0.795]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.22, 0.025, 6, 28]} />
        <primitive object={DARK_MAT} attach="material" />
      </mesh>

      {/* ── Prop shaft stub ──────────────────────────────────────────────── */}
      <mesh position={[0, 0, 0.88]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.045, 0.055, 0.12, 16]} />
        <primitive object={DARK_MAT} attach="material" />
      </mesh>

      {/* ── Rear magneto / accessory housing ─────────────────────────────── */}
      <mesh position={[0, 0, -0.70]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.150, 0.155, 0.100, 24]} />
        <primitive object={DARK_MAT} attach="material" />
      </mesh>

      {/* ── Intake manifold box (between carbs and heads) ─────────────────── */}
      <mesh position={[0, 0.29, 0]}>
        <boxGeometry args={[0.20, 0.08, 0.80]} />
        <primitive object={DARK_MAT} attach="material" />
      </mesh>

      {/* Left intake runner */}
      <mesh position={[-0.16, 0.28, 0.28]} rotation={[0, 0, Math.PI / 6]}>
        <cylinderGeometry args={[0.024, 0.024, 0.22, 10]} />
        <primitive object={DARK_MAT} attach="material" />
      </mesh>
      <mesh position={[-0.16, 0.28, -0.28]} rotation={[0, 0, Math.PI / 6]}>
        <cylinderGeometry args={[0.024, 0.024, 0.22, 10]} />
        <primitive object={DARK_MAT} attach="material" />
      </mesh>

      {/* Right intake runner */}
      <mesh position={[0.16, 0.28, 0.28]} rotation={[0, 0, -Math.PI / 6]}>
        <cylinderGeometry args={[0.024, 0.024, 0.22, 10]} />
        <primitive object={DARK_MAT} attach="material" />
      </mesh>
      <mesh position={[0.16, 0.28, -0.28]} rotation={[0, 0, -Math.PI / 6]}>
        <cylinderGeometry args={[0.024, 0.024, 0.22, 10]} />
        <primitive object={DARK_MAT} attach="material" />
      </mesh>

      {/* ── Breather / coolant overflow bottle ───────────────────────────── */}
      <mesh position={[0.26, 0.10, -0.55]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.038, 0.038, 0.14, 10]} />
        <primitive object={RUBBER_MAT} attach="material" />
      </mesh>

      {/* ── Throttle body connector hoses ─────────────────────────────────── */}
      <mesh position={[-0.12, 0.265, 0.28]} rotation={[0.3, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.028, 0.028, 0.10, 8]} />
        <primitive object={RUBBER_MAT} attach="material" />
      </mesh>
      <mesh position={[0.12, 0.265, 0.28]} rotation={[0.3, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.028, 0.028, 0.10, 8]} />
        <primitive object={RUBBER_MAT} attach="material" />
      </mesh>

      {/* ══════════════════════════════════════════════════════════════════════
          HEALTH COMPONENTS — cylinder_1 to cylinder_4
          Left bank: cylinders 1 (front) and 2 (rear)
          Right bank: cylinders 3 (front) and 4 (rear)
         ══════════════════════════════════════════════════════════════════════ */}

      {cylinderAssembly('cylinder_1', -1,  0.36, m('cylinder_1'),
        e => handleClick('cylinder_1', e))}

      {cylinderAssembly('cylinder_2', -1, -0.36, m('cylinder_2'),
        e => handleClick('cylinder_2', e))}

      {cylinderAssembly('cylinder_3',  1,  0.36, m('cylinder_3'),
        e => handleClick('cylinder_3', e))}

      {cylinderAssembly('cylinder_4',  1, -0.36, m('cylinder_4'),
        e => handleClick('cylinder_4', e))}

      {/* ══════════════════════════════════════════════════════════════════════
          OIL SYSTEM — oil_system component
         ══════════════════════════════════════════════════════════════════════ */}
      <group onClick={e => handleClick('oil_system', e)}>
        {/* Oil cooler radiator block */}
        <mesh position={[0, -0.40, 0.10]}>
          <boxGeometry args={[0.55, 0.115, 0.68]} />
          <primitive object={m('oil_system')} attach="material" />
        </mesh>

        {/* Oil cooler fins (front face) */}
        {Array.from({ length: 10 }, (_, i) => (
          <mesh key={i} position={[0, -0.40, 0.46 - i * 0.07]}>
            <boxGeometry args={[0.54, 0.010, 0.006]} />
            <primitive object={m('oil_system')} attach="material" />
          </mesh>
        ))}

        {/* Oil filter canister */}
        <mesh position={[-0.30, -0.26, -0.48]} rotation={[0.4, 0, 0.3]}>
          <cylinderGeometry args={[0.048, 0.048, 0.130, 12]} />
          <primitive object={m('oil_system')} attach="material" />
        </mesh>

        {/* Oil pump housing */}
        <mesh position={[0, -0.30, -0.60]}>
          <boxGeometry args={[0.14, 0.11, 0.10]} />
          <primitive object={m('oil_system')} attach="material" />
        </mesh>

        {/* Oil feed line */}
        <mesh position={[0, -0.33, -0.30]} rotation={[0.5, 0, 0]}>
          <cylinderGeometry args={[0.018, 0.018, 0.28, 8]} />
          <primitive object={RUBBER_MAT} attach="material" />
        </mesh>
      </group>

      {/* ══════════════════════════════════════════════════════════════════════
          EXHAUST — all four pipes as one health component
         ══════════════════════════════════════════════════════════════════════ */}
      <group onClick={e => handleClick('exhaust', e)}>
        {exhaustPipe('ex1', headX_left,  0.36, -1, m('exhaust'))}
        {exhaustPipe('ex2', headX_left, -0.36, -1, m('exhaust'))}
        {exhaustPipe('ex3', headX_right, 0.36,  1, m('exhaust'))}
        {exhaustPipe('ex4', headX_right,-0.36,  1, m('exhaust'))}

        {/* Exhaust collector / muffler stub */}
        <mesh position={[0, -0.30, -0.74]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.042, 0.042, 0.08, 10]} />
          <primitive object={m('exhaust')} attach="material" />
        </mesh>
      </group>

      {/* ══════════════════════════════════════════════════════════════════════
          INJECTORS / CARBURETORS — injectors component
         ══════════════════════════════════════════════════════════════════════ */}
      <group onClick={e => handleClick('injectors', e)}>
        {/* Front carburetor body */}
        <mesh position={[0, 0.42, 0.26]} rotation={[0.10, 0, 0]}>
          <cylinderGeometry args={[0.065, 0.070, 0.160, 14]} />
          <primitive object={m('injectors')} attach="material" />
        </mesh>

        {/* Front carb bowl */}
        <mesh position={[0.058, 0.36, 0.28]}>
          <sphereGeometry args={[0.040, 10, 8]} />
          <primitive object={m('injectors')} attach="material" />
        </mesh>

        {/* Rear carburetor body */}
        <mesh position={[0, 0.42, -0.26]} rotation={[-0.10, 0, 0]}>
          <cylinderGeometry args={[0.065, 0.070, 0.160, 14]} />
          <primitive object={m('injectors')} attach="material" />
        </mesh>

        {/* Rear carb bowl */}
        <mesh position={[0.058, 0.36, -0.28]}>
          <sphereGeometry args={[0.040, 10, 8]} />
          <primitive object={m('injectors')} attach="material" />
        </mesh>

        {/* Air filter box */}
        <mesh position={[0, 0.52, 0]}>
          <boxGeometry args={[0.18, 0.07, 0.62]} />
          <primitive object={m('injectors')} attach="material" />
        </mesh>

        {/* Throttle cable bracket */}
        <mesh position={[0.10, 0.40, 0]}>
          <boxGeometry args={[0.04, 0.05, 0.08]} />
          <primitive object={DARK_MAT} attach="material" />
        </mesh>
      </group>

    </group>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// GLTF SHADER — attaches ShaderMaterials to named meshes in the Rotax 914 GLB
// ─────────────────────────────────────────────────────────────────────────────

// ── Rotax 914 GLB node name → health component mapping ───────────────────────
// The GLB uses Russian/transliterated node names from the original CAD.
// Each health component is mapped to all GLB node names that visually
// represent it.  The traversal walks ancestors, so children of named group
// nodes (e.g. Koleno1's sub-meshes) are coloured automatically.
const GLTF_MESH_MAP: Record<string, string[]> = {
  cylinder_1: ['Cylindr_1'],
  cylinder_2: ['Cylindr_2'],
  cylinder_3: ['Cylindr_001'],
  cylinder_4: ['Cylindr_002'],
  oil_system: [
    'Korpus_maslo',           // oil housing body
    'Filtr',                  // oil filter canister
    'Bachek',                 // oil reservoir / expansion tank
    'Datchik',                // oil pressure sensor
    'Patrubok_voda_1',        // coolant hoses (shared oil/coolant system)
    'Patrubok_voda_2',
    'Patrubok_voda_3',
    'Patrubok_voda_4',
    'Patrubok_voda_niz_1_m3d',
    'Patrubok_voda_niz_2',
    'Patrubok_voda_niz_3',
    'Patrubok_voda_niz_4',
  ],
  exhaust: [
    'Truba_vyhlopnaya',       // main exhaust tube
    'Summator_vyhlopa',       // exhaust collector / muffler
    'Koleno1',                // exhaust elbow 1 (group — children inherit)
    'Koleno2',
    'Koleno3',
    'Koleno4',
    'Koleno_summatora_1',     // collector elbows
    'Koleno_summatora_2',
    'Колено_сумматора_3',     // Cyrillic variants from the CAD file
    'Колено_сумматора_001',
    'Koleno1_sldasm-Part-001',
    'Koleno1_sldasm-Part-002',
    'Koleno1_sldasm-Part-003',
    'Koleno1_sldasm-Part-1',
    'Koleno1_sldasm-Part-2',
  ],
  turbo: [
    'Turbina_1',
  ],
  intake: [
    'Vhodnoy_kollektor',      // intake collector
    'Patrubok_vhodnoy',       // intake pipe / throttle body
  ],
  electrical: [
    'Korpus_magneto',         // magneto housing (electrical generation)
    'Starter',                // starter motor
  ],
  injectors: [],
}

// Reverse map: GLB node name → health component name (for O(1) lookup)
const NAME_TO_COMPONENT = new Map<string, string>()
for (const [comp, names] of Object.entries(GLTF_MESH_MAP)) {
  for (const n of names) NAME_TO_COMPONENT.set(n, comp)
}

export function GltfModelShader({
  path,
  message,
  onComponentClick,
}: EngineModelShaderProps & { path: string }) {
  const { scene }  = useGLTF(path)
  const materials  = useHealthMaterials(KNOWN_MESH_NAMES)
  const msgRef     = useRef(message)
  msgRef.current   = message

  // Clone scene once to avoid mutating the shared GLTF cache
  const clonedScene = useMemo(() => scene.clone(true), [scene])

  // ── Material assignment: walk each mesh's ancestors to find component ──────
  useMemo(() => {
    clonedScene.traverse(obj => {
      if (!(obj instanceof THREE.Mesh)) return

      // Walk up the hierarchy until we find a node name in the map
      let cur: THREE.Object3D | null = obj
      while (cur) {
        const compName = NAME_TO_COMPONENT.get(cur.name)
        if (compName) {
          const mat = materials.get(compName)
          if (mat) obj.material = mat
          break
        }
        cur = cur.parent
      }
    })
  }, [clonedScene, materials])

  // ── Per-frame uniform updates ─────────────────────────────────────────────
  useFrame(({ clock }) => {
    const t       = clock.elapsedTime
    const compMap = buildCompMap(msgRef.current)
    for (const [name, mat] of materials.entries()) {
      const comp = compMap.get(name)
      mat.uniforms.uHealthScore.value = comp?.health_score ?? 100
      mat.uniforms.uIsRed.value       = comp?.status === 'red' ? 1 : 0
      mat.uniforms.uTime.value        = t
      // uYMin / uYMax are set once below; no need to update them every frame
    }
  })

  // ── Auto-scale + center from bounding box ────────────────────────────────
  // Compute once from the cloned scene so the model always fills TARGET_SIZE
  // units regardless of the original CAD unit system (mm, m, inches…).
  // We also derive yMin/yMax (in LOCAL space after centering) for the shader.
  const TARGET_SIZE = 5   // longest engine dimension will fill 5 scene units

  const { autoScale, autoOffset, yMin, yMax } = useMemo(() => {
    const box  = new THREE.Box3().setFromObject(clonedScene)
    const size = new THREE.Vector3()
    box.getSize(size)

    const maxDim = Math.max(size.x, size.y, size.z)
    const s      = maxDim > 0 ? TARGET_SIZE / maxDim : 1

    const center = new THREE.Vector3()
    box.getCenter(center)

    // After centering the cloned scene, local Y runs from -(size.y*s)/2 to +(size.y*s)/2
    const halfH = (size.y * s) / 2

    return {
      autoScale:  s,
      autoOffset: center.clone().negate(),  // move center → origin
      yMin: -halfH,
      yMax:  halfH,
    }
  }, [clonedScene])

  // Push Y bounds into every material's uniforms so the shader knows the range
  useMemo(() => {
    for (const mat of materials.values()) {
      mat.uniforms.uYMin.value = yMin
      mat.uniforms.uYMax.value = yMax
    }
  }, [materials, yMin, yMax])

  // ── Click handler: walk ancestors to find the component ───────────────────
  const handleClick = useCallback(
    (e: { stopPropagation: () => void; object: THREE.Object3D }) => {
      e.stopPropagation()
      let cur: THREE.Object3D | null = e.object
      while (cur) {
        const compName = NAME_TO_COMPONENT.get(cur.name)
        if (compName) {
          const comp = msgRef.current?.components.find(c => c.component === compName)
          if (comp) { onComponentClick(comp); return }
        }
        cur = cur.parent
      }
    },
    [onComponentClick]
  )

  // ── Render ────────────────────────────────────────────────────────────────
  // The group applies:
  //   1. Position offset  — moves the model's bounding-box centre to origin
  //   2. Uniform scale    — fits the longest dimension to TARGET_SIZE units
  //   3. X rotation       — SolidWorks uses Z-up; Three.js uses Y-up
  return (
    <group
      position={[autoOffset.x * autoScale, autoOffset.y * autoScale, autoOffset.z * autoScale]}
      scale={[autoScale, autoScale, autoScale]}
      rotation={[-Math.PI / 2, 0, 0]}
    >
      <primitive object={clonedScene} onClick={handleClick} />
    </group>
  )
}

