// ─── EngineScene ──────────────────────────────────────────────────────────────
// The R3F Canvas wrapper. Owns lighting and camera controls.
// Passes the health message and click handler down to EngineModel.

import { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment, Grid } from '@react-three/drei'
import { EngineModel } from './EngineModel'
import type { HealthIndexMessage, ComponentHealth } from '../types/healthIndex'

interface EngineSceneProps {
  message: HealthIndexMessage | null
  onComponentClick: (component: ComponentHealth) => void
}

export function EngineScene({ message, onComponentClick }: EngineSceneProps) {
  return (
    <Canvas
      camera={{ position: [0, 3, 12], fov: 50, near: 0.01, far: 2000 }}
      shadows
      gl={{ antialias: true }}
      style={{ background: '#0f1117' }}
    >
      {/* ── Lighting ── */}
      <ambientLight intensity={0.5} />
      <directionalLight
        castShadow
        position={[8, 12, 8]}
        intensity={1.4}
        shadow-mapSize={[2048, 2048]}
      />
      <directionalLight position={[-6, 5, -6]} intensity={0.5} color="#6080ff" />
      <pointLight position={[0, -4, 3]} intensity={0.4} color="#ff8040" />

      {/* ── Environment (provides IBL reflections on metallic meshes) ── */}
      <Suspense fallback={null}>
        <Environment preset="city" />
      </Suspense>

      {/* ── Ground grid for spatial reference ── */}
      <Grid
        position={[0, -3.5, 0]}
        args={[40, 40]}
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#2e3250"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#3d4470"
        fadeDistance={30}
        fadeStrength={1}
        infiniteGrid
      />

      {/* ── Engine model (GLTF or placeholder) ── */}
      <Suspense fallback={null}>
        <EngineModel message={message} onComponentClick={onComponentClick} />
      </Suspense>

      {/* ── Camera controls ── */}
      {/* minDistance 0.5 lets you zoom right into individual parts */}
      {/* maxDistance 40  lets you pull back to see the whole engine */}
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.05}
        minDistance={0.5}
        maxDistance={40}
        maxPolarAngle={Math.PI * 0.85}
      />
    </Canvas>
  )
}
