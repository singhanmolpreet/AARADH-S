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
      camera={{ position: [0, 2, 8], fov: 50, near: 0.1, far: 1000 }}
      shadows
      gl={{ antialias: true }}
      style={{ background: '#0f1117' }}
    >
      {/* ── Lighting ── */}
      <ambientLight intensity={0.4} />
      <directionalLight
        castShadow
        position={[5, 10, 5]}
        intensity={1.2}
        shadow-mapSize={[2048, 2048]}
      />
      <directionalLight position={[-5, 4, -5]} intensity={0.4} color="#6080ff" />
      <pointLight position={[0, -3, 2]} intensity={0.3} color="#ff8040" />

      {/* ── Environment (provides IBL reflections on metallic meshes) ── */}
      <Suspense fallback={null}>
        <Environment preset="city" />
      </Suspense>

      {/* ── Ground grid for spatial reference ── */}
      <Grid
        position={[0, -2.2, 0]}
        args={[20, 20]}
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#2e3250"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#3d4470"
        fadeDistance={18}
        fadeStrength={1}
        infiniteGrid
      />

      {/* ── Engine model (GLTF or placeholder) ── */}
      <Suspense fallback={null}>
        <EngineModel message={message} onComponentClick={onComponentClick} />
      </Suspense>

      {/* ── Camera controls — right-drag to pan, scroll to zoom ── */}
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.05}
        minDistance={3}
        maxDistance={20}
        maxPolarAngle={Math.PI * 0.85}
      />
    </Canvas>
  )
}
