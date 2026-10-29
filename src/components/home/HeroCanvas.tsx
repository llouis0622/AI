import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import * as THREE from 'three';
import { useVizColors } from '../viz/core/colors';

/* A bumpy loss landscape with one deep valley. */
const f = (x: number, z: number) =>
  0.055 * (x * x + z * z) + 0.55 * Math.sin(0.8 * x + 0.5) * Math.cos(0.7 * z) - 1.2 * Math.exp(-((x - 1.2) ** 2 + (z - 0.8) ** 2) / 2.4);

const grad = (x: number, z: number): [number, number] => {
  const h = 1e-3;
  return [(f(x + h, z) - f(x - h, z)) / (2 * h), (f(x, z + h) - f(x, z - h)) / (2 * h)];
};

const SIZE = 10;
const SEG = 56;

export default function HeroCanvas() {
  const c = useVizColors();
  return (
    <Canvas className="hero-canvas" dpr={[1, 1.5]} camera={{ position: [8.5, 6.2, 8.5], fov: 38 }} gl={{ antialias: true, alpha: true }}>
      <ambientLight intensity={c.theme === 'dark' ? 0.5 : 0.95} />
      <directionalLight position={[6, 10, 4]} intensity={c.theme === 'dark' ? 0.9 : 1.1} />
      <Rig>
        <Terrain />
        <Roller />
        <Dust />
      </Rig>
    </Canvas>
  );
}

function Rig({ children }: { children: React.ReactNode }) {
  const g = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (g.current) g.current.rotation.y += dt * 0.08;
  });
  return (
    <group ref={g} position={[0, -0.6, 0]}>
      {children}
    </group>
  );
}

function Terrain() {
  const c = useVizColors();
  const dark = c.theme === 'dark';
  const { geom, wire } = useMemo(() => {
    const g = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
    g.rotateX(-Math.PI / 2);
    const pos = g.attributes.position as THREE.BufferAttribute;
    let lo = Infinity;
    let hi = -Infinity;
    const hs: number[] = [];
    for (let i = 0; i < pos.count; i++) {
      const h = f(pos.getX(i), pos.getZ(i));
      hs.push(h);
      lo = Math.min(lo, h);
      hi = Math.max(hi, h);
    }
    const a = new THREE.Color(c.low);
    const m = new THREE.Color(c.mid);
    const b = new THREE.Color(c.high);
    const tmp = new THREE.Color();
    const col = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      pos.setY(i, hs[i]);
      const t = (hs[i] - lo) / (hi - lo);
      if (t < 0.5) tmp.copy(a).lerp(m, t * 2);
      else tmp.copy(m).lerp(b, (t - 0.5) * 2);
      col.set([tmp.r, tmp.g, tmp.b], i * 3);
    }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.computeVertexNormals();
    const wg = new THREE.PlaneGeometry(SIZE, SIZE, 28, 28);
    wg.rotateX(-Math.PI / 2);
    const wp = wg.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < wp.count; i++) wp.setY(i, f(wp.getX(i), wp.getZ(i)) + 0.012);
    return { geom: g, wire: new THREE.WireframeGeometry(wg) };
  }, [c.low, c.mid, c.high]);
  return (
    <group>
      <mesh geometry={geom}>
        <meshStandardMaterial
          vertexColors
          roughness={dark ? 0.3 : 0.65}
          metalness={dark ? 0.3 : 0.05}
          transparent
          opacity={dark ? 0.85 : 0.95}
          side={THREE.DoubleSide}
          emissive={dark ? '#1a0b3a' : '#000000'}
          emissiveIntensity={dark ? 0.7 : 0}
        />
      </mesh>
      <lineSegments geometry={wire}>
        <lineBasicMaterial color={dark ? '#5ce8ff' : '#6a58f0'} transparent opacity={dark ? 0.3 : 0.14} />
      </lineSegments>
    </group>
  );
}

/** Gradient descent with momentum, restarting from a new hilltop when it settles. */
function Roller() {
  const c = useVizColors();
  const ball = useRef<THREE.Group>(null);
  const state = useRef({ x: -3.8, z: -3.2, vx: 0, vz: 0, t: 0, fade: 1, trail: [] as [number, number, number][], seed: 1 });
  const lineRef = useRef<{ geometry: { setPositions: (p: number[]) => void } } | null>(null);
  const restart = () => {
    const s = state.current;
    s.seed += 1;
    const ang = s.seed * 2.39996;
    s.x = Math.cos(ang) * 4.2;
    s.z = Math.sin(ang) * 4.2;
    s.vx = 0;
    s.vz = 0;
    s.t = 0;
    s.trail = [];
  };
  useFrame((_, dt) => {
    const s = state.current;
    const steps = 2;
    for (let k = 0; k < steps; k++) {
      const [gx, gz] = grad(s.x, s.z);
      s.vx = 0.86 * s.vx - 0.055 * gx;
      s.vz = 0.86 * s.vz - 0.055 * gz;
      s.x = Math.max(-4.8, Math.min(4.8, s.x + s.vx));
      s.z = Math.max(-4.8, Math.min(4.8, s.z + s.vz));
    }
    s.t += dt;
    const y = f(s.x, s.z) + 0.22;
    s.trail.push([s.x, y + 0.02, s.z]);
    if (s.trail.length > 140) s.trail.shift();
    const settled = Math.hypot(s.vx, s.vz) < 0.002 && s.t > 2;
    if (settled || s.t > 14) {
      s.fade -= dt * 1.5;
      if (s.fade <= 0) {
        restart();
        s.fade = 0.01;
      }
    } else if (s.fade < 1) s.fade = Math.min(1, s.fade + dt * 2);
    if (ball.current) {
      ball.current.position.set(s.x, y, s.z);
      ball.current.scale.setScalar(Math.max(0.01, s.fade));
    }
    if (lineRef.current && s.trail.length > 1) lineRef.current.geometry.setPositions(s.trail.flat());
  });
  return (
    <group>
      <Line ref={lineRef as never} points={[[0, 0, 0], [0, 0.01, 0]]} color={c.series[2]} lineWidth={3} transparent opacity={0.9} />
      <group ref={ball}>
        <mesh>
          <sphereGeometry args={[0.22, 32, 32]} />
          <meshStandardMaterial color={c.series[0]} emissive={c.series[0]} emissiveIntensity={0.8} roughness={0.2} />
        </mesh>
        <mesh>
          <sphereGeometry args={[0.46, 24, 24]} />
          <meshBasicMaterial color={c.series[0]} transparent opacity={0.18} depthWrite={false} />
        </mesh>
      </group>
    </group>
  );
}

function Dust() {
  const c = useVizColors();
  const ref = useRef<THREE.Points>(null);
  const geom = useMemo(() => {
    const n = 260;
    const p = new Float32Array(n * 3);
    let s = 7;
    const r = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    for (let i = 0; i < n; i++) {
      p[i * 3] = (r() - 0.5) * 14;
      p[i * 3 + 1] = r() * 6 - 0.5;
      p[i * 3 + 2] = (r() - 0.5) * 14;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    return g;
  }, []);
  useFrame((st) => {
    if (ref.current) ref.current.position.y = Math.sin(st.clock.elapsedTime * 0.4) * 0.15;
  });
  return (
    <points ref={ref} geometry={geom}>
      <pointsMaterial size={c.theme === 'dark' ? 0.05 : 0.06} color={c.theme === 'dark' ? '#bdf6ff' : '#b7abff'} transparent opacity={c.theme === 'dark' ? 0.9 : 0.6} sizeAttenuation />
    </points>
  );
}
