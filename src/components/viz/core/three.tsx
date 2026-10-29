import { useMemo, useRef, type ReactNode } from 'react';
import { Canvas, type ThreeEvent } from '@react-three/fiber';
import { Html, Line, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useInView } from './hooks';
import { contourSegments } from './math';
import { useVizColors } from './colors';

/* ---------- Scene ---------- */

/**
 * Standard 3D stage: theme-aware lights, orbit controls (drag to rotate, wheel /
 * pinch to zoom), and rendering paused while scrolled off-screen.
 */
export function Scene3D({
  children,
  height = 420,
  camera = [7, 6, 8],
  target = [0, 0.8, 0],
  fov = 42,
  autoRotate = false,
  minDistance = 4,
  maxDistance = 22,
  enablePan = false
}: {
  children: ReactNode;
  height?: number | string;
  camera?: [number, number, number];
  target?: [number, number, number];
  fov?: number;
  autoRotate?: boolean;
  minDistance?: number;
  maxDistance?: number;
  enablePan?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, '100px');
  const c = useVizColors();
  const dark = c.theme === 'dark';
  return (
    <div ref={ref} className="vz-3d" style={{ height }}>
      <Canvas
        dpr={[1, 1.75]}
        camera={{ position: camera, fov, near: 0.1, far: 200 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        frameloop={inView ? 'always' : 'never'}
      >
        <ambientLight intensity={dark ? 0.55 : 0.9} />
        <hemisphereLight args={[dark ? '#7fe9ff' : '#ffffff', dark ? '#2a0f4a' : '#f3e6d6', dark ? 0.6 : 0.55]} />
        <directionalLight position={[6, 12, 8]} intensity={dark ? 1.0 : 1.25} />
        <directionalLight position={[-8, 6, -6]} intensity={dark ? 0.35 : 0.3} />
        {children}
        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.08}
          target={target}
          autoRotate={autoRotate}
          autoRotateSpeed={0.7}
          enablePan={enablePan}
          maxPolarAngle={Math.PI * 0.495}
          minDistance={minDistance}
          maxDistance={maxDistance}
        />
      </Canvas>
    </div>
  );
}

/* ---------- Coordinate mapping (function space ⇄ scene space) ---------- */

export interface Mapper {
  uRange: [number, number];
  vRange: [number, number];
  hScale: number;
  size: [number, number];
  /** function (u, v, h) → scene [x, y, z] (y is up) */
  toScene: (u: number, v: number, h: number) => [number, number, number];
  /** scene x, z → function (u, v) */
  fromScene: (x: number, z: number) => [number, number];
}

/** Map a u×v domain onto a scene rectangle whose longer side is `size` units. */
export function makeMapper(uRange: [number, number], vRange: [number, number], size = 8, hScale = 1): Mapper {
  const du = uRange[1] - uRange[0];
  const dv = vRange[1] - vRange[0];
  const k = size / Math.max(du, dv);
  const sx = du * k;
  const sz = dv * k;
  const cu = (uRange[0] + uRange[1]) / 2;
  const cv = (vRange[0] + vRange[1]) / 2;
  return {
    uRange,
    vRange,
    hScale,
    size: [sx, sz],
    toScene: (u, v, h) => [(u - cu) * k, h * hScale, -(v - cv) * k],
    fromScene: (x, z) => [x / k + cu, -z / k + cv]
  };
}

/* ---------- Surface ---------- */

/**
 * Height-field surface h = fn(u, v) with a low→mid→high color ramp and an
 * optional wire grid. Heights above `maxH` are clipped (handy for divergence).
 */
export function Surface({
  fn,
  mapper,
  segments = 72,
  maxH = Infinity,
  opacity = 0.92,
  wire = true,
  onPick
}: {
  fn: (u: number, v: number) => number;
  mapper: Mapper;
  segments?: number;
  maxH?: number;
  opacity?: number;
  wire?: boolean;
  /** Click (not drag) on the surface → function coordinates. */
  onPick?: (u: number, v: number) => void;
}) {
  const c = useVizColors();
  const { geom, wireGeom } = useMemo(() => {
    const [sx, sz] = mapper.size;
    const g = new THREE.PlaneGeometry(sx, sz, segments, segments);
    g.rotateX(-Math.PI / 2);
    const pos = g.attributes.position as THREE.BufferAttribute;
    const hs: number[] = [];
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      const [u, v] = mapper.fromScene(pos.getX(i), pos.getZ(i));
      const h = Math.min(maxH, fn(u, v));
      hs.push(h);
      lo = Math.min(lo, h);
      hi = Math.max(hi, h);
    }
    const cl = new THREE.Color(c.low);
    const cm = new THREE.Color(c.mid);
    const ch = new THREE.Color(c.high);
    const tmp = new THREE.Color();
    const colors = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      pos.setY(i, hs[i] * mapper.hScale);
      const t = hi > lo ? (hs[i] - lo) / (hi - lo) : 0;
      const tt = Math.sqrt(t);
      if (tt < 0.5) tmp.copy(cl).lerp(cm, tt * 2);
      else tmp.copy(cm).lerp(ch, (tt - 0.5) * 2);
      colors[i * 3] = tmp.r;
      colors[i * 3 + 1] = tmp.g;
      colors[i * 3 + 2] = tmp.b;
    }
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    g.computeVertexNormals();
    // sparse wire grid
    const wireSeg = Math.max(8, Math.round(segments / 4));
    const wg = new THREE.PlaneGeometry(sx, sz, wireSeg, wireSeg);
    wg.rotateX(-Math.PI / 2);
    const wp = wg.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < wp.count; i++) {
      const [u, v] = mapper.fromScene(wp.getX(i), wp.getZ(i));
      wp.setY(i, Math.min(maxH, fn(u, v)) * mapper.hScale + 0.01);
    }
    return { geom: g, wireGeom: new THREE.WireframeGeometry(wg) };
  }, [fn, mapper, segments, maxH, c.low, c.mid, c.high]);

  const dark = c.theme === 'dark';
  const pick = (e: ThreeEvent<MouseEvent>) => {
    if (!onPick || e.delta > 6) return;
    e.stopPropagation();
    const [u, v] = mapper.fromScene(e.point.x, e.point.z);
    onPick(u, v);
  };

  return (
    <group>
      <mesh geometry={geom} onClick={pick}>
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          roughness={dark ? 0.35 : 0.6}
          metalness={dark ? 0.25 : 0.05}
          transparent={opacity < 1}
          opacity={opacity}
          emissive={dark ? new THREE.Color('#1a0b3a') : new THREE.Color('#000000')}
          emissiveIntensity={dark ? 0.6 : 0}
        />
      </mesh>
      {wire && (
        <lineSegments geometry={wireGeom}>
          <lineBasicMaterial color={dark ? '#7fe9ff' : '#2a2540'} transparent opacity={dark ? 0.22 : 0.12} />
        </lineSegments>
      )}
    </group>
  );
}

/* ---------- Contours ---------- */

/** Level lines of fn drawn on the surface (y = fn + lift) or flat on the floor (floorY). */
export function Contours({
  fn,
  mapper,
  levels,
  color,
  floorY,
  lift = 0.02,
  res = 70,
  opacity = 0.75
}: {
  fn: (u: number, v: number) => number;
  mapper: Mapper;
  levels: number[];
  color: string;
  floorY?: number;
  lift?: number;
  res?: number;
  opacity?: number;
}) {
  const geom = useMemo(() => {
    const pts: number[] = [];
    for (const level of levels) {
      const seg = contourSegments(fn, mapper.uRange, mapper.vRange, level, res);
      for (let i = 0; i < seg.length; i += 2) {
        const [x, y, z] = mapper.toScene(seg[i], seg[i + 1], level);
        pts.push(x, floorY ?? y + lift, z);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    return g;
  }, [fn, mapper, levels, floorY, lift, res]);
  return (
    <lineSegments geometry={geom}>
      <lineBasicMaterial color={color} transparent opacity={opacity} />
    </lineSegments>
  );
}

/* ---------- Small primitives ---------- */

export function Ball({ position, color, radius = 0.2, glow = true }: { position: [number, number, number]; color: string; radius?: number; glow?: boolean }) {
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[radius, 32, 32]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={glow ? 0.55 : 0.1} roughness={0.25} metalness={0.1} />
      </mesh>
      {glow && (
        <mesh>
          <sphereGeometry args={[radius * 1.9, 24, 24]} />
          <meshBasicMaterial color={color} transparent opacity={0.16} depthWrite={false} />
        </mesh>
      )}
    </group>
  );
}

export function Arrow3D({
  from,
  to,
  color,
  width = 3,
  head = 0.22
}: {
  from: [number, number, number];
  to: [number, number, number];
  color: string;
  width?: number;
  head?: number;
}) {
  const { quat, tip, len } = useMemo(() => {
    const a = new THREE.Vector3(...from);
    const b = new THREE.Vector3(...to);
    const dir = b.clone().sub(a);
    const l = dir.length();
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
    return { quat: q, tip: b, len: l };
  }, [from, to]);
  if (len < 1e-4) return null;
  const base: [number, number, number] = [
    from[0] + (to[0] - from[0]) * Math.max(0, 1 - head / len),
    from[1] + (to[1] - from[1]) * Math.max(0, 1 - head / len),
    from[2] + (to[2] - from[2]) * Math.max(0, 1 - head / len)
  ];
  return (
    <group>
      <Line points={[from, base]} color={color} lineWidth={width} />
      <mesh position={[tip.x, tip.y, tip.z]} quaternion={quat}>
        <coneGeometry args={[head * 0.45, head, 16]} />
        <meshBasicMaterial color={color} />
      </mesh>
    </group>
  );
}

export function Label3D({ position, children }: { position: [number, number, number]; children: ReactNode }) {
  return (
    <Html position={position} center zIndexRange={[10, 0]}>
      <span className="vz-3d-label">{children}</span>
    </Html>
  );
}

/** Floor outline + axis arrows with labels for a mapped domain. */
export function DomainAxes({
  mapper,
  uLabel,
  vLabel,
  hLabel,
  hMax,
  color
}: {
  mapper: Mapper;
  uLabel: string;
  vLabel: string;
  hLabel?: string;
  hMax?: number;
  color: string;
}) {
  const [u0, u1] = mapper.uRange;
  const [v0, v1] = mapper.vRange;
  const p = (u: number, v: number, h = 0) => mapper.toScene(u, v, h);
  return (
    <group>
      <Line points={[p(u0, v0), p(u1, v0), p(u1, v1), p(u0, v1), p(u0, v0)]} color={color} lineWidth={1} transparent opacity={0.5} />
      <Arrow3D from={p(u0, v0)} to={p(u1 + (u1 - u0) * 0.08, v0)} color={color} width={2} head={0.25} />
      <Arrow3D from={p(u0, v0)} to={p(u0, v1 + (v1 - v0) * 0.08)} color={color} width={2} head={0.25} />
      <Label3D position={p(u1 + (u1 - u0) * 0.16, v0)}>{uLabel}</Label3D>
      <Label3D position={p(u0, v1 + (v1 - v0) * 0.16)}>{vLabel}</Label3D>
      {hLabel && hMax !== undefined && (
        <>
          <Arrow3D from={p(u0, v0)} to={p(u0, v0, hMax)} color={color} width={2} head={0.25} />
          <Label3D position={p(u0, v0, hMax * 1.08)}>{hLabel}</Label3D>
        </>
      )}
    </group>
  );
}

export { Line, Html };
