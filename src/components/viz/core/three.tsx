import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Canvas, useThree, type ThreeEvent } from '@react-three/fiber';
import { Html, Line, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useInView } from './hooks';
import { contourSegments } from './math';
import { useVizColors } from './colors';

/* ---------- Scene ---------- */

/**
 * Standard 3D stage: theme-aware lights, orbit controls, and rendering paused
 * while scrolled off-screen.
 *
 * Inline (inside an article) the stage must not hijack page scrolling, like an
 * embedded map: mouse wheel zooms only with Ctrl/⌘, and on touch screens one
 * finger scrolls the page while two fingers rotate / zoom. In the expanded
 * (full-screen) frame every gesture goes to the scene.
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
  const [expanded, setExpanded] = useState(false);
  const [coarse, setCoarse] = useState(false);
  const [hint, setHint] = useState<null | 'wheel' | 'touch'>(null);
  const hintTimer = useRef(0);

  useEffect(() => {
    const frame = ref.current?.closest('.vz-frame');
    const mq = window.matchMedia('(pointer: coarse)');
    setCoarse(mq.matches);
    if (!frame) return;
    const update = () => setExpanded(frame.classList.contains('is-expanded'));
    update();
    const mo = new MutationObserver(update);
    mo.observe(frame, { attributes: true, attributeFilter: ['class'] });
    return () => mo.disconnect();
  }, []);

  const flash = (kind: 'wheel' | 'touch') => {
    setHint(kind);
    window.clearTimeout(hintTimer.current);
    hintTimer.current = window.setTimeout(() => setHint(null), 1800);
  };

  useEffect(() => {
    const el = ref.current;
    if (!el || expanded) return;
    // Capture phase: keep plain wheel scrolling the page instead of zooming.
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) return;
      e.stopPropagation();
      flash('wheel');
    };
    let touchStart: { x: number; y: number } | null = null;
    const onTouchStart = (e: TouchEvent) => {
      touchStart = e.touches.length === 1 ? { x: e.touches[0].clientX, y: e.touches[0].clientY } : null;
    };
    const onTouchMove = (e: TouchEvent) => {
      if (!touchStart || e.touches.length !== 1) return;
      const dx = Math.abs(e.touches[0].clientX - touchStart.x);
      const dy = Math.abs(e.touches[0].clientY - touchStart.y);
      if (dx + dy > 12) {
        flash('touch');
        touchStart = null;
      }
    };
    el.addEventListener('wheel', onWheel, { capture: true, passive: true });
    el.addEventListener('touchstart', onTouchStart, { passive: true });
    el.addEventListener('touchmove', onTouchMove, { passive: true });
    return () => {
      el.removeEventListener('wheel', onWheel, { capture: true });
      el.removeEventListener('touchstart', onTouchStart);
      el.removeEventListener('touchmove', onTouchMove);
    };
  }, [expanded]);

  useEffect(() => () => window.clearTimeout(hintTimer.current), []);

  // On touch screens, explain the two-finger gesture once when the scene appears.
  useEffect(() => {
    if (!coarse || expanded) return;
    const id = window.setTimeout(() => flash('touch'), 700);
    return () => window.clearTimeout(id);
  }, [coarse]); // eslint-disable-line react-hooks/exhaustive-deps

  const twoFinger = coarse && !expanded;
  return (
    <div ref={ref} className={`vz-3d${twoFinger ? ' is-two-finger' : ''}`} style={{ height }}>
      {hint && (
        <div className="vz-3d-hint" role="status">
          {hint === 'wheel' ? 'Ctrl(⌘) + 스크롤로 확대해요' : '두 손가락으로 돌리고 확대해요 · ⤢ 크게 보기에서는 한 손가락으로!'}
        </div>
      )}
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
        <CameraFit position={camera} target={target} />
        {children}
        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.08}
          target={target}
          autoRotate={autoRotate}
          autoRotateSpeed={0.7}
          enablePan={enablePan}
          touches={twoFinger ? { ONE: undefined as unknown as THREE.TOUCH, TWO: THREE.TOUCH.DOLLY_ROTATE } : { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN }}
          maxPolarAngle={Math.PI * 0.495}
          minDistance={minDistance}
          maxDistance={maxDistance}
        />
      </Canvas>
    </div>
  );
}

/**
 * Narrow (portrait) stages see less horizontally, so pull the camera back along
 * its view direction until the scene fits. Runs on resize only, so user orbiting
 * is otherwise left alone.
 */
function CameraFit({ position, target }: { position: [number, number, number]; target: [number, number, number] }) {
  const { camera, size } = useThree();
  const aspect = size.width / Math.max(1, size.height);
  useEffect(() => {
    const k = Math.min(2, Math.max(1, 1.6 / aspect));
    camera.position.set(
      target[0] + (position[0] - target[0]) * k,
      target[1] + (position[1] - target[1]) * k,
      target[2] + (position[2] - target[2]) * k
    );
    camera.updateProjectionMatrix();
  }, [aspect]); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
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
