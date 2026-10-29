import { useEffect, useMemo, useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { VizProps } from '../Viz';
import { Scene3D, Arrow3D, Label3D, Line, Html } from '../core/three';
import { Button, Controls, Note, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { useElementSize, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors } from '../core/colors';
import { rng } from '../core/math';

type V3 = [number, number, number];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const add = (a: V3, b: V3, k = 1): V3 => [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k];
const scale = (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];

/* ---------- data: three groups spread mostly along one direction, a little along a second, barely along a third ---------- */

function rotation(ax: number, ay: number, az: number) {
  const [cx, sx, cy, sy, cz, sz] = [Math.cos(ax), Math.sin(ax), Math.cos(ay), Math.sin(ay), Math.cos(az), Math.sin(az)];
  // R = Rz · Ry · Rx
  return [
    [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
    [-sy, cy * sx, cy * cx]
  ];
}
const R = rotation(-0.62, 0.48, 0.42);
const rot = (v: V3): V3 => [dot(R[0] as V3, v), dot(R[1] as V3, v), dot(R[2] as V3, v)];

const CLOUD = (() => {
  const r = rng(17);
  const centers = [
    [-2.05, 0.62],
    [0.05, -0.72],
    [2.05, 0.48]
  ];
  const pts: V3[] = [];
  const group: number[] = [];
  for (let g = 0; g < 3; g++)
    for (let i = 0; i < 50; i++) {
      pts.push(rot([centers[g][0] + r.normal() * 0.52, centers[g][1] + r.normal() * 0.4, r.normal() * 0.17]));
      group.push(g);
    }
  const mu = pts.reduce<V3>((m, p) => add(m, p, 1 / pts.length), [0, 0, 0]);
  return { pts: pts.map((p) => add(p, mu, -1)), group };
})();

/** Eigen-decomposition of a symmetric 3×3 matrix (cyclic Jacobi). */
function eigenSym(A: number[][]) {
  const a = A.map((row) => [...row]);
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ];
  for (let sweep = 0; sweep < 60; sweep++) {
    let p = 0;
    let q = 1;
    if (Math.abs(a[0][2]) > Math.abs(a[p][q])) [p, q] = [0, 2];
    if (Math.abs(a[1][2]) > Math.abs(a[p][q])) [p, q] = [1, 2];
    if (Math.abs(a[p][q]) < 1e-13) break;
    const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
    const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
    const cs = 1 / Math.sqrt(t * t + 1);
    const sn = t * cs;
    for (let k = 0; k < 3; k++) {
      const akp = a[k][p];
      const akq = a[k][q];
      a[k][p] = cs * akp - sn * akq;
      a[k][q] = sn * akp + cs * akq;
    }
    for (let k = 0; k < 3; k++) {
      const apk = a[p][k];
      const aqk = a[q][k];
      a[p][k] = cs * apk - sn * aqk;
      a[q][k] = sn * apk + cs * aqk;
    }
    for (let k = 0; k < 3; k++) {
      const vkp = v[k][p];
      const vkq = v[k][q];
      v[k][p] = cs * vkp - sn * vkq;
      v[k][q] = sn * vkp + cs * vkq;
    }
  }
  const pairs = [0, 1, 2].map((i) => ({ value: a[i][i], vec: [v[0][i], v[1][i], v[2][i]] as V3 }));
  pairs.sort((x, y) => y.value - x.value);
  return pairs;
}

const PCA = (() => {
  const C = [0, 1, 2].map((i) => [0, 1, 2].map((j) => CLOUD.pts.reduce((s, p) => s + p[i] * p[j], 0) / CLOUD.pts.length));
  const e = eigenSym(C);
  // stable, readable signs: PC1 points right-ish, PC2 up-ish
  if (e[0].vec[0] < 0) e[0].vec = scale(e[0].vec, -1);
  if (e[1].vec[1] < 0) e[1].vec = scale(e[1].vec, -1);
  const c3 = e[0].vec;
  const c4 = e[1].vec;
  e[2].vec = [c3[1] * c4[2] - c3[2] * c4[1], c3[2] * c4[0] - c3[0] * c4[2], c3[0] * c4[1] - c3[1] * c4[0]];
  const total = e.reduce((s, x) => s + x.value, 0);
  return { e, total };
})();

/** Plane normal for tilt θ: θ = 0 → normal = PC3 (best plane ①②), θ = 90° → normal = PC1 (bad plane ②③). */
function planeFor(deg: number) {
  const t = (deg * Math.PI) / 180;
  const [e1, e2, e3] = PCA.e.map((x) => x.vec);
  const n = add(scale(e3, Math.cos(t)), e1, Math.sin(t));
  const w = add(scale(e1, Math.cos(t)), e3, -Math.sin(t));
  const lost = Math.cos(t) ** 2 * PCA.e[2].value + Math.sin(t) ** 2 * PCA.e[0].value;
  return { n, u: e2, w, kept: 1 - lost / PCA.total };
}

/** PCA in 3D: find the plane that keeps the most spread, then flatten the cloud onto it. */
export default function PCA3D({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [tilt, setTilt] = useState(0);
  const [flat, setFlat] = useState(false);
  const [shadows, setShadows] = useState(true);
  const [arrows, setArrows] = useState(true);
  const [box, { width }] = useElementSize<HTMLDivElement>();
  const plane = planeFor(tilt);
  const share = PCA.e.map((x) => x.value / PCA.total);
  const groupColors = [c.series[0], c.series[1], c.series[2]];
  const stageH = expanded ? 'min(66vh, 740px)' : Math.round(Math.max(330, Math.min(480, width * 0.62)));

  // separation of the three groups in the 2D shadow (between-group spread / within-group spread)
  const sep = useMemo(() => {
    const proj = CLOUD.pts.map((p) => [dot(p, plane.w), dot(p, plane.u)]);
    const means = [0, 1, 2].map((g) => {
      const mine = proj.filter((_, i) => CLOUD.group[i] === g);
      return [mine.reduce((s, q) => s + q[0], 0) / mine.length, mine.reduce((s, q) => s + q[1], 0) / mine.length];
    });
    let within = 0;
    proj.forEach((q, i) => {
      const m = means[CLOUD.group[i]];
      within += (q[0] - m[0]) ** 2 + (q[1] - m[1]) ** 2;
    });
    within /= proj.length;
    const between = means.reduce((s, m) => s + m[0] ** 2 + m[1] ** 2, 0) / 3;
    return between / within;
  }, [plane.w, plane.u]);

  const best = tilt < 8;
  const worst = tilt > 75;
  let note: { text: string; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' | 'bad' };
  if (best && flat)
    note = { text: `가장 넓게 퍼진 두 방향(①②)을 담은 평면에 그림자를 찍었어요. 3차원 → 2차원으로 줄였는데도 분산(정보)의 ${Math.round(plane.kept * 100)}%가 남았고, 세 무리도 또렷하게 구별돼요! 🎉`, mood: 'cheer', tone: 'good' };
  else if (worst && flat)
    note = { text: `퍼짐이 가장 큰 ①번 방향을 버린 평면이에요. 그림자에서 세 무리가 겹쳐 버렸죠? 분산이 ${Math.round(plane.kept * 100)}%밖에 안 남았어요 — 정보를 많이 잃었어요.`, mood: 'wow', tone: 'bad' };
  else if (flat)
    note = { text: `평면을 기울이면 남는 분산이 ${Math.round(plane.kept * 100)}%로 변해요. 오른쪽 그림자 그림에서 무리들이 얼마나 구별되는지 보세요.`, mood: 'think' };
  else
    note = { text: '점 150개가 3차원에 떠 있어요. 화살표 ①은 가장 넓게 퍼진 방향(제1주성분), ②는 그다음, ③은 거의 안 퍼진 방향이에요. "🫓 납작하게 누르기"로 평면에 그림자를 찍어 보세요!', mood: 'smile' };

  return (
    <div ref={box}>
      <Scene3D height={stageH} camera={[7.4, 5.0, 8.6]} target={[0, 0, 0]} minDistance={5} maxDistance={22}>
        {/* the first <Html> overlay in a scene currently never shows up (gradient-descent-3d loses its first axis label too);
            this empty one absorbs that so the ① label is visible */}
        <Html position={[0, 0, 0]}>
          <span aria-hidden="true" />
        </Html>
        <gridHelper args={[10, 10, c.theme === 'dark' ? '#4a5590' : '#d9cdbd', c.theme === 'dark' ? '#2a3466' : '#efe5d8']} position={[0, -2.6, 0]} />
        <Cloud n={plane.n} flat={flat} reduce={reduce} colors={groupColors} shadows={shadows} lineColor={c.ink} />
        <PlaneMesh n={plane.n} u={plane.u} w={plane.w} color={c.ink} dark={c.theme === 'dark'} best={best} />
        {arrows &&
          PCA.e.map((x, i) => {
            const len = Math.max(0.55, Math.sqrt(x.value) * 2.1);
            const tip = scale(x.vec, len);
            const label = ['① 가장 넓게 퍼짐', '② 두 번째', '③ 거의 안 퍼짐'][i];
            // labels sit on the arrow (not past its tip) so narrow screens don't clip them
            const at = add(scale(x.vec, i === 0 ? len * 0.62 : len + 0.3), [0, 0.32, 0]);
            return (
              <group key={i}>
                <Arrow3D from={[0, 0, 0]} to={tip} color={c.ink} width={i === 0 ? 4 : 3} head={0.28} />
                <Label3D position={at}>{label}</Label3D>
              </group>
            );
          })}
      </Scene3D>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14, alignItems: 'start' }}>
          <div style={{ display: 'grid', gap: 10 }}>
            <div className="vz-stats">
              <Stat label="남는 분산 (정보)" value={`${Math.round(plane.kept * 100)}%`} tone={plane.kept > 0.9 ? 'good' : plane.kept < 0.6 ? 'bad' : 'warn'} />
              <Stat label="무리 구별도" value={sep.toFixed(1)} sub="클수록 또렷" tone={sep > 4 ? 'good' : sep < 1.5 ? 'bad' : undefined} />
            </div>
            <div>
              <div className="vz-panel-title" style={{ marginBottom: 6 }}>
                방향별로 담은 분산 (성적표)
              </div>
              <div style={{ display: 'flex', height: 28, borderRadius: 10, overflow: 'hidden', border: '1px solid var(--border)', fontSize: 12.5, fontWeight: 850 }}>
                {share.map((s, i) => (
                  <div
                    key={i}
                    style={{
                      width: `${s * 100}%`,
                      display: 'grid',
                      placeItems: 'center',
                      background: i === 0 ? 'var(--brand)' : i === 1 ? 'var(--brand-soft)' : 'var(--surface-3)',
                      color: i === 0 ? 'var(--brand-ink)' : 'var(--text)',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden'
                    }}
                  >
                    {s > 0.06 ? `${['①', '②', '③'][i]} ${Math.round(s * 100)}%` : ''}
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 12.5, color: 'var(--text-2)', marginTop: 6 }}>
                ③은 겨우 {Math.round(share[2] * 100)}% — 버려도 거의 손해가 없어요. 그래서 ①②만 남기는 게 PCA예요.
              </div>
            </div>
          </div>
          <Shadow2D plane={plane} colors={groupColors} />
        </div>
      </div>

      <Controls>
        <Button variant="primary" onClick={() => setFlat((f) => !f)}>
          {flat ? '↩ 다시 부풀리기' : '🫓 납작하게 누르기'}
        </Button>
        <Segmented
          label="평면 고르기"
          options={[
            { value: 0, label: '✅ 최선 (①②)' },
            { value: 90, label: '❌ 나쁨 (②③)' }
          ]}
          value={tilt === 0 || tilt === 90 ? tilt : -1}
          onChange={setTilt}
        />
        <Slider label="평면 기울이기" value={tilt} min={0} max={90} step={1} onChange={setTilt} format={(v) => `${Math.round(v)}°`} />
        <Toggle label="그림자 선" checked={shadows} onChange={setShadows} />
        <Toggle label="주성분 화살표" checked={arrows} onChange={setArrows} />
      </Controls>
    </div>
  );
}

/* ---------- scene parts ---------- */

function Cloud({ n, flat, reduce, colors, shadows, lineColor }: { n: V3; flat: boolean; reduce: boolean; colors: string[]; shadows: boolean; lineColor: string }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const lines = useRef<THREE.LineSegments>(null);
  const t = useRef(0);
  const dirty = useRef(true);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const count = CLOUD.pts.length;
  const lineGeom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(count * 6), 3));
    return g;
  }, [count]);

  useEffect(() => {
    dirty.current = true;
  }, [n, flat, shadows]);

  useEffect(() => {
    const m = mesh.current;
    if (!m) return;
    const col = new THREE.Color();
    CLOUD.group.forEach((g, i) => m.setColorAt(i, col.set(colors[g])));
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  }, [colors]);

  useFrame((_, dt) => {
    const target = flat ? 1 : 0;
    const before = t.current;
    t.current = reduce ? target : t.current + (target - t.current) * (1 - Math.exp(-dt * 4.5));
    if (Math.abs(t.current - target) < 0.002) t.current = target;
    if (before === t.current && !dirty.current) return;
    dirty.current = false;
    const m = mesh.current;
    const pos = lineGeom.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < count; i++) {
      const p = CLOUD.pts[i];
      const h = dot(p, n);
      const q = add(p, n, -h * t.current);
      dummy.position.set(q[0], q[1], q[2]);
      dummy.updateMatrix();
      m?.setMatrixAt(i, dummy.matrix);
      const proj = add(p, n, -h);
      pos.setXYZ(i * 2, p[0], p[1], p[2]);
      pos.setXYZ(i * 2 + 1, proj[0], proj[1], proj[2]);
    }
    if (m) m.instanceMatrix.needsUpdate = true;
    pos.needsUpdate = true;
    if (lines.current) {
      const mat = lines.current.material as THREE.LineBasicMaterial;
      mat.opacity = shadows ? 0.12 + 0.2 * t.current : 0;
      lines.current.visible = shadows && t.current > 0.02;
    }
  });

  return (
    <group>
      <instancedMesh ref={mesh} args={[undefined, undefined, count]}>
        <sphereGeometry args={[0.085, 14, 14]} />
        <meshStandardMaterial roughness={0.4} metalness={0.05} />
      </instancedMesh>
      <lineSegments ref={lines} geometry={lineGeom}>
        <lineBasicMaterial color={lineColor} transparent opacity={0.2} />
      </lineSegments>
    </group>
  );
}

function PlaneMesh({ n, u, w, color, dark, best }: { n: V3; u: V3; w: V3; color: string; dark: boolean; best: boolean }) {
  const quat = useMemo(() => {
    const m = new THREE.Matrix4().makeBasis(new THREE.Vector3(...w), new THREE.Vector3(...u), new THREE.Vector3(...n));
    return new THREE.Quaternion().setFromRotationMatrix(m);
  }, [n, u, w]);
  const lemon = useMemo(() => {
    const v = typeof window === 'undefined' ? '' : getComputedStyle(document.documentElement).getPropertyValue('--lemon').trim();
    return v || (dark ? '#ffd23f' : '#e7a90f');
  }, [dark]);
  const S = 2.9;
  const corner = (a: number, b: number): V3 => add(scale(w, a * S), u, b * S);
  return (
    <group>
      <mesh quaternion={quat}>
        <planeGeometry args={[S * 2, S * 2]} />
        <meshBasicMaterial color={color} transparent opacity={dark ? 0.08 : 0.05} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <Line points={[corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1), corner(-1, -1)]} color={best ? lemon : color} lineWidth={2} transparent opacity={0.8} />
    </group>
  );
}

function Shadow2D({ plane, colors }: { plane: ReturnType<typeof planeFor>; colors: string[] }) {
  const c = useVizColors();
  const S = 200;
  const D = 4.2;
  const X = (v: number) => S / 2 + (v / D) * (S / 2);
  const Y = (v: number) => S / 2 - (v / D) * (S / 2);
  return (
    <div>
      <div className="vz-panel-title" style={{ marginBottom: 6 }}>
        평면에 찍힌 그림자 = 2차원 결과
      </div>
      <svg className="vz-svg" viewBox={`0 0 ${S} ${S}`} style={{ width: '100%', maxWidth: 300, display: 'block', borderRadius: 14, border: '1px solid var(--border)', background: 'var(--surface-2)' }} role="img" aria-label="평면에 투영한 2차원 산점도">
        <line x1={0} x2={S} y1={S / 2} y2={S / 2} stroke={c.grid} />
        <line x1={S / 2} x2={S / 2} y1={0} y2={S} stroke={c.grid} />
        {CLOUD.pts.map((p, i) => (
          <circle key={i} cx={X(dot(p, plane.w))} cy={Y(dot(p, plane.u))} r={3.2} fill={colors[CLOUD.group[i]]} opacity={0.85} />
        ))}
        <text x={S - 6} y={S - 7} textAnchor="end" style={{ fontSize: 10.5, fontWeight: 750, fill: c.text }}>
          평면 축 1 →
        </text>
        <text x={6} y={13} style={{ fontSize: 10.5, fontWeight: 750, fill: c.text }}>
          ↑ 평면 축 2 (②)
        </text>
      </svg>
    </div>
  );
}
