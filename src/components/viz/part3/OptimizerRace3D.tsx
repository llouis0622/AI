import { useEffect, useMemo, useRef, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { VizProps } from '../Viz';
import { Scene3D, Surface, Contours, Ball, Label3D, DomainAxes, Line, makeMapper } from '../core/three';
import { Controls, Note, PlayBar, Segmented, Slider, Toggle } from '../core/ui';
import { useElementSize, useInterval } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { rng } from '../core/math';

type TerrainId = 'ravine' | 'saddle' | 'banana';
type OptId = 'sgd' | 'mom' | 'rms' | 'adam';

interface Terrain {
  label: string;
  f: (x: number, y: number) => number;
  g: (x: number, y: number) => [number, number];
  /** display height (may be compressed) */
  h: (x: number, y: number) => number;
  xr: [number, number];
  yr: [number, number];
  hScale: number;
  start: [number, number];
  min: [number, number];
  lr: Record<OptId, number>;
  noise: number;
  story: string;
}

const TERRAINS: Record<TerrainId, Terrain> = {
  ravine: {
    label: '좁은 협곡',
    f: (x, y) => 0.04 * x * x + 2 * y * y,
    g: (x, y) => [0.08 * x, 4 * y],
    h: (x, y) => 0.04 * x * x + 2 * y * y,
    xr: [-4, 4],
    yr: [-2.6, 2.6],
    hScale: 0.3,
    start: [-3.7, 2.2],
    min: [0, 0],
    lr: { sgd: 0.42, mom: 0.08, rms: 0.1, adam: 0.2 },
    noise: 0.5,
    story: '좁고 긴 협곡이에요. SGD는 가파른 벽 사이를 지그재그로 튕기고, 관성(모멘텀)은 좌우 흔들림을 상쇄하며 협곡 방향으로 가속해요.'
  },
  saddle: {
    label: '안장점',
    f: (x, y) => 0.5 * x * x - 0.5 * y * y + 0.08 * y ** 4 + 1,
    g: (x, y) => [x, -y + 0.32 * y ** 3],
    h: (x, y) => 0.5 * x * x - 0.5 * y * y + 0.08 * y ** 4 + 1,
    xr: [-3, 3],
    yr: [-2.6, 2.6],
    hScale: 0.75,
    start: [-2.8, 0.002],
    min: [0, 1.768],
    lr: { sgd: 0.1, mom: 0.03, rms: 0.1, adam: 0.1 },
    noise: 0.25,
    story: '말 안장처럼 한쪽은 오르막, 한쪽은 내리막인 곳이에요. 기울기가 거의 0인 안장 근처에서 SGD는 한참 머뭇거리고, 좌표별로 보폭을 키우는 Adam·RMSProp은 금방 빠져나가요.'
  },
  banana: {
    label: '바나나 골짜기',
    f: (x, y) => (1 - x) ** 2 + 3 * (y - x * x) ** 2,
    g: (x, y) => [-2 * (1 - x) - 12 * x * (y - x * x), 6 * (y - x * x)],
    h: (x, y) => Math.sqrt((1 - x) ** 2 + 3 * (y - x * x) ** 2),
    xr: [-2, 2],
    yr: [-1, 3],
    hScale: 0.45,
    start: [-1.5, 2.0],
    min: [1, 1],
    lr: { sgd: 0.05, mom: 0.02, rms: 0.1, adam: 0.2 },
    noise: 1,
    story: '휘어진 바나나 모양의 골짜기(유명한 로젠브록 함수)예요. 바닥이 완만하게 휘어서 SGD는 큰 보폭을 못 쓰고 느릿느릿 가요. (높이는 보기 좋게 √로 눌렀어요)'
  }
};

const OPTS: { id: OptId; name: string; desc: string }[] = [
  { id: 'sgd', name: 'SGD', desc: '기울기만큼 그대로' },
  { id: 'mom', name: '모멘텀', desc: '관성으로 가속' },
  { id: 'rms', name: 'RMSProp', desc: '좌표별 보폭' },
  { id: 'adam', name: 'Adam', desc: '관성 + 좌표별 보폭' }
];

interface Runner {
  x: number;
  y: number;
  vx: number;
  vy: number;
  sx: number;
  sy: number;
  k: number;
  path: [number, number][];
  doneAt: number | null;
  out: boolean;
}

const fresh = (s: [number, number]): Runner => ({ x: s[0], y: s[1], vx: 0, vy: 0, sx: 0, sy: 0, k: 0, path: [s], doneAt: null, out: false });

function stepRunner(r: Runner, opt: OptId, t: Terrain, lr: number, noise: number, rand: () => number, n: number): Runner {
  if (r.out) return r;
  let [gx, gy] = t.g(r.x, r.y);
  if (noise > 0) {
    gx += noise * rand();
    gy += noise * rand();
  }
  const s = { ...r };
  if (opt === 'sgd') {
    s.x -= lr * gx;
    s.y -= lr * gy;
  } else if (opt === 'mom') {
    s.vx = 0.9 * s.vx + gx;
    s.vy = 0.9 * s.vy + gy;
    s.x -= lr * s.vx;
    s.y -= lr * s.vy;
  } else if (opt === 'rms') {
    s.sx = 0.9 * s.sx + 0.1 * gx * gx;
    s.sy = 0.9 * s.sy + 0.1 * gy * gy;
    s.x -= (lr * gx) / (Math.sqrt(s.sx) + 1e-8);
    s.y -= (lr * gy) / (Math.sqrt(s.sy) + 1e-8);
  } else {
    s.k += 1;
    s.vx = 0.9 * s.vx + 0.1 * gx;
    s.vy = 0.9 * s.vy + 0.1 * gy;
    s.sx = 0.999 * s.sx + 0.001 * gx * gx;
    s.sy = 0.999 * s.sy + 0.001 * gy * gy;
    const c1 = 1 - 0.9 ** s.k;
    const c2 = 1 - 0.999 ** s.k;
    s.x -= (lr * (s.vx / c1)) / (Math.sqrt(s.sx / c2) + 1e-8);
    s.y -= (lr * (s.vy / c1)) / (Math.sqrt(s.sy / c2) + 1e-8);
  }
  const span = Math.max(t.xr[1] - t.xr[0], t.yr[1] - t.yr[0]);
  if (!Number.isFinite(s.x) || !Number.isFinite(s.y) || Math.abs(s.x) > span * 3 || Math.abs(s.y) > span * 3) s.out = true;
  s.path = [...r.path, [s.x, s.y]];
  const fmin = t.f(t.min[0], t.min[1]);
  if (s.doneAt === null && !s.out && t.f(s.x, s.y) - fmin < 0.02) s.doneAt = n;
  return s;
}

export default function OptimizerRace3D({ expanded }: VizProps) {
  const c = useVizColors();
  const [tid, setTid] = useState<TerrainId>('ravine');
  const T = TERRAINS[tid];
  const [lrScale, setLrScale] = useState(1);
  const [noise, setNoise] = useState(false);
  const [show, setShow] = useState<Record<OptId, boolean>>({ sgd: true, mom: true, rms: true, adam: true });
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [stepN, setStepN] = useState(0);
  const [topView, setTopView] = useState(false);
  const [start, setStart] = useState<[number, number]>(T.start);
  const [runners, setRunners] = useState<Record<OptId, Runner>>(() => ({ sgd: fresh(T.start), mom: fresh(T.start), rms: fresh(T.start), adam: fresh(T.start) }));
  const randRef = useRef(rng(11));

  const mapper = useMemo(() => makeMapper(T.xr, T.yr, 8, T.hScale), [T]);
  const hFn = T.h;
  const fmin = T.f(T.min[0], T.min[1]);
  const levels = useMemo(() => {
    const hMin = T.h(T.min[0], T.min[1]);
    return [0.15, 0.4, 0.8, 1.4, 2.2, 3.2, 4.5, 6].map((d) => hMin + d * (tid === 'banana' ? 0.6 : 1));
  }, [T, tid]);

  const reset = (s: [number, number] = start) => {
    setPlaying(false);
    setStepN(0);
    randRef.current = rng(11);
    setRunners({ sgd: fresh(s), mom: fresh(s), rms: fresh(s), adam: fresh(s) });
  };

  useEffect(() => {
    setStart(T.start);
    reset(T.start);
  }, [tid]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => reset(), [lrScale, noise]); // eslint-disable-line react-hooks/exhaustive-deps

  const allDone = OPTS.every((o) => !show[o.id] || runners[o.id].doneAt !== null || runners[o.id].out);
  const step = () => {
    const n = stepN + 1;
    const rand = () => randRef.current.normal();
    setRunners((rs) => {
      const next = { ...rs };
      for (const o of OPTS) next[o.id] = stepRunner(rs[o.id], o.id, T, T.lr[o.id] * lrScale, noise ? T.noise : 0, rand, n);
      return next;
    });
    setStepN(n);
  };
  useInterval(
    () => {
      if (allDone || stepN >= 300) setPlaying(false);
      else step();
    },
    playing ? 260 / speed : null
  );

  const [box, { width }] = useElementSize<HTMLDivElement>();
  const stageH = expanded ? 'min(66vh, 740px)' : Math.round(Math.max(330, Math.min(500, width * 0.64)));
  const colors: Record<OptId, string> = { sgd: c.series[2], mom: c.series[0], rms: c.series[3], adam: c.series[1] };
  const clampU = (u: number) => Math.min(T.xr[1], Math.max(T.xr[0], u));
  const clampV = (v: number) => Math.min(T.yr[1], Math.max(T.yr[0], v));
  const lift = 0.24 / T.hScale;
  const toScene = (x: number, y: number, up = lift) => {
    const cx = clampU(x);
    const cy = clampV(y);
    return mapper.toScene(cx, cy, hFn(cx, cy) + up);
  };

  const ranking = OPTS.filter((o) => show[o.id])
    .map((o) => ({ ...o, r: runners[o.id], loss: T.f(runners[o.id].x, runners[o.id].y) - fmin }))
    .sort((a, b) => {
      const da = a.r.doneAt ?? (a.r.out ? 1e9 : 1e6 + a.loss);
      const db = b.r.doneAt ?? (b.r.out ? 1e9 : 1e6 + b.loss);
      return da - db;
    });

  const winner = ranking.find((r) => r.r.doneAt !== null);
  let note: { text: string; mood: 'smile' | 'happy' | 'think' | 'cheer' | 'wow'; tone?: 'good' | 'warn' };
  if (stepN === 0) note = { text: T.story, mood: 'smile' };
  else if (allDone && winner) note = { text: `🏁 ${winner.name}가 ${winner.r.doneAt}걸음 만에 1등! ${noise ? '노이즈(미니배치)가 있어도 ' : ''}지형에 따라 강자가 달라져요. 다른 지형도 골라 보세요.`, mood: 'cheer', tone: 'good' };
  else if (OPTS.some((o) => show[o.id] && runners[o.id].out)) note = { text: '앗, 누군가 지형 밖으로 튕겨 나갔어요(발산). 학습률 배율을 줄여 보세요.', mood: 'wow', tone: 'warn' };
  else note = { text: '같은 출발점, 같은 지형에서 걸음걸이만 달라요. 공의 궤적 모양을 비교해 보세요.', mood: 'happy' };

  return (
    <div ref={box}>
      <Scene3D height={stageH} camera={[8.5, 8, 11]} target={[0, 1, 0]} maxDistance={26}>
        <CameraRig top={topView} />
        <Surface fn={hFn} mapper={mapper} onPick={(u, v) => { setStart([u, v]); reset([u, v]); }} />
        <Contours fn={hFn} mapper={mapper} levels={levels} color={c.theme === 'dark' ? '#7fe9ff' : '#7c6cf6'} floorY={0.01} opacity={0.4} />
        <DomainAxes mapper={mapper} uLabel="파라미터 1" vLabel="파라미터 2" color={c.theme === 'dark' ? '#b4c3ff' : '#5b5570'} />
        <group>
          <mesh position={mapper.toScene(T.min[0], T.min[1], hFn(T.min[0], T.min[1]) + 0.02)} rotation={[-Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.18, 0.3, 32]} />
            <meshBasicMaterial color={c.pos} side={THREE.DoubleSide} />
          </mesh>
          <Label3D position={mapper.toScene(T.min[0], T.min[1], hFn(T.min[0], T.min[1]) + 1.6 / T.hScale)}>🏁 목표</Label3D>
        </group>
        {OPTS.filter((o) => show[o.id]).map((o) => {
          const r = runners[o.id];
          const pts = r.path.map(([x, y]) => toScene(x, y, lift * 0.6));
          return (
            <group key={o.id}>
              {pts.length > 1 && <Line points={pts} color={colors[o.id]} lineWidth={3} />}
              <RollingBall target={toScene(r.x, r.y)} color={colors[o.id]} />
            </group>
          );
        })}
      </Scene3D>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <PlayBar
          playing={playing}
          onToggle={() => {
            if (allDone || stepN >= 300) reset();
            setPlaying((p) => !p);
          }}
          onStep={allDone ? undefined : step}
          onReset={() => reset()}
          speed={speed}
          onSpeed={setSpeed}
        />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8 }}>
          {OPTS.map((o) => {
            const r = runners[o.id];
            const rank = ranking.findIndex((x) => x.id === o.id);
            return (
              <button
                key={o.id}
                type="button"
                onClick={() => setShow((s) => ({ ...s, [o.id]: !s[o.id] }))}
                aria-pressed={show[o.id]}
                style={{
                  textAlign: 'left',
                  padding: '10px 12px',
                  borderRadius: 14,
                  border: `2px solid ${show[o.id] ? colors[o.id] : 'var(--border)'}`,
                  background: show[o.id] ? withAlpha(colors[o.id], 0.1) : 'var(--surface-solid)',
                  opacity: show[o.id] ? 1 : 0.55
                }}
              >
                <span style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 850, fontSize: 14.5 }}>
                  <span style={{ width: 12, height: 12, borderRadius: 99, background: colors[o.id] }} />
                  {o.name}
                  {show[o.id] && stepN > 0 && rank >= 0 && <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-3)' }}>{rank + 1}위</span>}
                </span>
                <span style={{ display: 'block', fontSize: 12, color: 'var(--text-2)', marginTop: 2 }}>{o.desc}</span>
                <span style={{ display: 'block', fontSize: 12.5, fontWeight: 800, marginTop: 4, color: r.out ? c.neg : r.doneAt !== null ? c.pos : 'var(--text)' }}>
                  {r.out ? '💥 튕겨 나감' : r.doneAt !== null ? `🏁 ${r.doneAt}걸음` : `손실 ${(T.f(r.x, r.y) - fmin).toFixed(3)}`}
                </span>
              </button>
            );
          })}
        </div>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        <Segmented
          label="지형"
          options={(Object.keys(TERRAINS) as TerrainId[]).map((k) => ({ value: k, label: TERRAINS[k].label }))}
          value={tid}
          onChange={(v) => setTid(v as TerrainId)}
        />
        <Slider label="학습률 배율" value={lrScale} min={0.25} max={2} step={0.05} onChange={setLrScale} format={(v) => `×${v.toFixed(2)}`} />
        <Toggle label="미니배치 노이즈" checked={noise} onChange={setNoise} />
        <Toggle label="위에서 보기" checked={topView} onChange={setTopView} />
      </Controls>
    </div>
  );
}

function RollingBall({ target, color }: { target: [number, number, number]; color: string }) {
  const ref = useRef<THREE.Group>(null);
  const goal = useRef(new THREE.Vector3(...target));
  goal.current.set(...target);
  useFrame((_, dt) => {
    const g = ref.current;
    if (g) g.position.lerp(goal.current, 1 - Math.exp(-dt * 12));
  });
  return (
    <group ref={ref} position={target}>
      <Ball position={[0, 0, 0]} color={color} radius={0.2} />
    </group>
  );
}

function CameraRig({ top }: { top: boolean }) {
  const { camera, controls } = useThree() as unknown as { camera: THREE.Camera; controls: { target: THREE.Vector3; update: () => void } | null };
  const anim = useRef<{ t: number; from: THREE.Vector3; to: THREE.Vector3 } | null>(null);
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    anim.current = { t: 0, from: camera.position.clone(), to: top ? new THREE.Vector3(0.001, 15, 0.4) : new THREE.Vector3(8.5, 8, 11) };
  }, [top, camera]);
  useFrame((_, dt) => {
    const a = anim.current;
    if (!a) return;
    a.t = Math.min(1, a.t + dt * 1.6);
    const e = 1 - (1 - a.t) ** 3;
    camera.position.lerpVectors(a.from, a.to, e);
    if (controls) {
      controls.target.set(0, top ? 0 : 1, 0);
      controls.update();
    }
    if (a.t >= 1) anim.current = null;
  });
  return null;
}
