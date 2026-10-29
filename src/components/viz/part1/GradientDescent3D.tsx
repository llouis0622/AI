import { useEffect, useMemo, useRef, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { VizProps } from '../Viz';
import { Scene3D, Surface, Contours, Ball, Arrow3D, Label3D, DomainAxes, Line, makeMapper } from '../core/three';
import { Controls, Segmented, Slider, Stat, Note, PlayBar, Toggle, Button } from '../core/ui';
import { MiniChart, Plot, linePath } from '../core/plot';
import { useElementSize, useInterval } from '../core/hooks';
import { useVizColors } from '../core/colors';
import { fmt } from '../core/math';

/* House data from chapter 2, with size normalised to x' = (평 − 20) / 10 ∈ [−1, 1]. */
const SIZES = [10, 15, 20, 25, 30];
const PRICES = [3.2, 4.1, 5.5, 6.1, 7.4];
const XN = SIZES.map((s) => (s - 20) / 10);

type Shape = 'bowl' | 'valley';
const SCALE: Record<Shape, number> = { bowl: 1, valley: 2.6 };

function lossFn(s: number) {
  return (w: number, b: number) => {
    let acc = 0;
    for (let i = 0; i < XN.length; i++) {
      const r = w * s * XN[i] + b - PRICES[i];
      acc += r * r;
    }
    return acc / XN.length;
  };
}

function grad(s: number, w: number, b: number): [number, number] {
  let gw = 0;
  let gb = 0;
  for (let i = 0; i < XN.length; i++) {
    const r = w * s * XN[i] + b - PRICES[i];
    gw += 2 * r * s * XN[i];
    gb += 2 * r;
  }
  return [gw / XN.length, gb / XN.length];
}

const B_STAR = PRICES.reduce((a, v) => a + v, 0) / PRICES.length; // 5.26
const W_STAR_1 = XN.reduce((a, x, i) => a + x * PRICES[i], 0) / XN.reduce((a, x) => a + x * x, 0); // 2.08

const LR_PRESETS = [
  { value: 0.03, label: '아주 작게' },
  { value: 0.35, label: '적당히' },
  { value: 0.9, label: '크게' },
  { value: 1.08, label: '너무 크게' }
];

const START: [number, number] = [-1.4, 8.7];

export default function GradientDescent3D({ expanded }: VizProps) {
  const c = useVizColors();
  const [shape, setShape] = useState<Shape>('bowl');
  const s = SCALE[shape];
  const wStar = W_STAR_1 / s;
  const f = useMemo(() => lossFn(s), [s]);
  const mapper = useMemo(() => makeMapper([wStar - 4, wStar + 4], [B_STAR - 4, B_STAR + 4], 8, shape === 'bowl' ? 0.17 : 0.15), [wStar, shape]);
  const maxH = shape === 'bowl' ? 26 : 30;

  const [lr, setLr] = useState(0.35);
  const [path, setPath] = useState<[number, number][]>([START]);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [topView, setTopView] = useState(false);
  const [showMin, setShowMin] = useState(false);

  const [w, b] = path[path.length - 1];
  const L = f(w, b);
  const [gw, gb] = grad(s, w, b);
  const gnorm = Math.hypot(gw, gb);
  const diverged = !Number.isFinite(L) || L > 400 || Math.abs(w - wStar) > 40 || Math.abs(b - B_STAR) > 40;
  const converged = !diverged && gnorm < 2e-3;
  const losses = path.map(([pw, pb]) => f(pw, pb));
  const zigzag = useMemo(() => {
    if (path.length < 5) return false;
    let flips = 0;
    for (let i = path.length - 4; i < path.length - 1; i++) {
      const d1 = path[i][1] - path[i - 1][1];
      const d2 = path[i + 1][1] - path[i][1];
      if (d1 * d2 < 0) flips++;
    }
    return flips >= 2;
  }, [path]);

  const step = () =>
    setPath((p) => {
      const [pw, pb] = p[p.length - 1];
      const [dw, db] = grad(s, pw, pb);
      return [...p, [pw - lr * dw, pb - lr * db]];
    });

  useInterval(
    () => {
      if (diverged || converged || path.length > 400) setPlaying(false);
      else step();
    },
    playing ? 450 / speed : null
  );

  const reset = (start: [number, number] = START) => {
    setPlaying(false);
    setPath([start]);
  };

  // Changing the landscape restarts from the same corner.
  useEffect(() => reset(), [shape]); // eslint-disable-line react-hooks/exhaustive-deps

  const [box, { width }] = useElementSize<HTMLDivElement>();
  const stageH = expanded ? 'min(68vh, 760px)' : Math.round(Math.max(320, Math.min(500, width * 0.66)));

  const clampU = (u: number) => Math.min(mapper.uRange[1], Math.max(mapper.uRange[0], u));
  const clampV = (v: number) => Math.min(mapper.vRange[1], Math.max(mapper.vRange[0], v));
  // lift the ball by its radius (scene units → function units)
  const lift = 0.27 / mapper.hScale;
  const scenePts = path.map(([pw, pb]) => {
    const cw = clampU(pw);
    const cb = clampV(pb);
    return mapper.toScene(cw, cb, Math.min(maxH, f(cw, cb)) + lift);
  });
  const ballPos = scenePts[scenePts.length - 1];
  const floorPts = path.map(([pw, pb]) => {
    const [x, , z] = mapper.toScene(clampU(pw), clampV(pb), 0);
    return [x, 0.02, z] as [number, number, number];
  });

  // gradient arrow: a short step down the slope, following the surface
  const arrow = useMemo(() => {
    if (gnorm < 1e-3 || diverged) return null;
    const len = Math.min(1.6, 0.35 + gnorm * 0.12);
    const k = len / (gnorm || 1);
    const nw = clampU(w - gw * k * 0.5);
    const nb = clampV(b - gb * k * 0.5);
    const up = 0.12 / mapper.hScale;
    const from = mapper.toScene(clampU(w), clampV(b), Math.min(maxH, f(clampU(w), clampV(b))) + up);
    const to = mapper.toScene(nw, nb, Math.min(maxH, f(nw, nb)) + up);
    return { from, to };
  }, [w, b, gw, gb, gnorm, diverged, mapper, f, maxH]); // eslint-disable-line react-hooks/exhaustive-deps

  const Lstar = f(wStar, B_STAR);
  const levels = useMemo(() => [0.3, 1, 2.5, 5, 9, 14, 20].map((d) => Lstar + d), [Lstar]);

  // the fitted line in original units (평 → 억)
  const slope = (w * s) / 10;
  const intercept = b - 2 * w * s;

  let note: { text: string; mood: 'smile' | 'happy' | 'wow' | 'think' | 'cheer'; tone?: 'good' | 'bad' | 'warn' };
  if (diverged) note = { text: '발산했어요! 🚀 보폭(학습률)이 너무 커서 공이 골짜기를 건너뛸 때마다 더 높이 튀어 올랐어요. 학습률을 줄이고 ↺ 처음으로를 눌러 보세요.', mood: 'wow', tone: 'bad' };
  else if (converged) note = { text: `골짜기 바닥 도착! 🎉 ${path.length - 1}걸음 만에 오차가 ${fmt(L, 3)}까지 줄었어요. 이게 바로 "학습"이에요.`, mood: 'cheer', tone: 'good' };
  else if (zigzag) note = { text: '지그재그! 보폭이 커서 골짜기 양쪽 벽을 왔다 갔다 해요. 그래도 조금씩 바닥으로 가고 있어요.', mood: 'think', tone: 'warn' };
  else if (path.length > 60 && lr < 0.05) note = { text: '보폭이 너무 작아서 한참 걸려요… 🐢 학습률을 키우면 더 빨라져요.', mood: 'think' };
  else if (path.length === 1) note = { text: '지형 위 아무 곳이나 클릭해서 공을 놓거나, ▶ 시작을 눌러 보세요. 화살표는 "발밑에서 가장 가파른 내리막"이에요.', mood: 'smile' };
  else note = { text: '공은 발밑의 경사(화살표)만 보고, 그 방향으로 학습률만큼 한 걸음씩 내려가요.', mood: 'happy' };

  if (shape === 'valley' && path.length === 1) {
    note = { text: '길쭉한 골짜기에서는 같은 학습률도 위험할 수 있어요. "적당히"로 시작해 보고, 왜 그런지 생각해 보세요 (힌트: 11장).', mood: 'think', tone: 'warn' };
  }

  return (
    <div ref={box}>
      <div style={{ position: 'relative' }}>
        <Scene3D height={stageH} camera={[8.5, 8.2, 11]} target={[0, 1.2, 0]} maxDistance={26}>
          <CameraRig top={topView} />
          <Surface fn={f} mapper={mapper} maxH={maxH} onPick={(u, v) => reset([u, v])} />
          <Contours fn={f} mapper={mapper} levels={levels} color={c.theme === 'dark' ? '#7fe9ff' : '#7c6cf6'} floorY={0.01} opacity={0.45} />
          <DomainAxes mapper={mapper} uLabel="w (기울기)" vLabel="b (절편)" hLabel="오차(손실)" hMax={maxH * 1.04} color={c.theme === 'dark' ? '#b4c3ff' : '#5b5570'} />
          {scenePts.length > 1 && <Line points={scenePts} color={c.series[2]} lineWidth={3.5} />}
          {floorPts.length > 1 && <Line points={floorPts} color={c.series[2]} lineWidth={2} dashed dashSize={0.15} gapSize={0.1} transparent opacity={0.7} />}
          {scenePts.slice(0, -1).map((p, i) => (
            <mesh key={i} position={p}>
              <sphereGeometry args={[0.07, 12, 12]} />
              <meshBasicMaterial color={c.series[2]} />
            </mesh>
          ))}
          <AnimatedBall target={ballPos} color={c.series[0]} />
          {arrow && <Arrow3D from={arrow.from} to={arrow.to} color={c.series[1]} width={4} head={0.3} />}
          {showMin && (
            <group>
              <Line points={[mapper.toScene(wStar, B_STAR, 0), mapper.toScene(wStar, B_STAR, Lstar + 6)]} color={c.pos} lineWidth={2} />
              <Label3D position={mapper.toScene(wStar, B_STAR, Lstar + 7)}>🏁 가장 낮은 곳</Label3D>
            </group>
          )}
        </Scene3D>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <PlayBar
          playing={playing}
          onToggle={() => {
            if (diverged || converged) reset();
            setPlaying((p) => !p);
          }}
          onStep={diverged || converged ? undefined : step}
          onReset={() => reset()}
          speed={speed}
          onSpeed={setSpeed}
        />
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
        <div className="vz-stats">
          <Stat label="걸음 수" value={path.length - 1} />
          <Stat label="오차 (손실)" value={diverged ? '∞' : fmt(L, 3)} tone={diverged ? 'bad' : converged ? 'good' : 'brand'} />
          <Stat label="w (기울기)" value={fmt(w, 2)} />
          <Stat label="b (절편)" value={fmt(b, 2)} />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: 12 }}>
          <div>
            <div className="vz-panel-title">지금 공의 위치 = 이 직선이에요</div>
            <Plot xDomain={[5, 35]} yDomain={[0, 10]} height={170} xLabel="크기 (평)" yLabel="가격 (억)" margin={{ top: 10, right: 12, bottom: 38, left: 40 }} xTicks={4} yTicks={4}>
              {({ x, y, clip }) => (
                <g clipPath={clip}>
                  {SIZES.map((sz, i) => {
                    const pred = slope * sz + intercept;
                    return (
                      <line key={`r${i}`} x1={x(sz)} x2={x(sz)} y1={y(PRICES[i])} y2={y(Math.max(-5, Math.min(15, pred)))} stroke={c.neg} strokeWidth={1.5} strokeDasharray="3 3" />
                    );
                  })}
                  <path d={linePath([[x(5), y(slope * 5 + intercept)], [x(35), y(slope * 35 + intercept)]])} stroke={c.series[0]} strokeWidth={3} fill="none" />
                  {SIZES.map((sz, i) => (
                    <circle key={sz} cx={x(sz)} cy={y(PRICES[i])} r={5} fill={c.series[1]} stroke={c.surface} strokeWidth={2} />
                  ))}
                </g>
              )}
            </Plot>
          </div>
          <div>
            <div className="vz-panel-title">걸음마다 오차가 어떻게 변했나</div>
            <MiniChart series={[losses.map((v) => Math.min(v, 60))]} colors={[c.series[2]]} height={170} xLabel="걸음" />
          </div>
        </div>
      </div>

      <Controls>
        <Segmented label="학습률(보폭)" options={LR_PRESETS} value={LR_PRESETS.find((p) => Math.abs(p.value - lr) < 1e-9)?.value ?? -1} onChange={(v) => setLr(v)} />
        <Slider label="학습률 직접 조절" value={lr} min={0.01} max={1.2} step={0.01} onChange={setLr} format={(v) => v.toFixed(2)} />
        <Segmented
          label="지형 모양"
          options={[
            { value: 'bowl', label: '둥근 그릇' },
            { value: 'valley', label: '길쭉한 골짜기' }
          ]}
          value={shape}
          onChange={(v) => setShape(v as Shape)}
        />
        <Toggle label="위에서 보기" checked={topView} onChange={setTopView} />
        <Toggle label="정답 위치 보기" checked={showMin} onChange={setShowMin} />
        <Button size="sm" variant="ghost" onClick={() => reset([wStar + (Math.random() * 2 - 1) * 3.6, B_STAR + (Math.random() * 2 - 1) * 3.6])}>
          🎲 무작위 출발
        </Button>
      </Controls>
    </div>
  );
}

/** Ball eases toward its target each frame so steps look like rolling. */
function AnimatedBall({ target, color }: { target: [number, number, number]; color: string }) {
  const ref = useRef<THREE.Group>(null);
  const goal = useRef(new THREE.Vector3(...target));
  goal.current.set(...target);
  useFrame((_, dt) => {
    const g = ref.current;
    if (!g) return;
    g.position.lerp(goal.current, 1 - Math.exp(-dt * 10));
  });
  return (
    <group ref={ref} position={target}>
      <Ball position={[0, 0, 0]} color={color} radius={0.24} />
    </group>
  );
}

/**
 * Smoothly fly the camera between the starting view and a top-down view. The
 * starting view is whatever the stage set up (Scene3D fits it to the stage), so
 * the first render is left untouched.
 */
function CameraRig({ top }: { top: boolean }) {
  const { camera, controls } = useThree() as unknown as { camera: THREE.Camera; controls: { target: THREE.Vector3; update: () => void } | null };
  const anim = useRef<{ t: number; from: THREE.Vector3; to: THREE.Vector3 } | null>(null);
  const home = useRef<THREE.Vector3 | null>(null);
  useEffect(() => {
    if (!home.current) {
      home.current = camera.position.clone();
      if (!top) return;
    }
    const dist = home.current.distanceTo(new THREE.Vector3(0, 1.2, 0));
    anim.current = {
      t: 0,
      from: camera.position.clone(),
      to: top ? new THREE.Vector3(0.001, dist * 1.02, 0.4) : home.current.clone()
    };
  }, [top, camera]);
  useFrame((_, dt) => {
    const a = anim.current;
    if (!a) return;
    a.t = Math.min(1, a.t + dt * 1.6);
    const e = 1 - (1 - a.t) ** 3;
    camera.position.lerpVectors(a.from, a.to, e);
    if (controls) {
      controls.target.set(0, top ? 0 : 1.2, 0);
      controls.update();
    }
    if (a.t >= 1) anim.current = null;
  });
  return null;
}
