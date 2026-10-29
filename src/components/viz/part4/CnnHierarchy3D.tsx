import { useEffect, useMemo, useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { VizProps } from '../Viz';
import { Scene3D, Label3D, Line } from '../core/three';
import { Button, Controls, Note, Segmented } from '../core/ui';
import { useElementSize, useRafLoop, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { clamp, rng } from '../core/math';
import { conv3, drawCat, hexToRgb, maxPool, paintMap, toGray } from './_draw';

/* The exact network from chapter 13's code: 32×32×3 → conv32 → pool → conv64 → pool → conv128 → GAP → 10 */
interface LayerDef {
  id: string;
  name: string;
  short: string;
  spatial: number;
  ch: number;
  what: string;
  rf: number;
  params: number;
}
const LAYERS: LayerDef[] = [
  { id: 'input', name: '입력 사진', short: '입력', spatial: 32, ch: 3, what: '픽셀 그대로예요. 빨강·초록·파랑 세 장이 겹쳐 있어요.', rf: 1, params: 0 },
  { id: 'conv1', name: '합성곱 1', short: '합성곱 1', spatial: 32, ch: 32, what: '3×3 픽셀만 보니까 방향별 모서리·색 대비·점 같은 원초적인 무늬를 찾아요.', rf: 3, params: 3 * 3 * 3 * 32 + 32 },
  { id: 'pool1', name: '맥스 풀링', short: '풀링', spatial: 16, ch: 32, what: '2×2 구역에서 가장 큰 값만 남겨 크기를 절반으로. "이 근처에 무늬가 있었다"만 기억해요.', rf: 4, params: 0 },
  { id: 'conv2', name: '합성곱 2', short: '합성곱 2', spatial: 16, ch: 64, what: '모서리들의 배치를 봐요: 모퉁이, 곡선, 줄무늬 같은 조합이에요.', rf: 8, params: 3 * 3 * 32 * 64 + 64 },
  { id: 'pool2', name: '맥스 풀링', short: '풀링', spatial: 8, ch: 64, what: '한 번 더 절반으로 요약해요. 한 칸이 원래 사진의 더 넓은 곳을 대표해요.', rf: 10, params: 0 },
  { id: 'conv3', name: '합성곱 3', short: '합성곱 3', spatial: 8, ch: 128, what: '이제 부품을 알아봐요: 눈, 귀, 수염처럼 "고양이의 부분"에 반응해요.', rf: 18, params: 3 * 3 * 64 * 128 + 128 },
  { id: 'gap', name: '전역 평균 풀링', short: '평균', spatial: 1, ch: 128, what: '지도마다 평균을 내서 숫자 128개짜리 벡터로 만들어요. "어디에"는 지우고 "무엇이"만 남겨요.', rf: 32, params: 0 },
  { id: 'fc', name: '분류기 (소프트맥스)', short: '출력', spatial: 1, ch: 10, what: '128개 숫자로 10가지 클래스의 확률을 내요. 5장의 그 출구예요!', rf: 32, params: 128 * 10 + 10 }
];
const CLASSES = [
  ['고양이', 0.86],
  ['개', 0.06],
  ['여우', 0.03],
  ['토끼', 0.02],
  ['새', 0.01],
  ['자동차', 0.005],
  ['배', 0.005],
  ['비행기', 0.004],
  ['개구리', 0.003],
  ['말', 0.003]
] as const;

const S = (spatial: number) => (spatial >= 32 ? 2.8 : spatial >= 16 ? 1.75 : 1.15);
const D = (ch: number) => 0.12 + 0.11 * Math.sqrt(ch);

function layout() {
  const gap = 0.6;
  const widths = LAYERS.map((l) => (l.id === 'gap' ? 0.32 : l.id === 'fc' ? 1.6 : S(l.spatial)));
  const total = widths.reduce((a, b) => a + b, 0) + gap * (LAYERS.length - 1);
  let x = -total / 2;
  return widths.map((w) => {
    const cx = x + w / 2;
    x += w + gap;
    return { cx, w };
  });
}
const POS = layout();
// the cat's left eye in input pixel coordinates (0..32)
const EYE: [number, number] = [12, 17];

export default function CnnHierarchy3D({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [sel, setSel] = useState(5);
  const [t, setT] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [box, { width }] = useElementSize<HTMLDivElement>();

  useRafLoop((dt) => setT((v) => Math.min(1, v + dt / 3.4)), playing);
  useEffect(() => {
    if (playing && t >= 1) setPlaying(false);
  }, [playing, t]);

  // feature-map textures computed from a drawn cat (recomputed when the theme flips)
  const maps = useMemo(() => {
    const cv = document.createElement('canvas');
    cv.width = 32;
    cv.height = 32;
    const ctx = cv.getContext('2d')!;
    drawCat(ctx, 32);
    const gray = toGray(ctx, 32);
    const big = document.createElement('canvas');
    big.width = 128;
    big.height = 128;
    drawCat(big.getContext('2d')!, 128);
    const edgeV = conv3(gray, 32, [-1, 0, 1, -2, 0, 2, -1, 0, 1]);
    const edgeH = conv3(gray, 32, [-1, -2, -1, 0, 0, 0, 1, 2, 1]);
    const conv1 = edgeV.map((v, i) => Math.hypot(v, edgeH[i]));
    const pool1 = maxPool(conv1, 32);
    const conv2 = conv3(pool1, 16, [1, 2, 1, 2, 4, 2, 1, 2, 1]).map((v) => Math.max(0, v - 1.2));
    const pool2 = maxPool(conv2, 16);
    const conv3m = new Float32Array(64);
    const blobs: [number, number, number][] = [
      [3, 4.2, 1],
      [5, 4.2, 1],
      [2.3, 1.6, 0.8],
      [5.7, 1.6, 0.8],
      [4, 5.5, 0.6]
    ];
    for (let y = 0; y < 8; y++)
      for (let x = 0; x < 8; x++) conv3m[y * 8 + x] = blobs.reduce((s, [bx, by, a]) => s + a * Math.exp(-((x + 0.5 - bx) ** 2 + (y + 0.5 - by) ** 2) / 0.9), 0);
    const r = rng(4);
    const gap = new Float32Array(128).map(() => r.next() ** 2);
    const base = hexToRgb(c.surface.startsWith('#') ? c.surface : '#ffffff');
    const tone = (i: number) => hexToRgb(c.series[i]);
    return {
      input: big,
      conv1: paintMap(conv1, 32, base, tone(0), 4),
      pool1: paintMap(pool1, 16, base, tone(0), 8),
      conv2: paintMap(conv2, 16, base, tone(3), 8),
      pool2: paintMap(pool2, 8, base, tone(3), 16),
      conv3: paintMap(conv3m, 8, base, tone(2), 16),
      gap: paintMap(gap, 128, base, tone(1), 1)
    } as Record<string, HTMLCanvasElement>;
  }, [c.surface, c.series]);

  const active = playing ? Math.min(LAYERS.length - 1, Math.floor(t * LAYERS.length)) : -1;
  const layer = LAYERS[sel];
  const stageH = expanded ? 'min(62vh, 680px)' : Math.round(Math.max(340, Math.min(460, width * 0.6)));
  // keep the whole stack in view: the narrower (taller) the stage, the farther the camera
  const aspect = typeof stageH === 'number' && width > 0 ? width / stageH : 1.7;
  const dist = clamp(26.5 / aspect, 15, 32);
  const camPos: [number, number, number] = [0.4, -0.1 + dist * 0.32, -0.6 + dist * 0.95];
  const thumb = useMemo(() => {
    const cv = maps[layer.id];
    return cv ? cv.toDataURL() : null;
  }, [maps, layer.id]);

  let note: { text: string; mood: 'smile' | 'happy' | 'wow' | 'cheer'; tone?: 'good' };
  if (playing) note = { text: `${LAYERS[Math.max(0, active)].name}: ${LAYERS[Math.max(0, active)].what}`, mood: 'wow' };
  else if (t >= 1 && sel === LAYERS.length - 1) note = { text: '마지막엔 "고양이 86%"! 모서리 → 무늬 → 부품 → 고양이로, 층마다 점점 추상적인 표현이 자라났어요. 🎉', mood: 'cheer', tone: 'good' };
  else note = { text: '판을 눌러 층마다 무엇을 보는지 확인해 보세요. 오른쪽으로 갈수록 판은 작아지고(공간↓) 두꺼워져요(채널↑). 판 위 무늬는 이해를 돕는 예시 지도예요.', mood: 'smile' };

  return (
    <div ref={box}>
      {width > 0 ? (
        <Scene3D key={aspect < 1.4 ? 'tall' : 'wide'} height={stageH} camera={camPos} target={[0, -0.1, -0.6]} minDistance={5} maxDistance={36} enablePan>
          {LAYERS.map((l, i) => (
            <Slab key={l.id} index={i} def={l} map={maps[l.id]} selected={sel === i} lit={active === i} onSelect={() => setSel(i)} c={c} />
          ))}
          <ReceptiveField sel={sel} c={c} />
          {playing && <Pulse t={t} color={c.series[2]} />}
        </Scene3D>
      ) : (
        <div style={{ height: stageH }} />
      )}

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: thumb ? 'auto minmax(0, 1fr)' : '1fr', gap: 14, alignItems: 'center', padding: 12, borderRadius: 16, background: 'var(--surface-solid)', border: '1px solid var(--border)' }}>
          {thumb && layer.id !== 'fc' && (
            <img src={thumb} alt={`${layer.name}의 특징 맵 한 장`} width={84} height={84} style={{ width: 84, height: layer.id === 'gap' ? 22 : 84, borderRadius: 10, imageRendering: layer.spatial <= 16 ? 'pixelated' : 'auto', border: '1px solid var(--border)' }} />
          )}
          {layer.id === 'fc' && (
            <div style={{ display: 'grid', gap: 3, width: 140 }}>
              {CLASSES.slice(0, 4).map(([name, p]) => (
                <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 750 }}>
                  <span style={{ width: 40 }}>{name}</span>
                  <span style={{ flex: 1, height: 8, borderRadius: 4, background: 'var(--surface-3)', overflow: 'hidden' }}>
                    <span style={{ display: 'block', height: '100%', width: `${p * 100}%`, background: c.series[2] }} />
                  </span>
                </div>
              ))}
            </div>
          )}
          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'baseline' }}>
              <strong style={{ fontSize: 16 }}>{layer.name}</strong>
              <span className="chip" style={{ fontFamily: 'var(--font-mono)' }}>
                {layer.id === 'fc' ? '10' : layer.id === 'gap' ? '128' : `${layer.spatial}×${layer.spatial}×${layer.ch}`}
              </span>
            </div>
            <p style={{ margin: '6px 0 4px', fontSize: 14, color: 'var(--text-2)', lineHeight: 1.65 }}>{layer.what}</p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, fontSize: 12.5, fontWeight: 750, color: 'var(--text-3)' }}>
              <span>👁️ 한 칸이 보는 범위: {layer.rf >= 32 ? '사진 전체' : `${layer.rf}×${layer.rf} 픽셀`}</span>
              <span>🎛️ 학습할 숫자: {layer.params ? layer.params.toLocaleString() : '없음'}</span>
            </div>
          </div>
        </div>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        <Button
          variant="primary"
          onClick={() => {
            if (reduce) {
              setSel(LAYERS.length - 1);
              return;
            }
            setT(0);
            setPlaying(true);
            setSel(LAYERS.length - 1);
          }}
        >
          ▶ 사진 한 장 통과시키기
        </Button>
        <Segmented label="층 고르기" options={LAYERS.map((l, i) => ({ value: i, label: l.short }))} value={sel} onChange={(v) => setSel(Number(v))} />
      </Controls>
    </div>
  );
}

function Slab({
  index,
  def,
  map,
  selected,
  lit,
  onSelect,
  c
}: {
  index: number;
  def: LayerDef;
  map?: HTMLCanvasElement;
  selected: boolean;
  lit: boolean;
  onSelect: () => void;
  c: ReturnType<typeof useVizColors>;
}) {
  const { cx, w } = POS[index];
  const isFc = def.id === 'fc';
  const isGap = def.id === 'gap';
  const h = isFc ? 2.6 : isGap ? 2.8 : S(def.spatial);
  const d = isFc ? 0.3 : isGap ? 0.32 : D(def.ch);
  const tex = useMemo(() => {
    if (!map) return null;
    const t = new THREE.CanvasTexture(map);
    t.colorSpace = THREE.SRGBColorSpace;
    t.magFilter = def.spatial <= 16 ? THREE.NearestFilter : THREE.LinearFilter;
    return t;
  }, [map, def.spatial]);
  useEffect(() => () => tex?.dispose(), [tex]);
  const tint = c.series[index % 2 === 0 ? 0 : 3];
  const dark = c.theme === 'dark';
  const glow = selected || lit;

  if (isFc) {
    return (
      <group position={[cx, 0, -0.2]}>
        {CLASSES.map(([name, p], k) => {
          const bw = 0.15 + 1.4 * Math.sqrt(p);
          const y = 1.15 - k * 0.26;
          return (
            <mesh
              key={name}
              position={[-w / 2 + bw / 2, y, 0]}
              onClick={(e) => {
                if (e.delta > 6) return;
                e.stopPropagation();
                onSelect();
              }}
            >
              <boxGeometry args={[bw, 0.18, 0.3]} />
              <meshStandardMaterial color={k === 0 ? c.series[2] : tint} emissive={k === 0 ? c.series[2] : '#000000'} emissiveIntensity={glow && k === 0 ? 0.6 : dark ? 0.2 : 0} />
            </mesh>
          );
        })}
        <Label3D position={[0.2, 1.6, 0]}>🐱 고양이 86%</Label3D>
        <Label3D position={[0, -1.75, 0]}>{def.short}</Label3D>
      </group>
    );
  }

  return (
    <group position={[cx, 0, -d / 2]}>
      <mesh
        onClick={(e) => {
          if (e.delta > 6) return;
          e.stopPropagation();
          onSelect();
        }}
      >
        <boxGeometry args={[w, h, d]} />
        {[0, 1, 2, 3, 4, 5].map((face) =>
          face === 4 && tex ? (
            <meshStandardMaterial key={face} attach={`material-${face}`} map={tex} emissive={glow ? tint : '#000000'} emissiveIntensity={glow ? 0.35 : 0} roughness={0.7} />
          ) : (
            <meshStandardMaterial
              key={face}
              attach={`material-${face}`}
              color={withAlpha(tint, 1)}
              transparent
              opacity={dark ? 0.75 : 0.85}
              emissive={tint}
              emissiveIntensity={glow ? 0.7 : dark ? 0.25 : 0.05}
              roughness={0.6}
            />
          )
        )}
      </mesh>
      {glow && (
        <lineSegments>
          <edgesGeometry args={[new THREE.BoxGeometry(w + 0.04, h + 0.04, d + 0.04)]} />
          <lineBasicMaterial color={dark ? '#ffffff' : c.ink} />
        </lineSegments>
      )}
      <Label3D position={[0, -h / 2 - 0.35, d / 2]}>{def.short}</Label3D>
    </group>
  );
}

/** Square on the input that one neuron of the selected layer "sees", with rays to that neuron. */
function ReceptiveField({ sel, c }: { sel: number; c: ReturnType<typeof useVizColors> }) {
  if (sel <= 0) return null;
  const def = LAYERS[sel];
  const inW = S(32);
  const inX = POS[0].cx;
  const pxToWorld = (px: number, py: number): [number, number] => [inX - inW / 2 + (px / 32) * inW, inW / 2 - (py / 32) * inW];
  const half = Math.min(32, def.rf) / 2;
  const cxp = def.rf >= 32 ? 16 : EYE[0];
  const cyp = def.rf >= 32 ? 16 : EYE[1];
  const corners = [
    pxToWorld(cxp - half, cyp - half),
    pxToWorld(cxp + half, cyp - half),
    pxToWorld(cxp + half, cyp + half),
    pxToWorld(cxp - half, cyp + half)
  ];
  const z = 0.02;
  // target neuron on the selected layer's front face
  const tgt = POS[sel];
  const tH = def.id === 'gap' || def.id === 'fc' ? 2.6 : S(def.spatial);
  const tw = tgt.w;
  const target: [number, number, number] =
    def.id === 'gap' || def.id === 'fc' ? [tgt.cx, 0, z] : [tgt.cx - tw / 2 + (cxp / 32) * tw, tH / 2 - (cyp / 32) * tH, z];
  const sq: [number, number, number][] = [...corners, corners[0]].map(([x, y]) => [x, y, z] as [number, number, number]);
  const color = c.series[2];
  return (
    <group>
      <Line points={sq} color={color} lineWidth={3.5} />
      {corners.map(([x, y], i) => (
        <Line key={i} points={[[x, y, z], target]} color={color} lineWidth={1.2} transparent opacity={0.65} />
      ))}
      <mesh position={target}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshBasicMaterial color={color} />
      </mesh>
    </group>
  );
}

function Pulse({ t, color }: { t: number; color: string }) {
  const ref = useRef<THREE.Mesh>(null);
  const x0 = POS[0].cx;
  const x1 = POS[POS.length - 1].cx;
  useFrame(() => {
    if (ref.current) ref.current.position.set(x0 + (x1 - x0) * t, 0, 0.35);
  });
  return (
    <mesh ref={ref} position={[x0, 0, 0.35]}>
      <sphereGeometry args={[0.22, 24, 24]} />
      <meshBasicMaterial color={color} transparent opacity={0.85} />
    </mesh>
  );
}
