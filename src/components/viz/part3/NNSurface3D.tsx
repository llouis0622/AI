import { useEffect, useMemo, useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { VizProps } from '../Viz';
import { Scene3D, Label3D, Line } from '../core/three';
import { Controls, Note, PlayBar, Segmented, Stat, Toggle } from '../core/ui';
import { useInterval } from '../core/hooks';
import { useVizColors } from '../core/colors';
import { contourSegments, fmt, pct } from '../core/math';
import { createNet, evaluate, forward, makeDataset, trainBatch, type DatasetId, type Net } from './_nn';

/* Input plane [-D, D]² maps onto a SIZE×SIZE floor; probability p ∈ [0, 1] becomes height p·H. */
const D = 1.2;
const SIZE = 6;
const H = 3;
const SEG = 48;
const toScene = (u: number, v: number, p: number): [number, number, number] => [(u / D) * (SIZE / 2), p * H, (-v / D) * (SIZE / 2)];

const DATASETS: { value: DatasetId; label: string }[] = [
  { value: 'xor', label: 'XOR' },
  { value: 'circle', label: '동그라미' },
  { value: 'spiral', label: '나선' }
];

/**
 * The network's answer as a landscape: height = "class 1" probability over the
 * input plane. Training lifts the terrain under one class and presses it down
 * under the other; the 0.5 water line is the decision boundary.
 */
export default function NNSurface3D({ expanded }: VizProps) {
  const c = useVizColors();
  const [dataset, setDataset] = useState<DatasetId>('xor');
  const [hidden, setHidden] = useState(4);
  const [playing, setPlaying] = useState(false);
  const [showPlane, setShowPlane] = useState(true);
  const data = useMemo(() => makeDataset(dataset, 120, 0.35, 3), [dataset]);
  const net = useRef<Net>(createNet([2, 4, 1], 'tanh', 5));
  const [version, setVersion] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [stats, setStats] = useState({ loss: NaN, acc: 0 });

  const reset = (seed = 5) => {
    net.current = createNet(hidden > 4 ? [2, hidden, hidden, 1] : [2, hidden, 1], 'tanh', seed);
    setEpoch(0);
    setStats(evaluate(net.current, data.X, data.Y));
    setVersion((v) => v + 1);
    setPlaying(false);
  };
  useEffect(() => reset(), [dataset, hidden]); // eslint-disable-line react-hooks/exhaustive-deps

  const train = (epochs: number) => {
    const n = data.X.length;
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let e = 0; e < epochs; e++) {
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      for (let k = 0; k < n; k += 20) trainBatch(net.current, data.X, data.Y, idx.slice(k, k + 20), 0.03);
    }
    setEpoch((v) => v + epochs);
    setStats(evaluate(net.current, data.X, data.Y));
    setVersion((v) => v + 1);
  };

  useInterval(
    () => {
      if (epoch >= 800 || stats.acc >= 0.995) setPlaying(false);
      else train(3);
    },
    playing ? 90 : null
  );

  const class1 = c.series[1];
  const class0 = c.series[3];
  const stageH = expanded ? 'min(66vh, 760px)' : 'clamp(330px, 70cqw, 520px)';

  let note: { text: string; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' };
  if (epoch === 0)
    note = { text: '지금은 신경망이 아무것도 몰라서 지형이 거의 평평해요(모든 곳이 반반 확률). ▶ 시작을 누르면 지형이 움직이기 시작해요!', mood: 'smile' };
  else if (stats.acc >= 0.97)
    note = { text: `완성! 🎉 ${pct(stats.acc)}를 맞혀요. 위쪽 점들 아래로는 지형이 솟고, 아래쪽 점들 위로는 꺼졌죠? 물에 잠긴 선(0.5)이 바로 경계예요.`, mood: 'cheer', tone: 'good' };
  else if (epoch > 300 && hidden === 2 && dataset !== 'xor')
    note = { text: '뉴런 2개로는 이 모양을 감싸기가 어려워요. 뉴런 수를 늘리면 지형이 더 구불구불해질 수 있어요!', mood: 'think', tone: 'warn' };
  else note = { text: '학습할수록 지형이 "1번 무리" 아래에서는 솟아오르고 "0번 무리" 아래에서는 내려가요. 경사하강법이 지형을 빚는 중이에요.', mood: 'happy' };

  return (
    <div>
      <Scene3D height={stageH} camera={[7.2, 6.4, 8.4]} target={[0, 1.2, 0]} minDistance={5} maxDistance={20}>
        <gridHelper args={[SIZE, 12, c.theme === 'dark' ? '#4a5590' : '#d9cdbd', c.theme === 'dark' ? '#27305e' : '#efe5d8']} position={[0, 0.001, 0]} />
        <LiveSurface net={net} version={version} class0={class0} class1={class1} dark={c.theme === 'dark'} />
        {showPlane && <WaterLine net={net} version={version} color={c.theme === 'dark' ? '#e9f6ff' : '#2a2540'} />}
        {data.X.map(([u, v], k) => (
          <mesh key={k} position={toScene(u, v, data.Y[k] === 1 ? 1 : 0)}>
            <sphereGeometry args={[0.075, 12, 12]} />
            <meshStandardMaterial color={data.Y[k] === 1 ? class1 : class0} emissive={data.Y[k] === 1 ? class1 : class0} emissiveIntensity={0.35} />
          </mesh>
        ))}
        <Line points={[toScene(-D, -D, 0), toScene(-D, -D, 1)]} color={c.theme === 'dark' ? '#b4c3ff' : '#5b5570'} lineWidth={2} />
        <Label3D position={toScene(-D, -D, 1.12)}>확률 1 (위쪽 무리)</Label3D>
        <Label3D position={toScene(D * 1.05, -D * 1.15, 0)}>입력 x₁</Label3D>
        <Label3D position={toScene(-D * 1.15, D * 1.05, 0)}>입력 x₂</Label3D>
      </Scene3D>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <PlayBar
          playing={playing}
          onToggle={() => {
            if (stats.acc >= 0.995 || epoch >= 800) reset(Math.floor(Math.random() * 1e6));
            setPlaying((p) => !p);
          }}
          onStep={() => train(10)}
          onReset={() => reset(Math.floor(Math.random() * 1e6))}
        />
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
        <div className="vz-stats">
          <Stat label="학습 횟수(에포크)" value={epoch} />
          <Stat label="손실" value={Number.isFinite(stats.loss) ? fmt(stats.loss, 3) : '—'} tone="brand" />
          <Stat label="정확도" value={pct(stats.acc)} tone={stats.acc >= 0.97 ? 'good' : undefined} />
        </div>
      </div>

      <Controls>
        <Segmented label="데이터 모양" options={DATASETS} value={dataset} onChange={(v) => setDataset(v as DatasetId)} />
        <Segmented
          label="숨은 뉴런 수"
          options={[
            { value: 2, label: '2개' },
            { value: 4, label: '4개' },
            { value: 8, label: '8개 × 2층' }
          ]}
          value={hidden}
          onChange={(v) => setHidden(Number(v))}
        />
        <Toggle label="0.5 물높이(경계) 보기" checked={showPlane} onChange={setShowPlane} />
      </Controls>
    </div>
  );
}

/** Height field rebuilt in place whenever the network changes (no new geometry per frame). */
function LiveSurface({
  net,
  version,
  class0,
  class1,
  dark
}: {
  net: React.MutableRefObject<Net>;
  version: number;
  class0: string;
  class1: string;
  dark: boolean;
}) {
  const geom = useMemo(() => {
    const g = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
    g.rotateX(-Math.PI / 2);
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(g.attributes.position.count * 3), 3));
    return g;
  }, []);
  const seen = useRef(-1);
  const c0 = useMemo(() => new THREE.Color(class0), [class0]);
  const c1 = useMemo(() => new THREE.Color(class1), [class1]);

  useFrame(() => {
    if (seen.current === version) return;
    seen.current = version;
    const pos = geom.attributes.position as THREE.BufferAttribute;
    const col = geom.attributes.color as THREE.BufferAttribute;
    const tmp = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const u = (pos.getX(i) / (SIZE / 2)) * D;
      const v = (-pos.getZ(i) / (SIZE / 2)) * D;
      const p = forward(net.current, u, v);
      pos.setY(i, p * H);
      tmp.copy(c0).lerp(c1, p);
      col.setXYZ(i, tmp.r, tmp.g, tmp.b);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    geom.computeVertexNormals();
  });

  useEffect(() => {
    seen.current = -1;
  }, [c0, c1]);

  return (
    <mesh geometry={geom}>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        transparent
        opacity={0.86}
        roughness={dark ? 0.35 : 0.6}
        metalness={dark ? 0.2 : 0.05}
        emissive={dark ? '#120a2c' : '#000000'}
        emissiveIntensity={dark ? 0.6 : 0}
      />
    </mesh>
  );
}

/** Translucent plane at p = 0.5 plus the boundary curve where the terrain crosses it. */
function WaterLine({ net, version, color }: { net: React.MutableRefObject<Net>; version: number; color: string }) {
  const segs = useMemo(() => {
    const raw = contourSegments((u, v) => forward(net.current, u, v), [-D, D], [-D, D], 0.5, 44);
    const pts: number[] = [];
    for (let i = 0; i < raw.length; i += 2) {
      const [x, , z] = toScene(raw[i], raw[i + 1], 0.5);
      pts.push(x, 0.5 * H + 0.015, z);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    return g;
  }, [version]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <group>
      <mesh position={[0, 0.5 * H, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[SIZE, SIZE]} />
        <meshBasicMaterial color="#6fb8ff" transparent opacity={0.16} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <lineSegments geometry={segs}>
        <lineBasicMaterial color={color} />
      </lineSegments>
    </group>
  );
}
