import { useMemo, useRef, useState, type ReactNode } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { VizProps } from '../Viz';
import { Scene3D, Label3D } from '../core/three';
import { Button, Controls, Icon, Note, PlayBar, Slider, Stat, Toggle } from '../core/ui';
import { useInterval } from '../core/hooks';
import { useVizColors, mixHex } from '../core/colors';
import { fmt, rng } from '../core/math';
import { DEFAULT_ROWS, argmaxQ, chooseAction, envStep, isTerminal, maxQ, parseWorld, qUpdate } from './_gridworld';

/* Room to wander: with 50 steps the first gem often takes ~30 episodes, with 100 about 18. */
const MAX_EP_STEPS = 100;
/* A value of 1 (the gem's reward) stands HMAX tall. */
const HMAX = 1.8;
/* Cone rotations that turn +Y into up / right / down / left on the board (−z, +x, +z, −x). */
const CONE_ROT: [number, number, number][] = [
  [-Math.PI / 2, 0, 0],
  [0, 0, -Math.PI / 2],
  [Math.PI / 2, 0, 0],
  [0, 0, Math.PI / 2]
];

/** Relative luminance of a #rrggbb color (0 = black, 1 = white). */
function luminance(hex: string): number {
  const m = hex.trim().match(/^#([0-9a-f]{6})$/i);
  if (!m) return 0;
  const [r, g, b] = [0, 2, 4].map((i) => {
    const v = parseInt(m[1].slice(i, i + 2), 16) / 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Q-learning seen as a landscape: each free cell is a column whose height is its
 * value V(s) = max Q(s, a). Value starts at the gem and spreads backwards one cell
 * at a time; γ decides how gently the mountain slopes away from it.
 */
export default function ValueTerrain3D({ expanded }: VizProps) {
  const c = useVizColors();
  const dark = c.theme === 'dark';
  const world = useMemo(() => parseWorld(DEFAULT_ROWS), []);
  const Q = useRef(new Float64Array(world.w * world.h * 4));
  // seed 42 finds the gem by episode 9 and lifts the start by ~52: two presses of "50판 한 번에" tell the story
  const rand = useRef(rng(42).next);
  const pos = useRef(world.start);
  const ep = useRef({ steps: 0, reward: 0, pendingReset: false });
  const [, setVersion] = useState(0);
  const [history, setHistory] = useState<number[]>([]);
  const [firstGoalAt, setFirstGoalAt] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [demo, setDemo] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [eps, setEps] = useState(0.2);
  const [gamma, setGamma] = useState(0.9);
  const [arrows, setArrows] = useState(true);

  const bump = () => setVersion((v) => v + 1);

  const toStart = () => {
    pos.current = world.start;
    ep.current = { steps: 0, reward: 0, pendingReset: false };
  };

  const resetLearning = () => {
    Q.current = new Float64Array(world.w * world.h * 4);
    rand.current = rng(Math.floor(Math.random() * 1e6)).next;
    toStart();
    setHistory([]);
    setFirstGoalAt(null);
    setPlaying(false);
    setDemo(false);
    bump();
  };

  /** One environment step; finished episodes push their reward into `records`. */
  const stepOnce = (greedy: boolean, records: number[]): boolean => {
    const e = ep.current;
    if (e.pendingReset) {
      toStart();
      return false;
    }
    const s = pos.current;
    const a = chooseAction(Q.current, s, greedy ? 0 : eps, rand.current);
    const { s2, r, done } = envStep(world, s, a);
    if (!greedy) qUpdate(Q.current, s, a, r, s2, done, 0.5, gamma);
    e.steps++;
    e.reward += r;
    pos.current = s2;
    if (done || e.steps >= MAX_EP_STEPS) {
      records.push(e.reward);
      e.pendingReset = true;
      return true;
    }
    return false;
  };

  const commit = (records: number[]) => {
    if (records.length) {
      if (firstGoalAt === null) {
        const k = records.findIndex((r) => r > 0);
        if (k >= 0) setFirstGoalAt(history.length + k + 1);
      }
      setHistory((h) => [...h, ...records].slice(-600));
    }
    bump();
  };

  const tick = () => {
    if (demo) {
      if (stepOnce(true, [])) setDemo(false);
      bump();
      return;
    }
    const records: number[] = [];
    const n = speed >= 3 ? 12 : 1;
    for (let i = 0; i < n; i++) stepOnce(false, records);
    commit(records);
  };

  useInterval(tick, playing || demo ? (demo ? 300 : speed <= 0.5 ? 320 : speed >= 3 ? 60 : 140) : null);

  const runEpisodes = (k: number) => {
    setPlaying(false);
    setDemo(false);
    const records: number[] = [];
    let guard = 0;
    while (records.length < k && guard < k * (MAX_EP_STEPS + 2)) {
      stepOnce(false, records);
      guard++;
    }
    toStart();
    commit(records);
  };

  const startDemo = () => {
    setPlaying(false);
    toStart();
    setDemo(true);
    bump();
  };

  const V = world.cells.map((cell, s) => (cell === '#' || isTerminal(world, s) ? 0 : maxQ(Q.current, s)));
  const recent = history.slice(-20);
  const success = recent.length ? recent.filter((r) => r > 0).length / recent.length : 0;
  const vStart = V[world.start];
  const stageH = expanded ? 'min(66vh, 760px)' : 'clamp(330px, 70cqw, 520px)';

  const cx = (s: number) => (s % world.w) - (world.w - 1) / 2;
  const cz = (s: number) => Math.floor(s / world.w) - (world.h - 1) / 2;
  const heightOf = (v: number) => Math.max(0.06, Math.min(1.2, Math.abs(v)) * HMAX);
  // neutral board: tiles and walls are greys taken from the theme; only value is colored
  const slab = mixHex(c.panel, c.ink, dark ? 0.05 : 0.08);
  const tile = mixHex(c.panel, c.ink, dark ? 0.13 : 0.03);
  const wall = mixHex(c.panel, c.ink, dark ? 0.3 : 0.5);
  const colorOf = (v: number) =>
    Math.abs(v) < 1e-6 ? tile : mixHex(tile, v > 0 ? c.pos : c.neg, 0.25 + 0.75 * Math.min(1, Math.abs(v)));
  // arrowheads switch between a dark and a light ink so they stay readable on any column color
  const inkDark = dark ? c.bg : c.ink;
  const inkLight = dark ? c.ink : c.bg;
  const coneColorOf = (v: number) => (luminance(colorOf(v)) > 0.18 ? inkDark : inkLight);
  const gemColor = mixHex(c.pos, dark ? c.ink : c.bg, 0.35);
  const boxGeom = useMemo(() => new THREE.BoxGeometry(0.86, 1, 0.86), []);

  let note: { t: ReactNode; tone?: 'good' | 'warn' };
  if (demo) note = { t: '학습한 대로만 움직여요. 주변에서 가장 높은 칸 쪽 화살표를 따라 산을 올라가요.' };
  else if (eps >= 0.8) note = { t: <>모험심(ε)이 너무 크면 배운 길을 거의 쓰지 않아요. 산은 솟아도 로봇은 <strong>엉뚱한 곳</strong>을 헤매요.</>, tone: 'warn' };
  else if (gamma < 0.6 && history.length > 30) note = { t: <>γ가 작으면 산이 <strong>뾰족</strong>해져요. 보석 근처만 높고 조금만 멀어져도 가치가 0에 가까워져서, 출발점에서는 어디로 갈지 잘 몰라요.</> };
  else if (vStart > 0.25) note = { t: <>산이 출발점까지 이어졌어요. 이제 로봇은 <strong>더 높은 칸으로 오르기만</strong> 하면 보석에 닿아요. <strong>학습한 대로 이동</strong>을 눌러 보세요.</>, tone: 'good' };
  else if (firstGoalAt !== null) note = { t: <>{firstGoalAt}번째 판에서 보석을 처음 찾았어요. 보석 바로 옆 칸부터 땅이 솟고, 가치는 보석에서 <strong>한 칸씩 거꾸로</strong> 퍼져 나가요.</>, tone: 'good' };
  else if (history.length > 0 || ep.current.steps > 0) note = { t: '로봇이 이리저리 돌아다니며 배우는 중이에요. 함정에 빠지면 -1, 보석을 찾으면 +1을 받고, 그때마다 바로 앞 칸의 높이가 조금씩 바뀌어요.' };
  else note = { t: <>지금은 땅이 평평해요. 로봇이 아직 어느 칸이 좋은지 몰라서 모든 칸의 가치가 0이에요. <strong>시작</strong>이나 <strong>50판 한 번에</strong>를 눌러 보세요.</> };

  const robotCell = pos.current;
  const robotY = (world.cells[robotCell] === '.' ? heightOf(V[robotCell]) : world.cells[robotCell] === 'G' ? HMAX : 0.04) + 0.26;

  return (
    <div>
      <Scene3D height={stageH} camera={[-4.4, 5.8, 6.8]} target={[0.2, 0.8, -0.2]} minDistance={4.5} maxDistance={20}>
        <mesh position={[0, -0.09, 0]}>
          <boxGeometry args={[world.w + 0.4, 0.12, world.h + 0.4]} />
          <meshStandardMaterial color={slab} roughness={0.8} />
        </mesh>

        {world.cells.map((cell, s) => {
          const x = cx(s);
          const z = cz(s);
          if (cell === '#')
            return (
              <mesh key={s} position={[x, 0.36, z]}>
                <boxGeometry args={[0.9, 0.72, 0.9]} />
                <meshStandardMaterial color={wall} roughness={0.7} />
              </mesh>
            );
          if (cell === 'F')
            return (
              <group key={s}>
                <mesh position={[x, 0.01, z]}>
                  <boxGeometry args={[0.86, 0.06, 0.86]} />
                  <meshStandardMaterial color={c.neg} emissive={c.neg} emissiveIntensity={dark ? 0.2 : 0.08} />
                </mesh>
                <Label3D position={[x, 0.55, z]}>함정 -1</Label3D>
              </group>
            );
          if (cell === 'G')
            return (
              <group key={s}>
                <mesh position={[x, HMAX / 2, z]}>
                  <boxGeometry args={[0.86, HMAX, 0.86]} />
                  <meshStandardMaterial color={c.pos} emissive={c.pos} emissiveIntensity={dark ? 0.15 : 0.06} roughness={0.4} />
                </mesh>
                <Gem position={[x, HMAX + 0.45, z]} color={gemColor} emissive={dark ? 0.2 : 0.08} />
                <Label3D position={[x, HMAX + 1.0, z]}>보석 +1</Label3D>
              </group>
            );
          const v = V[s];
          const known = Math.abs(v) > 1e-3;
          return (
            <Column
              key={s}
              x={x}
              z={z}
              h={heightOf(v)}
              color={colorOf(v)}
              geometry={boxGeom}
              arrow={arrows && known ? argmaxQ(Q.current, s, () => 0) : -1}
              arrowColor={coneColorOf(v)}
            />
          );
        })}

        <Label3D position={[cx(world.start), heightOf(vStart) + 0.75, cz(world.start)]}>출발</Label3D>
        <Robot target={[cx(robotCell), robotY, cz(robotCell)]} color={c.ink} />
      </Scene3D>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <PlayBar
          playing={playing}
          onToggle={() => {
            setDemo(false);
            setPlaying((p) => !p);
          }}
          onStep={() => {
            const r: number[] = [];
            stepOnce(false, r);
            commit(r);
          }}
          onReset={resetLearning}
          speed={speed}
          onSpeed={setSpeed}
          extra={
            <>
              <Button onClick={() => runEpisodes(50)}>
                <Icon name="fast" size={16} />50판 한 번에
              </Button>
              <Button onClick={startDemo} disabled={demo}>
                <Icon name="route" size={16} />학습한 대로 이동
              </Button>
            </>
          }
        />
        <Note tone={note.tone}>{note.t}</Note>
        <div className="vz-stats">
          <Stat label="에피소드(판)" value={history.length} />
          <Stat label="최근 20판 성공률" value={`${Math.round(success * 100)}%`} tone={success > 0.7 ? 'good' : undefined} />
          <Stat label="출발점의 가치" value={fmt(vStart, 2)} tone={vStart > 0.25 ? 'good' : 'brand'} />
        </div>
      </div>

      <Controls>
        <Slider label="할인율 γ (참을성)" value={gamma} min={0.3} max={0.99} step={0.01} onChange={setGamma} format={(v) => v.toFixed(2)} />
        <Slider label="탐색 ε (모험심)" value={eps} min={0} max={1} step={0.05} onChange={setEps} format={(v) => v.toFixed(2)} accent="var(--accent-2)" />
        <Toggle label="배운 방향(화살표) 보기" checked={arrows} onChange={setArrows} />
      </Controls>
    </div>
  );
}

/** A value column that grows toward its target height, with the greedy action as a cone on top. */
function Column({
  x,
  z,
  h,
  color,
  geometry,
  arrow,
  arrowColor
}: {
  x: number;
  z: number;
  h: number;
  color: string;
  geometry: THREE.BufferGeometry;
  arrow: number;
  arrowColor: string;
}) {
  const box = useRef<THREE.Mesh>(null);
  const cone = useRef<THREE.Mesh>(null);
  const cur = useRef(h);
  // Height props stay at their first value: re-sending the target on every render
  // would snap the column for a frame before useFrame eases it back.
  const h0 = useRef(h).current;
  useFrame((_, dt) => {
    cur.current += (h - cur.current) * Math.min(1, dt * 7);
    if (box.current) {
      box.current.scale.y = cur.current;
      box.current.position.y = cur.current / 2;
    }
    if (cone.current) cone.current.position.y = cur.current + 0.07;
  });
  return (
    <group position={[x, 0, z]}>
      <mesh ref={box} geometry={geometry} scale={[1, h0, 1]} position={[0, h0 / 2, 0]}>
        <meshStandardMaterial color={color} roughness={0.5} metalness={0.05} />
      </mesh>
      {arrow >= 0 && (
        // a cone squashed flat becomes an arrowhead lying on the column's top face
        <mesh ref={cone} rotation={CONE_ROT[arrow]} scale={arrow % 2 ? [0.3, 1, 1] : [1, 1, 0.3]} position={[0, cur.current + 0.07, 0]}>
          <coneGeometry args={[0.2, 0.46, 24]} />
          <meshStandardMaterial color={arrowColor} roughness={0.4} />
        </mesh>
      )}
    </group>
  );
}

function Gem({ position, color, emissive }: { position: [number, number, number]; color: string; emissive: number }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame((state) => {
    if (!ref.current) return;
    ref.current.rotation.y = state.clock.elapsedTime * 1.2;
    ref.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2) * 0.06;
  });
  return (
    <mesh ref={ref} position={position}>
      <octahedronGeometry args={[0.28, 0]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={emissive} roughness={0.15} metalness={0.3} />
    </mesh>
  );
}

/** The robot glides to its cell; a jump back to the start (new episode) snaps instead of flying. */
function Robot({ target, color }: { target: [number, number, number]; color: string }) {
  const ref = useRef<THREE.Group>(null);
  const goal = useMemo(() => new THREE.Vector3(), []);
  useFrame((_, dt) => {
    const g = ref.current;
    if (!g) return;
    goal.set(target[0], target[1], target[2]);
    if (g.position.distanceTo(goal) > 2.5) g.position.copy(goal);
    else g.position.lerp(goal, 1 - Math.exp(-dt * 12));
  });
  return (
    <group ref={ref} position={target}>
      <mesh>
        <sphereGeometry args={[0.2, 28, 28]} />
        <meshStandardMaterial color={color} roughness={0.35} />
      </mesh>
    </group>
  );
}
