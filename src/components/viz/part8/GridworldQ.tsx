import { useRef, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { MiniChart } from '../core/plot';
import { useInterval, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, mixHex } from '../core/colors';
import { fmt, rng } from '../core/math';
import { ARROWS, DEFAULT_ROWS, argmaxQ, chooseAction, envStep, isTerminal, maxQ, parseWorld, qUpdate, type Cell, type World } from './_gridworld';

type Speed = 'slow' | 'normal' | 'fast';
type Brush = 'none' | '#' | 'F' | 'G' | 'S';
const MAX_EP_STEPS = 50;
const CS = 64;

export default function GridworldQ({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [world, setWorld] = useState<World>(() => parseWorld(DEFAULT_ROWS));
  const Q = useRef(new Float64Array(world.w * world.h * 4));
  const rand = useRef(rng(7).next);
  const pos = useRef(world.start);
  const ep = useRef({ steps: 0, reward: 0, pendingReset: false, trail: [world.start] as number[] });
  const [, setVersion] = useState(0);
  const [history, setHistory] = useState<{ r: number; steps: number }[]>([]);
  const [firstGoalAt, setFirstGoalAt] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [demo, setDemo] = useState(false);
  const [speed, setSpeed] = useState<Speed>('normal');
  const [eps, setEps] = useState(0.2);
  const [gamma, setGamma] = useState(0.9);
  const [alpha, setAlpha] = useState(0.5);
  const [showValues, setShowValues] = useState(false);
  const [brush, setBrush] = useState<Brush>('none');

  const bump = () => setVersion((v) => v + 1);

  const resetLearning = (w: World = world) => {
    Q.current = new Float64Array(w.w * w.h * 4);
    rand.current = rng(Math.floor(Math.random() * 1e6)).next;
    pos.current = w.start;
    ep.current = { steps: 0, reward: 0, pendingReset: false, trail: [w.start] };
    setHistory([]);
    setFirstGoalAt(null);
    setDemo(false);
    bump();
  };

  /** Advance the simulation by one environment step. Returns true if an episode ended. */
  const stepOnce = (greedy: boolean, records: { r: number; steps: number }[]): boolean => {
    const e = ep.current;
    if (e.pendingReset) {
      pos.current = world.start;
      ep.current = { steps: 0, reward: 0, pendingReset: false, trail: [world.start] };
      return false;
    }
    const s = pos.current;
    const a = chooseAction(Q.current, s, greedy ? 0 : eps, rand.current);
    const { s2, r, done } = envStep(world, s, a);
    if (!greedy) qUpdate(Q.current, s, a, r, s2, done, alpha, gamma);
    e.steps++;
    e.reward += r;
    e.trail.push(s2);
    pos.current = s2;
    if (done || e.steps >= MAX_EP_STEPS) {
      records.push({ r: e.reward, steps: e.steps });
      e.pendingReset = true;
      return true;
    }
    return false;
  };

  const commit = (records: { r: number; steps: number }[]) => {
    if (records.length) {
      if (firstGoalAt === null) {
        const k = records.findIndex((x) => x.r > 0);
        if (k >= 0) setFirstGoalAt(history.length + k + 1);
      }
      setHistory((h) => [...h, ...records].slice(-600));
    }
    bump();
  };

  const tick = () => {
    const records: { r: number; steps: number }[] = [];
    if (demo) {
      const ended = stepOnce(true, []);
      if (ended) setDemo(false);
      commit([]);
      return;
    }
    const n = speed === 'fast' ? 30 : 1;
    for (let i = 0; i < n; i++) stepOnce(false, records);
    commit(records);
  };

  useInterval(tick, playing || demo ? (demo ? 260 : speed === 'slow' ? 360 : speed === 'normal' ? 150 : 40) : null);

  const runEpisodes = (k: number) => {
    setPlaying(false);
    setDemo(false);
    const records: { r: number; steps: number }[] = [];
    let guard = 0;
    while (records.length < k && guard < k * (MAX_EP_STEPS + 2)) {
      stepOnce(false, records);
      guard++;
    }
    // finish on a clean start position
    pos.current = world.start;
    ep.current = { steps: 0, reward: 0, pendingReset: false, trail: [world.start] };
    commit(records);
  };

  const startDemo = () => {
    setPlaying(false);
    pos.current = world.start;
    ep.current = { steps: 0, reward: 0, pendingReset: false, trail: [world.start] };
    setDemo(true);
    bump();
  };

  const paint = (idx: number) => {
    if (brush === 'none') return;
    const cells = [...world.cells];
    let start = world.start;
    if (brush === 'S') {
      if (cells[idx] !== '.') cells[idx] = '.';
      start = idx;
    } else {
      if (idx === world.start) return;
      cells[idx] = cells[idx] === brush ? '.' : (brush as Cell);
    }
    if (!cells.includes('G')) return; // always keep at least one goal
    const nw = { ...world, cells, start };
    setWorld(nw);
    setPlaying(false);
    resetLearning(nw);
  };

  // derived values for rendering
  const V = world.cells.map((_, s) => (world.cells[s] === '#' || isTerminal(world, s) ? 0 : maxQ(Q.current, s)));
  const recent = history.slice(-20);
  const success = recent.length ? recent.filter((h) => h.r > 0).length / recent.length : 0;
  const vStart = maxQ(Q.current, world.start);
  const ma: number[] = [];
  for (let i = 0; i < history.length; i++) {
    const win = history.slice(Math.max(0, i - 9), i + 1);
    ma.push(win.reduce((a, b) => a + b.r, 0) / win.length);
  }
  const rx = pos.current % world.w;
  const ry = Math.floor(pos.current / world.w);
  const animate = !reduce && speed !== 'fast';
  const dark = c.theme === 'dark';

  const cellFill = (s: number) => {
    const cell = world.cells[s];
    if (cell === '#') return dark ? '#2a3366' : '#5b5570';
    if (cell === 'G') return mixHex(c.surface.startsWith('#') ? c.surface : '#ffffff', c.pos, 0.35);
    if (cell === 'F') return mixHex(c.surface.startsWith('#') ? c.surface : '#ffffff', c.neg, 0.3);
    const v = V[s];
    const base = c.surface.startsWith('#') ? c.surface : '#ffffff';
    return v >= 0 ? mixHex(base, c.pos, Math.min(1, v) * 0.75) : mixHex(base, c.neg, Math.min(1, -v) * 0.75);
  };

  let note: { t: ReactNode; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' | 'bad' };
  if (brush !== 'none') note = { t: '칸을 눌러 지도를 바꿔 보세요. 지도가 바뀌면 로봇은 처음부터 다시 배워요.', mood: 'smile' };
  else if (demo) note = { t: '배운 대로만(가치가 가장 큰 화살표를 따라) 움직여요. 탐색 없이!', mood: 'happy' };
  else if (eps === 0) note = { t: <>탐색을 0으로 하면 <strong>아는 길만</strong> 가요. 처음엔 아는 게 없어서 같은 곳만 맴돌거나, 더 좋은 길을 영영 못 찾을 수 있어요.</>, mood: 'think', tone: 'warn' };
  else if (eps >= 0.9) note = { t: <>거의 <strong>무작위로만</strong> 움직여요. 배운 걸 써먹지 못하니 보석에 잘 못 가요. 탐색과 활용의 균형이 중요해요!</>, mood: 'wow', tone: 'warn' };
  else if (gamma < 0.5 && history.length > 30) note = { t: <>γ가 작으면 멀리 있는 보석의 가치가 거의 0이 돼서, 출발점까지 <strong>가치가 잘 퍼지지 않아요</strong>.</>, mood: 'think' };
  else if (vStart > 0.25) note = { t: <>가치가 보석에서 <strong>한 칸씩 거꾸로</strong> 퍼져 출발점까지 왔어요! 🎉 화살표만 따라가면 보석이에요. <strong>🏃 배운 대로 가 보기</strong>를 눌러 보세요.</>, mood: 'cheer', tone: 'good' };
  else if (firstGoalAt !== null) note = { t: <>💎 {firstGoalAt}번째 판에서 처음으로 보석을 찾았어요! 이제 <strong>보석 바로 옆 칸</strong>부터 가치(초록색)가 올라가요. 계속 지켜보세요.</>, mood: 'wow', tone: 'good' };
  else if (history.length > 0 || ep.current.steps > 0) note = { t: '로봇은 아직 아무것도 몰라요. 무작위로 돌아다니며 보상을 찾는 중… 🔥에 빠지면 -1, 💎를 찾으면 +1!', mood: 'think' };
  else note = { t: <>로봇 🤖이 시행착오로 💎까지 가는 길을 배워요. 정답은 아무도 안 알려 줘요 — <strong>보상만</strong> 있어요. <strong>▶ 시작</strong>을 눌러 보세요.</>, mood: 'smile' };

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: expanded ? 'minmax(0, 1.2fr) minmax(0, 1fr)' : '1fr', gap: 12, alignItems: 'start' }}>
          <svg
            viewBox={`0 0 ${world.w * CS} ${world.h * CS}`}
            className="vz-svg"
            role="img"
            aria-label="Q-학습 그리드월드: 로봇, 보석, 함정, 벽"
            style={{ maxWidth: expanded ? 640 : 520, margin: '0 auto', borderRadius: 14, overflow: 'hidden', touchAction: 'pan-y' }}
          >
            {world.cells.map((cell, s) => {
              const x = (s % world.w) * CS;
              const y = Math.floor(s / world.w) * CS;
              const free = cell === '.';
              const v = V[s];
              const known = free && Math.abs(v) > 1e-6;
              const a = known ? argmaxQ(Q.current, s, () => 0) : -1;
              return (
                <g key={s} onClick={() => paint(s)} style={{ cursor: brush !== 'none' ? 'pointer' : 'default' }}>
                  <rect x={x + 1.5} y={y + 1.5} width={CS - 3} height={CS - 3} rx={10} fill={cellFill(s)} stroke="var(--border)" style={{ transition: 'fill .3s' }} />
                  {cell === 'G' && (
                    <text x={x + CS / 2} y={y + CS / 2 + 10} textAnchor="middle" style={{ fontSize: 28 }}>
                      💎
                    </text>
                  )}
                  {cell === 'F' && (
                    <text x={x + CS / 2} y={y + CS / 2 + 10} textAnchor="middle" style={{ fontSize: 26 }}>
                      🔥
                    </text>
                  )}
                  {cell === '#' && (
                    <text x={x + CS / 2} y={y + CS / 2 + 9} textAnchor="middle" style={{ fontSize: 22, opacity: 0.85 }}>
                      🧱
                    </text>
                  )}
                  {known && (
                    <text x={x + CS / 2} y={y + CS / 2 + (showValues ? 2 : 9)} textAnchor="middle" style={{ fontSize: 24, fontWeight: 900, fill: v >= 0 ? (dark ? '#d8fff2' : '#1d6e57') : dark ? '#ffd0d9' : '#9b2335', opacity: 0.35 + 0.65 * Math.min(1, Math.abs(v) * 2) }}>
                      {ARROWS[a]}
                    </text>
                  )}
                  {showValues && free && (
                    <text x={x + CS / 2} y={y + CS - 8} textAnchor="middle" style={{ fontSize: 11, fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>
                      {fmt(v, 2)}
                    </text>
                  )}
                  {s === world.start && (
                    <text x={x + 9} y={y + 17} style={{ fontSize: 12 }}>
                      🏁
                    </text>
                  )}
                </g>
              );
            })}
            {ep.current.trail.slice(-24).map((s, i, arr) => (
              <circle key={`t${i}`} cx={(s % world.w) * CS + CS / 2} cy={Math.floor(s / world.w) * CS + CS / 2} r={4} fill={c.series[0]} opacity={0.15 + (0.5 * i) / arr.length} />
            ))}
            <g style={{ transform: `translate(${rx * CS + CS / 2}px, ${ry * CS + CS / 2}px)`, transition: animate ? 'transform .13s ease-out' : undefined }}>
              <circle r={21} fill={c.series[0]} opacity={0.18} />
              <text y={10} textAnchor="middle" style={{ fontSize: 28 }}>
                🤖
              </text>
            </g>
          </svg>

          <div style={{ display: 'grid', gap: 10 }}>
            <div className="vz-stats">
              <Stat label="에피소드(판)" value={history.length} />
              <Stat label="이번 판 걸음" value={ep.current.steps} />
              <Stat label="최근 20판 성공률" value={`${Math.round(success * 100)}%`} tone={success > 0.7 ? 'good' : undefined} />
              <Stat label="출발점의 가치" value={fmt(vStart, 2)} tone={vStart > 0.2 ? 'good' : undefined} />
            </div>
            <div>
              <div className="vz-panel-title">판마다 받은 보상 (최근 10판 평균 · 가로축: 판)</div>
              <MiniChart series={[ma.length > 1 ? ma : [0, 0]]} colors={[c.series[1]]} height={130} yMin={-1} yMax={1} />
            </div>
          </div>
        </div>
        <Note mood={note.mood} tone={note.tone}>
          {note.t}
        </Note>
      </div>

      <Controls>
        <Button variant="primary" onClick={() => { setDemo(false); setPlaying((p) => !p); }}>
          {playing ? '❚❚ 멈춤' : '▶ 시작'}
        </Button>
        <Button onClick={() => { const r: { r: number; steps: number }[] = []; stepOnce(false, r); commit(r); }} disabled={playing || demo}>
          한 걸음 ⟶
        </Button>
        <Button onClick={() => runEpisodes(50)}>⚡ 50판 순식간에</Button>
        <Button onClick={startDemo} disabled={demo}>
          🏃 배운 대로 가 보기
        </Button>
        <Button variant="ghost" onClick={() => { setPlaying(false); resetLearning(); }}>
          ↺ 기억 지우기
        </Button>
        <Segmented
          label="속도"
          options={[
            { value: 'slow', label: '느리게' },
            { value: 'normal', label: '보통' },
            { value: 'fast', label: '빠르게' }
          ]}
          value={speed}
          onChange={(v) => setSpeed(v as Speed)}
        />
        <Slider label="탐색 ε (모험심)" value={eps} min={0} max={1} step={0.05} onChange={setEps} format={(v) => v.toFixed(2)} accent="var(--peach)" />
        <Slider label="할인율 γ (참을성)" value={gamma} min={0.1} max={0.99} step={0.01} onChange={setGamma} format={(v) => v.toFixed(2)} accent="var(--mint)" />
        <Slider label="학습률 α" value={alpha} min={0.05} max={1} step={0.05} onChange={setAlpha} format={(v) => v.toFixed(2)} />
        <Toggle label="가치 숫자 보기" checked={showValues} onChange={setShowValues} />
        <Segmented
          label="지도 고치기"
          options={[
            { value: 'none', label: '보기' },
            { value: '#', label: '🧱 벽' },
            { value: 'F', label: '🔥 함정' },
            { value: 'G', label: '💎 보석' },
            { value: 'S', label: '🏁 출발' }
          ]}
          value={brush}
          onChange={(v) => setBrush(v as Brush)}
        />
      </Controls>
    </div>
  );
}
