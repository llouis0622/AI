import { useRef, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Icon, Note, Segmented, Slider, Stat, Toggle } from '../core/ui';
import type { IconName } from '@/components/ui/Icon';
import { MiniChart } from '../core/plot';
import { useInterval, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, mixHex } from '../core/colors';
import { fmt, rng } from '../core/math';
import { ACTIONS, DEFAULT_ROWS, argmaxQ, chooseAction, envStep, isTerminal, maxQ, parseWorld, qUpdate, type Cell, type World } from './_gridworld';

type Speed = 'slow' | 'normal' | 'fast';
type Brush = 'none' | '#' | 'F' | 'G' | 'S';
const MAX_EP_STEPS = 50;
const CS = 64;

/** The greedy action as a thin arrow (shaft and open head) centred on (cx, cy). */
function arrowPath(cx: number, cy: number, a: number) {
  const [dx, dy] = ACTIONS[a];
  const L = 11;
  const h = 6.5;
  const tx = cx + dx * L;
  const ty = cy + dy * L;
  return `M${cx - dx * L} ${cy - dy * L}L${tx} ${ty}M${tx - dx * h - dy * h} ${ty - dy * h + dx * h}L${tx} ${ty}L${tx - dx * h + dy * h} ${ty - dy * h - dx * h}`;
}

const brushLabel = (icon: IconName, text: string) => (
  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
    <Icon name={icon} size={14} />
    {text}
  </span>
);

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
  const base = c.surface.startsWith('#') ? c.surface : '#ffffff';

  const cellFill = (s: number) => {
    const cell = world.cells[s];
    if (cell === '#') return mixHex(base, c.ink, dark ? 0.16 : 0.1);
    if (cell === 'G') return mixHex(base, c.pos, 0.3);
    if (cell === 'F') return mixHex(base, c.neg, 0.26);
    const v = V[s];
    return v >= 0 ? mixHex(base, c.pos, Math.min(1, v) * 0.75) : mixHex(base, c.neg, Math.min(1, -v) * 0.75);
  };
  const cellStroke = (cell: Cell) => (cell === 'G' ? c.pos : cell === 'F' ? c.neg : 'var(--border)');

  let note: { t: ReactNode; tone?: 'good' | 'warn' | 'bad' };
  if (brush !== 'none') note = { t: '칸을 눌러 지도를 바꿔 보세요. 지도가 바뀌면 로봇은 처음부터 다시 배워요.' };
  else if (demo) note = { t: '탐색 없이, 학습한 대로만(가치가 가장 큰 화살표를 따라) 움직여요.' };
  else if (eps === 0) note = { t: <>탐색을 0으로 하면 <strong>아는 길만</strong> 가요. 처음엔 아는 게 없어서 같은 곳만 맴돌거나, 더 좋은 길을 영영 못 찾을 수 있어요.</>, tone: 'warn' };
  else if (eps >= 0.9) note = { t: <>거의 <strong>무작위로만</strong> 움직여요. 배운 것을 써먹지 못해서 보석에 잘 닿지 못해요. 탐색과 활용 사이의 균형이 중요해요.</>, tone: 'warn' };
  else if (gamma < 0.5 && history.length > 30) note = { t: <>γ가 작으면 멀리 있는 보석의 가치가 거의 0이 돼서, 출발점까지 <strong>가치가 잘 퍼지지 않아요</strong>.</> };
  else if (vStart > 0.25) note = { t: <>가치가 보석에서 <strong>한 칸씩 거꾸로</strong> 퍼져 출발점까지 왔어요. 이제 화살표만 따라가면 보석에 닿아요. <strong>학습한 대로 이동</strong>을 눌러 보세요.</>, tone: 'good' };
  else if (firstGoalAt !== null) note = { t: <>{firstGoalAt}번째 판에서 처음으로 보석을 찾았어요. 이제 <strong>보석 바로 옆 칸</strong>부터 가치(파란색)가 올라가요.</>, tone: 'good' };
  else if (history.length > 0 || ep.current.steps > 0) note = { t: '로봇은 아직 아무것도 몰라서 무작위로 돌아다니며 보상을 찾고 있어요. 함정에 빠지면 -1, 보석을 찾으면 +1이에요.' };
  else note = { t: <>로봇이 시행착오로 보석까지 가는 길을 배워요. 정답은 아무도 알려 주지 않고 <strong>보상</strong>만 있어요. <strong>시작</strong>을 눌러 보세요.</> };

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
              const mx = x + CS / 2;
              const my = y + CS / 2;
              const free = cell === '.';
              const v = V[s];
              const known = free && Math.abs(v) > 1e-6;
              const a = known ? argmaxQ(Q.current, s, () => 0) : -1;
              return (
                <g key={s} onClick={() => paint(s)} style={{ cursor: brush !== 'none' ? 'pointer' : 'default' }}>
                  <rect
                    x={x + 1.5}
                    y={y + 1.5}
                    width={CS - 3}
                    height={CS - 3}
                    rx={10}
                    fill={cellFill(s)}
                    stroke={cellStroke(cell)}
                    strokeWidth={cell === 'G' || cell === 'F' ? 1.5 : 1}
                    style={{ transition: 'fill .3s' }}
                  />
                  {cell === 'G' && <Icon name="gem" x={mx - 14} y={my - 14} size={28} color={c.pos} />}
                  {cell === 'F' && <Icon name="flame" x={mx - 13} y={my - 13} size={26} color={c.neg} />}
                  {cell === '#' && <Icon name="brick" x={mx - 12} y={my - 12} size={24} color={c.text} />}
                  {known && (
                    <path
                      d={arrowPath(mx, my - (showValues ? 6 : 0), a)}
                      fill="none"
                      stroke={c.ink}
                      strokeWidth={2.25}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      opacity={0.35 + 0.65 * Math.min(1, Math.abs(v) * 2)}
                    />
                  )}
                  {showValues && free && (
                    <text x={mx} y={y + CS - 8} textAnchor="middle" style={{ fontSize: 11, fontWeight: 700, fontVariantNumeric: 'tabular-nums', fill: c.ink }}>
                      {fmt(v, 2)}
                    </text>
                  )}
                  {s === world.start && <Icon name="flag" x={x + 6} y={y + 6} size={13} color={c.text} />}
                </g>
              );
            })}
            {ep.current.trail.slice(-24).map((s, i, arr) => (
              <circle key={`t${i}`} cx={(s % world.w) * CS + CS / 2} cy={Math.floor(s / world.w) * CS + CS / 2} r={3.5} fill={c.ink} opacity={0.12 + (0.45 * i) / arr.length} />
            ))}
            <g style={{ transform: `translate(${rx * CS + CS / 2}px, ${ry * CS + CS / 2}px)`, transition: animate ? 'transform .13s ease-out' : undefined }}>
              <circle r={19} fill={c.surface} stroke={c.ink} strokeWidth={1.75} />
              <Icon name="bot" x={-11} y={-11} size={22} color={c.ink} />
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
        <Note tone={note.tone}>{note.t}</Note>
      </div>

      <Controls>
        <Button variant="primary" onClick={() => { setDemo(false); setPlaying((p) => !p); }}>
          <Icon name={playing ? 'pause' : 'play'} size={16} />
          {playing ? '멈춤' : '시작'}
        </Button>
        <Button onClick={() => { const r: { r: number; steps: number }[] = []; stepOnce(false, r); commit(r); }} disabled={playing || demo}>
          <Icon name="step" size={16} />한 걸음
        </Button>
        <Button onClick={() => runEpisodes(50)}>
          <Icon name="fast" size={16} />50판 한 번에
        </Button>
        <Button onClick={startDemo} disabled={demo}>
          <Icon name="route" size={16} />학습한 대로 이동
        </Button>
        <Button variant="ghost" onClick={() => { setPlaying(false); resetLearning(); }}>
          <Icon name="reset" size={16} />기억 지우기
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
        <Slider label="탐색 ε (모험심)" value={eps} min={0} max={1} step={0.05} onChange={setEps} format={(v) => v.toFixed(2)} accent="var(--accent-2)" />
        <Slider label="할인율 γ (참을성)" value={gamma} min={0.1} max={0.99} step={0.01} onChange={setGamma} format={(v) => v.toFixed(2)} />
        <Slider label="학습률 α" value={alpha} min={0.05} max={1} step={0.05} onChange={setAlpha} format={(v) => v.toFixed(2)} />
        <Toggle label="가치 숫자 보기" checked={showValues} onChange={setShowValues} />
        <Segmented
          label="지도 고치기"
          options={[
            { value: 'none', label: '보기' },
            { value: '#', label: brushLabel('brick', '벽') },
            { value: 'F', label: brushLabel('flame', '함정') },
            { value: 'G', label: brushLabel('gem', '보석') },
            { value: 'S', label: brushLabel('flag', '출발') }
          ]}
          value={brush}
          onChange={(v) => setBrush(v as Brush)}
        />
      </Controls>
    </div>
  );
}
