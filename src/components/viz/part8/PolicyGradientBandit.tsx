import { useMemo, useRef, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Legend, Note, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { MiniChart, Plot, linePath } from '../core/plot';
import { useInterval } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { fmt, rng } from '../core/math';

/*
 * REINFORCE on a 3-armed bandit. Policy π = softmax(θ); after pulling arm a and
 * receiving reward G:  θ ← θ + lr · (G − b) · (onehot(a) − π).
 * Every payout is positive on average, so without a baseline every pulled arm
 * gets pushed up — the classic high-variance case from the chapter.
 */
const ARMS = [
  { icon: '🍎', name: '사과 기계', mu: 4 },
  { icon: '🍋', name: '레몬 기계', mu: 7 },
  { icon: '🍇', name: '포도 기계', mu: 5.5 }
];
const BEST = 1;
const SIGMA = 3;
const SEEDS = 16;
const STEPS = 300;

const softmax = (z: number[]) => {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};
function sampleIdx(p: number[], u: number) {
  let acc = 0;
  for (let i = 0; i < p.length; i++) {
    acc += p[i];
    if (u <= acc) return i;
  }
  return p.length - 1;
}

function simulate(seed: number, lr: number, useBaseline: boolean): number[] {
  const R = rng(seed);
  let th = [0, 0, 0];
  let b = 0;
  let started = false;
  const out: number[] = [];
  for (let t = 0; t < STEPS; t++) {
    const p = softmax(th);
    const a = sampleIdx(p, R.next());
    const g = ARMS[a].mu + SIGMA * R.normal();
    if (!started) {
      b = g;
      started = true;
    }
    const adv = useBaseline ? g - b : g;
    th = th.map((x, k) => x + lr * adv * ((k === a ? 1 : 0) - p[k]));
    if (useBaseline) b += 0.1 * (g - b);
    out.push(softmax(th)[BEST]);
  }
  return out;
}

function quantile(sorted: number[], q: number) {
  const i = (sorted.length - 1) * q;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

export default function PolicyGradientBandit({ expanded }: VizProps) {
  const [mode, setMode] = useState<'play' | 'compare'>('play');
  const [lr, setLr] = useState(0.05);
  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <Segmented
          options={[
            { value: 'play', label: '🎰 직접 당겨 보기' },
            { value: 'compare', label: '📊 기준선 비교 실험' }
          ]}
          value={mode}
          onChange={(v) => setMode(v as 'play' | 'compare')}
        />
      </div>
      {mode === 'play' ? <Play lr={lr} setLr={setLr} /> : <Compare lr={lr} setLr={setLr} expanded={expanded} />}
    </div>
  );
}

function Play({ lr, setLr }: { lr: number; setLr: (v: number) => void }) {
  const c = useVizColors();
  const dark = c.theme === 'dark';
  const R = useRef(rng(2024));
  const [theta, setTheta] = useState([0, 0, 0]);
  const [baseline, setBaseline] = useState<number | null>(null);
  const [useB, setUseB] = useState(true);
  const [last, setLast] = useState<{ a: number; g: number; b: number | null; dp: number } | null>(null);
  const [pBest, setPBest] = useState<number[]>([1 / 3]);
  const [auto, setAuto] = useState(0);
  const [reveal, setReveal] = useState(false);
  const pi = softmax(theta);

  const pull = () => {
    const a = sampleIdx(pi, R.current.next());
    const g = ARMS[a].mu + SIGMA * R.current.normal();
    const b = useB ? (baseline ?? g) : null;
    const adv = b === null ? g : g - b;
    const th = theta.map((x, k) => x + lr * adv * ((k === a ? 1 : 0) - pi[k]));
    const np = softmax(th);
    setTheta(th);
    if (useB) setBaseline((b ?? g) + 0.1 * (g - (b ?? g)));
    setLast({ a, g, b, dp: np[a] - pi[a] });
    setPBest((h) => [...h, np[BEST]].slice(-400));
  };

  useInterval(
    () => {
      pull();
      setAuto((n) => n - 1);
    },
    auto > 0 ? 70 : null
  );

  const reset = () => {
    R.current = rng(Math.floor(Math.random() * 1e6));
    setTheta([0, 0, 0]);
    setBaseline(null);
    setLast(null);
    setPBest([1 / 3]);
    setAuto(0);
  };

  let note: { t: ReactNode; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' };
  if (!last) note = { t: <>세 기계 중 어떤 게 제일 좋은지 아무도 안 알려 줘요. 정책은 확률대로 하나를 당기고, <strong>결과가 좋았던 행동의 확률을 결과만큼</strong> 올려요.</>, mood: 'smile' };
  else if (last.b === null)
    note =
      last.g >= 0
        ? { t: <>{ARMS[last.a].icon}에서 <strong>+{fmt(last.g, 1)}</strong>! 결과가 양수라서 이 기계 확률을 <strong>무조건 올렸어요</strong>. 그런데 모든 기계가 대체로 양수를 주니… 나쁜 기계도 늘 올라가요. 기준선을 켜 보세요.</>, mood: 'think', tone: 'warn' }
        : { t: <>{ARMS[last.a].icon}에서 <strong>{fmt(last.g, 1)}</strong>. 이번엔 음수라서 내렸지만, 결과는 대부분 양수라 거의 늘 "올리기"만 해요. 그래서 진짜 좋은 기계를 가려내는 게 느리고 들쭉날쭉해요.</>, mood: 'think', tone: 'warn' };
  else {
    const diff = last.g - last.b;
    note = {
      t: (
        <>
          {ARMS[last.a].icon}에서 <strong>{fmt(last.g, 1)}</strong>. 평균(기준선) {fmt(last.b, 1)}보다 <strong>{diff >= 0 ? `${fmt(diff, 1)} 높아서 ↑` : `${fmt(-diff, 1)} 낮아서 ↓`}</strong> — 이 차이가 바로 <strong>이점(advantage)</strong>이에요.
        </>
      ),
      mood: diff >= 0 ? 'happy' : 'think'
    };
  }
  if (pi[BEST] > 0.9) note = { t: <>🍋 레몬 기계가 가장 좋다는 걸 스스로 알아냈어요! 🎉 가치를 계산하지 않고, <strong>정책(확률) 자체</strong>를 보상의 기울기로 다듬었어요.</>, mood: 'cheer', tone: 'good' };

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 10 }}>
          {ARMS.map((arm, k) => {
            const chosen = last?.a === k;
            return (
              <div
                key={arm.name}
                style={{
                  position: 'relative',
                  display: 'grid',
                  justifyItems: 'center',
                  gap: 4,
                  padding: '12px 8px',
                  borderRadius: 18,
                  border: `2px solid ${chosen ? c.series[k] : 'var(--border)'}`,
                  background: chosen ? withAlpha(c.series[k], dark ? 0.18 : 0.1) : 'var(--surface-solid)',
                  transition: 'all .2s'
                }}
              >
                <span style={{ fontSize: 34 }} aria-hidden="true">
                  🎰
                </span>
                <strong style={{ fontSize: 14 }}>
                  {arm.icon} {arm.name}
                </strong>
                <span style={{ fontSize: 24, fontWeight: 900, fontVariantNumeric: 'tabular-nums' }}>{Math.round(pi[k] * 100)}%</span>
                <span style={{ width: '100%', height: 10, borderRadius: 99, background: 'var(--surface-3)', overflow: 'hidden' }}>
                  <span style={{ display: 'block', height: '100%', width: `${pi[k] * 100}%`, borderRadius: 99, background: c.series[k], transition: 'width .2s' }} />
                </span>
                {chosen && (
                  <span key={pBest.length} style={{ fontSize: 13, fontWeight: 850, color: last!.dp >= 0 ? c.pos : c.neg, animation: 'vz-note-in .3s cubic-bezier(.34,1.56,.64,1)' }}>
                    {last!.g >= 0 ? '+' : ''}{fmt(last!.g, 1)}점 · 확률 {last!.dp >= 0 ? '↑' : '↓'}
                  </span>
                )}
                {reveal && <span style={{ fontSize: 11.5, color: 'var(--text-3)' }}>평균 {arm.mu}점</span>}
              </div>
            );
          })}
        </div>
        <Note mood={note.mood} tone={note.tone}>
          {note.t}
        </Note>
        <div className="vz-stats">
          <Stat label="당긴 횟수" value={pBest.length - 1} />
          <Stat label="기준선 (평균 보상)" value={useB ? (baseline === null ? '—' : fmt(baseline, 2)) : '안 씀'} />
          <Stat label="🍋 최고 기계 확률" value={`${Math.round(pi[BEST] * 100)}%`} tone={pi[BEST] > 0.8 ? 'good' : undefined} />
        </div>
        <div>
          <div className="vz-panel-title">🍋 레몬 기계를 고를 확률의 변화 (가로축: 당긴 횟수)</div>
          <MiniChart series={[pBest.length > 1 ? pBest : [1 / 3, 1 / 3]]} colors={[c.series[1]]} height={120} yMin={0} yMax={1} />
        </div>
      </div>
      <Controls>
        <Button variant="primary" onClick={pull} disabled={auto > 0}>
          🎲 정책대로 한 번 당기기
        </Button>
        <Button onClick={() => setAuto(50)} disabled={auto > 0}>
          ⏩ 50번 자동
        </Button>
        <Button variant="ghost" onClick={reset}>
          ↺ 처음부터
        </Button>
        <Toggle label="기준선 쓰기 (평균과 비교)" checked={useB} onChange={(v) => { setUseB(v); setBaseline(null); }} />
        <Toggle label="기계의 진짜 평균 보기" checked={reveal} onChange={setReveal} />
        <Slider label="학습률" value={lr} min={0.01} max={0.2} step={0.01} onChange={setLr} format={(v) => v.toFixed(2)} />
      </Controls>
    </div>
  );
}

function Compare({ lr, setLr, expanded }: { lr: number; setLr: (v: number) => void; expanded?: boolean }) {
  const c = useVizColors();
  const [seedBase, setSeedBase] = useState(1);
  const runs = useMemo(() => {
    const variants = [false, true].map((useB) => Array.from({ length: SEEDS }, (_, s) => simulate(seedBase * 100 + s, lr, useB)));
    return variants.map((rs) => {
      const mean: number[] = [];
      const lo: number[] = [];
      const hi: number[] = [];
      for (let t = 0; t < STEPS; t++) {
        const col = rs.map((r) => r[t]).sort((a, b) => a - b);
        mean.push(col.reduce((a, b) => a + b, 0) / col.length);
        lo.push(quantile(col, 0.1));
        hi.push(quantile(col, 0.9));
      }
      const finals = rs.map((r) => r[STEPS - 1]);
      const m = finals.reduce((a, b) => a + b, 0) / finals.length;
      const sd = Math.sqrt(finals.reduce((a, b) => a + (b - m) ** 2, 0) / finals.length);
      return { mean, lo, hi, finalMean: m, finalSd: sd, wrong: finals.filter((f) => f < 0.5).length };
    });
  }, [lr, seedBase]);
  const [noB, withB] = runs;
  const cols = [c.series[2], c.series[0]];

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <Plot xDomain={[0, STEPS - 1]} yDomain={[0, 1]} xLabel="당긴 횟수" yLabel="🍋 최고 기계 확률" aspect={expanded ? 2.3 : 1.7} maxHeight={expanded ? 480 : 340} yFormat={(v) => `${Math.round(v * 100)}%`} ariaLabel="기준선 유무에 따른 학습 곡선 비교">
          {({ x, y, clip }) => (
            <g clipPath={clip}>
              {runs.map((r, i) => (
                <g key={i}>
                  <path
                    d={`${linePath(r.hi.map((v, t) => [x(t), y(v)]))}L${r.lo
                      .map((v, t) => [x(t), y(v)] as [number, number])
                      .reverse()
                      .map(([px, py]) => `${px.toFixed(1)},${py.toFixed(1)}`)
                      .join('L')}Z`}
                    fill={withAlpha(cols[i], 0.18)}
                    stroke="none"
                  />
                  <path d={linePath(r.mean.map((v, t) => [x(t), y(v)]))} fill="none" stroke={cols[i]} strokeWidth={3} />
                </g>
              ))}
            </g>
          )}
        </Plot>
        <Legend
          items={[
            { color: cols[0], label: '기준선 없음 (결과를 그대로 곱함)' },
            { color: cols[1], label: '기준선 있음 (평균과의 차이를 곱함)' }
          ]}
        />
        <Note mood="happy" tone="good">
          같은 기계, 같은 횟수로 {SEEDS}번씩 다시 해 봤어요. 진한 선은 평균, 옅은 띠는 운에 따라 갈린 범위예요. <strong>기준선이 있으면 띠가 좁아요</strong> — 운에 덜 휘둘린다(분산이 작다)는 뜻이에요.
        </Note>
        <div className="vz-stats">
          <Stat label="300번 뒤 평균 (기준선 없음)" value={`${Math.round(noB.finalMean * 100)}%`} sub={`흔들림 ±${Math.round(noB.finalSd * 100)}%p · 엉뚱한 기계 ${noB.wrong}번`} />
          <Stat label="300번 뒤 평균 (기준선 있음)" value={`${Math.round(withB.finalMean * 100)}%`} sub={`흔들림 ±${Math.round(withB.finalSd * 100)}%p · 엉뚱한 기계 ${withB.wrong}번`} tone="good" />
        </div>
      </div>
      <Controls>
        <Slider label="학습률" value={lr} min={0.01} max={0.2} step={0.01} onChange={setLr} format={(v) => v.toFixed(2)} />
        <Button onClick={() => setSeedBase((s) => s + 1)}>🎲 다른 운으로 다시</Button>
      </Controls>
    </div>
  );
}
