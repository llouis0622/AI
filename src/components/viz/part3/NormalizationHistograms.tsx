import { useMemo, useState } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Legend, Note, Segmented, Slider, Stat } from '../core/ui';
import { Plot, linePath } from '../core/plot';
import { useVizColors } from '../core/colors';
import { rng } from '../core/math';
import { PowTen, Sci } from './_sci';

type Mode = 'small' | 'big' | 'he' | 'norm';
const MODES: { value: Mode; label: string }[] = [
  { value: 'small', label: '너무 작게' },
  { value: 'big', label: '너무 크게' },
  { value: 'he', label: 'He 초기화' },
  { value: 'norm', label: '정규화 층' }
];
/** weight std = mult × √(2/n); "정규화 층" keeps the too-big weights but re-standardises every layer. */
const MULT: Record<Mode, number> = { small: 0.5, big: 1.6, he: 1, norm: 1.6 };

const WIDTH = 64;
const BATCH = 200;
const BINS = 32;
const RANGE = 4;

interface LayerStat {
  hist: number[];
  under: number;
  over: number;
  std: number;
}

function simulate(mode: Mode, depth: number, seed: number): LayerStat[] {
  const r = rng(seed);
  const n = WIDTH;
  let X = new Float32Array(BATCH * n);
  for (let k = 0; k < X.length; k++) X[k] = r.normal();
  const out: LayerStat[] = [];
  const wStd = MULT[mode] * Math.sqrt(2 / n);
  for (let l = 0; l < depth; l++) {
    const W = new Float32Array(n * n);
    for (let k = 0; k < W.length; k++) W[k] = r.normal() * wStd;
    const Z = new Float32Array(BATCH * n);
    for (let s = 0; s < BATCH; s++) {
      for (let j = 0; j < n; j++) {
        let acc = 0;
        for (let i = 0; i < n; i++) acc += X[s * n + i] * W[i * n + j];
        Z[s * n + j] = acc;
      }
      if (mode === 'norm') {
        // layer norm: each sample's features → mean 0, variance 1
        let mu = 0;
        for (let j = 0; j < n; j++) mu += Z[s * n + j];
        mu /= n;
        let v = 0;
        for (let j = 0; j < n; j++) v += (Z[s * n + j] - mu) ** 2;
        const sd = Math.sqrt(v / n + 1e-5);
        for (let j = 0; j < n; j++) Z[s * n + j] = (Z[s * n + j] - mu) / sd;
      }
    }
    const hist = new Array(BINS).fill(0);
    let under = 0;
    let over = 0;
    let sum = 0;
    let sq = 0;
    for (let k = 0; k < Z.length; k++) {
      const v = Z[k];
      sum += v;
      sq += v * v;
      if (v < -RANGE) under++;
      else if (v >= RANGE) over++;
      else hist[Math.floor(((v + RANGE) / (2 * RANGE)) * BINS)]++;
    }
    const mean = sum / Z.length;
    out.push({ hist, under, over, std: Math.sqrt(Math.max(0, sq / Z.length - mean * mean)) });
    const A = new Float32Array(Z.length);
    for (let k = 0; k < Z.length; k++) A[k] = Z[k] > 0 ? Z[k] : 0;
    X = A;
  }
  return out;
}


export default function NormalizationHistograms({ expanded }: VizProps) {
  const c = useVizColors();
  const [mode, setMode] = useState<Mode>('small');
  const [depth, setDepth] = useState(10);
  const [seed, setSeed] = useState(3);

  const all = useMemo(() => {
    const res = {} as Record<Mode, LayerStat[]>;
    for (const m of MODES) res[m.value] = simulate(m.value, depth, seed);
    return res;
  }, [depth, seed]);
  const layers = all[mode];
  const last = layers[layers.length - 1];
  const health = last.std < 0.05 ? 'vanish' : last.std > 20 ? 'explode' : 'ok';
  const modeColor: Record<Mode, string> = { small: c.series[3], big: c.series[2], he: c.series[1], norm: c.series[0] };

  const notes: Record<Mode, { text: string; mood: 'think' | 'wow' | 'cheer' | 'happy'; tone: 'good' | 'bad' }> = {
    small: { text: '너무 작게 시작하면 층을 지날 때마다 신호가 쪼그라들어요. 뒤쪽 층의 그래프가 0에 꽂힌 바늘처럼 변하죠? 기울기도 같이 사라져요.', mood: 'think', tone: 'bad' },
    big: { text: '너무 크게 시작하면 층마다 신호가 불어나 눈금 밖으로 넘쳐요(폭발). 빨간 "넘침" 막대가 점점 커지는 걸 보세요.', mood: 'wow', tone: 'bad' },
    he: { text: 'He 초기화(가중치 분산 2/n)는 층을 몇 개 지나도 모양과 크기가 거의 그대로예요. "출발선"을 맞춘 거예요! 🎉', mood: 'cheer', tone: 'good' },
    norm: { text: '가중치는 "너무 크게"와 똑같은데도, 정규화 층이 매 층에서 평균 0·분산 1로 다시 맞춰 줘서 끝까지 건강해요.', mood: 'happy', tone: 'good' }
  };

  const cols = expanded ? 'repeat(auto-fill, minmax(150px, 1fr))' : 'repeat(auto-fill, minmax(118px, 1fr))';
  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gridTemplateColumns: cols, gap: 8 }}>
        {layers.map((L, i) => {
          const maxBin = Math.max(1, ...L.hist);
          const total = BATCH * WIDTH;
          const overPct = Math.round(((L.over + L.under) / total) * 100);
          return (
            <div key={i} style={{ padding: '8px 8px 6px', borderRadius: 12, background: 'var(--surface-solid)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, fontWeight: 800, color: 'var(--text-2)' }}>
                <span>{i + 1}층</span>
                <span style={{ color: L.std < 0.05 || L.std > 20 ? c.neg : 'var(--text-2)' }}>σ=<Sci v={L.std} /></span>
              </div>
              <svg viewBox="0 0 120 54" className="vz-svg" role="img" aria-label={`${i + 1}층 활성값 분포`} style={{ touchAction: 'pan-y' }}>
                <line x1={60} x2={60} y1={0} y2={46} stroke={c.grid} />
                {L.hist.map((h, k) => {
                  const bh = (h / maxBin) * 40;
                  return <rect key={k} x={8 + k * (104 / BINS)} y={46 - bh} width={104 / BINS - 0.6} height={bh} fill={modeColor[mode]} opacity={0.85} />;
                })}
                {L.under > 0 && <rect x={0} y={46 - Math.min(40, (L.under / total) * 80)} width={6} height={Math.min(40, (L.under / total) * 80)} fill={c.neg} />}
                {L.over > 0 && <rect x={114} y={46 - Math.min(40, (L.over / total) * 80)} width={6} height={Math.min(40, (L.over / total) * 80)} fill={c.neg} />}
                <line x1={0} x2={120} y1={46} y2={46} stroke={c.axis} />
                <text x={8} y={53} style={{ fontSize: 7, fill: 'var(--text-3)' }}>-4</text>
                <text x={60} y={53} textAnchor="middle" style={{ fontSize: 7, fill: 'var(--text-3)' }}>0</text>
                <text x={112} y={53} textAnchor="end" style={{ fontSize: 7, fill: 'var(--text-3)' }}>4</text>
              </svg>
              {overPct > 0 && <div style={{ fontSize: 10.5, fontWeight: 800, color: c.neg }}>넘침 {overPct}%</div>}
            </div>
          );
        })}
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12, paddingTop: 0 }}>
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
            <span className="vz-panel-title">층별 신호 크기 σ (로그 눈금) — 1 근처로 유지되는 게 건강해요</span>
            <Legend items={MODES.map((m) => ({ color: modeColor[m.value], label: m.label }))} />
          </div>
          <Plot
            xDomain={[1, depth]}
            yDomain={[-4, 4]}
            height={170}
            margin={{ top: 10, right: 12, bottom: 32, left: 46 }}
            xLabel="층"
            xTicks={Math.min(depth, 8)}
            yTicks={8}
            yFormat={() => ''}
            ariaLabel="층별 표준편차 비교"
          >
            {({ x, y, clip, inner }) => (
              <g>
              {[-4, -2, 0, 2, 4].map((e) => (
                <PowTen key={e} e={e} x={inner.left - 8} y={y(e) + 4} />
              ))}
              <g clipPath={clip}>
                <line x1={x(1)} x2={x(depth)} y1={y(0)} y2={y(0)} stroke={c.axis} strokeDasharray="5 4" />
                {MODES.map((m) => {
                  const pts = all[m.value].map((L, i) => [x(i + 1), y(Math.max(-4, Math.min(4, Math.log10(Math.max(L.std, 1e-6)))))] as [number, number]);
                  const on = m.value === mode;
                  return (
                    <g key={m.value} opacity={on ? 1 : 0.35}>
                      <path d={linePath(pts)} fill="none" stroke={modeColor[m.value]} strokeWidth={on ? 3.5 : 2} />
                      {on && pts.map(([px, py], k) => <circle key={k} cx={px} cy={py} r={3.5} fill={modeColor[m.value]} />)}
                    </g>
                  );
                })}
              </g>
              </g>
            )}
          </Plot>
        </div>
        <div className="vz-stats">
          <Stat label={`${depth}층의 신호 크기 σ`} value={<Sci v={last.std} />} tone={health === 'ok' ? 'good' : 'bad'} />
          <Stat label="상태" value={health === 'vanish' ? '사라짐 😵' : health === 'explode' ? '폭발 💥' : '건강 💚'} tone={health === 'ok' ? 'good' : 'bad'} />
        </div>
        <Note mood={notes[mode].mood} tone={notes[mode].tone}>
          {notes[mode].text}
        </Note>
      </div>

      <Controls>
        <Segmented label="출발 방법" options={MODES} value={mode} onChange={(v) => setMode(v as Mode)} />
        <Slider label="층 수" value={depth} min={4} max={16} step={1} onChange={setDepth} format={(v) => `${v}층`} />
        <Button size="sm" variant="ghost" onClick={() => setSeed((s) => s + 1)}>
          🎲 다시 뽑기
        </Button>
      </Controls>
    </div>
  );
}
