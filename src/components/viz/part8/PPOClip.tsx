import { useMemo, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Controls, Legend, Note, Segmented, Slider, Stat } from '../core/ui';
import { Plot, linePath } from '../core/plot';
import { useVizColors, withAlpha } from '../core/colors';
import { fmt } from '../core/math';

const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
/** PPO clipped surrogate for one sample. */
const ppo = (rho: number, A: number, eps: number) => Math.min(rho * A, clip(rho, 1 - eps, 1 + eps) * A);
/** Is the gradient switched off at this ratio? */
const flat = (rho: number, A: number, eps: number) => (A > 0 ? rho > 1 + eps : rho < 1 - eps);

export default function PPOClip({ expanded }: VizProps) {
  const [mode, setMode] = useState<'graph' | 'epochs'>('graph');
  const [sign, setSign] = useState<1 | -1>(1);
  const [eps, setEps] = useState(0.2);
  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <Segmented
          options={[
            { value: 'graph', label: '① 클리핑 그래프' },
            { value: 'epochs', label: '② 같은 데이터로 10번 배우기' }
          ]}
          value={mode}
          onChange={(v) => setMode(v as 'graph' | 'epochs')}
        />
      </div>
      {mode === 'graph' ? <Graph sign={sign} setSign={setSign} eps={eps} setEps={setEps} expanded={expanded} /> : <Epochs sign={sign} setSign={setSign} eps={eps} setEps={setEps} expanded={expanded} />}
    </div>
  );
}

interface Shared {
  sign: 1 | -1;
  setSign: (v: 1 | -1) => void;
  eps: number;
  setEps: (v: number) => void;
  expanded?: boolean;
}

function SignControl({ sign, setSign }: Pick<Shared, 'sign' | 'setSign'>) {
  return (
    <Segmented
      label="이점 A"
      options={[
        { value: 1, label: '👍 좋았던 행동 (A > 0)' },
        { value: -1, label: '👎 나빴던 행동 (A < 0)' }
      ]}
      value={sign}
      onChange={(v) => setSign(v as 1 | -1)}
    />
  );
}

function Graph({ sign, setSign, eps, setEps, expanded }: Shared) {
  const c = useVizColors();
  const [rho, setRho] = useState(1.1);
  const A = sign;
  const L = ppo(rho, A, eps);
  const isFlat = flat(rho, A, eps);
  const pts = (f: (r: number) => number): [number, number][] => Array.from({ length: 121 }, (_, i) => [0.05 + (1.9 * i) / 120, f(0.05 + (1.9 * i) / 120)]);

  let note: { t: ReactNode; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' };
  if (isFlat)
    note = {
      t:
        A > 0 ? (
          <>확률을 이미 <strong>{Math.round(eps * 100)}% 넘게 키웠어요</strong>. 여기서부턴 그래프가 평평해서 기울기가 0 — 더 밀어도 점수가 안 올라가요. "좋은 방향이라도 한 번에 조금만!" ✋</>
        ) : (
          <>확률을 이미 <strong>{Math.round(eps * 100)}% 넘게 줄였어요</strong>. 그래프가 평평해서 더 줄여도 이득이 없어요. 나쁜 행동도 한 번에 너무 깎지 않아요. ✋</>
        ),
      mood: 'wow',
      tone: 'warn'
    };
  else
    note = {
      t: (
        <>
          비율 ρ = 새 정책 확률 ÷ 옛 정책 확률. 지금은 안전 범위 [{fmt(1 - eps, 2)}, {fmt(1 + eps, 2)}] 안이라 {A > 0 ? '확률을 키우는' : '확률을 줄이는'} 쪽으로 <strong>계속 밀어요</strong> (기울기 = A). 슬라이더로 ρ를 범위 밖으로 옮겨 보세요.
        </>
      ),
      mood: 'smile'
    };

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <Plot
          xDomain={[0, 2]}
          yDomain={[-2.1, 2.1]}
          xLabel="확률 비율 ρ = π새 / π옛"
          yLabel="목적 L (클수록 좋음)"
          aspect={expanded ? 2.3 : 1.65}
          maxHeight={expanded ? 480 : 360}
          xTicks={5}
          yTicks={5}
          ariaLabel="PPO 클리핑 목적 함수 그래프"
        >
          {({ x, y, inner, clip: clipId }) => (
            <g>
              <rect x={x(1 - eps)} y={inner.top} width={x(1 + eps) - x(1 - eps)} height={inner.height} fill={withAlpha(c.pos, 0.12)} />
              <text x={x(1)} y={A > 0 ? inner.top + inner.height - 8 : inner.top + 14} textAnchor="middle" style={{ fontSize: 11.5, fontWeight: 800, fill: c.pos }}>
                안전 범위 1±{eps.toFixed(2)}
              </text>
              <line x1={inner.left} x2={inner.left + inner.width} y1={y(0)} y2={y(0)} stroke="var(--viz-axis)" strokeDasharray="3 4" />
              <g clipPath={clipId}>
                <path d={linePath(pts((r) => r * A).map(([r, v]) => [x(r), y(v)]))} fill="none" stroke={c.series[3]} strokeWidth={2.5} strokeDasharray="7 6" />
                <path d={linePath(pts((r) => ppo(r, A, eps)).map(([r, v]) => [x(r), y(v)]))} fill="none" stroke={c.series[0]} strokeWidth={4.5} strokeLinejoin="round" />
              </g>
              <line x1={x(rho)} x2={x(rho)} y1={inner.top} y2={inner.top + inner.height} stroke={c.series[2]} strokeDasharray="2 4" />
              <circle cx={x(rho)} cy={y(L)} r={9} fill={isFlat ? c.neg : c.series[2]} stroke={c.surface} strokeWidth={3} />
              {!isFlat && (
                <path
                  d={`M${x(rho)} ${y(L)} L${x(rho + 0.22 * A)} ${y(ppo(rho + 0.22 * A, A, eps))}`}
                  stroke={c.series[2]}
                  strokeWidth={3}
                  markerEnd="url(#ppo-arrow)"
                />
              )}
              <defs>
                <marker id="ppo-arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                  <path d="M0 0 L10 5 L0 10z" fill={c.series[2]} />
                </marker>
              </defs>
              <text x={x(rho) + (rho > 1.5 ? -12 : 12)} y={y(L) - 14} textAnchor={rho > 1.5 ? 'end' : 'start'} style={{ fontSize: 12.5, fontWeight: 850, fill: isFlat ? c.neg : c.series[2] }}>
                {isFlat ? '기울기 0 ✋' : `기울기 ${A > 0 ? '+1' : '-1'}`}
              </text>
            </g>
          )}
        </Plot>
        <Legend
          items={[
            { color: c.series[0], label: 'PPO 목적 (잘라 낸 것)' },
            { color: c.series[3], label: '자르지 않은 목적 ρ×A', dashed: true }
          ]}
        />
        <Note mood={note.mood} tone={note.tone}>
          {note.t}
        </Note>
        <div className="vz-stats">
          <Stat label="비율 ρ" value={fmt(rho, 2)} />
          <Stat label="목적 L" value={fmt(L, 2)} tone="brand" />
          <Stat label="밀어 주는 힘(기울기)" value={isFlat ? '0' : A > 0 ? '+1' : '-1'} tone={isFlat ? 'bad' : 'good'} />
        </div>
      </div>
      <Controls>
        <Slider label="비율 ρ (새 정책이 얼마나 바뀌었나)" value={rho} min={0.1} max={1.9} step={0.01} onChange={setRho} format={(v) => v.toFixed(2)} accent="var(--peach)" />
        <Slider label="클리핑 범위 ε" value={eps} min={0.05} max={0.5} step={0.01} onChange={setEps} format={(v) => v.toFixed(2)} accent="var(--mint)" />
        <SignControl sign={sign} setSign={setSign} />
      </Controls>
    </div>
  );
}

const P0 = 0.3;
const EPOCHS = 10;

function Epochs({ sign, setSign, eps, setEps, expanded }: Shared) {
  const c = useVizColors();
  const [lr, setLr] = useState(0.4);
  const A = sign;
  const traj = useMemo(() => {
    const run = (clipped: boolean) => {
      let th = Math.log(P0 / (1 - P0));
      const ps = [P0];
      for (let e = 0; e < EPOCHS; e++) {
        const p = 1 / (1 + Math.exp(-th));
        const rho = p / P0;
        const g = clipped && flat(rho, A, eps) ? 0 : (A * p * (1 - p)) / P0;
        th += lr * g;
        ps.push(1 / (1 + Math.exp(-th)));
      }
      return ps;
    };
    return { raw: run(false), clipped: run(true) };
  }, [A, eps, lr]);
  const rawEnd = traj.raw[EPOCHS];
  const clipEnd = traj.clipped[EPOCHS];

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <Plot xDomain={[0, EPOCHS]} yDomain={[0, 1]} xLabel="같은 데이터로 다시 배운 횟수 (에포크)" yLabel="그 행동을 할 확률" aspect={expanded ? 2.3 : 1.7} maxHeight={expanded ? 460 : 340} xTicks={10} yTicks={5} yFormat={(v) => `${Math.round(v * 100)}%`} ariaLabel="클리핑 유무에 따른 정책 확률 변화">
          {({ x, y, inner }) => (
            <g>
              <rect x={inner.left} y={y(P0 * (1 + eps))} width={inner.width} height={y(P0 * (1 - eps)) - y(P0 * (1 + eps))} fill={withAlpha(c.pos, 0.14)} />
              <line x1={inner.left} x2={inner.left + inner.width} y1={y(P0)} y2={y(P0)} stroke={c.pos} strokeDasharray="4 4" />
              <text x={inner.left + inner.width - 6} y={y(P0 * (1 + eps)) - 6} textAnchor="end" style={{ fontSize: 11.5, fontWeight: 800, fill: c.pos }}>
                안전 범위 (원래 {Math.round(P0 * 100)}% ±{Math.round(eps * 100)}%)
              </text>
              {[
                { ps: traj.raw, col: c.series[3], dash: '7 6' },
                { ps: traj.clipped, col: c.series[0], dash: undefined }
              ].map((s, i) => (
                <g key={i}>
                  <path d={linePath(s.ps.map((p, e) => [x(e), y(p)]))} fill="none" stroke={s.col} strokeWidth={3.5} strokeDasharray={s.dash} />
                  {s.ps.map((p, e) => (
                    <circle key={e} cx={x(e)} cy={y(p)} r={4.5} fill={s.col} />
                  ))}
                </g>
              ))}
            </g>
          )}
        </Plot>
        <Legend
          items={[
            { color: c.series[3], label: '클리핑 없음', dashed: true },
            { color: c.series[0], label: 'PPO (클리핑)' }
          ]}
        />
        <Note mood={Math.abs(rawEnd - P0) > 0.3 ? 'wow' : 'think'} tone={Math.abs(rawEnd - P0) > 0.3 ? 'warn' : undefined}>
          {A > 0 ? '운 좋게 한 번 잘된 행동이에요.' : '운 나쁘게 한 번 망한 행동이에요.'} 클리핑이 없으면 그 한 판을 10번이나 다시 믿어서 확률이 {Math.round(P0 * 100)}% → <strong>{Math.round(rawEnd * 100)}%</strong>로 급변해요(정책 붕괴의 시작!). PPO는 안전 범위를 넘으면 기울기를 0으로 만들어 <strong>{Math.round(clipEnd * 100)}%</strong>에서 멈춰요 — 그래서 같은 데이터를 여러 번 재사용할 수 있어요.
        </Note>
        <div className="vz-stats">
          <Stat label="10번 뒤 확률 (클리핑 없음)" value={`${Math.round(rawEnd * 100)}%`} tone="bad" />
          <Stat label="10번 뒤 확률 (PPO)" value={`${Math.round(clipEnd * 100)}%`} tone="good" />
          <Stat label="PPO의 비율 ρ" value={fmt(clipEnd / P0, 2)} />
        </div>
      </div>
      <Controls>
        <Slider label="학습률" value={lr} min={0.1} max={2} step={0.05} onChange={setLr} format={(v) => v.toFixed(2)} />
        <Slider label="클리핑 범위 ε" value={eps} min={0.05} max={0.5} step={0.01} onChange={setEps} format={(v) => v.toFixed(2)} accent="var(--mint)" />
        <SignControl sign={sign} setSign={setSign} />
      </Controls>
    </div>
  );
}
