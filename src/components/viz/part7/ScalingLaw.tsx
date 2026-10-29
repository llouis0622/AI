import { useState } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { Plot, linePath } from '../core/plot';
import { useRafLoop } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { clamp, fmt } from '../core/math';

/*
 * Toy scaling law (simplified from the published fits so the compute-optimal
 * ratio is exactly 20 tokens per parameter, as the chapter states):
 *   L(N, D) = E + A / N^a + B / D^a,   compute C ≈ 6 · N · D
 */
const a = 0.3;
const A = 400;
const B = A * Math.pow(20, a);
const E = 1.7;

const lossND = (N: number, D: number, floor: number) => floor + A / Math.pow(N, a) + B / Math.pow(D, a);
const nOpt = (C: number) => Math.sqrt(C / 120);
const lossC = (C: number, floor: number) => lossND(nOpt(C), 20 * nOpt(C), floor);

const ABILITIES = [
  { x: 21.6, icon: '🧮', label: '세 자리 덧셈' },
  { x: 22.8, icon: '📝', label: '예시 보고 따라 하기' },
  { x: 24.0, icon: '🧠', label: '단계별 추론' },
  { x: 25.2, icon: '💻', label: '코드 도우미' }
];

const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const sup = (n: number) => String(n).split('').map((d) => (d === '-' ? '⁻' : SUP[Number(d)] ?? d)).join('');
const pow10 = (x: number) => (Number.isInteger(x) ? `10${sup(x)}` : `10${sup(Math.round(x * 10) / 10).replace('.', '·')}`);

function kor(v: number): string {
  if (v >= 1e12) return `${fmt(v / 1e12, v >= 1e13 ? 0 : 1)}조`;
  if (v >= 1e8) return `${fmt(v / 1e8, v >= 1e9 ? 0 : 1)}억`;
  if (v >= 1e4) return `${fmt(v / 1e4, v >= 1e5 ? 0 : 1)}만`;
  return fmt(v, 0);
}

export default function ScalingLaw({ expanded }: VizProps) {
  const [mode, setMode] = useState<'grow' | 'budget'>('grow');
  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <Segmented
          options={[
            { value: 'grow', label: '① 키우면 좋아진다' },
            { value: 'budget', label: '② 예산을 어떻게 나눌까' }
          ]}
          value={mode}
          onChange={(v) => setMode(v as 'grow' | 'budget')}
        />
      </div>
      {mode === 'grow' ? <Grow expanded={expanded} /> : <Budget expanded={expanded} />}
    </div>
  );
}

function Grow({ expanded }: { expanded?: boolean }) {
  const c = useVizColors();
  const [target, setTarget] = useState(19);
  const [x, setX] = useState(19);
  const [logAxes, setLogAxes] = useState(true);
  const [floor, setFloor] = useState(false);
  const fl = floor ? E : 0;

  useRafLoop((dt) => setX((v) => (Math.abs(target - v) < 0.002 ? target : v + (target - v) * Math.min(1, dt * 7))), Math.abs(target - x) > 0.001);

  const C = Math.pow(10, x);
  const L = lossC(C, fl);
  const N = nOpt(C);
  const D = 20 * N;
  const red = (lossC(C * 10, fl) - fl) / (L - fl);
  const passed = ABILITIES.filter((ab) => x >= ab.x);

  const curve: [number, number][] = [];
  for (let k = 0; k <= 160; k++) {
    const xx = 18 + (8 * k) / 160;
    curve.push([xx, lossC(Math.pow(10, xx), fl)]);
  }

  const yTicks = [0.2, 0.5, 1, 2, 5];
  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 4 }}>
        {logAxes ? (
          <Plot
            xDomain={[18, 26]}
            yDomain={[Math.log10(0.15), Math.log10(8)]}
            xLabel="계산량 (FLOP, 로그 눈금)"
            yLabel="손실 (로그 눈금)"
            aspect={expanded ? 2.1 : 1.55}
            maxHeight={expanded ? 560 : 400}
            yTicks={1}
            xTicks={8}
            grid={false}
            xFormat={(v) => pow10(v)}
            yFormat={() => ''}
            margin={{ top: 46, right: 18, bottom: 44, left: 48 }}
            ariaLabel="계산량에 따른 손실을 로그-로그 눈금으로 그린 그래프"
          >
            {({ x: sx, y: sy, inner, clip }) => (
              <g>
                {yTicks.map((t) => (
                  <g key={t}>
                    <line x1={inner.left} x2={inner.left + inner.width} y1={sy(Math.log10(t))} y2={sy(Math.log10(t))} stroke="var(--viz-grid)" />
                    <text className="vz-tick" x={inner.left - 8} y={sy(Math.log10(t)) + 4} textAnchor="end">
                      {t}
                    </text>
                  </g>
                ))}
                {[19, 20, 21, 22, 23, 24, 25].map((t) => (
                  <line key={t} x1={sx(t)} x2={sx(t)} y1={inner.top} y2={inner.top + inner.height} stroke="var(--viz-grid)" />
                ))}
                {floor && (
                  <g>
                    <line x1={inner.left} x2={inner.left + inner.width} y1={sy(Math.log10(E))} y2={sy(Math.log10(E))} stroke={c.neg} strokeDasharray="6 5" strokeWidth={1.5} />
                    <text x={inner.left + inner.width - 4} y={sy(Math.log10(E)) + 16} textAnchor="end" style={{ fontSize: 11.5, fontWeight: 750, fill: c.neg }}>
                      아무리 키워도 남는 오차
                    </text>
                  </g>
                )}
                {ABILITIES.map((ab) => {
                  const on = x >= ab.x;
                  return (
                    <g key={ab.label} opacity={on ? 1 : 0.4}>
                      <line x1={sx(ab.x)} x2={sx(ab.x)} y1={inner.top - 4} y2={inner.top + inner.height} stroke={on ? c.series[3] : 'var(--viz-axis)'} strokeDasharray="3 5" />
                      <text x={sx(ab.x)} y={inner.top - 12} textAnchor="middle" style={{ fontSize: 17 }}>
                        {ab.icon}
                      </text>
                    </g>
                  );
                })}
                <g clipPath={clip}>
                  <path d={linePath(curve.map(([xx, l]) => [sx(xx), sy(Math.log10(l))]))} fill="none" stroke={c.series[0]} strokeWidth={3.5} strokeLinecap="round" />
                </g>
                <line x1={sx(x)} x2={sx(x)} y1={sy(Math.log10(L))} y2={inner.top + inner.height} stroke={c.series[2]} strokeWidth={1.5} strokeDasharray="2 4" />
                <circle cx={sx(x)} cy={sy(Math.log10(L))} r={9} fill={c.series[2]} stroke={c.surface} strokeWidth={3} />
                <text x={sx(x) + (x > 24.5 ? -14 : 14)} y={sy(Math.log10(L)) - 12} textAnchor={x > 24.5 ? 'end' : 'start'} style={{ fontSize: 13, fontWeight: 850, fill: c.series[2] }}>
                  손실 {fmt(L, 2)}
                </text>
              </g>
            )}
          </Plot>
        ) : (
          <Plot
            xDomain={[0, 1e26]}
            yDomain={[0, 5.4]}
            xLabel="계산량 (보통 눈금)"
            yLabel="손실"
            aspect={expanded ? 2.1 : 1.55}
            maxHeight={expanded ? 560 : 400}
            xTicks={4}
            yTicks={5}
            xFormat={(v) => (v === 0 ? '0' : `${(v / 1e25).toFixed(0)}×10²⁵`)}
            margin={{ top: 20, right: 18, bottom: 44, left: 48 }}
            ariaLabel="계산량에 따른 손실을 보통 눈금으로 그린 그래프"
          >
            {({ x: sx, y: sy, clip }) => (
              <g>
                <g clipPath={clip}>
                  <path d={linePath(curve.map(([xx, l]) => [sx(Math.pow(10, xx)), sy(l)]))} fill="none" stroke={c.series[0]} strokeWidth={3.5} />
                </g>
                <circle cx={sx(C)} cy={sy(L)} r={9} fill={c.series[2]} stroke={c.surface} strokeWidth={3} />
              </g>
            )}
          </Plot>
        )}
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12, paddingTop: 4 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {ABILITIES.map((ab) => {
            const on = x >= ab.x;
            return (
              <span
                key={ab.label}
                style={{
                  padding: '4px 10px',
                  borderRadius: 99,
                  fontSize: 12.5,
                  fontWeight: 750,
                  border: `1.5px solid ${on ? c.series[3] : 'var(--border)'}`,
                  background: on ? withAlpha(c.series[3], 0.16) : 'transparent',
                  color: on ? 'var(--text)' : 'var(--text-3)',
                  transition: 'all .3s'
                }}
              >
                {ab.icon} {ab.label} {on ? '✓' : ''}
              </span>
            );
          })}
        </div>
        <Note mood={passed.length === ABILITIES.length ? 'cheer' : !logAxes ? 'think' : 'happy'} tone={passed.length === ABILITIES.length ? 'good' : undefined}>
          {!logAxes ? (
            <>보통 눈금에서는 처음에만 확 떨어지고 나머지는 평평해 보여요. <strong>로그 눈금</strong>으로 펴면 같은 비율로 꾸준히 줄어드는 <strong>직선</strong>이 보여요!</>
          ) : (
            <>
              계산량을 <strong>10배</strong> 키울 때마다 줄일 수 있는 손실이 약 <strong>{Math.round((1 - red) * 100)}%</strong>씩 줄어요. 이렇게 <strong>미리 계산할 수 있다</strong>는 게 스케일링 법칙의 힘이에요.
              {passed.length > 0 && ` 지금 규모에선 ${passed.map((p) => p.label).join(', ')}도 할 수 있게 됐어요.`}
            </>
          )}
        </Note>
        <div className="vz-stats">
          <Stat label="계산량" value={`${pow10(Math.round(x * 4) / 4)} FLOP`} />
          <Stat label="모델 크기 (파라미터)" value={`${kor(N)}개`} />
          <Stat label="데이터 (토큰)" value={`${kor(D)}개`} sub="파라미터 1개당 20개" />
          <Stat label="손실" value={fmt(L, 2)} tone="brand" />
        </div>
      </div>

      <Controls>
        <Slider label="계산량 (로그 눈금, 1칸 = 10배)" value={target} min={18} max={26} step={0.25} onChange={setTarget} format={(v) => pow10(Math.round(v * 4) / 4)} />
        <Button variant="primary" onClick={() => setTarget((t) => clamp(Math.round(t) + 1, 18, 26))} disabled={target >= 26}>
          ×10 키우기 🚀
        </Button>
        <Button variant="ghost" onClick={() => setTarget(19)}>
          ↺ 처음으로
        </Button>
        <Toggle label="로그 눈금" checked={logAxes} onChange={setLogAxes} />
        <Toggle label="줄지 않는 바닥 보기" checked={floor} onChange={setFloor} />
      </Controls>
    </div>
  );
}

const BUDGETS = [21, 22, 23];

function Budget({ expanded }: { expanded?: boolean }) {
  const c = useVizColors();
  const [cExp, setCExp] = useState(22);
  const C = Math.pow(10, cExp);
  const opt = Math.log10(nOpt(C));
  const [offset, setOffset] = useState(1.3); // log10(N) − log10(N_opt)
  const logN = opt + offset;
  const N = Math.pow(10, logN);
  const D = C / (6 * N);
  const ratio = D / N;
  const L = lossND(N, D, E);
  const Lopt = lossND(nOpt(C), 20 * nOpt(C), E);

  const curve: [number, number][] = [];
  for (let k = 0; k <= 120; k++) {
    const ln = opt - 2.2 + (4.4 * k) / 120;
    const n = Math.pow(10, ln);
    curve.push([ln, lossND(n, C / (6 * n), E)]);
  }
  const yMax = Math.max(...curve.map(([, l]) => l));
  const tilt = clamp(Math.log10(ratio / 20) * 9, -24, 24); // + : data side heavier

  let note: { t: React.ReactNode; mood: 'think' | 'cheer' | 'wow'; tone?: 'good' | 'warn' };
  if (ratio < 4) note = { t: <>모델은 큰데 데이터가 모자라요 (토큰/파라미터 {fmt(ratio, 1)}). 한때의 거대 모델들이 딱 이랬어요 — <strong>더 작은 모델을 더 많은 데이터로</strong>!</>, mood: 'wow', tone: 'warn' };
  else if (ratio > 100) note = { t: <>데이터는 넘치는데 모델이 너무 작아 다 담지 못해요 (토큰/파라미터 {fmt(ratio, 0)}). 모델을 키워 보세요.</>, mood: 'think', tone: 'warn' };
  else if (Math.abs(Math.log10(ratio / 20)) < 0.25) note = { t: <>균형 잡혔어요! ⚖️ <strong>파라미터 1개당 토큰 약 20개</strong>일 때 같은 예산으로 손실이 가장 낮아요.</>, mood: 'cheer', tone: 'good' };
  else note = { t: <>시소가 기울었어요. 같은 계산 예산(C ≈ 6 × 모델 × 데이터)이라면, 한쪽만 키우기보다 <strong>함께 키워야</strong> 손실이 가장 낮아요.</>, mood: 'think' };

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12, paddingBottom: 4 }}>
        <Plot
          xDomain={[opt - 2.2, opt + 2.2]}
          yDomain={[Lopt - 0.08, yMax + 0.05]}
          xLabel="모델 크기 (파라미터 수, 로그 눈금)"
          yLabel="손실"
          aspect={expanded ? 2.3 : 1.8}
          maxHeight={expanded ? 460 : 330}
          xTicks={5}
          yTicks={4}
          xFormat={(v) => `${kor(Math.pow(10, v))}`}
          margin={{ top: 16, right: 18, bottom: 44, left: 48 }}
          ariaLabel="같은 계산 예산에서 모델 크기에 따른 손실 곡선"
        >
          {({ x: sx, y: sy, inner, clip }) => (
            <g>
              <g clipPath={clip}>
                <path d={linePath(curve.map(([ln, l]) => [sx(ln), sy(l)]))} fill="none" stroke={c.series[0]} strokeWidth={3.5} />
              </g>
              <line x1={sx(opt)} x2={sx(opt)} y1={inner.top} y2={inner.top + inner.height} stroke={c.pos} strokeDasharray="4 4" />
              <text x={sx(opt)} y={inner.top + 12} textAnchor="middle" style={{ fontSize: 11.5, fontWeight: 800, fill: c.pos }}>
                ⚖️ 최적 (×20)
              </text>
              <circle cx={sx(logN)} cy={sy(L)} r={9} fill={c.series[2]} stroke={c.surface} strokeWidth={3} />
            </g>
          )}
        </Plot>

        {/* seesaw: model vs data */}
        <svg viewBox="0 0 400 120" className="vz-svg" role="img" aria-label={`모델과 데이터의 시소, 토큰/파라미터 비율 ${fmt(ratio, 1)}`} style={{ maxWidth: 460, margin: '0 auto' }}>
          <polygon points="200,112 186,88 214,88" fill="var(--viz-axis)" />
          <g transform={`rotate(${tilt} 200 86)`} style={{ transition: 'transform .35s cubic-bezier(.34,1.56,.64,1)' }}>
            <rect x={40} y={82} width={320} height={8} rx={4} fill={c.ink} opacity={0.75} />
            <g transform="translate(78 82)">
              <rect x={-36} y={-46} width={72} height={46} rx={12} fill={withAlpha(c.series[0], 0.2)} stroke={c.series[0]} strokeWidth={2} />
              <text x={0} y={-24} textAnchor="middle" style={{ fontSize: 18 }}>🧠</text>
              <text x={0} y={-6} textAnchor="middle" style={{ fontSize: 11, fontWeight: 800 }}>모델 {kor(N)}</text>
            </g>
            <g transform="translate(322 82)">
              <rect x={-36} y={-46} width={72} height={46} rx={12} fill={withAlpha(c.series[1], 0.2)} stroke={c.series[1]} strokeWidth={2} />
              <text x={0} y={-24} textAnchor="middle" style={{ fontSize: 18 }}>📚</text>
              <text x={0} y={-6} textAnchor="middle" style={{ fontSize: 11, fontWeight: 800 }}>데이터 {kor(D)}</text>
            </g>
          </g>
        </svg>

        <Note mood={note.mood} tone={note.tone}>
          {note.t}
        </Note>
        <div className="vz-stats">
          <Stat label="모델 크기" value={`${kor(N)}개`} />
          <Stat label="데이터" value={`${kor(D)} 토큰`} />
          <Stat label="토큰 / 파라미터" value={fmt(ratio, ratio < 10 ? 1 : 0)} tone={Math.abs(Math.log10(ratio / 20)) < 0.25 ? 'good' : 'warn'} />
          <Stat label="손실" value={fmt(L, 3)} sub={L - Lopt < 0.002 ? '최저!' : `최저보다 +${fmt(L - Lopt, 3)}`} tone="brand" />
        </div>
      </div>
      <Controls>
        <Segmented label="계산 예산" options={BUDGETS.map((b) => ({ value: b, label: `${pow10(b)} FLOP` }))} value={cExp} onChange={setCExp} />
        <Slider label="모델 크기 ← 작게 · 크게 →" value={offset} min={-2} max={2} step={0.05} onChange={setOffset} format={() => `${kor(N)}`} />
        <Button variant="ghost" onClick={() => setOffset(0)}>
          ⚖️ 최적으로
        </Button>
      </Controls>
    </div>
  );
}
