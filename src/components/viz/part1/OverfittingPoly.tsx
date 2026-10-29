import { useMemo, useState } from 'react';
import type { VizProps } from '../Viz';
import { Plot, linePath } from '../core/plot';
import { Controls, Legend, Note, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { useVizColors } from '../core/colors';
import { fmt, linspace, rng } from '../core/math';

const XR: [number, number] = [-1.6, 1.6];
const truth = (x: number) => 0.6 * Math.sin(2 * x) + 0.25 * x;
const NOISE = 0.16;
const LAMBDAS = [0, 1e-4, 1e-3, 1e-2, 0.1, 1];
const LAMBDA_NAMES = ['없음', '아주 약하게', '약하게', '보통', '강하게', '아주 강하게'];
const MAX_DEG = 12;

function makeData(n: number, seed: number) {
  const r = rng(seed);
  return Array.from({ length: n }, (_, i) => {
    const x = XR[0] + ((i + 0.5) / n) * (XR[1] - XR[0]) + (r.next() - 0.5) * ((XR[1] - XR[0]) / n) * 0.8;
    return [x, truth(x) + r.normal() * NOISE] as [number, number];
  });
}

/** Least-squares polynomial fit with an L2 penalty (bias not penalised). x is scaled to [-1, 1]. */
function fitPoly(pts: [number, number][], deg: number, lambda: number): number[] {
  const m = deg + 1;
  const A = Array.from({ length: m }, () => new Array(m).fill(0));
  const b = new Array(m).fill(0);
  for (const [x, y] of pts) {
    const t = x / XR[1];
    const pw = [1];
    for (let k = 1; k < m; k++) pw.push(pw[k - 1] * t);
    for (let i = 0; i < m; i++) {
      b[i] += pw[i] * y;
      for (let j = 0; j < m; j++) A[i][j] += pw[i] * pw[j];
    }
  }
  for (let i = 1; i < m; i++) A[i][i] += lambda * pts.length + 1e-10;
  A[0][0] += 1e-10;
  // Gaussian elimination with partial pivoting
  for (let col = 0; col < m; col++) {
    let piv = col;
    for (let r = col + 1; r < m; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    [A[col], A[piv]] = [A[piv], A[col]];
    [b[col], b[piv]] = [b[piv], b[col]];
    const d = A[col][col] || 1e-12;
    for (let r = col + 1; r < m; r++) {
      const f = A[r][col] / d;
      if (!f) continue;
      for (let k = col; k < m; k++) A[r][k] -= f * A[col][k];
      b[r] -= f * b[col];
    }
  }
  const w = new Array(m).fill(0);
  for (let i = m - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < m; k++) s -= A[i][k] * w[k];
    w[i] = s / (A[i][i] || 1e-12);
  }
  return w;
}

const evalPoly = (w: number[], x: number) => {
  const t = x / XR[1];
  let s = 0;
  for (let k = w.length - 1; k >= 0; k--) s = s * t + w[k];
  return s;
};
const mse = (w: number[], pts: [number, number][]) => pts.reduce((a, [x, y]) => a + (evalPoly(w, x) - y) ** 2, 0) / pts.length;

/** Polynomial degree slider: watch underfitting → just right → overfitting, with train vs validation error. */
export default function OverfittingPoly({ expanded }: VizProps) {
  const c = useVizColors();
  const [deg, setDeg] = useState(1);
  const [nData, setNData] = useState(8);
  const [lamIdx, setLamIdx] = useState(0);
  const [showTruth, setShowTruth] = useState(false);
  const train = useMemo(() => makeData(nData, 100 + nData), [nData]);
  const val = useMemo(() => makeData(40, 7), []);
  const lambda = LAMBDAS[lamIdx];
  const w = useMemo(() => fitPoly(train, deg, lambda), [train, deg, lambda]);
  const curves = useMemo(() => {
    const tr: number[] = [];
    const va: number[] = [];
    for (let d = 0; d <= MAX_DEG; d++) {
      const wd = fitPoly(train, d, lambda);
      tr.push(mse(wd, train));
      va.push(mse(wd, val));
    }
    return { tr, va };
  }, [train, val, lambda]);
  const trE = curves.tr[deg];
  const vaE = curves.va[deg];
  const minVa = Math.min(...curves.va);
  const best = curves.va.indexOf(minVa);
  const status = deg < best && vaE > minVa * 1.3 ? 'under' : deg > best && vaE > minVa * 1.5 ? 'over' : 'good';
  const xs = linspace(XR[0], XR[1], 220);
  const logY = (v: number) => Math.log10(Math.max(1e-3, Math.min(v, 1e3)));

  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <Plot xDomain={[-1.7, 1.7]} yDomain={[-1.5, 1.5]} xLabel="입력 x" yLabel="출력 y" aspect={expanded ? 2.1 : 1.6} maxHeight={expanded ? 540 : 380} ariaLabel="점들과 다항식 곡선">
          {({ x, y, clip }) => (
            <g>
              <g clipPath={clip}>
                {showTruth && <path d={linePath(xs.map((v) => [x(v), y(truth(v))]))} fill="none" stroke={c.pos} strokeWidth={2.5} strokeDasharray="7 6" />}
                <path d={linePath(xs.map((v) => [x(v), y(Math.max(-5, Math.min(5, evalPoly(w, v))))]))} fill="none" stroke={c.series[0]} strokeWidth={4} strokeLinejoin="round" />
              </g>
              {val.map(([vx, vy], i) => (
                <circle key={`v${i}`} cx={x(vx)} cy={y(vy)} r={4} fill="none" stroke={c.series[2]} strokeWidth={1.8} opacity={0.75} />
              ))}
              {train.map(([tx, ty], i) => (
                <circle key={`t${i}`} cx={x(tx)} cy={y(ty)} r={6.5} fill={c.series[1]} stroke={c.surface} strokeWidth={2} />
              ))}
            </g>
          )}
        </Plot>
        <Legend
          items={[
            { color: c.series[1], label: '훈련 데이터 (공부한 문제)' },
            { color: c.series[2], label: '검증 데이터 (처음 보는 문제)', dashed: true },
            { color: c.series[0], label: `${deg}차 곡선 (모델)` },
            ...(showTruth ? [{ color: c.pos, label: '진짜 규칙', dashed: true }] : [])
          ]}
        />
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: 12, alignItems: 'start' }}>
          <div className="vz-stats">
            <Stat label="훈련 오차" value={fmt(trE, 3)} tone={trE < 0.03 ? 'good' : undefined} />
            <Stat label="검증 오차" value={vaE > 99 ? '99+' : fmt(vaE, 3)} tone={status === 'over' ? 'bad' : status === 'good' ? 'good' : 'warn'} />
          </div>
          <div>
            <div className="vz-panel-title">차수에 따른 오차 (U자 곡선)</div>
            <Plot
              xDomain={[0, MAX_DEG]}
              yDomain={[-3, 2]}
              height={150}
              margin={{ top: 8, right: 10, bottom: 30, left: 40 }}
              xTicks={6}
              yTicks={5}
              xLabel="차수 (모델의 힘)"
              yFormat={(v) => String(+(10 ** v).toPrecision(1))}
            >
              {({ x, y }) => (
                <g>
                  <path d={linePath(curves.tr.map((v, d) => [x(d), y(logY(v))]))} fill="none" stroke={c.series[1]} strokeWidth={2.5} />
                  <path d={linePath(curves.va.map((v, d) => [x(d), y(logY(v))]))} fill="none" stroke={c.series[2]} strokeWidth={2.5} strokeDasharray="6 4" />
                  <line x1={x(deg)} x2={x(deg)} y1={y(-3)} y2={y(2)} stroke={c.series[0]} strokeWidth={2} />
                  <circle cx={x(deg)} cy={y(logY(vaE))} r={5} fill={c.series[2]} />
                  <circle cx={x(best)} cy={y(logY(curves.va[best]))} r={4} fill="none" stroke={c.pos} strokeWidth={2} />
                </g>
              )}
            </Plot>
          </div>
        </div>
        <Note mood={status === 'good' ? 'happy' : status === 'under' ? 'think' : 'wow'} tone={status === 'good' ? 'good' : status === 'over' ? 'bad' : 'warn'}>
          {status === 'under'
            ? '과소적합! 😴 곡선이 너무 단순해서 공부한 점들조차 못 따라가요. 차수를 올려 보세요.'
            : status === 'over'
              ? `과적합! 😵 훈련 점은 거의 다 지나가지만(훈련 오차 ${fmt(trE, 3)}), 처음 보는 점에선 크게 빗나가요. 족보만 외운 학생이에요. 데이터를 늘리거나 벌점을 켜 보세요.`
              : `딱 좋아요! 👍 훈련 오차와 검증 오차가 둘 다 작아요. 이번 데이터에선 ${best}차가 검증 오차가 가장 작아요.`}
        </Note>
      </div>

      <Controls>
        <Slider label="곡선의 차수 (모델의 힘)" value={deg} min={0} max={MAX_DEG} step={1} onChange={setDeg} format={(v) => `${v}차`} />
        <Segmented
          label="데이터 양"
          options={[
            { value: 8, label: '8개' },
            { value: 16, label: '16개' },
            { value: 40, label: '40개' }
          ]}
          value={nData}
          onChange={setNData}
        />
        <Slider label="벌점 (L2 정규화)" value={lamIdx} min={0} max={LAMBDAS.length - 1} step={1} onChange={setLamIdx} format={(v) => LAMBDA_NAMES[v]} accent="var(--mint)" />
        <Toggle label="진짜 규칙 보기" checked={showTruth} onChange={setShowTruth} />
      </Controls>
    </div>
  );
}
