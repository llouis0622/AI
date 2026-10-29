import { useMemo, useState } from 'react';
import type { VizProps } from '../Viz';
import { Plot, linePath } from '../core/plot';
import { Controls, Note, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { useVizColors, withAlpha } from '../core/colors';
import { fmt, linspace } from '../core/math';

type Target = 'wave' | 'hill' | 'abs' | 'step';
const TARGETS: Record<Target, { label: string; f: (x: number) => number }> = {
  wave: { label: '물결', f: (x) => Math.sin(1.6 * x) },
  hill: { label: '언덕', f: (x) => 1.8 * Math.exp(-x * x) - 0.6 },
  abs: { label: 'V자', f: (x) => Math.abs(x) - 1.2 },
  step: { label: '계단', f: (x) => Math.tanh(4 * x) }
};

const XS = linspace(-3, 3, 241);

/** Least squares with a tiny ridge: solve (AᵀA + λI) w = Aᵀy by Gaussian elimination. */
function fit(basis: ((x: number) => number)[], f: (x: number) => number): number[] {
  const m = basis.length;
  const A = XS.map((x) => basis.map((g) => g(x)));
  const y = XS.map(f);
  const M = Array.from({ length: m }, (_, i) => Array.from({ length: m + 1 }, (_, j) => (j === m ? 0 : i === j ? 1e-6 : 0)));
  for (let r = 0; r < A.length; r++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) M[i][j] += A[r][i] * A[r][j];
      M[i][m] += A[r][i] * y[r];
    }
  }
  for (let col = 0; col < m; col++) {
    let piv = col;
    for (let r = col + 1; r < m; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col] || 1e-12;
    for (let j = col; j <= m; j++) M[col][j] /= d;
    for (let r = 0; r < m; r++) {
      if (r === col) continue;
      const k = M[r][col];
      if (k === 0) continue;
      for (let j = col; j <= m; j++) M[r][j] -= k * M[col][j];
    }
  }
  return M.map((row) => row[m]);
}

export default function ReluApprox({ expanded }: VizProps) {
  const c = useVizColors();
  const [target, setTarget] = useState<Target>('wave');
  const [n, setN] = useState(3);
  const [linearOnly, setLinearOnly] = useState(false);
  const [showPieces, setShowPieces] = useState(true);
  const f = TARGETS[target].f;

  const model = useMemo(() => {
    // knots spread over [-3, 3): the first knot at the left edge doubles as the linear part
    const knots = Array.from({ length: n }, (_, i) => -3 + (6 * i) / n);
    const act = (z: number) => (linearOnly ? z : Math.max(0, z));
    const basis = [() => 1, ...knots.map((k) => (x: number) => act(x - k))];
    const w = fit(basis, f);
    const predict = (x: number) => basis.reduce((s, g, i) => s + w[i] * g(x), 0);
    const mse = XS.reduce((s, x) => s + (predict(x) - f(x)) ** 2, 0) / XS.length;
    const pieces = knots.map((k, i) => (x: number) => w[i + 1] * act(x - k));
    return { knots, w, predict, mse, pieces };
  }, [n, f, linearOnly]);

  const good = model.mse < 0.005;
  let note: { text: string; mood: 'think' | 'cheer' | 'wow' | 'happy'; tone?: 'good' | 'warn' | 'bad' };
  if (linearOnly)
    note = { text: `ReLU를 빼면 뉴런이 ${n}개든 100개든 결과는 직선 하나뿐이에요. 직선을 더해도 직선이니까요! 비선형 함수가 "관절" 역할을 하는 이유예요.`, mood: 'wow', tone: 'bad' };
  else if (n <= 2) note = { text: '꺾인 막대기 몇 개로는 곡선을 흉내 내기 어려워요. 뉴런 수를 늘려 보세요!', mood: 'think' };
  else if (good) note = { text: `뉴런 ${n}개, 즉 직선 조각 ${n}개를 이어 붙였더니 곡선에 딱 붙었어요! 🎉 조각이 충분하면 어떤 곡선이든 흉내 낼 수 있다 — 이게 "보편 근사 정리"예요.`, mood: 'cheer', tone: 'good' };
  else note = { text: '뉴런 하나가 꺾인 직선(ReLU) 하나를 만들어요. 조각이 늘수록 목표 곡선에 가까워지는 걸 보세요.', mood: 'happy' };

  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <Plot
          xDomain={[-3, 3]}
          yDomain={[-2, 2]}
          aspect={expanded ? 2.1 : 1.7}
          maxHeight={expanded ? 560 : 380}
          xLabel="입력 x"
          yLabel="출력"
          ariaLabel="ReLU 조각들을 더해 목표 곡선을 흉내 내는 그래프"
        >
          {({ x, y, clip }) => (
            <g clipPath={clip}>
              {showPieces &&
                model.pieces.map((g, i) => (
                  <path
                    key={i}
                    d={linePath(XS.map((v) => [x(v), y(g(v))]))}
                    fill="none"
                    stroke={c.series[(i % 4) + 1]}
                    strokeOpacity={0.45}
                    strokeWidth={1.5}
                    strokeDasharray="5 4"
                  />
                ))}
              <path d={linePath(XS.map((v) => [x(v), y(f(v))]))} fill="none" stroke={c.axis} strokeWidth={7} strokeOpacity={0.35} strokeLinecap="round" />
              <path d={linePath(XS.map((v) => [x(v), y(model.predict(v))]))} fill="none" stroke={c.series[0]} strokeWidth={3.5} strokeLinejoin="round" />
              {!linearOnly &&
                model.knots.slice(1).map((k) => (
                  <circle key={k} cx={x(k)} cy={y(model.predict(k))} r={5} fill={c.surface} stroke={c.series[0]} strokeWidth={2.5} />
                ))}
            </g>
          )}
        </Plot>
      </div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 12.5, fontWeight: 700, color: 'var(--text-2)' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 22, height: 7, borderRadius: 4, background: withAlpha(c.axis.startsWith('#') ? c.axis : '#8c86a0', 0.5) }} />
            목표 곡선
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 22, height: 4, borderRadius: 4, background: c.series[0] }} />
            신경망의 답 (조각들의 합)
          </span>
          {showPieces && (
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 22, borderTop: `2px dashed ${c.series[1]}` }} />
              뉴런 하나하나의 조각
            </span>
          )}
        </div>
        <div className="vz-stats">
          <Stat label="뉴런(조각) 수" value={n} />
          <Stat label="오차" value={fmt(model.mse, 4)} tone={good ? 'good' : model.mse < 0.05 ? 'warn' : 'bad'} />
        </div>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>
      <Controls>
        <Slider label="뉴런 수" value={n} min={1} max={16} step={1} onChange={setN} format={(v) => `${v}개`} />
        <Segmented
          label="목표 곡선"
          options={(Object.keys(TARGETS) as Target[]).map((k) => ({ value: k, label: TARGETS[k].label }))}
          value={target}
          onChange={(v) => setTarget(v as Target)}
        />
        <Toggle label="조각 보기" checked={showPieces} onChange={setShowPieces} />
        <Toggle label="ReLU 빼기 (선형만)" checked={linearOnly} onChange={setLinearOnly} />
      </Controls>
    </div>
  );
}
