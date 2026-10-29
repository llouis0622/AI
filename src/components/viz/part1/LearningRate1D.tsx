import { useState } from 'react';
import type { VizProps } from '../Viz';
import { Plot, MiniChart, linePath } from '../core/plot';
import { Controls, Note, PlayBar, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { useInterval, useRafLoop, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors } from '../core/colors';
import { fmt, linspace } from '../core/math';

const W_STAR = 3;
const L0 = 0.6;
const loss = (w: number) => (w - W_STAR) ** 2 + L0;
const slope = (w: number) => 2 * (w - W_STAR);
const START = -1;
const X_DOMAIN: [number, number] = [-3, 9];
const Y_MAX = 30;

const PRESETS = [
  { value: 0.05, label: '🐢 아주 작게' },
  { value: 0.3, label: '👍 적당히' },
  { value: 0.85, label: '↔️ 크게' },
  { value: 1.05, label: '🚀 너무 크게' }
];

/** One-dimensional valley: watch the step size (learning rate) decide slow / smooth / zigzag / divergence. */
export default function LearningRate1D({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [lr, setLr] = useState(0.3);
  const [path, setPath] = useState<number[]>([START]);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [prog, setProg] = useState(1);
  const [showTangent, setShowTangent] = useState(true);

  const w = path[path.length - 1];
  const prev = path.length > 1 ? path[path.length - 2] : w;
  const diverged = Math.abs(w - W_STAR) > 9 || !Number.isFinite(w);
  const converged = !diverged && Math.abs(w - W_STAR) < 0.01;

  const step = () => {
    setPath((p) => [...p, p[p.length - 1] - lr * slope(p[p.length - 1])]);
    setProg(reduce ? 1 : 0);
  };

  useRafLoop((dt) => setProg((v) => Math.min(1, v + (dt * 2.6 * speed))), prog < 1);
  useInterval(
    () => {
      if (prog < 1) return;
      if (diverged || converged || path.length > 80) setPlaying(false);
      else step();
    },
    playing ? 120 : null
  );

  const reset = () => {
    setPlaying(false);
    setPath([START]);
    setProg(1);
  };

  // ball position along the latest hop arc
  const ax = prev + (w - prev) * prog;
  const arcLift = Math.min(8, Math.abs(w - prev) * 1.2);
  const ay = (1 - prog) ** 2 * loss(prev) + 2 * (1 - prog) * prog * (Math.max(loss(prev), loss(w)) + arcLift) + prog ** 2 * loss(w);
  const clampY = (y: number) => Math.min(Y_MAX * 1.2, y);

  const curve = linspace(X_DOMAIN[0], X_DOMAIN[1], 120).map((x) => [x, loss(x)] as [number, number]);
  const losses = path.map(loss);

  let note: { text: string; mood: 'smile' | 'happy' | 'wow' | 'think' | 'cheer'; tone?: 'good' | 'bad' | 'warn' };
  if (diverged) note = { text: '발산! 🚀 보폭이 너무 커서 골짜기를 건너뛸 때마다 반대편 벽의 더 높은 곳에 떨어졌어요. 오차가 오히려 폭발해요.', mood: 'wow', tone: 'bad' };
  else if (converged) note = { text: `바닥 도착! 🎉 ${path.length - 1}걸음 걸렸어요. 다른 학습률로도 몇 걸음 걸리는지 비교해 보세요.`, mood: 'cheer', tone: 'good' };
  else if (lr > 0.5 && path.length > 2) note = { text: '지그재그! ↔️ 보폭이 커서 바닥을 지나쳐 반대편으로 갔다가 돌아와요. 그래도 점점 바닥에 가까워지고 있어요.', mood: 'think', tone: 'warn' };
  else if (lr < 0.1 && path.length > 6) note = { text: '보폭이 너무 작아서 거북이처럼 느려요 🐢 — 발산은 안 하지만, 바닥까지 한참 걸려요.', mood: 'think' };
  else note = { text: '공은 발밑의 기울기(초록 선)만 보고, 내리막 쪽으로 "학습률 × 기울기"만큼 점프해요.', mood: 'smile' };

  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <Plot xDomain={X_DOMAIN} yDomain={[0, Y_MAX]} xLabel="w (파라미터)" yLabel="오차 (손실)" aspect={expanded ? 2.1 : 1.55} maxHeight={expanded ? 560 : 400} ariaLabel="골짜기 모양의 손실 곡선과 그 위를 뛰는 공">
          {({ x, y, clip }) => (
            <g>
              <defs>
                <linearGradient id="lr1d-fill" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor={c.series[0]} stopOpacity={0.05} />
                  <stop offset="100%" stopColor={c.series[0]} stopOpacity={0.22} />
                </linearGradient>
              </defs>
              <g clipPath={clip}>
                <path d={`${linePath(curve.map(([a, b]) => [x(a), y(b)]))}L${x(X_DOMAIN[1])},${y(0)}L${x(X_DOMAIN[0])},${y(0)}Z`} fill="url(#lr1d-fill)" />
                <path d={linePath(curve.map(([a, b]) => [x(a), y(b)]))} fill="none" stroke={c.series[0]} strokeWidth={3.5} />
                {/* hop arcs */}
                {path.slice(1).map((w1, i) => {
                  const w0 = path[i];
                  const isLast = i === path.length - 2;
                  if (isLast && prog < 1) return null;
                  const lift = Math.min(8, Math.abs(w1 - w0) * 1.2);
                  const cy = Math.max(loss(w0), loss(w1)) + lift;
                  return (
                    <path
                      key={i}
                      d={`M${x(w0)},${y(clampY(loss(w0)))} Q${x((w0 + w1) / 2)},${y(clampY(cy * 1.0))} ${x(w1)},${y(clampY(loss(w1)))}`}
                      fill="none"
                      stroke={c.series[2]}
                      strokeWidth={2}
                      strokeDasharray="5 4"
                      opacity={0.85}
                    />
                  );
                })}
                {path.slice(0, -1).map((pw, i) => (
                  <circle key={i} cx={x(pw)} cy={y(clampY(loss(pw)))} r={4} fill={c.series[2]} />
                ))}
                {/* tangent at current point */}
                {showTangent && !diverged && prog >= 1 && (
                  <g>
                    <line x1={x(w - 1.2)} x2={x(w + 1.2)} y1={y(loss(w) - slope(w) * 1.2)} y2={y(loss(w) + slope(w) * 1.2)} stroke={c.series[1]} strokeWidth={3} strokeLinecap="round" />
                    {Math.abs(slope(w)) > 0.02 && (
                      <g>
                        <line x1={x(w)} x2={x(w - lr * slope(w))} y1={y(0) - 12} y2={y(0) - 12} stroke={c.series[3]} strokeWidth={3} />
                        <path
                          d={`M${x(w - lr * slope(w))},${y(0) - 12} l${slope(w) > 0 ? 9 : -9},-6 l0,12 Z`}
                          fill={c.series[3]}
                        />
                      </g>
                    )}
                  </g>
                )}
                <line x1={x(W_STAR)} x2={x(W_STAR)} y1={y(0)} y2={y(L0)} stroke={c.pos} strokeWidth={2} strokeDasharray="3 3" />
              </g>
              <circle cx={x(Math.max(X_DOMAIN[0], Math.min(X_DOMAIN[1], ax)))} cy={y(clampY(ay))} r={11} fill={c.series[4]} stroke={c.surface} strokeWidth={3} />
              {showTangent && !diverged && prog >= 1 && Math.abs(slope(w)) > 0.02 && (
                <text x={x(w - (lr * slope(w)) / 2)} y={y(0) - 22} textAnchor="middle" style={{ fontSize: 12, fontWeight: 800, fill: c.series[3] }}>
                  한 걸음 = 학습률 × 기울기
                </text>
              )}
              <text x={x(W_STAR) + 6} y={y(0) - 6} textAnchor="start" style={{ fontSize: 11.5, fontWeight: 800, fill: c.pos }}>
                🏁 바닥
              </text>
            </g>
          )}
        </Plot>
      </div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <PlayBar
          playing={playing}
          onToggle={() => {
            if (diverged || converged) reset();
            setPlaying((p) => !p);
          }}
          onStep={diverged || converged ? undefined : step}
          onReset={reset}
          speed={speed}
          onSpeed={setSpeed}
        />
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, alignItems: 'start' }}>
          <div className="vz-stats">
            <Stat label="걸음 수" value={path.length - 1} />
            <Stat label="오차" value={diverged ? '폭발!' : fmt(loss(w), 2)} tone={diverged ? 'bad' : converged ? 'good' : 'brand'} />
            <Stat label="기울기" value={diverged ? '—' : fmt(slope(w), 2)} />
          </div>
          <div>
            <div className="vz-panel-title">걸음마다 오차</div>
            <MiniChart series={[losses.map((v) => Math.min(v, 60))]} colors={[c.series[2]]} height={120} xLabel="걸음" />
          </div>
        </div>
      </div>
      <Controls>
        <Segmented
          label="학습률 고르기"
          options={PRESETS}
          value={PRESETS.find((p) => Math.abs(p.value - lr) < 1e-9)?.value ?? -1}
          onChange={(v) => {
            setLr(v);
            reset();
          }}
        />
        <Slider label="학습률 직접 조절" value={lr} min={0.01} max={1.2} step={0.01} onChange={(v) => setLr(v)} format={(v) => v.toFixed(2)} />
        <Toggle label="기울기 선 보기" checked={showTangent} onChange={setShowTangent} />
      </Controls>
    </div>
  );
}
