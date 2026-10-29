import { useEffect, useMemo, useState } from 'react';
import type { VizProps } from '../Viz';
import { Plot, linePath } from '../core/plot';
import { Button, Controls, Note, Segmented, Slider, Stat } from '../core/ui';
import { useRafLoop, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { clamp, linspace } from '../core/math';

/* XOR: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0 */
const POINTS: { x: number; y: number; label: 0 | 1 }[] = [
  { x: 0, y: 0, label: 0 },
  { x: 0, y: 1, label: 1 },
  { x: 1, y: 0, label: 1 },
  { x: 1, y: 1, label: 0 }
];

const relu = (v: number) => (v > 0 ? v : 0);
/** The two hidden ReLU neurons from the chapter: h1 = ReLU(x1+x2), h2 = ReLU(x1+x2-1). */
const hidden = (x: number, y: number): [number, number] => [relu(x + y), relu(x + y - 1)];
const morph = (x: number, y: number, t: number): [number, number] => {
  const [h1, h2] = hidden(x, y);
  return [x + (h1 - x) * t, y + (h2 - y) * t];
};

type Stage = 'line' | 'fold' | 'band';
const XD: [number, number] = [-0.6, 2.6];
const YD: [number, number] = [-0.6, 1.6];

export default function XorTransform({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [stage, setStage] = useState<Stage>('line');
  const [angle, setAngle] = useState(90);
  const [offset, setOffset] = useState(0.5);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);

  useRafLoop((dt) => setT((v) => Math.min(1, v + dt / 2.2)), playing);
  useEffect(() => {
    if (playing && t >= 1) setPlaying(false);
  }, [playing, t]);

  const col = (label: 0 | 1) => (label === 1 ? c.series[2] : c.series[0]);

  // single straight line in input space: cosθ·x + sinθ·y = d
  const th = (angle * Math.PI) / 180;
  const side = (x: number, y: number) => Math.cos(th) * x + Math.sin(th) * y - offset;
  const lineScore = useMemo(() => {
    const a = POINTS.filter((p) => (side(p.x, p.y) > 0 ? 1 : 0) === p.label).length;
    return Math.max(a, 4 - a);
  }, [angle, offset]); // eslint-disable-line react-hooks/exhaustive-deps
  const flipped = POINTS.filter((p) => (side(p.x, p.y) > 0 ? 1 : 0) === p.label).length < 2;
  const sideLabel = (x: number, y: number): 0 | 1 => {
    const raw = side(x, y) > 0 ? 1 : 0;
    return (flipped ? 1 - raw : raw) as 0 | 1;
  };

  // folded grid lines of the input square [-0.5, 1.5]²
  const gridT = stage === 'fold' ? t : 0;
  const grid = useMemo(() => {
    const vals = linspace(-0.5, 1.5, 9);
    const samples = linspace(-0.5, 1.5, 81);
    const lines: [number, number][][] = [];
    for (const v of vals) {
      lines.push(samples.map((s) => morph(v, s, gridT)));
      lines.push(samples.map((s) => morph(s, v, gridT)));
    }
    return lines;
  }, [gridT]);

  const score = stage === 'line' ? lineScore : stage === 'fold' ? (t >= 0.999 ? 4 : null) : 4;

  let note: { text: string; mood: 'think' | 'cheer' | 'happy' | 'wow'; tone?: 'good' | 'warn' };
  if (stage === 'line')
    note =
      lineScore >= 3
        ? { text: '3개까지는 되는데… 4개는 아무리 돌려도 안 돼요! 이게 바로 "XOR의 벽". 후보 집합(직선들) 안에 답이 없어요.', mood: 'think', tone: 'warn' }
        : { text: '각도와 위치를 바꿔서 ■끼리, ●끼리 같은 편이 되게 갈라 보세요.', mood: 'happy' };
  else if (stage === 'fold')
    note =
      t >= 0.999
        ? { text: '접힌 공간에서는 직선 하나로 깔끔하게 갈라져요! 🎉 (0,1)과 (1,0)이 한 점으로 모인 게 보이나요?', mood: 'cheer', tone: 'good' }
        : { text: '은닉층 뉴런 두 개(h₁, h₂)가 종이를 접듯 공간을 접어요. ▶ 버튼이나 슬라이더로 접어 보세요.', mood: 'wow' };
  else note = { text: '원래 공간에서 보면 신경망의 경계는 직선 두 개 — 띠 모양이에요. 은닉층이 만든 "좋은 표현" 덕분이에요.', mood: 'happy', tone: 'good' };

  const hiddenAxes = stage === 'fold' && t > 0.5;

  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <Plot
          xDomain={XD}
          yDomain={YD}
          aspect={expanded ? 2 : 1.55}
          maxHeight={expanded ? 600 : 420}
          xLabel={hiddenAxes ? 'h₁ = ReLU(x₁+x₂)' : 'x₁'}
          yLabel={hiddenAxes ? 'h₂ = ReLU(x₁+x₂-1)' : 'x₂'}
          xTicks={6}
          yTicks={4}
          ariaLabel="XOR 점 네 개와 그것을 가르는 경계"
        >
          {({ x, y, clip, inner }) => {
            const box: [number, number][] = [
              [XD[0], YD[0]],
              [XD[1], YD[0]],
              [XD[1], YD[1]],
              [XD[0], YD[1]]
            ];
            const toPath = (poly: [number, number][]) => (poly.length ? `M${poly.map(([a, b]) => `${x(a)},${y(b)}`).join('L')}Z` : '');
            const half = (f: (px: number, py: number) => number) => clipPoly(box, f);

            let regions: { d: string; fill: string }[] = [];
            let bounds: [number, number][][] = [];
            if (stage === 'line') {
              const f1 = (px: number, py: number) => (flipped ? -side(px, py) : side(px, py));
              regions = [
                { d: toPath(half(f1)), fill: withAlpha(col(1), 0.13) },
                { d: toPath(half((px, py) => -f1(px, py))), fill: withAlpha(col(0), 0.13) }
              ];
              const dir: [number, number] = [-Math.sin(th), Math.cos(th)];
              const p0: [number, number] = [Math.cos(th) * offset, Math.sin(th) * offset];
              bounds = [[[p0[0] - dir[0] * 10, p0[1] - dir[1] * 10], [p0[0] + dir[0] * 10, p0[1] + dir[1] * 10]]];
            } else if (stage === 'fold' && t >= 0.999) {
              const f = (px: number, py: number) => px - 2 * py - 0.5;
              regions = [
                { d: toPath(half(f)), fill: withAlpha(col(1), 0.14) },
                { d: toPath(half((px, py) => -f(px, py))), fill: withAlpha(col(0), 0.14) }
              ];
              bounds = [[[0.5 - 10, -5], [0.5 + 10, 5]]];
            } else if (stage === 'band') {
              const band = clipPoly(half((px, py) => px + py - 0.5), (px, py) => 1.5 - px - py);
              regions = [
                { d: toPath(box), fill: withAlpha(col(0), 0.12) },
                { d: toPath(band), fill: withAlpha(col(1), 0.22) }
              ];
              bounds = [
                [[-10, 10.5], [10.5, -10]],
                [[-10, 11.5], [11.5, -10]]
              ];
            }

            return (
              <g>
                <g clipPath={clip}>
                  {regions.map((r, k) => r.d && <path key={k} d={r.d} fill={r.fill} />)}
                  {grid.map((ln, k) => (
                    <path
                      key={k}
                      d={linePath(ln.map(([a, b]) => [x(a), y(b)]))}
                      fill="none"
                      stroke={stage === 'fold' ? c.series[3] : c.axis}
                      strokeOpacity={stage === 'fold' ? 0.5 : 0.18}
                      strokeWidth={stage === 'fold' ? 1.4 : 1}
                    />
                  ))}
                  {bounds.map((b, k) => (
                    <path
                      key={`b${k}`}
                      d={linePath(b.map(([a, bb]) => [x(a), y(bb)]))}
                      stroke={c.ink}
                      strokeWidth={3}
                      strokeDasharray={stage === 'band' ? '8 5' : undefined}
                      strokeLinecap="round"
                    />
                  ))}
                </g>
                {stage === 'fold' &&
                  t > 0.02 &&
                  POINTS.map((p, k) => {
                    // ghost of the original position + a trail to where the point is moving
                    const [px, py] = morph(p.x, p.y, t);
                    return (
                      <g key={`ghost${k}`} opacity={Math.min(1, t * 3)}>
                        <line x1={x(p.x)} y1={y(p.y)} x2={x(px)} y2={y(py)} stroke={col(p.label)} strokeWidth={1.8} strokeDasharray="3 4" />
                        {p.label === 1 ? (
                          <circle cx={x(p.x)} cy={y(p.y)} r={9} fill="none" stroke={col(1)} strokeWidth={1.8} strokeDasharray="3 3" />
                        ) : (
                          <rect x={x(p.x) - 9} y={y(p.y) - 9} width={18} height={18} rx={4} fill="none" stroke={col(0)} strokeWidth={1.8} strokeDasharray="3 3" />
                        )}
                      </g>
                    );
                  })}
                {POINTS.map((p, k) => {
                  const tt = stage === 'fold' ? t : 0;
                  let [px, py] = morph(p.x, p.y, tt);
                  // (0,1) and (1,0) land on the same spot — nudge them apart so both stay visible
                  if (p.label === 1 && tt > 0.8) py += (p.x === 0 ? 0.07 : -0.07) * ((tt - 0.8) / 0.2);
                  const cx = x(px);
                  const cy = y(py);
                  const wrong = stage === 'line' && sideLabel(p.x, p.y) !== p.label;
                  const r = 11;
                  return (
                    <g key={k}>
                      {p.label === 1 ? (
                        <circle cx={cx} cy={cy} r={r} fill={col(1)} stroke={c.surface} strokeWidth={3} />
                      ) : (
                        <rect x={cx - r} y={cy - r} width={r * 2} height={r * 2} rx={4} fill={col(0)} stroke={c.surface} strokeWidth={3} />
                      )}
                      {wrong && (
                        <text x={cx} y={cy + 5} textAnchor="middle" style={{ fontSize: 15, fontWeight: 900, fill: c.surface }}>
                          ✕
                        </text>
                      )}
                      <text
                        x={cx + (p.x === 0 ? -16 : 16)}
                        y={cy - 14}
                        textAnchor={p.x === 0 ? 'end' : 'start'}
                        style={{ fontSize: 12, fontWeight: 800, fill: 'var(--text-2)' }}
                      >
                        ({p.x},{p.y})→{p.label}
                      </text>
                    </g>
                  );
                })}
                {stage === 'fold' && t >= 0.999 && (
                  <text x={x(1)} y={y(-0.38)} textAnchor="middle" style={{ fontSize: 12, fontWeight: 800, fill: col(1) }}>
                    두 점이 한곳으로!
                  </text>
                )}
                <text x={inner.left + inner.width - 6} y={inner.top + 16} textAnchor="end" style={{ fontSize: 12, fontWeight: 800, fill: 'var(--text-3)' }}>
                  {stage === 'line' ? '입력 공간' : stage === 'fold' ? (t < 0.5 ? '입력 공간 → 접는 중…' : '은닉층 공간 (h₁, h₂)') : '입력 공간 (신경망의 경계)'}
                </text>
              </g>
            );
          }}
        </Plot>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, alignItems: 'center' }}>
          <NetDiagram stage={stage} t={t} c={c} col={col} />
          <div className="vz-stats">
            <Stat label="맞힌 점" value={score === null ? '…' : `${score} / 4`} tone={score === 4 ? 'good' : score === 3 ? 'warn' : 'brand'} />
            <Stat label="쓴 경계" value={stage === 'line' ? '직선 1개' : stage === 'fold' ? '접은 뒤 직선 1개' : '직선 2개(띠)'} />
          </div>
        </div>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        <Segmented
          label="단계"
          options={[
            { value: 'line', label: '① 직선 하나로 도전' },
            { value: 'fold', label: '② 은닉층으로 접기' },
            { value: 'band', label: '③ 원래 공간에서 보기' }
          ]}
          value={stage}
          onChange={(v) => {
            setStage(v as Stage);
            setPlaying(false);
            if (v === 'fold') setT(0);
          }}
        />
        {stage === 'line' && (
          <>
            <Slider label="직선의 각도" value={angle} min={0} max={179} step={1} onChange={setAngle} format={(v) => `${v}°`} />
            <Slider label="직선의 위치" value={offset} min={-1.2} max={1.8} step={0.01} onChange={setOffset} format={(v) => v.toFixed(2)} accent="var(--mint)" />
          </>
        )}
        {stage === 'fold' && (
          <>
            <Slider label="접는 정도" value={t} min={0} max={1} step={0.01} onChange={(v) => { setPlaying(false); setT(v); }} format={(v) => `${Math.round(v * 100)}%`} />
            <Button
              variant="primary"
              onClick={() => {
                if (reduce) {
                  setT(1);
                  return;
                }
                if (t >= 0.999) setT(0);
                setPlaying(true);
              }}
            >
              ▶ 접기 애니메이션
            </Button>
          </>
        )}
      </Controls>
    </div>
  );
}

/** Sutherland–Hodgman: keep the part of a convex polygon where f(x, y) > 0. */
function clipPoly(poly: [number, number][], f: (px: number, py: number) => number): [number, number][] {
  const out: [number, number][] = [];
  for (let k = 0; k < poly.length; k++) {
    const p = poly[k];
    const q = poly[(k + 1) % poly.length];
    const fp = f(p[0], p[1]);
    const fq = f(q[0], q[1]);
    if (fp > 0) out.push(p);
    if (fp > 0 !== fq > 0) {
      const s = fp / (fp - fq);
      out.push([p[0] + (q[0] - p[0]) * s, p[1] + (q[1] - p[1]) * s]);
    }
  }
  return out;
}

function NetDiagram({
  stage,
  t,
  c,
  col
}: {
  stage: Stage;
  t: number;
  c: ReturnType<typeof useVizColors>;
  col: (l: 0 | 1) => string;
}) {
  const hiddenOn = stage !== 'line';
  const glow = stage === 'fold' ? clamp(t * 1.5, 0, 1) : hiddenOn ? 1 : 0;
  const node = (cx: number, cy: number, label: string, on: boolean, accent: string) => (
    <g>
      <circle cx={cx} cy={cy} r={17} fill={on ? withAlpha(accent, 0.18) : 'var(--surface-3)'} stroke={on ? accent : c.border} strokeWidth={2.5} />
      <text x={cx} y={cy + 5} textAnchor="middle" style={{ fontSize: 13, fontWeight: 850, fill: on ? 'var(--text)' : 'var(--text-3)' }}>
        {label}
      </text>
    </g>
  );
  const edge = (x1: number, y1: number, x2: number, y2: number, w: string, on: boolean) => (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={on ? c.series[3] : c.border} strokeWidth={on ? 2.2 : 1.5} strokeDasharray={on ? undefined : '4 4'} />
      {on && (
        <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 4} textAnchor="middle" style={{ fontSize: 10.5, fontWeight: 800, fill: 'var(--text-2)' }}>
          {w}
        </text>
      )}
    </g>
  );
  return (
    <svg className="vz-svg" viewBox="0 0 260 130" role="img" aria-label="XOR를 푸는 작은 신경망 그림" style={{ maxWidth: 320 }}>
      {hiddenOn ? (
        <>
          {edge(32, 35, 130, 32, '1', true)}
          {edge(32, 35, 130, 98, '1', true)}
          {edge(32, 98, 130, 32, '1', true)}
          {edge(32, 98, 130, 98, '1', true)}
          {edge(130, 32, 228, 65, '+1', true)}
          {edge(130, 98, 228, 65, '-2', true)}
        </>
      ) : (
        <>
          {edge(32, 35, 228, 65, '', false)}
          {edge(32, 98, 228, 65, '', false)}
        </>
      )}
      {node(32, 35, 'x₁', true, c.series[3])}
      {node(32, 98, 'x₂', true, c.series[3])}
      <g opacity={hiddenOn ? 0.35 + 0.65 * glow : 0.35}>
        {node(130, 32, 'h₁', hiddenOn, c.series[1])}
        {node(130, 98, 'h₂', hiddenOn, c.series[1])}
      </g>
      {node(228, 65, 'ŷ', true, col(1))}
      <text x={130} y={126} textAnchor="middle" style={{ fontSize: 10.5, fontWeight: 750, fill: 'var(--text-3)' }}>
        {hiddenOn ? '은닉층: h₂는 편향 -1' : '은닉층 없음 = 직선 하나'}
      </text>
    </svg>
  );
}
