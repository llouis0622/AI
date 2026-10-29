import { useEffect, useRef, useState } from 'react';
import type { VizProps } from '../Viz';
import { Plot, linePath, type PlotScales } from '../core/plot';
import { Button, Controls, Note, Slider, Stat, Toggle } from '../core/ui';
import { useInterval } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { clamp, fmt } from '../core/math';

const SIZES = [10, 15, 20, 25, 30];
const PRICES = [3.2, 4.1, 5.5, 6.1, 7.4];
const W_STAR = 0.208;
const B_STAR = 1.1;

const mse = (w: number, b: number) => SIZES.reduce((a, x, i) => a + (w * x + b - PRICES[i]) ** 2, 0) / SIZES.length;
const BEST = mse(W_STAR, B_STAR);

/** Drag the line through the houses; residuals and their squares show the loss. */
export default function LineFitPlayground({ expanded }: VizProps) {
  const c = useVizColors();
  const [w, setW] = useState(0.05);
  const [b, setB] = useState(4.2);
  const [squares, setSquares] = useState(true);
  const [auto, setAuto] = useState<null | { steps: number }>(null);
  const [best, setBest] = useState(Infinity);
  const drag = useRef<null | 'left' | 'right'>(null);

  const L = mse(w, b);
  useEffect(() => {
    setBest((v) => Math.min(v, L));
  }, [L]);

  const yAt = (x: number) => w * x + b;
  const setFromHandles = (y10: number, y30: number) => {
    const nw = (y30 - y10) / 20;
    setW(nw);
    setB(y10 - 10 * nw);
  };

  // gradient descent in normalised coordinates (x' = (x - 20) / 10) for a quick, stable glide
  useInterval(
    () => {
      setAuto((a) => (a ? { steps: a.steps + 1 } : a));
      const W = 10 * w;
      const B = b + 20 * w;
      let gW = 0;
      let gB = 0;
      SIZES.forEach((x, i) => {
        const r = W * ((x - 20) / 10) + B - PRICES[i];
        gW += (2 * r * (x - 20)) / 10;
        gB += 2 * r;
      });
      gW /= SIZES.length;
      gB /= SIZES.length;
      const lr = 0.3;
      const nW = W - lr * gW;
      const nB = B - lr * gB;
      setW(nW / 10);
      setB(nB - 2 * nW);
      if (Math.hypot(gW, gB) < 5e-3 || (auto?.steps ?? 0) > 80) setAuto(null);
    },
    auto ? 110 : null
  );

  const pointerTo = (e: React.PointerEvent<SVGSVGElement>, s: PlotScales) => {
    if (!drag.current) return;
    const [, py] = s.toData(e);
    const y = clamp(py, 0, 10);
    if (drag.current === 'left') setFromHandles(y, yAt(30));
    else setFromHandles(yAt(10), y);
  };

  const quality = L < BEST + 0.02 ? 'good' : L < 0.5 ? 'warn' : 'bad';
  const pred18 = yAt(18);
  const meterMax = 6;

  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 4 }}>
        <Plot
          xDomain={[5, 35]}
          yDomain={[0, 10]}
          xLabel="집 크기 (평)"
          yLabel="가격 (억)"
          aspect={expanded ? 2 : 1.45}
          maxHeight={expanded ? 620 : 440}
          ariaLabel="집 크기와 가격 데이터, 그리고 끌어서 움직일 수 있는 직선"
          onPointerMove={pointerTo}
          onPointerUp={() => (drag.current = null)}
        >
          {(s) => {
            const { x, y, clip } = s;
            const handle = (side: 'left' | 'right') => {
              const hx = side === 'left' ? 10 : 30;
              return (
                <g
                  key={side}
                  style={{ cursor: 'ns-resize' }}
                  onPointerDown={(e) => {
                    setAuto(null);
                    drag.current = side;
                    (e.currentTarget.ownerSVGElement as SVGSVGElement).setPointerCapture(e.pointerId);
                  }}
                >
                  <circle cx={x(hx)} cy={y(clamp(yAt(hx), 0, 10))} r={20} fill="transparent" />
                  <circle cx={x(hx)} cy={y(clamp(yAt(hx), 0, 10))} r={11} fill={c.surface} stroke={c.series[0]} strokeWidth={4} />
                </g>
              );
            };
            return (
              <g>
                <g clipPath={clip}>
                  {squares &&
                    SIZES.map((sx, i) => {
                      const top = Math.min(y(PRICES[i]), y(yAt(sx)));
                      const side = Math.abs(y(PRICES[i]) - y(yAt(sx)));
                      return (
                        <rect
                          key={`sq${sx}`}
                          x={x(sx)}
                          y={top}
                          width={side}
                          height={side}
                          fill={withAlpha(c.neg.startsWith('#') ? c.neg : '#f2626b', 0.16)}
                          stroke={c.neg}
                          strokeWidth={1.2}
                          rx={3}
                        />
                      );
                    })}
                  {SIZES.map((sx, i) => (
                    <line key={`r${sx}`} x1={x(sx)} x2={x(sx)} y1={y(PRICES[i])} y2={y(yAt(sx))} stroke={c.neg} strokeWidth={2} strokeDasharray="4 3" />
                  ))}
                  <path d={linePath([[x(5), y(yAt(5))], [x(35), y(yAt(35))]])} stroke={c.series[0]} strokeWidth={4} strokeLinecap="round" />
                  {/* 18평 prediction */}
                  <line x1={x(18)} x2={x(18)} y1={y(0)} y2={y(clamp(pred18, 0, 10))} stroke={c.series[3]} strokeWidth={1.5} strokeDasharray="2 4" />
                  <circle cx={x(18)} cy={y(clamp(pred18, 0, 10))} r={6} fill={c.series[3]} />
                </g>
                <text x={x(18) + 6} y={y(0) - 10} style={{ fontSize: 12.5, fontWeight: 800, fill: c.series[3] }}>
                  18평 → {fmt(pred18, 1)}억?
                </text>
                {SIZES.map((sx, i) => (
                  <g key={`p${sx}`}>
                    <circle cx={x(sx)} cy={y(PRICES[i])} r={7.5} fill={c.series[1]} stroke={c.surface} strokeWidth={2.5} />
                    <text x={x(sx) - 10} y={y(PRICES[i]) - 12} textAnchor="end" style={{ fontSize: 11, fontWeight: 700 }}>
                      🏠
                    </text>
                  </g>
                ))}
                {handle('left')}
                {handle('right')}
              </g>
            );
          }}
        </Plot>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12, paddingTop: 4 }}>
        <div className="vz-stats">
          <Stat label="오차 (평균제곱오차)" value={fmt(L, 3)} tone={quality === 'good' ? 'good' : quality === 'warn' ? 'warn' : 'bad'} sub={`가장 좋은 기록 ${fmt(Math.min(best, L), 3)}`} />
          <Stat label="w (한 평당 오르는 값)" value={`${fmt(w, 3)}억`} />
          <Stat label="b (기본값)" value={`${fmt(b, 2)}억`} />
        </div>
        <div aria-hidden="true" style={{ height: 12, borderRadius: 99, background: 'var(--surface-3)', overflow: 'hidden' }}>
          <div
            style={{
              height: '100%',
              width: `${Math.min(100, (L / meterMax) * 100)}%`,
              borderRadius: 99,
              background: quality === 'good' ? c.pos : quality === 'warn' ? 'var(--warn)' : c.neg,
              transition: 'width .25s, background-color .25s'
            }}
          />
        </div>
        {auto ? (
          <Note mood="happy">기계가 경사하강법으로 직선을 조금씩 옮기는 중… {auto.steps}걸음째!</Note>
        ) : quality === 'good' ? (
          <Note mood="cheer" tone="good">
            거의 완벽해요! 🎉 지금 직선이 "한 평에 약 {fmt(w * 10000, 0)}만 원씩 오른다"는 규칙이에요. 기계가 찾는 답도 바로 이거예요.
          </Note>
        ) : (
          <Note mood="smile">
            선 위 <strong>동그란 손잡이 두 개</strong>를 위아래로 끌어서 빨간 네모(오차²)가 최대한 작아지게 맞춰 보세요. 어렵다면 🤖 버튼!
          </Note>
        )}
      </div>

      <Controls>
        <Slider label="w (기울기)" value={w} min={-0.1} max={0.5} step={0.001} onChange={(v) => { setAuto(null); setW(v); }} format={(v) => v.toFixed(3)} />
        <Slider label="b (절편)" value={b} min={-3} max={8} step={0.01} onChange={(v) => { setAuto(null); setB(v); }} format={(v) => v.toFixed(2)} accent="var(--mint)" />
        <Toggle label="오차를 네모(제곱)로 보기" checked={squares} onChange={setSquares} />
        <Button variant="primary" onClick={() => setAuto({ steps: 0 })} disabled={!!auto}>
          🤖 기계에게 맡기기
        </Button>
        <Button variant="ghost" onClick={() => { setAuto(null); setW(0.05); setB(4.2); setBest(Infinity); }}>
          ↺ 처음으로
        </Button>
      </Controls>
    </div>
  );
}
