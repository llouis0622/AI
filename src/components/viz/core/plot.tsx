import { useId, type ReactNode, type PointerEvent as ReactPointerEvent } from 'react';
import { useElementSize } from './hooks';

export interface Scale {
  (v: number): number;
  invert: (px: number) => number;
  domain: [number, number];
  range: [number, number];
  ticks: (count?: number) => number[];
}

/** d3-style linear scale without the dependency. */
export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const s = ((v: number) => r0 + (v - d0) * k) as Scale;
  s.invert = (px: number) => (k === 0 ? d0 : d0 + (px - r0) / k);
  s.domain = domain;
  s.range = range;
  s.ticks = (count = 5) => niceTicks(Math.min(d0, d1), Math.max(d0, d1), count);
  return s;
}

export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const start = Math.ceil(lo / step - 1e-9) * step;
  const out: number[] = [];
  for (let v = start; v <= hi + step * 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}

export const linePath = (pts: [number, number][]) =>
  pts.length ? `M${pts.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join('L')}` : '';

export interface PlotScales {
  x: Scale;
  y: Scale;
  width: number;
  height: number;
  inner: { left: number; top: number; width: number; height: number };
  /** `clipPath` value that clips children to the plotting area. */
  clip: string;
  /** Convert a pointer event to data coordinates. */
  toData: (e: ReactPointerEvent<SVGElement>) => [number, number];
}

/**
 * Responsive SVG plot with axes, grid and labels. Children get the scales.
 *
 *   <Plot xDomain={[0, 10]} yDomain={[0, 5]} xLabel="크기" yLabel="가격">
 *     {({ x, y }) => <circle cx={x(3)} cy={y(2)} r={5} />}
 *   </Plot>
 */
export function Plot({
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  aspect = 1.6,
  minHeight = 220,
  maxHeight = 460,
  height: fixedHeight,
  margin = { top: 16, right: 18, bottom: 44, left: 52 },
  xTicks = 6,
  yTicks = 5,
  grid = true,
  xFormat = (v: number) => String(+v.toFixed(2)),
  yFormat = (v: number) => String(+v.toFixed(2)),
  onPointerDown,
  onPointerMove,
  onPointerUp,
  children,
  className = '',
  ariaLabel
}: {
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel?: ReactNode;
  yLabel?: ReactNode;
  aspect?: number;
  minHeight?: number;
  maxHeight?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  xTicks?: number;
  yTicks?: number;
  grid?: boolean;
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  onPointerDown?: (e: ReactPointerEvent<SVGSVGElement>, s: PlotScales) => void;
  onPointerMove?: (e: ReactPointerEvent<SVGSVGElement>, s: PlotScales) => void;
  onPointerUp?: (e: ReactPointerEvent<SVGSVGElement>, s: PlotScales) => void;
  children?: (s: PlotScales) => ReactNode;
  className?: string;
  ariaLabel?: string;
}) {
  const [ref, { width }] = useElementSize<HTMLDivElement>();
  const clipId = `vzclip${useId().replace(/[^a-zA-Z0-9]/g, '')}`;
  const w = Math.max(width, 10);
  const h = fixedHeight ?? Math.max(minHeight, Math.min(maxHeight, w / aspect));
  const inner = { left: margin.left, top: margin.top, width: Math.max(10, w - margin.left - margin.right), height: Math.max(10, h - margin.top - margin.bottom) };
  const x = linear(xDomain, [inner.left, inner.left + inner.width]);
  const y = linear(yDomain, [inner.top + inner.height, inner.top]);
  const scales: PlotScales = {
    x,
    y,
    width: w,
    height: h,
    inner,
    clip: `url(#${clipId})`,
    toData: (e) => {
      const r = (e.currentTarget.ownerSVGElement ?? (e.currentTarget as SVGSVGElement)).getBoundingClientRect();
      return [x.invert(e.clientX - r.left), y.invert(e.clientY - r.top)];
    }
  };
  const xt = x.ticks(xTicks);
  const yt = y.ticks(yTicks);
  return (
    <div ref={ref} className={`vz-plot ${className}`} style={{ width: '100%' }}>
      {width > 0 && (
        <svg
          className="vz-svg"
          width={w}
          height={h}
          viewBox={`0 0 ${w} ${h}`}
          role="img"
          aria-label={ariaLabel}
          onPointerDown={onPointerDown && ((e) => onPointerDown(e, scales))}
          onPointerMove={onPointerMove && ((e) => onPointerMove(e, scales))}
          onPointerUp={onPointerUp && ((e) => onPointerUp(e, scales))}
        >
          <defs>
            <clipPath id={clipId}>
              <rect x={inner.left} y={inner.top} width={inner.width} height={inner.height} />
            </clipPath>
          </defs>
          {grid && (
            <g>
              {xt.map((t) => (
                <line key={`gx${t}`} x1={x(t)} x2={x(t)} y1={inner.top} y2={inner.top + inner.height} stroke="var(--viz-grid)" />
              ))}
              {yt.map((t) => (
                <line key={`gy${t}`} x1={inner.left} x2={inner.left + inner.width} y1={y(t)} y2={y(t)} stroke="var(--viz-grid)" />
              ))}
            </g>
          )}
          <g>
            <line x1={inner.left} x2={inner.left + inner.width} y1={inner.top + inner.height} y2={inner.top + inner.height} stroke="var(--viz-axis)" strokeWidth={1.5} />
            <line x1={inner.left} x2={inner.left} y1={inner.top} y2={inner.top + inner.height} stroke="var(--viz-axis)" strokeWidth={1.5} />
            {xt.map((t) => (
              <text key={`tx${t}`} className="vz-tick" x={x(t)} y={inner.top + inner.height + 16} textAnchor="middle">
                {xFormat(t)}
              </text>
            ))}
            {yt.map((t) => (
              <text key={`ty${t}`} className="vz-tick" x={inner.left - 8} y={y(t) + 4} textAnchor="end">
                {yFormat(t)}
              </text>
            ))}
            {xLabel && (
              <text className="vz-axis-label" x={inner.left + inner.width / 2} y={h - 3} textAnchor="middle">
                {xLabel}
              </text>
            )}
            {yLabel && (
              <text className="vz-axis-label" transform={`translate(14 ${inner.top + inner.height / 2}) rotate(-90)`} textAnchor="middle">
                {yLabel}
              </text>
            )}
          </g>
          {children?.(scales)}
        </svg>
      )}
    </div>
  );
}

/** Compact line chart for "loss over steps" style readouts. */
export function MiniChart({
  series,
  height = 120,
  yMin,
  yMax,
  xLabel,
  yLabel,
  colors,
  logY = false
}: {
  series: number[][];
  height?: number;
  yMin?: number;
  yMax?: number;
  xLabel?: string;
  yLabel?: string;
  colors: string[];
  logY?: boolean;
}) {
  const all = series.flat().filter((v) => Number.isFinite(v));
  const tf = (v: number) => (logY ? Math.log10(Math.max(v, 1e-6)) : v);
  const lo = yMin ?? (all.length ? Math.min(...all.map(tf)) : 0);
  const hi = yMax ?? (all.length ? Math.max(...all.map(tf)) : 1);
  const n = Math.max(2, ...series.map((s) => s.length));
  if (Math.max(0, ...series.map((s) => s.length)) < 2) {
    return (
      <div
        style={{
          height,
          display: 'grid',
          placeItems: 'center',
          borderRadius: 12,
          border: '1.5px dashed var(--border-strong)',
          color: 'var(--text-3)',
          fontSize: 13,
          fontWeight: 650,
          textAlign: 'center',
          padding: 12
        }}
      >
        ▶ 시작하면 여기에 그래프가 그려져요
      </div>
    );
  }
  return (
    <Plot
      xDomain={[0, n - 1]}
      yDomain={[lo, hi === lo ? lo + 1 : hi]}
      height={height}
      margin={{ top: 8, right: 10, bottom: xLabel ? 38 : 22, left: 40 }}
      xTicks={4}
      yTicks={3}
      xLabel={xLabel}
      yLabel={yLabel}
      yFormat={(v) => (logY ? `1e${v.toFixed(0)}` : String(+v.toFixed(2)))}
    >
      {({ x, y }) =>
        series.map((s, i) => (
          <path
            key={i}
            d={linePath(s.map((v, k) => [x(k), y(Math.min(hi, Math.max(lo, tf(v))))] as [number, number]))}
            fill="none"
            stroke={colors[i % colors.length]}
            strokeWidth={2.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))
      }
    </Plot>
  );
}
