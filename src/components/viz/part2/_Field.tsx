import { type MouseEvent as ReactMouseEvent, type ReactNode } from 'react';
import { useCanvas } from '../core/hooks';
import { useVizColors, type VizColors } from '../core/colors';
import { paintField, paintPoints } from './_heat';
import type { Pt } from './_data';

/**
 * Unit-square canvas used by the part 2 classifiers: optional probability field,
 * labelled points, extra drawing hooks, click-to-pick and DOM overlays.
 * Clicks (not pointer-down) are used so a finger scrolling the page never adds points.
 */
export function FieldCanvas({
  height,
  points,
  predict,
  boundary = true,
  res,
  wrong,
  hollow,
  under,
  over,
  deps = [],
  onPick,
  onHover,
  ariaLabel,
  xLabel,
  yLabel,
  children
}: {
  height: number;
  points: Pt[];
  /** p(class 1) at (x, y); omit for no background field. */
  predict?: ((x: number, y: number) => number) | null;
  boundary?: boolean;
  res?: number;
  wrong?: (p: Pt) => boolean;
  /** Points drawn hollow (e.g. test data). */
  hollow?: Pt[];
  /** Drawn after the field, before the points. */
  under?: (ctx: CanvasRenderingContext2D, w: number, h: number, c: VizColors) => void;
  /** Drawn on top of everything. */
  over?: (ctx: CanvasRenderingContext2D, w: number, h: number, c: VizColors) => void;
  deps?: unknown[];
  onPick?: (x: number, y: number) => void;
  onHover?: (pt: [number, number] | null) => void;
  ariaLabel: string;
  xLabel?: ReactNode;
  yLabel?: ReactNode;
  children?: ReactNode;
}) {
  const c = useVizColors();
  const [ref] = useCanvas(
    (ctx, w, h) => {
      // faint grid so an empty field still reads as a map
      ctx.save();
      ctx.strokeStyle = c.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i < 10; i++) {
        ctx.moveTo((i / 10) * w, 0);
        ctx.lineTo((i / 10) * w, h);
        ctx.moveTo(0, (i / 10) * h);
        ctx.lineTo(w, (i / 10) * h);
      }
      ctx.stroke();
      ctx.restore();
      if (predict) paintField(ctx, w, h, predict, c, { res, boundary });
      under?.(ctx, w, h, c);
      if (hollow?.length) paintPoints(ctx, w, h, hollow, c, { hollow: true, r: Math.max(4.5, Math.min(6.5, w / 90)) });
      paintPoints(ctx, w, h, points, c, { wrong, r: Math.max(4.5, Math.min(6.5, w / 90)) });
      over?.(ctx, w, h, c);
    },
    [points, predict, hollow, c, boundary, res, ...deps]
  );

  const toUnit = (e: ReactMouseEvent<HTMLCanvasElement>): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, 1 - (e.clientY - r.top) / r.height))];
  };

  return (
    <div
      style={{
        position: 'relative',
        height,
        borderRadius: 14,
        overflow: 'hidden',
        border: '1px solid var(--border)',
        background: 'var(--surface-2)'
      }}
    >
      <canvas
        ref={ref}
        className="vz-canvas"
        role="img"
        aria-label={ariaLabel}
        style={{ cursor: onPick ? 'crosshair' : 'default', touchAction: 'pan-y' }}
        onClick={onPick ? (e) => onPick(...toUnit(e)) : undefined}
        onPointerMove={onHover ? (e) => e.pointerType === 'mouse' && onHover(toUnit(e as unknown as ReactMouseEvent<HTMLCanvasElement>)) : undefined}
        onPointerLeave={onHover ? () => onHover(null) : undefined}
      />
      {xLabel && <span style={{ ...axisChip, right: 6, bottom: 6 }}>{xLabel}</span>}
      {yLabel && <span style={{ ...axisChip, left: 6, top: 6 }}>{yLabel}</span>}
      {children}
    </div>
  );
}

const axisChip = {
  position: 'absolute' as const,
  padding: '1px 8px',
  borderRadius: 8,
  fontSize: 11.5,
  fontWeight: 750,
  color: 'var(--text-2)',
  background: 'color-mix(in srgb, var(--surface-solid) 78%, transparent)',
  pointerEvents: 'none' as const,
  whiteSpace: 'nowrap' as const
};

/** Little legend for the two classes, matching paintPoints' shapes. */
export function ClassLegend({ one = '악성', zero = '양성', extra }: { one?: ReactNode; zero?: ReactNode; extra?: ReactNode }) {
  const c = useVizColors();
  return (
    <div className="vz-legend" style={{ alignItems: 'center' }}>
      <span className="vz-legend-item">
        <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
          <path d="M7 1.5 L13 12 L1 12 Z" fill={c.series[2]} />
        </svg>
        {one}
      </span>
      <span className="vz-legend-item">
        <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
          <circle cx="7" cy="7" r="5.5" fill={c.series[3]} />
        </svg>
        {zero}
      </span>
      {extra}
    </div>
  );
}
