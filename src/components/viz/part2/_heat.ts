import type { VizColors } from '../core/colors';
import type { Pt } from './_data';

function hexToRgb(h: string): [number, number, number] {
  const m = h.trim().match(/^#([0-9a-f]{6})$/i);
  if (!m) return [128, 128, 128];
  const s = m[1];
  return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
}

/**
 * Paint a probability field p(x, y) ∈ [0, 1] on a canvas covering the unit square
 * (y up). Class 0 tint ← 0.5 → class 1 tint, plus an optional 0.5 boundary line.
 */
export function paintField(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  predict: (x: number, y: number) => number,
  c: VizColors,
  opts: { res?: number; boundary?: boolean; alpha?: number } = {}
) {
  const res = opts.res ?? 60;
  const alpha = opts.alpha ?? (c.theme === 'dark' ? 0.42 : 0.34);
  const c0 = hexToRgb(c.series[3]);
  const c1 = hexToRgb(c.series[2]);
  const cols = res;
  const rows = Math.max(8, Math.round((res * h) / w));
  const img = ctx.createImageData(cols, rows);
  const vals: number[] = new Array(cols * rows);
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const x = (i + 0.5) / cols;
      const y = 1 - (j + 0.5) / rows;
      const p = predict(x, y);
      vals[j * cols + i] = p;
      const t = Math.abs(p - 0.5) * 2;
      const base = p >= 0.5 ? c1 : c0;
      const k = (j * cols + i) * 4;
      img.data[k] = base[0];
      img.data[k + 1] = base[1];
      img.data[k + 2] = base[2];
      img.data[k + 3] = Math.round(255 * alpha * (0.25 + 0.75 * t));
    }
  }
  const off = document.createElement('canvas');
  off.width = cols;
  off.height = rows;
  off.getContext('2d')!.putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, w, h);
  ctx.restore();
  if (opts.boundary !== false) {
    // marching-squares-ish: mark cells where the class flips
    ctx.save();
    ctx.strokeStyle = c.theme === 'dark' ? 'rgba(235,240,255,0.85)' : 'rgba(42,37,64,0.75)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const cw = w / cols;
    const ch = h / rows;
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const v = vals[j * cols + i] >= 0.5;
        if (i + 1 < cols && v !== vals[j * cols + i + 1] >= 0.5) {
          ctx.moveTo((i + 1) * cw, j * ch);
          ctx.lineTo((i + 1) * cw, (j + 1) * ch);
        }
        if (j + 1 < rows && v !== vals[(j + 1) * cols + i] >= 0.5) {
          ctx.moveTo(i * cw, (j + 1) * ch);
          ctx.lineTo((i + 1) * cw, (j + 1) * ch);
        }
      }
    }
    ctx.stroke();
    ctx.restore();
  }
}

/** Draw labelled points (unit square → canvas pixels). */
export function paintPoints(ctx: CanvasRenderingContext2D, w: number, h: number, pts: Pt[], c: VizColors, opts: { r?: number; hollow?: boolean; wrong?: (p: Pt) => boolean } = {}) {
  const r = opts.r ?? 6;
  for (const p of pts) {
    const px = p.x * w;
    const py = (1 - p.y) * h;
    ctx.beginPath();
    if (p.label) {
      ctx.moveTo(px, py - r * 1.15);
      ctx.lineTo(px + r * 1.05, py + r * 0.75);
      ctx.lineTo(px - r * 1.05, py + r * 0.75);
      ctx.closePath();
    } else {
      ctx.arc(px, py, r, 0, Math.PI * 2);
    }
    const col = p.label ? c.series[2] : c.series[3];
    if (opts.hollow) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = col;
      ctx.stroke();
    } else {
      ctx.fillStyle = col;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = c.surface.startsWith('#') ? c.surface : '#ffffff';
      ctx.stroke();
    }
    if (opts.wrong?.(p)) {
      ctx.beginPath();
      ctx.arc(px, py, r + 5, 0, Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = c.neg;
      ctx.stroke();
    }
  }
}
