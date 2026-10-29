/** Procedural pictures for the vision visualizations (no image files needed). */

/** A friendly cat face filling a size×size canvas. */
export function drawCat(ctx: CanvasRenderingContext2D, size: number, bg = '#cfe9ff') {
  const s = size / 64;
  ctx.save();
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, size, size);
  // ears
  ctx.fillStyle = '#f2994a';
  tri(ctx, 14 * s, 26 * s, 20 * s, 6 * s, 30 * s, 20 * s);
  tri(ctx, 50 * s, 26 * s, 44 * s, 6 * s, 34 * s, 20 * s);
  ctx.fillStyle = '#ffc6a5';
  tri(ctx, 18 * s, 22 * s, 21 * s, 11 * s, 27 * s, 19 * s);
  tri(ctx, 46 * s, 22 * s, 43 * s, 11 * s, 37 * s, 19 * s);
  // head
  ctx.fillStyle = '#f2994a';
  ctx.beginPath();
  ctx.ellipse(32 * s, 36 * s, 21 * s, 19 * s, 0, 0, Math.PI * 2);
  ctx.fill();
  // stripes
  ctx.fillStyle = '#d9772b';
  ctx.fillRect(29 * s, 18 * s, 2 * s, 7 * s);
  ctx.fillRect(33 * s, 18 * s, 2 * s, 7 * s);
  // muzzle
  ctx.fillStyle = '#fff3e8';
  ctx.beginPath();
  ctx.ellipse(32 * s, 44 * s, 10 * s, 7 * s, 0, 0, Math.PI * 2);
  ctx.fill();
  // eyes
  ctx.fillStyle = '#2a2540';
  ctx.beginPath();
  ctx.ellipse(24 * s, 34 * s, 3 * s, 4 * s, 0, 0, Math.PI * 2);
  ctx.ellipse(40 * s, 34 * s, 3 * s, 4 * s, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(25 * s, 33 * s, 1 * s, 0, Math.PI * 2);
  ctx.arc(41 * s, 33 * s, 1 * s, 0, Math.PI * 2);
  ctx.fill();
  // nose + mouth
  ctx.fillStyle = '#e8577a';
  tri(ctx, 30 * s, 41 * s, 34 * s, 41 * s, 32 * s, 43.5 * s);
  ctx.strokeStyle = '#2a2540';
  ctx.lineWidth = 1.2 * s;
  ctx.beginPath();
  ctx.moveTo(32 * s, 43.5 * s);
  ctx.quadraticCurveTo(29 * s, 47 * s, 27 * s, 45 * s);
  ctx.moveTo(32 * s, 43.5 * s);
  ctx.quadraticCurveTo(35 * s, 47 * s, 37 * s, 45 * s);
  // whiskers
  ctx.moveTo(20 * s, 43 * s);
  ctx.lineTo(8 * s, 41 * s);
  ctx.moveTo(20 * s, 45 * s);
  ctx.lineTo(8 * s, 47 * s);
  ctx.moveTo(44 * s, 43 * s);
  ctx.lineTo(56 * s, 41 * s);
  ctx.moveTo(44 * s, 45 * s);
  ctx.lineTo(56 * s, 47 * s);
  ctx.stroke();
  ctx.restore();
}

function tri(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, x3: number, y3: number) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.lineTo(x3, y3);
  ctx.closePath();
  ctx.fill();
}

/** A tiny house-with-sun landscape on a size×size canvas (used for ViT patches). */
export function drawHouse(ctx: CanvasRenderingContext2D, size: number) {
  const s = size / 96;
  const sky = ctx.createLinearGradient(0, 0, 0, size);
  sky.addColorStop(0, '#8fd3ff');
  sky.addColorStop(0.65, '#d8f1ff');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, size, size);
  // sun
  ctx.fillStyle = '#ffd24a';
  ctx.beginPath();
  ctx.arc(74 * s, 20 * s, 10 * s, 0, Math.PI * 2);
  ctx.fill();
  // hills
  ctx.fillStyle = '#7ccf7a';
  ctx.beginPath();
  ctx.ellipse(20 * s, 92 * s, 50 * s, 26 * s, 0, 0, Math.PI * 2);
  ctx.ellipse(80 * s, 96 * s, 46 * s, 24 * s, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#5fb85d';
  ctx.fillRect(0, 80 * s, size, 16 * s);
  // house body
  ctx.fillStyle = '#fff4e0';
  ctx.fillRect(22 * s, 50 * s, 34 * s, 30 * s);
  // roof
  ctx.fillStyle = '#e8577a';
  tri(ctx, 18 * s, 52 * s, 39 * s, 30 * s, 60 * s, 52 * s);
  // door + window
  ctx.fillStyle = '#8a5a3b';
  ctx.fillRect(34 * s, 62 * s, 9 * s, 18 * s);
  ctx.fillStyle = '#6fb8ff';
  ctx.fillRect(25 * s, 57 * s, 7 * s, 7 * s);
  ctx.fillRect(46 * s, 57 * s, 7 * s, 7 * s);
  // tree
  ctx.fillStyle = '#8a5a3b';
  ctx.fillRect(70 * s, 58 * s, 5 * s, 22 * s);
  ctx.fillStyle = '#3fa45b';
  ctx.beginPath();
  ctx.arc(72.5 * s, 52 * s, 11 * s, 0, Math.PI * 2);
  ctx.fill();
  // cloud
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(22 * s, 20 * s, 7 * s, 0, Math.PI * 2);
  ctx.arc(31 * s, 17 * s, 9 * s, 0, Math.PI * 2);
  ctx.arc(40 * s, 21 * s, 6 * s, 0, Math.PI * 2);
  ctx.fill();
}

/** Grayscale (0..1) of a canvas region. */
export function toGray(ctx: CanvasRenderingContext2D, size: number): Float32Array {
  const d = ctx.getImageData(0, 0, size, size).data;
  const g = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) g[i] = (0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]) / 255;
  return g;
}

/** 3×3 correlation with zero padding. */
export function conv3(src: Float32Array, n: number, k: number[]): Float32Array {
  const out = new Float32Array(n * n);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      let s = 0;
      for (let v = -1; v <= 1; v++)
        for (let u = -1; u <= 1; u++) {
          const yy = y + v;
          const xx = x + u;
          if (yy < 0 || xx < 0 || yy >= n || xx >= n) continue;
          s += src[yy * n + xx] * k[(v + 1) * 3 + (u + 1)];
        }
      out[y * n + x] = s;
    }
  return out;
}

/** 2×2 max pooling. */
export function maxPool(src: Float32Array, n: number): Float32Array {
  const m = n / 2;
  const out = new Float32Array(m * m);
  for (let y = 0; y < m; y++)
    for (let x = 0; x < m; x++)
      out[y * m + x] = Math.max(src[2 * y * n + 2 * x], src[2 * y * n + 2 * x + 1], src[(2 * y + 1) * n + 2 * x], src[(2 * y + 1) * n + 2 * x + 1]);
  return out;
}

/** Paint a single-channel map onto a canvas with a two-color ramp (values normalised to [0, 1]). */
export function paintMap(map: Float32Array, n: number, lo: [number, number, number], hi: [number, number, number], px = 4): HTMLCanvasElement {
  let mn = Infinity;
  let mx = -Infinity;
  for (const v of map) {
    mn = Math.min(mn, v);
    mx = Math.max(mx, v);
  }
  const cv = document.createElement('canvas');
  cv.width = n * px;
  cv.height = n * px;
  const ctx = cv.getContext('2d')!;
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const t = mx > mn ? (map[y * n + x] - mn) / (mx - mn) : 0;
      const c = lo.map((v, i) => Math.round(v + (hi[i] - v) * t));
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fillRect(x * px, y * px, px, px);
    }
  return cv;
}

export function hexToRgb(hex: string): [number, number, number] {
  const m = hex.trim().match(/^#([0-9a-f]{6})$/i);
  if (!m) return [128, 128, 128];
  return [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)];
}

/** Filled ellipse. */
export function ell(ctx: CanvasRenderingContext2D, x: number, y: number, rx: number, ry: number, rot = 0) {
  ctx.beginPath();
  ctx.ellipse(x, y, rx, ry, rot, 0, Math.PI * 2);
  ctx.fill();
}

/** A friendly puppy face filling a size×size square (transparent background). */
export function drawDog(ctx: CanvasRenderingContext2D, size: number) {
  ctx.save();
  ctx.scale(size / 64, size / 64);
  ctx.fillStyle = '#7a4b2c';
  ell(ctx, 14, 31, 7.5, 15, 0.28);
  ell(ctx, 50, 31, 7.5, 15, -0.28);
  ctx.fillStyle = '#c98b4f';
  ell(ctx, 32, 33, 18, 19.5);
  ctx.fillStyle = '#f3dcc0';
  ell(ctx, 32, 22, 4.5, 8);
  ell(ctx, 32, 43.5, 11.5, 9);
  ctx.fillStyle = '#2a2540';
  ell(ctx, 24.5, 31, 3, 3.3);
  ell(ctx, 39.5, 31, 3, 3.3);
  ctx.fillStyle = '#ffffff';
  ell(ctx, 25.4, 30, 1, 1);
  ell(ctx, 40.4, 30, 1, 1);
  ctx.fillStyle = '#f5679b';
  ell(ctx, 32, 50.5, 3.6, 4.2);
  ctx.fillStyle = '#2a2540';
  ell(ctx, 32, 39.5, 4.4, 3.1);
  ctx.strokeStyle = '#2a2540';
  ctx.lineWidth = 1.3;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(32, 42.5);
  ctx.lineTo(32, 46);
  ctx.quadraticCurveTo(28.5, 49, 25.5, 46.5);
  ctx.moveTo(32, 46);
  ctx.quadraticCurveTo(35.5, 49, 38.5, 46.5);
  ctx.stroke();
  ctx.restore();
}

function pentagon(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, rot: number) {
  const pts: [number, number][] = [];
  for (let k = 0; k < 5; k++) pts.push([cx + r * Math.cos(rot + (k * 2 * Math.PI) / 5), cy + r * Math.sin(rot + (k * 2 * Math.PI) / 5)]);
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.fill();
  return pts;
}

/** A soccer ball filling a size×size square (transparent background). */
export function drawBall(ctx: CanvasRenderingContext2D, size: number) {
  ctx.save();
  ctx.scale(size / 64, size / 64);
  ctx.fillStyle = '#ffffff';
  ell(ctx, 32, 32, 28, 28);
  ctx.save();
  ctx.beginPath();
  ctx.arc(32, 32, 28, 0, Math.PI * 2);
  ctx.clip();
  ctx.fillStyle = '#2a2540';
  ctx.strokeStyle = '#2a2540';
  ctx.lineWidth = 1.4;
  const inner = pentagon(ctx, 32, 32, 9, -Math.PI / 2);
  const outer: [number, number][][] = [];
  for (let k = 0; k < 5; k++) {
    const a = -Math.PI / 2 + (k * 2 * Math.PI) / 5;
    outer.push(pentagon(ctx, 32 + 25 * Math.cos(a), 32 + 25 * Math.sin(a), 9, a + Math.PI));
    ctx.beginPath();
    ctx.moveTo(inner[k][0], inner[k][1]);
    ctx.lineTo(32 + 16 * Math.cos(a), 32 + 16 * Math.sin(a));
    ctx.stroke();
  }
  // seams between neighbouring rim pentagons (closest vertex pair)
  for (let k = 0; k < 5; k++) {
    const A = outer[k];
    const B = outer[(k + 1) % 5];
    let best: [number, number, number, number] = [0, 0, 0, 0];
    let bd = Infinity;
    for (const p of A)
      for (const q of B) {
        const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2;
        if (d < bd) {
          bd = d;
          best = [p[0], p[1], q[0], q[1]];
        }
      }
    ctx.beginPath();
    ctx.moveTo(best[0], best[1]);
    ctx.lineTo(best[2], best[3]);
    ctx.stroke();
  }
  ctx.restore();
  ctx.strokeStyle = 'rgba(42,37,64,0.45)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(32, 32, 28, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ell(ctx, 22, 19, 7, 4, -0.6);
  ctx.restore();
}

/** Sunny park backdrop (sky, sun, cloud, hills, lawn) scaled to w×h; designed on a 320×200 grid. */
export function drawPark(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.save();
  ctx.scale(w / 320, h / 200);
  const sky = ctx.createLinearGradient(0, 0, 0, 200);
  sky.addColorStop(0, '#bfe6ff');
  sky.addColorStop(0.62, '#eef9ff');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, 320, 200);
  ctx.fillStyle = '#ffe07a';
  ell(ctx, 34, 30, 15, 15);
  ctx.fillStyle = '#ffffff';
  ell(ctx, 150, 30, 14, 8);
  ell(ctx, 164, 26, 12, 9);
  ell(ctx, 177, 31, 11, 7);
  ctx.fillStyle = '#c6e8b4';
  ell(ctx, 70, 140, 140, 34);
  ell(ctx, 270, 138, 120, 30);
  ctx.fillStyle = '#9fd88a';
  ctx.fillRect(0, 128, 320, 72);
  ctx.fillStyle = 'rgba(255,255,255,0.13)';
  for (let x = 0; x < 320; x += 40) ctx.fillRect(x, 128, 20, 72);
  ctx.restore();
}
