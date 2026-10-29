import { useEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Toggle, VizLayout } from '../core/ui';
import { useCanvas, useElementSize, usePrefersReducedMotion, useRafLoop } from '../core/hooks';
import { useVizColors, withAlpha, type VizColors } from '../core/colors';
import { clamp, softmax } from '../core/math';
import { drawBall, drawCat, drawDog, drawPark, ell } from './_draw';

/* One photo, three questions: classification → detection → segmentation. */

const W = 320;
const H = 200;
/** Label-map grid for segmentation (one cell = 5×5 scene units = one "pixel"). */
const GW = 64;
const GH = 40;
const FONT = 'Pretendard Variable, Pretendard, system-ui, sans-serif';

type Mode = 'cls' | 'det' | 'seg';
type Kind = 'cat' | 'dog' | 'ball';
interface Obj {
  kind: Kind;
  x: number;
  y: number;
  s: number;
}

const KIND_ORDER: Kind[] = ['cat', 'dog', 'ball'];
/** box = visible extent inside the object's square, as fractions [x0, y0, x1, y1]. */
const KINDS: Record<Kind, { name: string; emoji: string; base: number; box: [number, number, number, number] }> = {
  cat: { name: '고양이', emoji: '🐱', base: 0.97, box: [10 / 64, 6 / 64, 54 / 64, 55 / 64] },
  dog: { name: '강아지', emoji: '🐶', base: 0.94, box: [6 / 64, 13 / 64, 58 / 64, 55 / 64] },
  ball: { name: '공', emoji: '⚽', base: 0.91, box: [4 / 64, 4 / 64, 60 / 64, 60 / 64] }
};

const START: Obj[] = [
  { kind: 'ball', x: 276, y: 162, s: 50 },
  { kind: 'cat', x: 92, y: 112, s: 100 },
  { kind: 'dog', x: 204, y: 104, s: 104 }
];

const RUNGS: { mode: Mode; floor: number; name: string; emoji: string; q: string; out: string }[] = [
  { mode: 'cls', floor: 1, name: '분류', emoji: '🏷️', q: '무엇이 있나?', out: '클래스 하나' },
  { mode: 'det', floor: 2, name: '탐지', emoji: '📦', q: '무엇이 어디에?', out: '상자 + 클래스' },
  { mode: 'seg', floor: 3, name: '분할', emoji: '🎨', q: '어떤 픽셀이 무엇?', out: '픽셀마다 클래스' }
];

/** Rough human labelling time for one photo like this (log-scale bars). */
const COST: Record<Mode, { time: string; sec: number; what: string }> = {
  cls: { time: '약 1초', sec: 1, what: '단어 하나 고르기' },
  det: { time: '30초쯤', sec: 30, what: '물체마다 상자 긋기' },
  seg: { time: '수십 분', sec: 1800, what: '픽셀마다 칠하기' }
};

/* ---------- drawing the "photo" ---------- */

function drawObj(ctx: CanvasRenderingContext2D, o: Obj) {
  ctx.save();
  ctx.translate(o.x - o.s / 2, o.y - o.s / 2);
  if (o.kind === 'cat') drawCat(ctx, o.s, 'rgba(0,0,0,0)');
  else if (o.kind === 'dog') drawDog(ctx, o.s);
  else drawBall(ctx, o.s);
  ctx.restore();
}

function drawShadow(ctx: CanvasRenderingContext2D, o: Obj) {
  ctx.fillStyle = 'rgba(30,60,30,0.16)';
  ell(ctx, o.x, o.y + o.s * 0.44, o.s * 0.34, o.s * 0.06);
}

/* ---------- label map (segmentation ground truth) ---------- */

let maskCanvas: HTMLCanvasElement | null = null;

/** Cells covered by one object, rendered at label-map resolution. */
function objectCells(o: Obj): Uint8Array {
  if (!maskCanvas) {
    maskCanvas = document.createElement('canvas');
    maskCanvas.width = GW;
    maskCanvas.height = GH;
  }
  const ctx = maskCanvas.getContext('2d', { willReadFrequently: true })!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, GW, GH);
  ctx.setTransform(GW / W, 0, 0, GH / H, 0, 0);
  drawObj(ctx, o);
  const d = ctx.getImageData(0, 0, GW, GH).data;
  const m = new Uint8Array(GW * GH);
  for (let i = 0; i < m.length; i++) m[i] = d[i * 4 + 3] > 110 ? 1 : 0;
  return m;
}

function labelMap(objs: Obj[]) {
  const map = new Uint8Array(GW * GH);
  const full: Record<Kind, number> = { cat: 0, dog: 0, ball: 0 };
  for (const o of objs) {
    const m = objectCells(o);
    const id = KIND_ORDER.indexOf(o.kind) + 1;
    for (let i = 0; i < m.length; i++)
      if (m[i]) {
        map[i] = id;
        full[o.kind]++;
      }
  }
  const vis: Record<Kind, number> = { cat: 0, dog: 0, ball: 0 };
  let bg = 0;
  for (const v of map) {
    if (v) vis[KIND_ORDER[v - 1]]++;
    else bg++;
  }
  return { map, full, vis, bg };
}

function boxOf(o: Obj): [number, number, number, number] {
  const [a, b, cc, d] = KINDS[o.kind].box;
  const x0 = o.x - o.s / 2;
  const y0 = o.y - o.s / 2;
  return [Math.max(0, x0 + a * o.s), Math.max(0, y0 + b * o.s), Math.min(W, x0 + cc * o.s), Math.min(H, y0 + d * o.s)];
}

const easeOut = (t: number) => 1 - (1 - t) ** 3;

export default function VisionTaskLadder(_props: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [mode, setMode] = useState<Mode>('cls');
  const [objs, setObjs] = useState<Obj[]>(START);
  const [hover, setHover] = useState<number | null>(null);
  const [mapOnly, setMapOnly] = useState(false);
  const [anim, setAnim] = useState(1);
  const [moved, setMoved] = useState(false);
  const drag = useRef<{ kind: Kind; dx: number; dy: number } | null>(null);
  const [box, { width }] = useElementSize<HTMLDivElement>();

  useEffect(() => setAnim(reduce ? 1 : 0), [mode, reduce]);
  useRafLoop((dt) => setAnim((a) => Math.min(1, a + dt / 0.9)), anim < 1);

  const colorOf: Record<Kind, string> = { cat: c.series[0], dog: c.series[1], ball: c.series[5] };
  const lab = useMemo(() => labelMap(objs), [objs]);
  const probs = useMemo(() => softmax(KIND_ORDER.map((k) => 2 * Math.log(lab.vis[k] + 1))), [lab]);
  const det = useMemo(() => {
    const out = {} as Record<Kind, { found: boolean; score: number; frac: number; box: [number, number, number, number] }>;
    for (const k of KIND_ORDER) {
      const o = objs.find((q) => q.kind === k)!;
      const frac = lab.full[k] ? lab.vis[k] / lab.full[k] : 0;
      out[k] = { found: frac > 0.15, score: KINDS[k].base * (0.55 + 0.45 * frac), frac, box: boxOf(o) };
    }
    return out;
  }, [objs, lab]);
  const top = KIND_ORDER[probs.indexOf(Math.max(...probs))];
  const total = GW * GH;
  const bgPct = Math.round((lab.bg / total) * 100);
  const nFound = KIND_ORDER.filter((k) => det[k].found).length;

  const [cv] = useCanvas(
    (ctx, w, h) => {
      const kx = w / W;
      const ky = h / H;
      ctx.save();
      ctx.scale(kx, ky);
      if (mode === 'seg' && mapOnly) {
        ctx.fillStyle = c.surface;
        ctx.fillRect(0, 0, W, H);
      } else {
        drawPark(ctx, W, H);
        for (const o of objs) drawShadow(ctx, o);
        for (const o of objs) drawObj(ctx, o);
      }
      ctx.restore();
      const t = easeOut(anim);

      if (mode === 'cls') {
        const label = `이 사진은 → ${KINDS[top].name} (${Math.round(probs[KIND_ORDER.indexOf(top)] * 100)}%)`;
        ctx.font = `800 14px ${FONT}`;
        const tw = ctx.measureText(label).width;
        ctx.globalAlpha = t;
        ctx.fillStyle = withAlpha(c.surface, 0.92);
        ctx.strokeStyle = colorOf[top];
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.roundRect(10, 10, tw + 24, 30, 10);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = c.ink;
        ctx.fillText(label, 22, 30);
        ctx.globalAlpha = 1;
      }

      if (mode === 'det') {
        for (const k of KIND_ORDER) {
          const d = det[k];
          if (!d.found) continue;
          const [x0, y0, x1, y1] = d.box;
          const cx = ((x0 + x1) / 2) * kx;
          const cy = ((y0 + y1) / 2) * ky;
          const bx0 = cx + (x0 * kx - cx) * t;
          const by0 = cy + (y0 * ky - cy) * t;
          const bw = (x1 - x0) * kx * t;
          const bh = (y1 - y0) * ky * t;
          ctx.fillStyle = withAlpha(colorOf[k], 0.1);
          ctx.fillRect(bx0, by0, bw, bh);
          ctx.strokeStyle = colorOf[k];
          ctx.lineWidth = 3;
          ctx.strokeRect(bx0, by0, bw, bh);
          if (t > 0.6) {
            const label = `${KINDS[k].name} ${d.score.toFixed(2)}`;
            ctx.font = `800 12px ${FONT}`;
            const tw = ctx.measureText(label).width + 12;
            const ly = by0 - 20 < 0 ? by0 + 2 : by0 - 20;
            const lx = Math.min(bx0 - 1.5, w - tw - 2);
            ctx.globalAlpha = (t - 0.6) / 0.4;
            ctx.fillStyle = colorOf[k];
            ctx.beginPath();
            ctx.roundRect(lx, ly, tw, 19, 5);
            ctx.fill();
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, lx + 6, ly + 14);
            ctx.globalAlpha = 1;
          }
        }
      }

      if (mode === 'seg') {
        const cw = w / GW;
        const ch = h / GH;
        const rows = Math.ceil(t * GH);
        for (let gy = 0; gy < rows; gy++)
          for (let gx = 0; gx < GW; gx++) {
            const v = lab.map[gy * GW + gx];
            if (!v) {
              if (mapOnly) {
                ctx.fillStyle = withAlpha(c.text, 0.07);
                ctx.fillRect(gx * cw, gy * ch, cw, ch);
              }
              continue;
            }
            ctx.fillStyle = withAlpha(colorOf[KIND_ORDER[v - 1]], mapOnly ? 0.92 : 0.6);
            ctx.fillRect(gx * cw, gy * ch, cw, ch);
          }
        ctx.strokeStyle = mapOnly ? withAlpha(c.text, 0.12) : 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let gx = 1; gx < GW; gx++) {
          ctx.moveTo(Math.round(gx * cw) + 0.5, 0);
          ctx.lineTo(Math.round(gx * cw) + 0.5, h);
        }
        for (let gy = 1; gy < GH; gy++) {
          ctx.moveTo(0, Math.round(gy * ch) + 0.5);
          ctx.lineTo(w, Math.round(gy * ch) + 0.5);
        }
        ctx.stroke();
        if (t < 1) {
          ctx.fillStyle = c.brand;
          ctx.fillRect(0, rows * ch - 2, w, 3);
        }
        if (hover !== null && t >= 1) {
          const gx = hover % GW;
          const gy = Math.floor(hover / GW);
          ctx.strokeStyle = c.ink;
          ctx.lineWidth = 2.5;
          ctx.strokeRect(gx * cw, gy * ch, cw, ch);
        }
      }
    },
    [mode, objs, lab, hover, mapOnly, anim, c, top, probs, det]
  );

  const toScene = (e: RPointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H] as const;
  };
  const hit = (sx: number, sy: number) => {
    for (let i = objs.length - 1; i >= 0; i--) {
      const [x0, y0, x1, y1] = boxOf(objs[i]);
      if (sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1) return objs[i];
    }
    return null;
  };
  const onDown = (e: RPointerEvent<HTMLCanvasElement>) => {
    const [sx, sy] = toScene(e);
    const o = hit(sx, sy);
    if (!o) return;
    drag.current = { kind: o.kind, dx: sx - o.x, dy: sy - o.y };
    e.currentTarget.setPointerCapture(e.pointerId);
    e.currentTarget.style.cursor = 'grabbing';
    setObjs((os) => [...os.filter((q) => q.kind !== o.kind), o]);
  };
  const onMove = (e: RPointerEvent<HTMLCanvasElement>) => {
    const [sx, sy] = toScene(e);
    const d = drag.current;
    if (d) {
      setMoved(true);
      setObjs((os) => os.map((o) => (o.kind === d.kind ? { ...o, x: clamp(sx - d.dx, 8, W - 8), y: clamp(sy - d.dy, 8, H - 8) } : o)));
      return;
    }
    e.currentTarget.style.cursor = hit(sx, sy) ? 'grab' : 'default';
    if (mode === 'seg') {
      const gx = clamp(Math.floor((sx / W) * GW), 0, GW - 1);
      const gy = clamp(Math.floor((sy / H) * GH), 0, GH - 1);
      setHover(gy * GW + gx);
    }
  };
  const onUp = (e: RPointerEvent<HTMLCanvasElement>) => {
    drag.current = null;
    e.currentTarget.style.cursor = 'grab';
  };

  const hoverLabel = hover === null ? null : lab.map[hover];
  const narrow = width > 0 && width < 560;

  /* ---------- side panel ---------- */
  const bar = (frac: number, color: string, label: string, right: string, key: string) => (
    <div key={key} style={{ display: 'grid', gridTemplateColumns: '64px 1fr 48px', alignItems: 'center', gap: 8, fontSize: 13, fontWeight: 700 }}>
      <span>{label}</span>
      <span style={{ height: 12, borderRadius: 6, background: 'var(--surface-3)', overflow: 'hidden' }}>
        <span style={{ display: 'block', height: '100%', width: `${clamp(frac, 0, 1) * 100}%`, background: color, borderRadius: 6, transition: 'width .35s' }} />
      </span>
      <span style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{right}</span>
    </div>
  );

  let output: React.ReactNode;
  let numbers: string;
  if (mode === 'cls') {
    output = (
      <>
        <div style={{ fontSize: 22, fontWeight: 900, color: colorOf[top] }}>
          {KINDS[top].emoji} {KINDS[top].name}
        </div>
        {KIND_ORDER.map((k, i) => bar(probs[i] * anim, colorOf[k], `${KINDS[k].emoji} ${KINDS[k].name}`, `${Math.round(probs[i] * 100)}%`, k))}
      </>
    );
    numbers = `${KIND_ORDER.length}개 (클래스별 확률)`;
  } else if (mode === 'det') {
    output = (
      <div style={{ display: 'grid', gap: 6, fontSize: 12.5 }}>
        {KIND_ORDER.map((k) => {
          const d = det[k];
          const [x0, y0, x1, y1] = d.box;
          return (
            <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px', borderRadius: 10, border: `1.5px solid ${d.found ? colorOf[k] : 'var(--border)'}`, opacity: d.found ? 1 : 0.6 }}>
              <span style={{ fontWeight: 850, minWidth: 64 }}>
                {KINDS[k].emoji} {KINDS[k].name}
              </span>
              {d.found ? (
                <>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--text-2)', flex: 1 }}>
                    ({Math.round(x0)}, {Math.round(y0)}, {Math.round(x1 - x0)}×{Math.round(y1 - y0)})
                  </span>
                  <span style={{ fontWeight: 900, color: colorOf[k] }}>{d.score.toFixed(2)}</span>
                </>
              ) : (
                <span style={{ color: 'var(--text-3)' }}>너무 가려져서 못 찾음</span>
              )}
            </div>
          );
        })}
        <div style={{ fontSize: 11.5, color: 'var(--text-3)' }}>(x, y, 너비×높이) · 확신도</div>
      </div>
    );
    numbers = `${nFound * 6}개 (상자마다 좌표 4 + 클래스 + 확신도)`;
  } else {
    output = (
      <>
        {KIND_ORDER.map((k) => bar(lab.vis[k] / total, colorOf[k], `${KINDS[k].emoji} ${KINDS[k].name}`, `${Math.round((lab.vis[k] / total) * 100)}%`, k))}
        {bar(lab.bg / total, withAlpha(c.text, 0.45), '배경', `${bgPct}%`, 'bg')}
        <div style={{ fontSize: 12.5, minHeight: 20, color: 'var(--text-2)' }}>
          {hover !== null ? (
            <>
              커서 아래 칸 → <b style={{ color: hoverLabel ? colorOf[KIND_ORDER[hoverLabel - 1]] : 'var(--text-2)' }}>{hoverLabel ? KINDS[KIND_ORDER[hoverLabel - 1]].name : '배경'}</b>
            </>
          ) : (
            '그림 위에 커서를 올려 칸마다 붙은 이름표를 확인해 보세요.'
          )}
        </div>
      </>
    );
    numbers = `${total.toLocaleString()}개 (${GW}×${GH}칸마다 하나)`;
  }

  const side = (
    <>
      <div className="vz-panel">
        <div className="vz-panel-title">모델의 출력</div>
        {output}
        <div style={{ fontSize: 12.5, color: 'var(--text-2)' }}>
          출력 숫자 개수: <b>{numbers}</b>
        </div>
      </div>
      <div className="vz-panel">
        <div className="vz-panel-title">사진 한 장 라벨 달기 (사람 손, 대략)</div>
        {RUNGS.map((r) => {
          const on = r.mode === mode;
          const frac = Math.log10(COST[r.mode].sec + 1) / Math.log10(3601);
          return (
            <div key={r.mode} style={{ display: 'grid', gap: 3, opacity: on ? 1 : 0.55 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, fontWeight: on ? 850 : 650 }}>
                <span>
                  {r.emoji} {r.name} · {COST[r.mode].what}
                </span>
                <span>{COST[r.mode].time}</span>
              </div>
              <span style={{ height: 9, borderRadius: 5, background: 'var(--surface-3)', overflow: 'hidden' }}>
                <span style={{ display: 'block', height: '100%', width: `${frac * 100}%`, background: on ? c.series[2] : c.axis, borderRadius: 5 }} />
              </span>
            </div>
          );
        })}
        <div style={{ fontSize: 11.5, color: 'var(--text-3)' }}>막대는 로그 눈금 — 한 칸 오를 때마다 수십 배씩 비싸져요.</div>
      </div>
    </>
  );

  let note: { text: React.ReactNode; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' };
  if (mode === 'cls')
    note = {
      text: (
        <>
          사진에 고양이·강아지·공이 다 있는데 답은 <b>딱 하나</b>예요. 물체를 끌어서 다른 물체를 가려 보면 1등이 바뀌어요. "어디에"는 전혀 알려 주지 않죠.
        </>
      ),
      mood: 'think'
    };
  else if (mode === 'det')
    note = {
      text: moved ? (
        <>상자가 물체를 따라다니죠? 많이 가려진 물체는 확신도가 떨어지고, 너무 가려지면 아예 놓쳐요.</>
      ) : (
        <>물체마다 <b>상자(좌표 4개) + 이름 + 확신도</b>가 나와요. 그림 속 물체를 손가락이나 마우스로 끌어 보세요!</>
      ),
      mood: 'happy'
    };
  else
    note = {
      text: (
        <>
          칸(픽셀)마다 이름표를 달아요. 그런데 배경이 <b>{bgPct}%</b>나 돼서, "전부 배경"이라고 찍어도 픽셀 정확도가 {bgPct}%예요. 그래서 분할은 IoU·Dice로 평가해요.
        </>
      ),
      mood: 'wow',
      tone: 'warn'
    };

  return (
    <div ref={box}>
      <div className="vz-pad" style={{ paddingBottom: 8 }}>
        <div role="radiogroup" aria-label="과제의 사다리" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', alignItems: 'end', gap: narrow ? 6 : 10 }}>
          {RUNGS.map((r, i) => {
            const on = r.mode === mode;
            return (
              <button
                key={r.mode}
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => setMode(r.mode)}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                  gap: 2,
                  padding: narrow ? '8px 8px' : '10px 12px',
                  paddingTop: (narrow ? 8 : 10) + i * (narrow ? 10 : 14),
                  borderRadius: 14,
                  border: `2px solid ${on ? c.brand : 'var(--border)'}`,
                  background: on ? withAlpha(c.brand, 0.12) : 'var(--surface-solid)',
                  color: 'var(--text)',
                  textAlign: 'left',
                  cursor: 'pointer',
                  boxShadow: on ? `0 6px 18px ${withAlpha(c.brand, 0.25)}` : 'none',
                  transition: 'all .2s'
                }}
              >
                <span style={{ fontSize: 11.5, fontWeight: 800, color: on ? c.brand : 'var(--text-3)' }}>{r.floor}층</span>
                <span style={{ fontSize: narrow ? 14 : 16, fontWeight: 900 }}>
                  {r.emoji} {r.name}
                </span>
                <span style={{ fontSize: 12, color: 'var(--text-2)', fontWeight: 650 }}>{r.q}</span>
                {!narrow && <span style={{ fontSize: 11.5, color: 'var(--text-3)' }}>→ {r.out}</span>}
              </button>
            );
          })}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, fontWeight: 700, color: 'var(--text-3)', marginTop: 6 }}>
          <span>간단한 답 · 싼 라벨</span>
          <span>자세한 답 · 비싼 라벨 →</span>
        </div>
      </div>

      <VizLayout
        sideWidth={300}
        stage={
          <div className="vz-pad" style={{ paddingTop: 4 }}>
            <canvas
              ref={cv}
              className="vz-canvas"
              style={{ aspectRatio: `${W} / ${H}`, height: 'auto', borderRadius: 14, border: '1px solid var(--border)', cursor: 'grab' }}
              role="img"
              aria-label={`고양이, 강아지, 공이 있는 사진. 지금 단계: ${RUNGS.find((r) => r.mode === mode)!.name}`}
              onPointerDown={onDown}
              onPointerMove={onMove}
              onPointerUp={onUp}
              onPointerCancel={onUp}
              onPointerLeave={() => setHover(null)}
            />
            <div style={{ fontSize: 12, color: 'var(--text-3)', marginTop: 6 }}>✋ 그림 속 물체를 끌어서 옮길 수 있어요. 출력 숫자는 개념을 보여 주기 위한 간이 모델이 계산해요.</div>
          </div>
        }
        side={side}
      />

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <HeadsDiagram mode={mode} narrow={narrow} c={c} reduce={reduce} />
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        <Button
          variant="ghost"
          onClick={() => {
            setObjs(START);
            setMoved(false);
          }}
        >
          ↺ 물체 제자리로
        </Button>
        {mode === 'seg' && <Toggle label="사진 빼고 라벨 지도만 보기" checked={mapOnly} onChange={setMapOnly} />}
      </Controls>
    </div>
  );
}

/** Shared backbone → three task heads. */
function HeadsDiagram({ mode, narrow, c, reduce }: { mode: Mode; narrow: boolean; c: VizColors; reduce: boolean }) {
  const heads: { mode: Mode; title: string; sub: string }[] = [
    { mode: 'cls', title: '🏷️ 분류 머리', sub: '선형 층' },
    { mode: 'det', title: '📦 탐지 머리', sub: '상자 예측 (YOLO 계열)' },
    { mode: 'seg', title: '🎨 분할 머리', sub: '디코더 (U-Net 계열)' }
  ];
  const flow = (d: string, on: boolean, key: string) => (
    <path key={key} d={d} fill="none" stroke={on ? c.brand : c.axis} strokeWidth={on ? 3 : 1.5} strokeDasharray={on ? '7 6' : '3 5'} opacity={on ? 1 : 0.55}>
      {on && !reduce && <animate attributeName="stroke-dashoffset" from="26" to="0" dur="0.9s" repeatCount="indefinite" />}
    </path>
  );
  const fmap = (x: number, y: number) => (
    <g>
      {[0, 1, 2].map((i) => (
        <rect key={i} x={x + i * 7} y={y - i * 7} width={30} height={30} rx={4} fill={withAlpha(c.series[1], 0.18 + i * 0.12)} stroke={c.series[1]} strokeWidth={1.2} />
      ))}
    </g>
  );

  if (!narrow) {
    const ys = [32, 86, 140];
    return (
      <svg className="vz-svg" viewBox="0 0 640 172" role="img" aria-label="공유 몸통 하나에 과제별 머리 세 개가 붙은 구조" style={{ touchAction: 'pan-y' }}>
        <rect x={6} y={60} width={78} height={52} rx={10} fill={withAlpha(c.series[3], 0.14)} stroke={c.series[3]} strokeWidth={1.5} />
        <text x={45} y={91} textAnchor="middle" style={{ fontSize: 13, fontWeight: 800 }}>📷 사진</text>
        {flow('M 86 86 L 110 86', true, 'a')}
        <rect x={112} y={44} width={150} height={84} rx={14} fill={withAlpha(c.brand, 0.12)} stroke={c.brand} strokeWidth={2} />
        <text x={187} y={80} textAnchor="middle" style={{ fontSize: 15, fontWeight: 900 }}>공유 몸통</text>
        <text x={187} y={101} textAnchor="middle" style={{ fontSize: 12, fill: 'var(--text-2)' }}>CNN · ViT (사전학습)</text>
        {flow('M 264 86 L 290 86', true, 'b')}
        {fmap(292, 86)}
        <text x={314} y={150} textAnchor="middle" style={{ fontSize: 11.5, fontWeight: 700, fill: 'var(--text-2)' }}>특징 맵</text>
        {heads.map((hd, i) => flow(`M 344 86 C 384 86, 380 ${ys[i]}, 418 ${ys[i]}`, hd.mode === mode, hd.mode))}
        {heads.map((hd, i) => {
          const on = hd.mode === mode;
          return (
            <g key={hd.mode}>
              <rect x={420} y={ys[i] - 21} width={214} height={42} rx={11} fill={on ? withAlpha(c.brand, 0.14) : 'var(--surface-solid)'} stroke={on ? c.brand : c.border} strokeWidth={on ? 2.5 : 1.5} />
              <text x={434} y={ys[i] + 5} style={{ fontSize: 13.5, fontWeight: on ? 900 : 700, opacity: on ? 1 : 0.7 }}>
                {hd.title}
                <tspan dx={8} style={{ fontSize: 11.5, fontWeight: 600, fill: 'var(--text-2)' }}>{hd.sub}</tspan>
              </text>
            </g>
          );
        })}
      </svg>
    );
  }

  const xs = [58, 170, 282];
  return (
    <svg className="vz-svg" viewBox="0 0 340 250" role="img" aria-label="공유 몸통 하나에 과제별 머리 세 개가 붙은 구조" style={{ touchAction: 'pan-y', maxWidth: 440, margin: '0 auto' }}>
      <rect x={8} y={12} width={88} height={50} rx={10} fill={withAlpha(c.series[3], 0.14)} stroke={c.series[3]} strokeWidth={1.5} />
      <text x={52} y={42} textAnchor="middle" style={{ fontSize: 13, fontWeight: 800 }}>📷 사진</text>
      {flow('M 98 37 L 122 37', true, 'a')}
      <rect x={124} y={8} width={208} height={58} rx={14} fill={withAlpha(c.brand, 0.12)} stroke={c.brand} strokeWidth={2} />
      <text x={228} y={33} textAnchor="middle" style={{ fontSize: 15, fontWeight: 900 }}>공유 몸통</text>
      <text x={228} y={53} textAnchor="middle" style={{ fontSize: 12, fill: 'var(--text-2)' }}>CNN · ViT (사전학습)</text>
      {flow('M 228 68 L 228 84', true, 'b')}
      {fmap(150, 112)}
      <text x={206} y={112} style={{ fontSize: 11.5, fontWeight: 700, fill: 'var(--text-2)' }}>특징 맵</text>
      {heads.map((hd, i) => flow(`M 170 146 C 170 160, ${xs[i]} 156, ${xs[i]} 176`, hd.mode === mode, hd.mode))}
      {heads.map((hd, i) => {
        const on = hd.mode === mode;
        return (
          <g key={hd.mode}>
            <rect x={xs[i] - 52} y={178} width={104} height={62} rx={11} fill={on ? withAlpha(c.brand, 0.14) : 'var(--surface-solid)'} stroke={on ? c.brand : c.border} strokeWidth={on ? 2.5 : 1.5} />
            <text x={xs[i]} y={204} textAnchor="middle" style={{ fontSize: 12.5, fontWeight: on ? 900 : 700, opacity: on ? 1 : 0.75 }}>
              {hd.title}
            </text>
            <text x={xs[i]} y={224} textAnchor="middle" style={{ fontSize: 10.5, fill: 'var(--text-2)' }}>
              {hd.sub.replace(/ \(.*\)/, '')}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
