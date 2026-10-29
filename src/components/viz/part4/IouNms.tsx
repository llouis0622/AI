import { useEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, PlayBar, Segmented, Slider, Stat, Toggle, VizLayout } from '../core/ui';
import { useElementSize, useInterval } from '../core/hooks';
import { useVizColors, withAlpha, type VizColors } from '../core/colors';
import { clamp, rng } from '../core/math';
import { drawBall, drawCat, drawDog, drawPark, ell } from './_draw';

/* IoU = intersection / union, and NMS = "keep the best box, delete its look-alikes, repeat". */

type Box = [number, number, number, number]; // x0, y0, x1, y1

function area(b: Box) {
  return Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
}
function interBox(a: Box, b: Box): Box | null {
  const x0 = Math.max(a[0], b[0]);
  const y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]);
  const y1 = Math.min(a[3], b[3]);
  return x1 > x0 && y1 > y0 ? [x0, y0, x1, y1] : null;
}
function iou(a: Box, b: Box) {
  const ib = interBox(a, b);
  const I = ib ? area(ib) : 0;
  const U = area(a) + area(b) - I;
  return U > 0 ? I / U : 0;
}

/* ================= IoU playground ================= */

const GX = 20;
const GY = 15;
const U = 20;
const OX = 30;
const OY = 26;

const PRESETS: { key: string; label: string; a: Box; b: Box }[] = [
  { key: 'book', label: '📘 책의 문제', a: [0, 0, 10, 10], b: [5, 5, 15, 15] },
  { key: 'half', label: '반만 겹침', a: [2, 3, 12, 13], b: [7, 3, 17, 13] },
  { key: 'inside', label: '안에 쏙', a: [3, 2, 15, 14], b: [6, 5, 12, 11] },
  { key: 'close', label: '거의 같음', a: [4, 3, 14, 12], b: [5, 4, 15, 13] },
  { key: 'apart', label: '안 겹침', a: [1, 2, 7, 8], b: [11, 6, 18, 13] }
];

type Part = 'move' | 'tl' | 'tr' | 'bl' | 'br';

function IouPlayground({ c, narrow }: { c: VizColors; narrow: boolean }) {
  const [A, setA] = useState<Box>(PRESETS[0].a);
  const [B, setB] = useState<Box>(PRESETS[0].b);
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ which: 'A' | 'B'; part: Part; u0: number; v0: number; orig: Box } | null>(null);

  const ib = interBox(A, B);
  const I = ib ? area(ib) : 0;
  const aA = area(A);
  const aB = area(B);
  const Un = aA + aB - I;
  const v = Un > 0 ? I / Un : 0;
  const ok = v >= 0.5;
  const preset = PRESETS.find((p) => p.a.every((x, i) => x === A[i]) && p.b.every((x, i) => x === B[i]))?.key ?? 'custom';
  const colA = c.series[0];
  const colB = c.series[2];
  const colI = c.series[4];

  const toUnits = (e: RPointerEvent) => {
    const svg = svgRef.current!;
    const m = svg.getScreenCTM();
    if (!m) return [0, 0];
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
    return [(pt.x - OX) / U, (pt.y - OY) / U];
  };
  const start = (which: 'A' | 'B', part: Part) => (e: RPointerEvent) => {
    e.stopPropagation();
    const [u, w] = toUnits(e);
    svgRef.current?.setPointerCapture(e.pointerId);
    drag.current = { which, part, u0: u, v0: w, orig: which === 'A' ? A : B };
  };
  const onMove = (e: RPointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const [u, w] = toUnits(e);
    const du = Math.round(u - d.u0);
    const dv = Math.round(w - d.v0);
    let [x0, y0, x1, y1] = d.orig;
    if (d.part === 'move') {
      const sx = clamp(du, -x0, GX - x1);
      const sy = clamp(dv, -y0, GY - y1);
      x0 += sx;
      x1 += sx;
      y0 += sy;
      y1 += sy;
    } else {
      if (d.part === 'tl' || d.part === 'bl') x0 = clamp(d.orig[0] + du, 0, x1 - 1);
      if (d.part === 'tr' || d.part === 'br') x1 = clamp(d.orig[2] + du, x0 + 1, GX);
      if (d.part === 'tl' || d.part === 'tr') y0 = clamp(d.orig[1] + dv, 0, y1 - 1);
      if (d.part === 'bl' || d.part === 'br') y1 = clamp(d.orig[3] + dv, y0 + 1, GY);
    }
    const nb: Box = [x0, y0, x1, y1];
    if (d.which === 'A') setA(nb);
    else setB(nb);
  };
  const end = () => (drag.current = null);

  const px = (u: number) => OX + u * U;
  const py = (w: number) => OY + w * U;

  const boxEl = (b: Box, which: 'A' | 'B', col: string) => {
    const corners: [Part, number, number][] = [
      ['tl', b[0], b[1]],
      ['tr', b[2], b[1]],
      ['bl', b[0], b[3]],
      ['br', b[2], b[3]]
    ];
    return (
      <g>
        <rect
          x={px(b[0])}
          y={py(b[1])}
          width={(b[2] - b[0]) * U}
          height={(b[3] - b[1]) * U}
          fill={withAlpha(col, 0.16)}
          stroke={col}
          strokeWidth={3}
          rx={3}
          style={{ cursor: 'move' }}
          onPointerDown={start(which, 'move')}
        />
        <text x={px(b[0]) + 7} y={py(b[1]) + 19} style={{ fontSize: 16, fontWeight: 900, fill: col, pointerEvents: 'none' }}>
          {which}
        </text>
        {corners.map(([part, u, w]) => (
          <g key={part} onPointerDown={start(which, part)} style={{ cursor: part === 'tl' || part === 'br' ? 'nwse-resize' : 'nesw-resize' }}>
            <circle cx={px(u)} cy={py(w)} r={15} fill="transparent" />
            <circle cx={px(u)} cy={py(w)} r={6.5} fill={c.surface} stroke={col} strokeWidth={3} />
          </g>
        ))}
      </g>
    );
  };

  const fmtBox = (b: Box) => `(${b[0]},${b[1]})~(${b[2]},${b[3]})`;
  const seg = (w: number, color: string, label: ReactNode, strong = false) =>
    w > 0 && (
      <div
        style={{
          flex: `${w} 0 0`,
          minWidth: 0,
          background: strong ? color : withAlpha(color, 0.35),
          color: strong ? '#fff' : 'var(--text)',
          fontSize: 11.5,
          fontWeight: 850,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          whiteSpace: 'nowrap',
          overflow: 'hidden'
        }}
      >
        {label}
      </div>
    );

  const side = (
    <>
      <div className="vz-panel">
        <div className="vz-panel-title">넓이 계산</div>
        <div style={{ display: 'grid', gap: 4, fontSize: 13, fontVariantNumeric: 'tabular-nums' }}>
          <div>
            <b style={{ color: colA }}>A</b> {fmtBox(A)} → {A[2] - A[0]}×{A[3] - A[1]} = <b>{aA}</b>
          </div>
          <div>
            <b style={{ color: colB }}>B</b> {fmtBox(B)} → {B[2] - B[0]}×{B[3] - B[1]} = <b>{aB}</b>
          </div>
          <div>
            <b style={{ color: colI }}>교집합</b> = {ib ? `${ib[2] - ib[0]}×${ib[3] - ib[1]} = ` : ''}
            <b>{I}</b>
          </div>
          <div>
            <b>합집합</b> = {aA} + {aB} - {I} = <b>{Un}</b>
          </div>
        </div>
        <div style={{ display: 'flex', height: 26, borderRadius: 8, overflow: 'hidden', border: '1px solid var(--border)' }} aria-hidden="true">
          {seg(aA - I, colA, 'A만')}
          {seg(I, colI, `교 ${I}`, true)}
          {seg(aB - I, colB, 'B만')}
        </div>
        <div style={{ fontSize: 11.5, color: 'var(--text-3)' }}>막대 전체 = 합집합. 진한 칸의 비율이 바로 IoU예요.</div>
      </div>
      <div className="vz-panel">
        <div className="vz-panel-title">IoU = 교집합 ÷ 합집합</div>
        <div style={{ fontSize: 24, fontWeight: 900, fontVariantNumeric: 'tabular-nums' }}>
          {I} / {Un} = <span style={{ color: ok ? c.pos : c.neg }}>{v.toFixed(3)}</span>
        </div>
        <svg viewBox="0 0 260 46" className="vz-svg" role="img" aria-label={`IoU 게이지 ${v.toFixed(2)}`} style={{ touchAction: 'pan-y', maxWidth: 360 }}>
          <rect x={4} y={8} width={252} height={16} rx={8} fill="var(--surface-3)" />
          <rect x={4} y={8} width={252 * v} height={16} rx={8} fill={ok ? c.pos : c.neg} style={{ transition: 'width .25s' }} />
          <line x1={130} x2={130} y1={2} y2={30} stroke={c.ink} strokeWidth={2} strokeDasharray="3 2" />
          <text x={4} y={42} style={{ fontSize: 10.5, fill: 'var(--text-3)' }}>0</text>
          <text x={130} y={42} textAnchor="middle" style={{ fontSize: 10.5, fontWeight: 800 }}>0.5 = 정답 인정선</text>
          <text x={256} y={42} textAnchor="end" style={{ fontSize: 10.5, fill: 'var(--text-3)' }}>1</text>
        </svg>
        <Stat label="탐지 채점이라면" value={ok ? '맞았다 ✅' : '틀렸다 ❌'} tone={ok ? 'good' : 'bad'} />
      </div>
    </>
  );

  let note: { text: ReactNode; mood: 'think' | 'wow' | 'cheer' | 'smile'; tone?: 'good' | 'bad' };
  if (preset === 'book')
    note = { text: <>책의 문제 1번 그대로예요. 교집합 25, 합집합 175 → IoU ≈ 0.14. 꽤 겹쳐 보이는데 생각보다 훨씬 작죠? 분모가 합집합이라서예요.</>, mood: 'think' };
  else if (preset === 'half') note = { text: <>딱 절반이 겹쳤는데 IoU는 0.33뿐! 겹친 부분(50)보다 합집합(150)이 훨씬 커요.</>, mood: 'wow' };
  else if (preset === 'inside') note = { text: <>작은 상자가 큰 상자 안에 쏙 들어가도, 큰 상자의 남는 부분이 합집합을 키워서 IoU가 작아요.</>, mood: 'think' };
  else if (v === 0) note = { text: <>하나도 안 겹치면 교집합이 0이라 IoU도 0이에요.</>, mood: 'smile' };
  else if (ok) note = { text: <>IoU {v.toFixed(2)} ≥ 0.5 — 탐지 대회라면 "맞혔다"로 인정받아요! 🎉</>, mood: 'cheer', tone: 'good' };
  else note = { text: <>상자 안을 끌면 옮겨지고, 모서리 동그라미를 끌면 크기가 바뀌어요. 0.5를 넘겨 보세요!</>, mood: 'smile' };

  return (
    <>
      <VizLayout
        sideWidth={300}
        stage={
          <div className="vz-pad">
            <svg
              ref={svgRef}
              className="vz-svg"
              viewBox={`0 0 ${OX + GX * U + 10} ${OY + GY * U + 10}`}
              role="img"
              aria-label={`상자 A ${fmtBox(A)}, 상자 B ${fmtBox(B)}, IoU ${v.toFixed(3)}`}
              onPointerMove={onMove}
              onPointerUp={end}
              onPointerCancel={end}
              style={{ maxWidth: narrow ? undefined : 560, margin: '0 auto' }}
            >
              {Array.from({ length: GX + 1 }, (_, i) => (
                <line key={`x${i}`} x1={px(i)} x2={px(i)} y1={py(0)} y2={py(GY)} stroke={i % 5 === 0 ? c.axis : c.grid} opacity={i % 5 === 0 ? 0.5 : 1} />
              ))}
              {Array.from({ length: GY + 1 }, (_, i) => (
                <line key={`y${i}`} x1={px(0)} x2={px(GX)} y1={py(i)} y2={py(i)} stroke={i % 5 === 0 ? c.axis : c.grid} opacity={i % 5 === 0 ? 0.5 : 1} />
              ))}
              {[0, 5, 10, 15, 20].map((i) => (
                <text key={`tx${i}`} x={px(i)} y={OY - 8} textAnchor="middle" className="vz-tick">
                  {i}
                </text>
              ))}
              {[0, 5, 10, 15].map((i) => (
                <text key={`ty${i}`} x={OX - 8} y={py(i) + 4} textAnchor="end" className="vz-tick">
                  {i}
                </text>
              ))}
              {boxEl(A, 'A', colA)}
              {boxEl(B, 'B', colB)}
              {ib && (
                <g pointerEvents="none">
                  <rect x={px(ib[0])} y={py(ib[1])} width={(ib[2] - ib[0]) * U} height={(ib[3] - ib[1]) * U} fill={withAlpha(colI, 0.55)} stroke={colI} strokeWidth={2} />
                  {(ib[2] - ib[0]) * U > 44 && (ib[3] - ib[1]) * U > 30 && (
                    <text x={px((ib[0] + ib[2]) / 2)} y={py((ib[1] + ib[3]) / 2) + 5} textAnchor="middle" style={{ fontSize: 14, fontWeight: 900, fill: c.ink }}>
                      교 {I}
                    </text>
                  )}
                </g>
              )}
            </svg>
          </div>
        }
        side={side}
      />
      <div className="vz-pad" style={{ paddingTop: 0 }}>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>
      <Controls>
        <Segmented
          label="예시 상자"
          options={PRESETS.map((p) => ({ value: p.key, label: p.label }))}
          value={preset}
          onChange={(k) => {
            const p = PRESETS.find((q) => q.key === k)!;
            setA(p.a);
            setB(p.b);
          }}
        />
      </Controls>
    </>
  );
}

/* ================= NMS step-through ================= */

const SW = 400;
const SH = 250;
type Kind = 'cat' | 'dog' | 'ball';
const NAMES: Record<Kind, string> = { cat: '고양이', dog: '강아지', ball: '공' };
const EMOJI: Record<Kind, string> = { cat: '🐱', dog: '🐶', ball: '⚽' };
/** Same visible-extent fractions as the task-ladder pictures. */
const FR: Record<Kind, Box> = { cat: [10 / 64, 6 / 64, 54 / 64, 55 / 64], dog: [6 / 64, 13 / 64, 58 / 64, 55 / 64], ball: [4 / 64, 4 / 64, 60 / 64, 60 / 64] };
/** The dog stands right behind the cat: their true boxes overlap with IoU ≈ 0.41. */
const OBJS: { kind: Kind; x: number; y: number; s: number; base: number }[] = [
  { kind: 'dog', x: 192, y: 122, s: 150, base: 0.92 },
  { kind: 'cat', x: 150, y: 132, s: 150, base: 0.95 },
  { kind: 'ball', x: 338, y: 190, s: 72, base: 0.88 }
];
const truthBox = (o: (typeof OBJS)[number]): Box => {
  const f = FR[o.kind];
  const x0 = o.x - o.s / 2;
  const y0 = o.y - o.s / 2;
  return [x0 + f[0] * o.s, y0 + f[1] * o.s, x0 + f[2] * o.s, y0 + f[3] * o.s];
};
const TRUTHS = OBJS.map((o) => ({ kind: o.kind, base: o.base, box: truthBox(o) }));

interface Cand {
  id: number;
  kind: Kind;
  box: Box;
  score: number;
}

/** Jittered candidate boxes around each object; the best-scoring one hugs the object. */
function makeCands(seed: number): Cand[] {
  const r = rng(seed);
  const out: Omit<Cand, 'id'>[] = [];
  for (const t of TRUTHS) {
    const [x0, y0, x1, y1] = t.box;
    const w = x1 - x0;
    const h = y1 - y0;
    const n = t.kind === 'ball' ? 4 : 5;
    let made = 0;
    let guard = 0;
    while (made < n && guard++ < 2000) {
      const sd = made === 0 ? 0.025 : 0.09;
      const b: Box = [x0 + r.normal() * sd * w, y0 + r.normal() * sd * h, x1 + r.normal() * sd * w, y1 + r.normal() * sd * h];
      const q = iou(b, t.box);
      if (made === 0 ? q < 0.88 : q < 0.6 || q > 0.86) continue;
      out.push({ kind: t.kind, box: b, score: clamp(t.base - (1 - q) * 0.75 + r.normal() * 0.025, 0.3, 0.99) });
      made++;
    }
  }
  return out.sort((a, b) => b.score - a.score).map((cd, i) => ({ ...cd, id: i }));
}

type St = 'idle' | 'current' | 'kept' | 'removed';
interface Run {
  status: St[];
  cur: number | null;
  phase: 'pick' | 'suppress' | 'done';
  removedIou: (number | null)[];
  lastRemoved: number;
}
const freshRun = (n: number): Run => ({ status: Array(n).fill('idle'), cur: null, phase: 'pick', removedIou: Array(n).fill(null), lastRemoved: 0 });

function stepRun(run: Run, cands: Cand[], thr: number): Run {
  if (run.phase === 'done') return run;
  const status = run.status.slice();
  if (run.phase === 'pick') {
    const cur = status.indexOf('idle');
    if (cur < 0) return { ...run, phase: 'done' };
    status[cur] = 'current';
    return { ...run, status, cur, phase: 'suppress' };
  }
  const cur = run.cur!;
  const removedIou = run.removedIou.slice();
  let removed = 0;
  for (let i = 0; i < cands.length; i++) {
    if (status[i] !== 'idle') continue;
    const q = iou(cands[i].box, cands[cur].box);
    if (q >= thr) {
      status[i] = 'removed';
      removedIou[i] = q;
      removed++;
    }
  }
  status[cur] = 'kept';
  const left = status.includes('idle');
  return { status, cur: null, phase: left ? 'pick' : 'done', removedIou, lastRemoved: removed };
}

let photoUrl: string | null = null;
function scenePhoto() {
  if (photoUrl) return photoUrl;
  const cv = document.createElement('canvas');
  cv.width = SW * 2;
  cv.height = SH * 2;
  const ctx = cv.getContext('2d')!;
  ctx.scale(2, 2);
  drawPark(ctx, SW, SH);
  for (const o of OBJS) {
    ctx.fillStyle = 'rgba(30,60,30,0.16)';
    ell(ctx, o.x, o.y + o.s * 0.44, o.s * 0.34, o.s * 0.06);
  }
  for (const o of OBJS) {
    ctx.save();
    ctx.translate(o.x - o.s / 2, o.y - o.s / 2);
    if (o.kind === 'cat') drawCat(ctx, o.s, 'rgba(0,0,0,0)');
    else if (o.kind === 'dog') drawDog(ctx, o.s);
    else drawBall(ctx, o.s);
    ctx.restore();
  }
  photoUrl = cv.toDataURL('image/png');
  return photoUrl;
}

/** Approximate label width at 10.5px (Hangul glyphs are about twice as wide as digits). */
const textW = (s: string) => [...s].reduce((w, ch) => w + (/[가-힣]/.test(ch) ? 10.6 : 6.1), 0);

function Chip({ x, y, text, fill, color = '#fff', anchor = 'start' }: { x: number; y: number; text: string; fill: string; color?: string; anchor?: 'start' | 'end' }) {
  const w = textW(text) + 10;
  const x0 = clamp(anchor === 'end' ? x - w : x, 0, SW - w);
  const y0 = clamp(y, 0, SH - 16);
  return (
    <g pointerEvents="none">
      <rect x={x0} y={y0} width={w} height={16} rx={4} fill={fill} />
      <text x={x0 + 5} y={y0 + 11.5} style={{ fontSize: 10.5, fontWeight: 800, fill: color }}>
        {text}
      </text>
    </g>
  );
}

function NmsStepper({ c, narrow }: { c: VizColors; narrow: boolean }) {
  const [seed, setSeed] = useState(3);
  const [thr, setThr] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showRemoved, setShowRemoved] = useState(false);
  const cands = useMemo(() => makeCands(seed), [seed]);
  const [run, setRun] = useState<Run>(() => freshRun(cands.length));
  const photo = useMemo(() => scenePhoto(), []);

  useEffect(() => {
    setRun(freshRun(cands.length));
    setPlaying(false);
  }, [cands, thr]);
  useInterval(
    () => {
      if (run.phase === 'done') setPlaying(false);
      else setRun((r) => stepRun(r, cands, thr));
    },
    playing ? 950 / speed : null
  );

  const cur = run.cur;
  const ious = cur !== null ? cands.map((cd) => iou(cd.box, cands[cur].box)) : null;
  const nKept = run.status.filter((s) => s === 'kept').length;
  const nRemoved = run.status.filter((s) => s === 'removed').length;
  const nIdle = run.status.filter((s) => s === 'idle').length;
  const started = run.status.some((s) => s !== 'idle');

  // final evaluation against the true objects
  const kept = cands.filter((_, i) => run.status[i] === 'kept');
  const evalRes = TRUTHS.map((t) => kept.filter((k) => iou(k.box, t.box) >= 0.5).length);
  const found = evalRes.filter((m) => m > 0).length;
  const dups = evalRes.reduce((s, m) => s + Math.max(0, m - 1), 0);

  const colIdle = c.series[3];
  const colCur = c.series[4];

  const boxes = cands.map((cd, i) => {
    const st = run.status[i];
    const [x0, y0, x1, y1] = cd.box;
    const common = { x: x0, y: y0, width: x1 - x0, height: y1 - y0, rx: 2, fill: 'none' as const };
    if (st === 'removed') {
      if (!showRemoved) return null;
      return <rect key={i} {...common} stroke={c.neg} strokeWidth={1.4} strokeDasharray="4 4" opacity={0.35} />;
    }
    if (st === 'kept') return <rect key={i} {...common} stroke={c.pos} strokeWidth={3.5} fill={withAlpha(c.pos, 0.08)} />;
    if (st === 'current') return <rect key={i} {...common} stroke={colCur} strokeWidth={4.5} fill={withAlpha(colCur, 0.12)} />;
    const q = ious?.[i] ?? 0;
    if (ious && q >= thr) return <rect key={i} {...common} stroke={c.neg} strokeWidth={2.6} strokeDasharray="6 4" />;
    return <rect key={i} {...common} stroke={colIdle} strokeWidth={1.6} opacity={0.3 + 0.7 * ((cd.score - 0.3) / 0.7)} />;
  });

  const chips: ReactNode[] = [];
  cands.forEach((cd, i) => {
    const st = run.status[i];
    const [x0, y0, x1, y1] = cd.box;
    if (st === 'kept') chips.push(<Chip key={`k${i}`} x={x0} y={y0 - 17} text={`${NAMES[cd.kind]} ${cd.score.toFixed(2)}`} fill={c.pos} />);
    if (st === 'current') chips.push(<Chip key={`c${i}`} x={x0} y={y0 - 17} text={`1등 #${i + 1} ${cd.score.toFixed(2)}`} fill={colCur} color="#2a2540" />);
    // deleted look-alikes are shown by their red dashed outline; label only the boxes that survive despite overlapping
    if (st === 'idle' && ious && ious[i] > 0.15 && ious[i] < thr) {
      const q = ious[i];
      chips.push(<Chip key={`i${i}`} x={x1} y={y1 - 16} anchor="end" text={`IoU ${q.toFixed(2)}${q >= thr ? ' 삭제' : ' 남김'}`} fill={q >= thr ? c.neg : withAlpha(c.ink, 0.72)} />);
    }
  });

  let note: { text: ReactNode; mood: 'smile' | 'think' | 'wow' | 'cheer' | 'happy'; tone?: 'good' | 'bad' | 'warn' };
  if (!started)
    note = {
      text: (
        <>
          후보 상자 {cands.length}개가 물체마다 우르르 겹쳐 있어요(탐지기 대신 만든 간이 후보예요). ▶ 시작이나 "한 걸음"을 눌러 NMS가 정리하는 걸 보세요.
        </>
      ),
      mood: 'smile'
    };
  else if (run.phase === 'suppress' && cur !== null) {
    const doomed = run.status.filter((s, i) => s === 'idle' && ious![i] >= thr).length;
    note = {
      text: (
        <>
          남은 후보 중 확신도 1등(#{cur + 1}, {cands[cur].score.toFixed(2)})을 골랐어요. 이 상자와 IoU가 {thr.toFixed(2)} 이상인 <b style={{ color: c.neg }}>빨간 점선 {doomed}개</b>는 "같은 물체의 중복"으로 보고 지울 거예요.
        </>
      ),
      mood: 'think'
    };
  } else if (run.phase === 'pick') note = { text: <>채택! 겹치던 {run.lastRemoved}개를 지웠어요. 남은 후보({nIdle}개) 중에서 다시 1등을 찾아요.</>, mood: 'happy' };
  else if (found < TRUTHS.length)
    note = {
      text: (
        <>
          물체를 놓쳤어요! 문턱이 {thr.toFixed(2)}로 너무 낮아서, 바짝 붙은 고양이와 강아지 상자(IoU ≈ 0.41)까지 "중복"으로 보고 지워 버렸어요.
        </>
      ),
      mood: 'wow',
      tone: 'bad'
    };
  else if (dups > 0) note = { text: <>중복 상자 {dups}개가 살아남았어요. 문턱이 {thr.toFixed(2)}로 너무 높으면 꽤 겹친 상자도 "다른 물체"로 봐 줘요.</>, mood: 'think', tone: 'warn' };
  else note = { text: <>🎉 물체 3개에 상자 3개! 후보 {cands.length}개가 깔끔하게 정리됐어요. 문턱을 0.3이나 0.8로 바꿔서도 해 보세요.</>, mood: 'cheer', tone: 'good' };

  const side = (
    <>
      <div className="vz-stats">
        <Stat label="후보" value={cands.length} />
        <Stat label="채택" value={nKept} tone="good" />
        <Stat label="삭제" value={nRemoved} tone={nRemoved ? 'bad' : undefined} />
      </div>
      {run.phase === 'done' && (
        <div className="vz-stats">
          <Stat label="찾은 물체" value={`${found} / ${TRUTHS.length}`} tone={found === TRUTHS.length ? 'good' : 'bad'} />
          <Stat label="중복" value={dups} tone={dups ? 'warn' : 'good'} />
        </div>
      )}
      <div className="vz-panel">
        <div className="vz-panel-title">확신도 순 후보 줄</div>
        <div style={{ display: 'grid', gap: 3, maxHeight: narrow ? 220 : 300, overflowY: 'auto', paddingRight: 2 }}>
          {cands.map((cd, i) => {
            const st = run.status[i];
            const doomed = st === 'idle' && ious && ious[i] >= thr;
            const color = st === 'kept' ? c.pos : st === 'current' ? colCur : st === 'removed' || doomed ? c.neg : 'var(--text-2)';
            return (
              <div
                key={i}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '28px 1fr 40px 88px',
                  alignItems: 'center',
                  gap: 6,
                  fontSize: 12,
                  padding: '3px 6px',
                  borderRadius: 8,
                  background: st === 'current' ? withAlpha(colCur, 0.18) : st === 'kept' ? withAlpha(c.pos, 0.1) : 'transparent',
                  opacity: st === 'removed' ? 0.5 : 1,
                  textDecoration: st === 'removed' ? 'line-through' : 'none'
                }}
              >
                <span style={{ color: 'var(--text-3)', fontWeight: 800 }}>#{i + 1}</span>
                <span style={{ fontWeight: 750 }}>
                  {EMOJI[cd.kind]} {NAMES[cd.kind]}
                </span>
                <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 800 }}>{cd.score.toFixed(2)}</span>
                <span style={{ color, fontWeight: 850, fontSize: 11.5 }}>
                  {st === 'kept' ? '✔ 채택' : st === 'current' ? '★ 1등' : st === 'removed' ? `✖ ${run.removedIou[i]!.toFixed(2)}` : doomed ? `삭제 예정 ${ious![i].toFixed(2)}` : '대기'}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );

  return (
    <>
      <VizLayout
        sideWidth={290}
        stage={
          <div className="vz-pad" style={{ display: 'grid', gap: 10 }}>
            <svg className="vz-svg" viewBox={`0 0 ${SW} ${SH}`} role="img" aria-label={`후보 상자 ${cands.length}개, 채택 ${nKept}개`} style={{ borderRadius: 14, border: '1px solid var(--border)', touchAction: 'pan-y' }}>
              <image href={photo} x={0} y={0} width={SW} height={SH} />
              {boxes}
              {chips}
            </svg>
            <PlayBar
              playing={playing}
              onToggle={() => {
                if (run.phase === 'done') setRun(freshRun(cands.length));
                setPlaying((p) => !p);
              }}
              onStep={run.phase === 'done' ? undefined : () => setRun((r) => stepRun(r, cands, thr))}
              onReset={() => {
                setPlaying(false);
                setRun(freshRun(cands.length));
              }}
              speed={speed}
              onSpeed={setSpeed}
            />
          </div>
        }
        side={side}
      />
      <div className="vz-pad" style={{ paddingTop: 0 }}>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>
      <Controls>
        <Slider label="겹침 문턱 (IoU)" value={thr} min={0.3} max={0.8} step={0.05} onChange={setThr} format={(x) => x.toFixed(2)} hint="이 값 이상 겹치면 중복으로 보고 지워요" />
        <Toggle label="지운 상자도 흐리게 보기" checked={showRemoved} onChange={setShowRemoved} />
        <Button variant="ghost" onClick={() => setSeed((s) => s + 1)}>
          🎲 새 후보
        </Button>
      </Controls>
    </>
  );
}

export default function IouNms(_props: VizProps) {
  const c = useVizColors();
  const [tab, setTab] = useState<'iou' | 'nms'>('iou');
  const [wrap, { width }] = useElementSize<HTMLDivElement>();
  const narrow = width > 0 && width < 560;

  return (
    <div ref={wrap}>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <Segmented
          options={[
            { value: 'iou', label: '① IoU 재기' },
            { value: 'nms', label: '② NMS로 정리하기' }
          ]}
          value={tab}
          onChange={setTab}
        />
      </div>
      {tab === 'iou' ? <IouPlayground c={c} narrow={narrow} /> : <NmsStepper c={c} narrow={narrow} />}
    </div>
  );
}
