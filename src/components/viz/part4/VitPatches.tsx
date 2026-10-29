import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Stat } from '../core/ui';
import { useElementSize, useInterval, usePrefersReducedMotion, useRafLoop } from '../core/hooks';
import { mixHex, useVizColors, withAlpha } from '../core/colors';
import { clamp } from '../core/math';
import { drawHouse } from './_draw';

/* ViT: cut the picture into patches → line them up → flatten → embed + number → Transformer. */

const SRC = 384;
const STAGES: { label: string; short: string }[] = [
  { label: '🖼️ 원본', short: '원본' },
  { label: '✂️ 자르기', short: '자르기' },
  { label: '🚂 줄 세우기', short: '줄 세우기' },
  { label: '📏 펼치기', short: '펼치기' },
  { label: '🔢 번호 붙이기', short: '번호' },
  { label: '🤖 Transformer', short: 'Transformer' }
];
const STRIPE_CELLS = 12;
const TOKEN_CELLS = 8;

interface PatchData {
  url: string;
  avg: [number, number, number];
  cells: string[];
}

let srcCanvas: HTMLCanvasElement | null = null;
let srcUrl = '';
const patchCache = new Map<number, PatchData[]>();

function source() {
  if (!srcCanvas) {
    srcCanvas = document.createElement('canvas');
    srcCanvas.width = SRC;
    srcCanvas.height = SRC;
    drawHouse(srcCanvas.getContext('2d')!, SRC);
    srcUrl = srcCanvas.toDataURL('image/png');
  }
  return srcCanvas;
}

/** Patch images, mean colours and a 3×4 "flattened pixels" summary for each patch. */
function patches(g: number): PatchData[] {
  const hit = patchCache.get(g);
  if (hit) return hit;
  const src = source();
  const data = src.getContext('2d')!.getImageData(0, 0, SRC, SRC).data;
  const ps = SRC / g;
  const out: PatchData[] = [];
  for (let r = 0; r < g; r++)
    for (let q = 0; q < g; q++) {
      const pc = document.createElement('canvas');
      const side = Math.min(ps, 128);
      pc.width = side;
      pc.height = side;
      pc.getContext('2d')!.drawImage(src, q * ps, r * ps, ps, ps, 0, 0, side, side);
      const cells: string[] = [];
      const tot = [0, 0, 0];
      for (let cy = 0; cy < 4; cy++)
        for (let cx = 0; cx < 3; cx++) {
          const acc = [0, 0, 0];
          let n = 0;
          const x0 = Math.floor(q * ps + (cx * ps) / 3);
          const x1 = Math.floor(q * ps + ((cx + 1) * ps) / 3);
          const y0 = Math.floor(r * ps + (cy * ps) / 4);
          const y1 = Math.floor(r * ps + ((cy + 1) * ps) / 4);
          for (let y = y0; y < y1; y += 2)
            for (let x = x0; x < x1; x += 2) {
              const k = (y * SRC + x) * 4;
              acc[0] += data[k];
              acc[1] += data[k + 1];
              acc[2] += data[k + 2];
              n++;
            }
          const rgb = acc.map((v) => Math.round(v / Math.max(1, n)));
          rgb.forEach((v, i) => (tot[i] += v));
          cells.push(`rgb(${rgb[0]},${rgb[1]},${rgb[2]})`);
        }
      out.push({ url: pc.toDataURL('image/png'), avg: tot.map((v) => v / 12 / 255) as [number, number, number], cells });
    }
  patchCache.set(g, out);
  return out;
}

interface Geo {
  cx: number;
  cy: number;
  w: number;
  h: number;
}
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const lerpGeo = (a: Geo, b: Geo, t: number): Geo => ({ cx: lerp(a.cx, b.cx, t), cy: lerp(a.cy, b.cy, t), w: lerp(a.w, b.w, t), h: lerp(a.h, b.h, t) });
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

const LAYOUTS = {
  wide: { vw: 720, vh: 356, IX: 22, IY: 64, IMG: 216, SX: 268, SY: 92, SW: 440 },
  narrow: { vw: 360, vh: 532, IX: 80, IY: 44, IMG: 200, SX: 12, SY: 318, SW: 336 }
};
type Lay = (typeof LAYOUTS)['wide'];

function seqDims(N: number, L: Lay) {
  const perRow = N <= 8 ? N : N === 16 ? 8 : 16;
  const cell = Math.min(54, L.SW / (perRow + 1));
  return { perRow, cell, ts: cell * 0.84, rowH: cell * 2.05, rows: Math.ceil(N / perRow) };
}

function geoFor(stage: number, i: number, g: number, L: Lay): Geo {
  const N = g * g;
  const ps = L.IMG / g;
  const r = Math.floor(i / g);
  const q = i % g;
  if (stage === 0) return { cx: L.IX + (q + 0.5) * ps, cy: L.IY + (r + 0.5) * ps, w: ps, h: ps };
  if (stage === 1) {
    const sp = ps * 1.1;
    return { cx: L.IX + L.IMG / 2 + (q - (g - 1) / 2) * sp, cy: L.IY + L.IMG / 2 + (r - (g - 1) / 2) * sp, w: ps * 0.9, h: ps * 0.9 };
  }
  const d = seqDims(N, L);
  const row = Math.floor(i / d.perRow);
  const col = i % d.perRow;
  const cx = L.SX + (col + 1.5) * d.cell;
  const cy = L.SY + row * d.rowH + d.ts * 0.75;
  if (stage === 2) return { cx, cy, w: d.ts, h: d.ts };
  if (stage === 3) return { cx, cy, w: d.ts * 0.42, h: d.ts * 1.5 };
  return { cx, cy, w: d.ts * 0.5, h: d.ts * 1.5 };
}

/** 0..1 weight of each visual form at a stage. */
const formAt = (s: number) => ({ tile: s <= 2 ? 1 : 0, stripe: s === 3 ? 1 : 0, token: s >= 4 ? 1 : 0 });

export default function VitPatches(_props: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [g, setG] = useState(4);
  const [stage, setStage] = useState(0);
  const [prev, setPrev] = useState(0);
  const [p, setP] = useState(1);
  const [sel, setSel] = useState<number | null>(null);
  const [auto, setAuto] = useState(false);
  const [box, { width }] = useElementSize<HTMLDivElement>();
  const L = width > 0 && width < 600 ? LAYOUTS.narrow : LAYOUTS.wide;
  const N = g * g;
  const data = useMemo(() => patches(g), [g]);
  const full = useMemo(() => {
    source();
    return srcUrl;
  }, []);
  const firstRender = useRef(true);

  const go = (s: number) => {
    const ns = clamp(s, 0, STAGES.length - 1);
    if (ns === stage) return;
    setPrev(stage);
    setStage(ns);
    setP(reduce ? 1 : 0);
  };
  useRafLoop((dt) => setP((v) => Math.min(1, v + dt / 1.1)), p < 1);
  useInterval(
    () => {
      if (stage >= STAGES.length - 1) setAuto(false);
      else go(stage + 1);
    },
    auto && p >= 1 ? 900 : null
  );
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    setPrev(stage);
    setP(1);
    setSel(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [g]);

  // default focus: a roof patch
  const roof = Math.floor(0.42 * g) * g + Math.floor(0.4 * g);
  const focus = sel ?? roof;
  const ps = L.IMG / g;
  const d = seqDims(N, L);

  // attention of the focused patch: similar-looking patches get more weight
  const attn = useMemo(() => {
    const a = data[focus]?.avg ?? [0, 0, 0];
    const raw = data.map((pd, j) => (j === focus ? 0 : Math.exp(-((pd.avg[0] - a[0]) ** 2 + (pd.avg[1] - a[1]) ** 2 + (pd.avg[2] - a[2]) ** 2) / 0.02)));
    const m = Math.max(1e-9, ...raw);
    return raw.map((v) => v / m);
  }, [data, focus]);

  const tokenColor = (pd: PatchData, k: number) => {
    const [r, gg, b] = pd.avg;
    const e = 0.5 + 0.5 * Math.sin(k * 2.1 + r * 4 + gg * 6.3 + b * 2.7 + k * r * 3);
    return mixHex(c.surface, k % 2 ? c.series[0] : c.series[1], 0.2 + 0.8 * e);
  };

  const settledOriginal = stage === 0 && p >= 1;
  const fa = formAt(prev);
  const fb = formAt(stage);
  const stageAlpha = (s: number) => (stage >= s && prev >= s ? 1 : stage >= s ? ease(p) : prev >= s ? 1 - ease(p) : 0);
  const clsA = stageAlpha(4);
  const trA = stageAlpha(5);
  const numA = stageAlpha(4);

  const tiles: ReactNode[] = [];
  for (let i = 0; i < N; i++) {
    const delay = N > 1 ? (i / (N - 1)) * 0.45 : 0;
    const lp = ease(clamp((p - delay) / 0.55, 0, 1));
    const geo = lerpGeo(geoFor(prev, i, g, L), geoFor(stage, i, g, L), lp);
    const wTile = lerp(fa.tile, fb.tile, lp);
    const wStripe = lerp(fa.stripe, fb.stripe, lp);
    const wToken = lerp(fa.token, fb.token, lp);
    const x = geo.cx - geo.w / 2;
    const y = geo.cy - geo.h / 2;
    const on = i === focus && (sel !== null || stage >= 2);
    const pd = data[i];
    tiles.push(
      <g key={`${g}-${i}`} onPointerEnter={() => setSel(i)} onClick={() => setSel(i)} style={{ cursor: 'pointer' }}>
        {wTile > 0.01 && <image href={pd.url} x={x} y={y} width={geo.w} height={geo.h} preserveAspectRatio="none" opacity={wTile} />}
        {wStripe > 0.01 && (
          <g opacity={wStripe}>
            {pd.cells.map((col, k) => (
              <rect key={k} x={x} y={y + (k * geo.h) / STRIPE_CELLS} width={geo.w} height={geo.h / STRIPE_CELLS + 0.4} fill={col} />
            ))}
          </g>
        )}
        {wToken > 0.01 && (
          <g opacity={wToken}>
            {Array.from({ length: TOKEN_CELLS }, (_, k) => (
              <rect key={k} x={x} y={y + (k * geo.h) / TOKEN_CELLS} width={geo.w} height={geo.h / TOKEN_CELLS + 0.4} fill={tokenColor(pd, k)} />
            ))}
          </g>
        )}
        {stage >= 5 && trA > 0.5 && i !== focus && (
          <rect x={x - 1.5} y={y - 1.5} width={geo.w + 3} height={geo.h + 3} rx={3} fill="none" stroke={c.brand} strokeWidth={2.2} opacity={attn[i] * trA} />
        )}
        <rect x={x} y={y} width={geo.w} height={geo.h} rx={wTile > 0.5 ? 2 : 3} fill="none" stroke={on ? c.series[4] : withAlpha(c.ink, 0.25)} strokeWidth={on ? 3 : settledOriginal ? 0 : 0.8} />
        {numA > 0.01 && (
          <text x={geo.cx} y={geo.cy + geo.h / 2 + 10} textAnchor="middle" opacity={numA} style={{ fontSize: d.cell < 24 ? 7.5 : 9.5, fontWeight: 800, fill: on ? c.series[4] : 'var(--text-2)' }}>
            {i + 1}
          </text>
        )}
      </g>
    );
  }

  const g0 = geoFor(4, 0, g, L);
  const cls = { cx: L.SX + 0.5 * d.cell, cy: g0.cy, w: g0.w, h: g0.h };
  const trBox = { x: L.SX - 4, y: L.SY - 30, w: L.SW + 8, h: d.rows * d.rowH + 32 };
  const focusCx = L.IX + ((focus % g) + 0.5) * ps;
  const focusCy = L.IY + (Math.floor(focus / g) + 0.5) * ps;
  const last = STAGES.length - 1;

  const rawPerPatch = (96 / g) ** 2 * 3;
  const notes: { text: ReactNode; mood: 'smile' | 'think' | 'wow' | 'happy' | 'cheer' }[] = [
    { text: <>ViT는 사진을 "문장"으로 바꿔 읽어요. 아래 "다음 ▶"을 눌러 한 단계씩 따라가 봐요.</>, mood: 'smile' },
    { text: <>✂️ 사진을 같은 크기의 조각(패치) {N}개로 잘라요. 진짜 ViT는 16×16 픽셀 조각을 써요.</>, mood: 'happy' },
    { text: <>🚂 조각을 왼쪽 위부터 차례로 한 줄로 세워요. 이제 사진이 "조각 단어 {N}개짜리 문장"이 됐어요. 조각에 마우스를 올리면 원래 자리가 보여요.</>, mood: 'wow' },
    { text: <>📏 조각 하나를 숫자 목록으로 쭉 펼쳐요. 이 그림에선 조각 하나가 {rawPerPatch.toLocaleString()}개 숫자(픽셀 × RGB 3)예요.</>, mood: 'think' },
    { text: <>🔢 모든 조각을 같은 선형 층에 통과시켜 길이가 같은 벡터(토큰)로 만들고, 몇 번째 조각인지 위치 번호를 더해요. 맨 앞엔 요약 담당 [CLS] 토큰을 붙여요.</>, mood: 'happy' },
    {
      text: (
        <>
          🤖 Transformer는 첫 층부터 <b>모든 조각 쌍</b>을 한 번에 비교해요(어텐션). 합성곱처럼 "가까운 픽셀끼리"라는 규칙이 없어서, 그 상식까지 데이터로 배워야 해요. 조각에 마우스를 올려 보세요! (선은 색이 비슷한 조각끼리 이은 간이 그림이에요.)
        </>
      ),
      mood: 'cheer'
    }
  ];
  const note = notes[stage];

  return (
    <div ref={box}>
      <div className="vz-pad" style={{ paddingBottom: 6 }}>
        <svg className="vz-svg" viewBox={`0 0 ${L.vw} ${L.vh}`} role="img" aria-label={`ViT 단계 ${stage + 1}: ${STAGES[stage].short}, 조각 ${N}개`} style={{ touchAction: 'pan-y' }} onPointerLeave={() => setSel(null)}>
          <text x={L.IX + L.IMG / 2} y={L.IY - 14} textAnchor="middle" style={{ fontSize: 13, fontWeight: 850, fill: 'var(--text-2)' }}>
            {stage < 2 ? '사진' : '원래 자리 지도'}
          </text>
          {stageAlpha(2) > 0.01 && (
            <g opacity={stageAlpha(2)}>
              <image href={full} x={L.IX} y={L.IY} width={L.IMG} height={L.IMG} opacity={0.4} />
              {Array.from({ length: g - 1 }, (_, k) => (
                <g key={k}>
                  <line x1={L.IX + (k + 1) * ps} x2={L.IX + (k + 1) * ps} y1={L.IY} y2={L.IY + L.IMG} stroke={withAlpha(c.ink, 0.35)} strokeDasharray="3 3" />
                  <line x1={L.IX} x2={L.IX + L.IMG} y1={L.IY + (k + 1) * ps} y2={L.IY + (k + 1) * ps} stroke={withAlpha(c.ink, 0.35)} strokeDasharray="3 3" />
                </g>
              ))}
              <rect x={L.IX} y={L.IY} width={L.IMG} height={L.IMG} fill="none" stroke={c.border} strokeWidth={1.5} />
              {trA > 0.01 &&
                data.map((_, j) =>
                  j === focus || attn[j] < 0.08 ? null : (
                    <line
                      key={j}
                      x1={focusCx}
                      y1={focusCy}
                      x2={L.IX + ((j % g) + 0.5) * ps}
                      y2={L.IY + (Math.floor(j / g) + 0.5) * ps}
                      stroke={c.brand}
                      strokeWidth={0.6 + attn[j] * 3.2}
                      strokeLinecap="round"
                      opacity={trA * (0.25 + 0.75 * attn[j])}
                    />
                  )
                )}
              <rect x={L.IX + (focus % g) * ps} y={L.IY + Math.floor(focus / g) * ps} width={ps} height={ps} fill={withAlpha(c.series[4], 0.25)} stroke={c.series[4]} strokeWidth={3} />
            </g>
          )}

          {stage >= 2 && (
            <text x={L.SX + 4} y={L.SY - (trA > 0.01 ? 40 : 18)} style={{ fontSize: 13, fontWeight: 850, fill: 'var(--text-2)' }}>
              {stage === 2 ? `조각 문장 (길이 ${N})` : stage === 3 ? '조각마다 숫자 목록으로 펼침' : `토큰 ${N + 1}개 = [CLS] + 조각 ${N}개`}
            </text>
          )}
          {trA > 0.01 && (
            <g opacity={trA}>
              <rect x={trBox.x} y={trBox.y} width={trBox.w} height={trBox.h} rx={16} fill={withAlpha(c.brand, 0.08)} stroke={c.brand} strokeWidth={2} />
              <text x={trBox.x + 12} y={trBox.y + 19} style={{ fontSize: 12.5, fontWeight: 900, fill: c.brand }}>
                Transformer — 모든 토큰이 서로를 한 번에 봐요
              </text>
            </g>
          )}
          {tiles}
          {clsA > 0.01 && (
            <g opacity={clsA}>
              <rect x={cls.cx - cls.w / 2} y={cls.cy - cls.h / 2} width={cls.w} height={cls.h} rx={3} fill={withAlpha(c.series[4], 0.35)} stroke={c.series[4]} strokeWidth={2} />
              <text x={cls.cx} y={cls.cy + 3.5} textAnchor="middle" style={{ fontSize: 10, fontWeight: 900 }}>
                ★
              </text>
              <text x={cls.cx} y={cls.cy + cls.h / 2 + 10} textAnchor="middle" style={{ fontSize: d.cell < 24 ? 7 : 9, fontWeight: 900, fill: c.series[4] }}>
                CLS
              </text>
            </g>
          )}
          {trA > 0.01 && (
            <g opacity={trA}>
              <line x1={cls.cx} y1={trBox.y + trBox.h} x2={cls.cx} y2={trBox.y + trBox.h + 14} stroke={c.series[4]} strokeWidth={2.5} />
              <rect x={cls.cx - 8} y={trBox.y + trBox.h + 14} width={170} height={24} rx={12} fill={c.series[4]} />
              <text x={cls.cx + 4} y={trBox.y + trBox.h + 30} style={{ fontSize: 12, fontWeight: 900, fill: '#2a2540' }}>
                [CLS] → 분류기 → 🏠 집!
              </text>
            </g>
          )}
        </svg>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12, paddingTop: 0 }}>
        <div role="radiogroup" aria-label="ViT 단계" style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {STAGES.map((s, k) => (
            <button
              key={k}
              type="button"
              role="radio"
              aria-checked={k === stage}
              onClick={() => {
                setAuto(false);
                go(k);
              }}
              style={{
                padding: '6px 10px',
                borderRadius: 999,
                border: `2px solid ${k === stage ? c.brand : 'var(--border)'}`,
                background: k === stage ? withAlpha(c.brand, 0.14) : k < stage ? withAlpha(c.pos, 0.1) : 'var(--surface-solid)',
                color: 'var(--text)',
                fontSize: 12.5,
                fontWeight: k === stage ? 900 : 700,
                cursor: 'pointer'
              }}
            >
              {k + 1}. {s.label}
            </button>
          ))}
        </div>
        <div className="vz-playbar">
          <Button
            onClick={() => {
              setAuto(false);
              go(stage - 1);
            }}
            disabled={stage === 0}
          >
            ◀ 이전
          </Button>
          <Button
            variant="primary"
            onClick={() => {
              setAuto(false);
              go(stage + 1);
            }}
            disabled={stage === last}
          >
            다음 ▶
          </Button>
          <Button
            variant="ghost"
            onClick={() => {
              if (stage === last) {
                setPrev(last);
                setStage(0);
                setP(1);
              }
              setAuto(true);
            }}
            disabled={auto}
          >
            🎬 차례로 다 보기
          </Button>
        </div>
        <div className="vz-stats">
          <Stat label="조각 수 = 문장 길이" value={`${N}개`} />
          <Stat label="조각 하나의 숫자" value={rawPerPatch.toLocaleString()} sub="픽셀 × RGB" />
          <Stat label="어텐션이 비교할 짝" value={(N * N).toLocaleString()} sub="조각 수의 제곱" tone={N >= 64 ? 'warn' : undefined} />
        </div>
        <Note mood={note.mood}>{note.text}</Note>
      </div>

      <Controls>
        <Segmented
          label="자르는 칸 수"
          options={[
            { value: 2, label: '2×2 (큰 조각)' },
            { value: 4, label: '4×4' },
            { value: 8, label: '8×8 (작은 조각)' }
          ]}
          value={g}
          onChange={setG}
        />
      </Controls>
    </div>
  );
}
