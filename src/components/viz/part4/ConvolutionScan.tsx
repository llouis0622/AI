import { useEffect, useMemo, useRef, useState } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, PlayBar, Segmented } from '../core/ui';
import { useElementSize, useInterval } from '../core/hooks';
import { mixHex, useVizColors, withAlpha } from '../core/colors';

const N = 12;
const M = N - 2; // valid convolution output size

type Img = number[][];
const blank = (): Img => Array.from({ length: N }, () => new Array(N).fill(0));
const fromRows = (rows: string[]): Img => rows.map((r) => r.split('').map((ch) => (ch === '#' ? 1 : 0)));

const IMAGES: Record<string, { label: string; img: Img }> = {
  edge: { label: '세로 경계', img: Array.from({ length: N }, () => Array.from({ length: N }, (_, x) => (x < 6 ? 0.1 : 0.9))) },
  square: {
    label: '네모',
    img: Array.from({ length: N }, (_, y) => Array.from({ length: N }, (_, x) => (x >= 3 && x <= 8 && y >= 3 && y <= 8 ? 1 : 0)))
  },
  smile: {
    label: '웃는 얼굴',
    img: fromRows([
      '............',
      '...######...',
      '..#......#..',
      '.#........#.',
      '.#..#..#..#.',
      '.#........#.',
      '.#.#....#.#.',
      '.#..####..#.',
      '..#......#..',
      '...######...',
      '............',
      '............'
    ])
  },
  seven: {
    label: '숫자 7',
    img: fromRows([
      '............',
      '.#########..',
      '.#########..',
      '........##..',
      '.......##...',
      '......##....',
      '.....##.....',
      '.....##.....',
      '....##......',
      '....##......',
      '....##......',
      '............'
    ])
  }
};

const KERNELS: Record<string, { label: string; k: number[]; show?: string[] }> = {
  vert: { label: '세로 경계', k: [-1, 0, 1, -1, 0, 1, -1, 0, 1] },
  horiz: { label: '가로 경계', k: [-1, -1, -1, 0, 0, 0, 1, 1, 1] },
  blur: { label: '흐리게', k: new Array(9).fill(1 / 9), show: new Array(9).fill('1/9') },
  sharpen: { label: '선명하게', k: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
  custom: { label: '직접 만들기', k: [0, 0, 0, 0, 1, 0, 0, 0, 0] }
};
const CYCLE = [-1, 0, 1, 2];

function convolve(img: Img, k: number[]): number[][] {
  const out: number[][] = [];
  for (let y = 0; y < M; y++) {
    const row: number[] = [];
    for (let x = 0; x < M; x++) {
      let s = 0;
      for (let v = 0; v < 3; v++) for (let u = 0; u < 3; u++) s += img[y + v][x + u] * k[v * 3 + u];
      row.push(s);
    }
    out.push(row);
  }
  return out;
}

const r2 = (v: number) => {
  const s = (Math.round(v * 100) / 100).toString();
  return s === '-0' ? '0' : s;
};

export default function ConvolutionScan(_props: VizProps) {
  const c = useVizColors();
  const [imgKey, setImgKey] = useState('edge');
  const [img, setImg] = useState<Img>(IMAGES.edge.img);
  const [kKey, setKKey] = useState('vert');
  const [custom, setCustom] = useState<number[]>(KERNELS.custom.k);
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const painting = useRef<null | number>(null);
  const [box, { width }] = useElementSize<HTMLDivElement>();

  const kernel = kKey === 'custom' ? custom : KERNELS[kKey].k;
  const out = useMemo(() => convolve(img, kernel), [img, kernel]);
  const maxAbs = Math.max(1e-6, ...out.flat().map(Math.abs));
  const total = M * M;
  const cur = Math.min(pos, total - 1);
  const cy = Math.floor(cur / M);
  const cx = cur % M;
  const done = pos >= total;

  useInterval(
    () => {
      if (pos >= total) setPlaying(false);
      else setPos((p) => p + 1);
    },
    playing ? 260 / speed : null
  );
  useEffect(() => {
    setPos(imgKey === 'draw' ? total : 0);
    setPlaying(false);
  }, [imgKey, kKey, total]);

  const wide = width >= 640;
  const cellIn = wide ? 18 : Math.max(11, Math.floor((width - 48) / 2 / N));
  const cellOut = wide ? 21 : cellIn;

  // pixel value = brightness (0 dark, 1 bright), like a grayscale photo in both themes
  const pixel = (v: number) => mixHex('#17142b', '#f7f7fb', Math.max(0, Math.min(1, v)));
  const heat = (v: number) => {
    const t = Math.max(-1, Math.min(1, v / maxAbs));
    return t >= 0 ? mixHex(c.surface, c.series[2], t) : mixHex(c.surface, c.series[0], -t);
  };

  const paintAt = (x: number, y: number, first: boolean) => {
    setImg((im) => {
      const next = im.map((row) => row.slice());
      if (first) painting.current = next[y][x] > 0.5 ? 0 : 1;
      next[y][x] = painting.current ?? 1;
      return next;
    });
    setImgKey('draw');
    setPos(total);
  };

  const inputSvg = (
    <svg
      className="vz-svg"
      viewBox={`0 0 ${N * cellIn} ${N * cellIn}`}
      style={{ maxWidth: N * cellIn, touchAction: 'none' }}
      role="img"
      aria-label="입력 그림 (누르거나 끌어서 그릴 수 있어요)"
      onPointerUp={() => (painting.current = null)}
      onPointerLeave={() => (painting.current = null)}
    >
      {img.map((row, y) =>
        row.map((v, x) => (
          <rect
            key={`${x}-${y}`}
            x={x * cellIn}
            y={y * cellIn}
            width={cellIn - 1}
            height={cellIn - 1}
            rx={2}
            fill={pixel(v)}
            onPointerDown={(e) => {
              (e.currentTarget.ownerSVGElement as SVGSVGElement).releasePointerCapture?.(e.pointerId);
              paintAt(x, y, true);
            }}
            onPointerEnter={() => painting.current !== null && paintAt(x, y, false)}
            style={{ cursor: 'crosshair' }}
          />
        ))
      )}
      {!done && (
        <rect x={cx * cellIn - 1.5} y={cy * cellIn - 1.5} width={cellIn * 3 + 2} height={cellIn * 3 + 2} rx={4} fill={withAlpha(c.series[3], 0.12)} stroke={c.series[3]} strokeWidth={3} pointerEvents="none" />
      )}
    </svg>
  );

  const outputSvg = (
    <svg className="vz-svg" viewBox={`0 0 ${M * cellOut} ${M * cellOut}`} style={{ maxWidth: M * cellOut }} role="img" aria-label="특징 맵 (필터를 찍은 결과)">
      {out.map((row, y) =>
        row.map((v, x) => {
          const idx = y * M + x;
          const shown = idx < pos || done;
          return (
            <g key={`${x}-${y}`}>
              <rect x={x * cellOut} y={y * cellOut} width={cellOut - 1} height={cellOut - 1} rx={2} fill={shown ? heat(v) : 'var(--surface-3)'} opacity={shown ? 1 : 0.5} />
              {shown && cellOut >= 20 && Math.abs(v) > maxAbs * 0.05 && (
                <text x={x * cellOut + cellOut / 2 - 0.5} y={y * cellOut + cellOut / 2 + 3.5} textAnchor="middle" style={{ fontSize: 8.5, fontWeight: 800, fill: 'var(--text)' }}>
                  {Math.abs(v) >= 10 ? Math.round(v) : r2(v)}
                </text>
              )}
            </g>
          );
        })
      )}
      {!done && <rect x={cx * cellOut - 1.5} y={cy * cellOut - 1.5} width={cellOut + 2} height={cellOut + 2} rx={4} fill="none" stroke={c.series[3]} strokeWidth={3} />}
    </svg>
  );

  const kernelGrid = (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 44px)', gap: 4, justifyContent: 'center' }}>
      {kernel.map((v, i) => (
        <button
          key={i}
          type="button"
          disabled={kKey !== 'custom'}
          onClick={() =>
            setCustom((k) => {
              const next = k.slice();
              next[i] = CYCLE[(CYCLE.indexOf(k[i]) + 1) % CYCLE.length];
              return next;
            })
          }
          aria-label={`필터 칸 ${i + 1}: ${r2(v)}`}
          style={{
            height: 44,
            borderRadius: 10,
            border: `2px solid ${v > 0 ? c.series[2] : v < 0 ? c.series[0] : 'var(--border)'}`,
            background: v > 0 ? withAlpha(c.series[2], 0.15) : v < 0 ? withAlpha(c.series[0], 0.15) : 'var(--surface-solid)',
            fontWeight: 850,
            fontSize: KERNELS[kKey].show ? 12 : 16,
            cursor: kKey === 'custom' ? 'pointer' : 'default',
            color: 'var(--text)'
          }}
        >
          {KERNELS[kKey].show?.[i] ?? r2(v)}
        </button>
      ))}
    </div>
  );

  // the 9 products for the current window
  const products = [] as { a: number; k: number }[];
  for (let v = 0; v < 3; v++) for (let u = 0; u < 3; u++) products.push({ a: img[cy + v][cx + u], k: kernel[v * 3 + u] });
  const sum = products.reduce((s, p) => s + p.a * p.k, 0);

  const panel = (title: string, body: React.ReactNode, sub?: string) => (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, minWidth: 0 }}>
      <div className="vz-panel-title" style={{ textAlign: 'center' }}>
        {title}
      </div>
      {body}
      {sub && <div style={{ fontSize: 12, color: 'var(--text-3)', textAlign: 'center' }}>{sub}</div>}
    </div>
  );

  let note: { text: React.ReactNode; mood: 'smile' | 'happy' | 'wow' | 'think' | 'cheer'; tone?: 'good' };
  if (imgKey === 'draw') note = { text: '직접 그린 그림에 필터가 바로바로 반응해요! 필터를 바꿔 가며 어떤 무늬를 찾는지 보세요.', mood: 'wow' };
  else if (!done && pos === 0) note = { text: '테두리 친 3×3 창(도장)이 그림 위를 훑으며, 칸마다 "그림 값 × 필터 값"을 모두 더해 특징 맵 한 칸을 채워요. ▶ 시작!', mood: 'smile' };
  else if (done && kKey === 'vert' && imgKey === 'edge') note = { text: '어두움→밝음이 바뀌는 세로 경계에서만 값이 커지고, 고른 곳은 0이에요. 이 "세로 경계 지도"가 바로 특징 맵! 🎉', mood: 'cheer', tone: 'good' };
  else if (done) note = { text: '같은 도장 하나(숫자 9개)로 그림 전체를 훑었어요. 그림이 커져도 필요한 숫자는 그대로 9개 — 이게 합성곱의 힘이에요.', mood: 'happy' };
  else note = { text: `지금 창 위치의 계산: 9칸을 곱해서 더하면 ${r2(sum)} → 특징 맵의 테두리 친 칸에 들어가요.`, mood: 'think' };

  return (
    <div ref={box}>
      <div className="vz-pad" style={{ display: 'grid', gridTemplateColumns: wide ? 'auto auto auto' : '1fr 1fr', justifyContent: 'center', alignItems: 'start', gap: wide ? 22 : 12 }}>
        {panel('① 입력 그림 (눌러서 그리기)', inputSvg)}
        {wide && panel('② 필터 (도장 3×3)', kernelGrid, kKey === 'custom' ? '칸을 눌러 값 바꾸기' : '학습되는 숫자 9개')}
        {panel('③ 특징 맵', outputSvg, `${M}×${M}`)}
      </div>
      {!wide && <div className="vz-pad" style={{ paddingTop: 0 }}>{panel('② 필터 (도장 3×3)', kernelGrid, kKey === 'custom' ? '칸을 눌러 값 바꾸기' : '학습되는 숫자 9개')}</div>}

      <div className="vz-pad" style={{ display: 'grid', gap: 12, paddingTop: 0 }}>
        {!done && (
          <div style={{ padding: '10px 12px', borderRadius: 14, background: 'var(--surface-solid)', border: '1px solid var(--border)', fontSize: 13, fontFamily: 'var(--font-mono)', lineHeight: 1.8, overflowX: 'auto' }}>
            {products.map((p, i) => (
              <span key={i} style={{ whiteSpace: 'nowrap' }}>
                <span style={{ color: p.a > 0.5 ? 'var(--text)' : 'var(--text-3)' }}>{r2(p.a)}</span>×
                <span style={{ color: p.k > 0 ? c.series[2] : p.k < 0 ? c.series[0] : 'var(--text-3)', fontWeight: 800 }}>{KERNELS[kKey].show ? '1/9' : r2(p.k)}</span>
                {i < 8 ? ' + ' : ''}
              </span>
            ))}
            <span style={{ fontWeight: 900 }}> = {r2(sum)}</span>
          </div>
        )}
        <PlayBar
          playing={playing}
          onToggle={() => {
            if (done) setPos(0);
            setPlaying((p) => !p);
          }}
          onStep={done ? undefined : () => setPos((p) => Math.min(total, p + 1))}
          onReset={() => {
            setPlaying(false);
            setPos(0);
          }}
          speed={speed}
          onSpeed={setSpeed}
          extra={
            <Button variant="ghost" onClick={() => { setPlaying(false); setPos(total); }}>
              ⚡ 한 번에 다 찍기
            </Button>
          }
        />
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        <Segmented
          label="입력 그림"
          options={[...Object.entries(IMAGES).map(([k, v]) => ({ value: k, label: v.label })), { value: 'draw', label: '✏️ 지우고 그리기' }]}
          value={imgKey}
          onChange={(v) => {
            setImgKey(v);
            if (v === 'draw') setImg(blank());
            else setImg(IMAGES[v].img);
          }}
        />
        <Segmented label="필터" options={Object.entries(KERNELS).map(([k, v]) => ({ value: k, label: v.label }))} value={kKey} onChange={setKKey} />
      </Controls>
    </div>
  );
}
