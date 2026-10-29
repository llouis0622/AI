import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Legend, Note, PlayBar, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { MiniChart } from '../core/plot';
import { useElementSize, useInView, useRafLoop } from '../core/hooks';
import { useVizColors, type VizColors } from '../core/colors';
import { fmt, pct, rng } from '../core/math';
import { createNet, evaluate, forward, makeBuffers, makeDataset, trainBatch, type Act, type DatasetId, type Net } from './_nn';

const DATASETS: { value: DatasetId; label: string }[] = [
  { value: 'blobs', label: '두 덩어리' },
  { value: 'circle', label: '동그라미' },
  { value: 'xor', label: 'XOR' },
  { value: 'spiral', label: '나선' }
];
const DOMAIN = 1.25;
const OUT_RES = 64;
const THUMB_RES = 18;

type Sel = { layer: number; idx: number } | null;

function parseColor(s: string): [number, number, number] {
  const m = s.trim().match(/^#([0-9a-f]{6})$/i);
  if (m) return [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)];
  const r = s.match(/rgba?\(([^)]+)\)/);
  if (r) {
    const [a, b, c] = r[1].split(',').map((v) => parseFloat(v));
    return [a, b, c];
  }
  return [128, 128, 128];
}

export default function NNPlayground({ expanded }: VizProps) {
  const c = useVizColors();
  const [dataset, setDataset] = useState<DatasetId>('circle');
  const [layers, setLayers] = useState(1);
  const [neurons, setNeurons] = useState(4);
  const [act, setAct] = useState<Act>('tanh');
  const [lr, setLr] = useState(0.03);
  const [noise, setNoise] = useState(0.15);
  const [seed, setSeed] = useState(7);
  const [showTest, setShowTest] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [epoch, setEpoch] = useState(0);
  const [stats, setStats] = useState({ train: NaN, test: NaN, acc: NaN });
  const [hist, setHist] = useState<{ train: number[]; test: number[] }>({ train: [], test: [] });
  const [sel, setSel] = useState<Sel>(null);
  const [ver, setVer] = useState(0);

  const netRef = useRef<Net | null>(null);
  const acc = useRef(0);
  const outRef = useRef<HTMLCanvasElement>(null);
  const netCanvas = useRef<HTMLCanvasElement>(null);
  const [wrap, { width }] = useElementSize<HTMLDivElement>();
  const inView = useInView(wrap, '120px');

  const data = useMemo(
    () => ({ train: makeDataset(dataset, 240, noise, seed), test: makeDataset(dataset, 120, noise, seed + 1000) }),
    [dataset, noise, seed]
  );
  const sizes = useMemo(() => [2, ...Array.from({ length: layers }, () => neurons), 1], [layers, neurons]);

  const measure = useCallback(() => {
    const net = netRef.current!;
    const tr = evaluate(net, data.train.X, data.train.Y);
    const te = evaluate(net, data.test.X, data.test.Y);
    return { train: tr.loss, test: te.loss, acc: te.acc };
  }, [data]);

  /* ---------- drawing ---------- */
  const draw = useCallback(() => {
    const net = netRef.current;
    if (!net) return;
    drawOutput(outRef.current, net, data, showTest, sel, c);
    drawNetwork(netCanvas.current, net, sel, c);
  }, [data, showTest, sel, c]);

  const reset = useCallback(() => {
    netRef.current = createNet(sizes, act, seed * 31 + layers * 7 + neurons);
    setEpoch(0);
    const m = measure();
    setStats(m);
    setHist({ train: [m.train], test: [m.test] });
    setSel(null);
    setVer((v) => v + 1);
  }, [sizes, act, seed, layers, neurons, measure]);

  useEffect(() => {
    setPlaying(false);
    reset();
  }, [reset]);

  useEffect(() => {
    draw();
  }, [draw, width, epoch, expanded, ver]);

  const trainEpoch = () => {
    const net = netRef.current!;
    const r = rng(epoch * 13 + 1);
    const idx = data.train.X.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(r.next() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (let b = 0; b < idx.length; b += 10) trainBatch(net, data.train.X, data.train.Y, idx.slice(b, b + 10), lr);
  };

  useRafLoop(
    (dt) => {
      acc.current += dt * 24 * speed;
      let ran = 0;
      while (acc.current >= 1 && ran < 6) {
        trainEpoch();
        acc.current -= 1;
        ran++;
      }
      if (!ran) return;
      const m = measure();
      setStats(m);
      setEpoch((e) => e + ran);
      setHist((h) => {
        const train = [...h.train, m.train];
        const test = [...h.test, m.test];
        return train.length > 400 ? { train: train.filter((_, i) => i % 2 === 0), test: test.filter((_, i) => i % 2 === 0) } : { train, test };
      });
    },
    playing && inView
  );

  const step = () => {
    trainEpoch();
    const m = measure();
    setStats(m);
    setEpoch((e) => e + 1);
    setHist((h) => ({ train: [...h.train, m.train], test: [...h.test, m.test] }));
  };

  const onNetClick = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const net = netRef.current;
    const cv = netCanvas.current;
    if (!net || !cv) return;
    const rect = cv.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const lay = networkLayout(net.sizes, rect.width, rect.height);
    for (const node of lay.nodes) {
      if (node.layer === 0 || node.layer === net.sizes.length - 1) continue;
      if (Math.abs(px - node.x) <= lay.size / 2 + 3 && Math.abs(py - node.y) <= lay.size / 2 + 3) {
        setSel((s) => (s && s.layer === node.layer && s.idx === node.idx ? null : { layer: node.layer, idx: node.idx }));
        return;
      }
    }
    setSel(null);
  };

  const total = layers * neurons;
  let note: { text: string; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' };
  if (sel) note = { text: `${sel.layer}층의 ${sel.idx + 1}번 뉴런이 "보는 것"이에요. 뉴런마다 공간을 서로 다르게 나누고, 다음 층이 이 그림들을 조합해요. 다시 누르면 출력으로 돌아가요.`, mood: 'wow' };
  else if (epoch === 0) note = { text: '▶ 시작을 누르면 신경망이 점들을 가르는 법을 스스로 배워요. 배경 색이 신경망의 "생각"이에요. 가운데 뉴런 그림을 눌러 보세요!', mood: 'smile' };
  else if (stats.acc >= 0.96) note = { text: `테스트 정확도 ${pct(stats.acc)}! 🎉 처음 보는 점(테스트)도 잘 맞혀요. 뉴런 그림을 눌러 "표현"이 어떻게 생겼는지 보세요.`, mood: 'cheer', tone: 'good' };
  else if (total <= 1 && dataset !== 'blobs' && epoch > 60) note = { text: '뉴런 하나는 직선 하나만 그을 수 있어요. 뉴런을 늘려 보세요!', mood: 'think', tone: 'warn' };
  else if (dataset === 'spiral' && epoch > 250 && stats.acc < 0.85) note = { text: '나선은 어려운 문제예요. 층을 2~3개로, 뉴런을 6개 이상으로 늘려 보세요.', mood: 'think', tone: 'warn' };
  else if (stats.train < 0.12 && stats.test > stats.train * 2.2) note = { text: '훈련 손실만 뚝 떨어지고 테스트는 그대로 — 과적합의 냄새가 나요(4장). 잡음을 줄이거나 뉴런을 줄여 보세요.', mood: 'think', tone: 'warn' };
  else note = { text: '경사하강법이 매 순간 가중치를 조금씩 고쳐요. 선의 굵기는 가중치의 크기, 색은 부호예요.', mood: 'happy' };

  const wide = width >= 620;
  return (
    <div ref={wrap}>
      <div className="vz-pad" style={{ display: 'grid', gridTemplateColumns: wide ? 'minmax(0, 1.05fr) minmax(0, 1fr)' : '1fr', gap: 14, alignItems: 'start' }}>
        <div style={{ order: wide ? 0 : 1 }}>
          <div className="vz-panel-title" style={{ marginBottom: 6 }}>
            신경망 속 (뉴런 그림을 눌러 보세요)
          </div>
          <canvas
            ref={netCanvas}
            className="vz-canvas"
            style={{ height: Math.max(expanded ? 300 : 210, Math.min(expanded ? 480 : 360, neurons * 50 + 70)), borderRadius: 16, background: 'var(--viz-panel)', border: '1px solid var(--border)', cursor: 'pointer', touchAction: 'pan-y' }}
            onPointerDown={onNetClick}
            aria-label="신경망 구조: 각 뉴런이 보는 그림과 가중치 선"
            role="img"
          />
        </div>
        <div style={{ order: wide ? 1 : 0 }}>
          <div className="vz-panel-title" style={{ marginBottom: 6, display: 'flex', justifyContent: 'space-between', gap: 8 }}>
            <span>{sel ? `${sel.layer}층 ${sel.idx + 1}번 뉴런이 보는 그림` : '출력: 신경망의 판단'}</span>
            {sel && (
              <button type="button" onClick={() => setSel(null)} style={{ color: 'var(--link)', fontWeight: 800 }}>
                출력 보기 ↩
              </button>
            )}
          </div>
          <canvas
            ref={outRef}
            className="vz-canvas"
            style={{ aspectRatio: '1', height: 'auto', borderRadius: 16, border: '1px solid var(--border)', touchAction: 'pan-y' }}
            aria-label="데이터 점과 신경망이 칠한 판단 영역"
            role="img"
          />
        </div>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12, paddingTop: 0 }}>
        <PlayBar
          playing={playing}
          onToggle={() => setPlaying((p) => !p)}
          onStep={step}
          onReset={reset}
          speed={speed}
          onSpeed={setSpeed}
        />
        <div className="vz-stats">
          <Stat label="에포크 (훑은 횟수)" value={epoch} />
          <Stat label="훈련 손실" value={fmt(stats.train, 3)} />
          <Stat label="테스트 손실" value={fmt(stats.test, 3)} />
          <Stat label="테스트 정확도" value={Number.isFinite(stats.acc) ? pct(stats.acc) : '—'} tone={stats.acc >= 0.96 ? 'good' : stats.acc >= 0.8 ? 'brand' : 'warn'} />
        </div>
        {hist.train.length > 1 && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
              <span className="vz-panel-title">손실 곡선</span>
              <Legend items={[{ color: c.series[0], label: '훈련' }, { color: c.series[2], label: '테스트' }]} />
            </div>
            <MiniChart series={[hist.train, hist.test]} colors={[c.series[0], c.series[2]]} height={110} yMin={0} />
          </div>
        )}
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        <Segmented label="데이터" options={DATASETS} value={dataset} onChange={(v) => setDataset(v as DatasetId)} />
        <Segmented
          label="은닉층 수"
          options={[
            { value: 1, label: '1층' },
            { value: 2, label: '2층' },
            { value: 3, label: '3층' }
          ]}
          value={layers}
          onChange={(v) => setLayers(Number(v))}
        />
        <Slider label="층마다 뉴런 수" value={neurons} min={1} max={8} step={1} onChange={setNeurons} format={(v) => `${v}개`} />
        <Segmented
          label="활성 함수"
          options={[
            { value: 'tanh', label: 'tanh' },
            { value: 'relu', label: 'ReLU' }
          ]}
          value={act}
          onChange={(v) => setAct(v as Act)}
        />
        <Segmented
          label="학습률"
          options={[
            { value: 0.003, label: '0.003' },
            { value: 0.01, label: '0.01' },
            { value: 0.03, label: '0.03' },
            { value: 0.1, label: '0.1' }
          ]}
          value={lr}
          onChange={(v) => setLr(Number(v))}
        />
        <Slider label="잡음" value={noise} min={0} max={0.6} step={0.05} onChange={setNoise} format={(v) => v.toFixed(2)} accent="var(--peach)" />
        <Toggle label="테스트 점도 보기" checked={showTest} onChange={setShowTest} />
        <Button size="sm" variant="ghost" onClick={() => setSeed((s) => s + 1)}>
          🎲 새 데이터
        </Button>
      </Controls>
    </div>
  );
}

/* ---------- canvas helpers ---------- */

function prepCanvas(cv: HTMLCanvasElement | null) {
  if (!cv) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth;
  const h = cv.clientHeight;
  if (!w || !h) return null;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

/** Heat colors: value v in [-1, 1] → blend background toward neg/pos class colors. */
function heatRGB(v: number, bg: number[], neg: number[], pos: number[], strength = 0.85): [number, number, number] {
  const t = Math.min(1, Math.abs(v)) * strength;
  const target = v >= 0 ? pos : neg;
  return [bg[0] + (target[0] - bg[0]) * t, bg[1] + (target[1] - bg[1]) * t, bg[2] + (target[2] - bg[2]) * t];
}

const offscreen = (n: number) => {
  const cv = document.createElement('canvas');
  cv.width = n;
  cv.height = n;
  return cv;
};

function hiddenMaps(net: Net, res: number) {
  // activation of every neuron on a res×res grid: maps[layer][idx][cell]
  const { acts, zs } = makeBuffers(net);
  const maps = net.sizes.map((n) => Array.from({ length: n }, () => new Float64Array(res * res)));
  for (let j = 0; j < res; j++) {
    for (let i = 0; i < res; i++) {
      const x1 = -DOMAIN + ((i + 0.5) / res) * 2 * DOMAIN;
      const x2 = DOMAIN - ((j + 0.5) / res) * 2 * DOMAIN;
      forward(net, x1, x2, acts, zs);
      for (let l = 0; l < net.sizes.length; l++) for (let k = 0; k < net.sizes[l]; k++) maps[l][k][j * res + i] = acts[l][k];
    }
  }
  return maps;
}

function drawOutput(
  cv: HTMLCanvasElement | null,
  net: Net,
  data: { train: { X: [number, number][]; Y: number[] }; test: { X: [number, number][]; Y: number[] } },
  showTest: boolean,
  sel: Sel,
  c: VizColors
) {
  const p = prepCanvas(cv);
  if (!p) return;
  const { ctx, w, h } = p;
  const bg = parseColor(c.surface);
  const pos = parseColor(c.series[2]);
  const neg = parseColor(c.series[0]);
  const img = offscreen(OUT_RES);
  const ictx = img.getContext('2d')!;
  const id = ictx.createImageData(OUT_RES, OUT_RES);
  if (sel) {
    const maps = hiddenMaps(net, OUT_RES);
    const m = maps[sel.layer][sel.idx];
    let mx = 1e-6;
    for (const v of m) mx = Math.max(mx, Math.abs(v));
    for (let k = 0; k < m.length; k++) {
      const v = net.act === 'relu' ? m[k] / mx : m[k];
      const [r, g, b] = heatRGB(v, bg, neg, pos, 0.95);
      id.data.set([r, g, b, 255], k * 4);
    }
  } else {
    for (let j = 0; j < OUT_RES; j++) {
      for (let i = 0; i < OUT_RES; i++) {
        const x1 = -DOMAIN + ((i + 0.5) / OUT_RES) * 2 * DOMAIN;
        const x2 = DOMAIN - ((j + 0.5) / OUT_RES) * 2 * DOMAIN;
        const prob = forward(net, x1, x2);
        const [r, g, b] = heatRGB((prob - 0.5) * 2, bg, neg, pos, 0.5);
        id.data.set([r, g, b, 255], (j * OUT_RES + i) * 4);
      }
    }
  }
  ictx.putImageData(id, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, w, h);
  // decision boundary hint: faint grid
  ctx.strokeStyle = c.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(w / 2, 0);
  ctx.lineTo(w / 2, h);
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
  const toPx = (x1: number, x2: number): [number, number] => [((x1 + DOMAIN) / (2 * DOMAIN)) * w, ((DOMAIN - x2) / (2 * DOMAIN)) * h];
  const r = Math.max(3, Math.min(5.5, w / 70));
  const dot = (X: [number, number][], Y: number[], hollow: boolean) => {
    for (let k = 0; k < X.length; k++) {
      const [px, py] = toPx(X[k][0], X[k][1]);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      const color = Y[k] === 1 ? c.series[2] : c.series[0];
      if (hollow) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
      } else {
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = c.surface;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  };
  if (!sel) {
    dot(data.train.X, data.train.Y, false);
    if (showTest) dot(data.test.X, data.test.Y, true);
  }
}

function networkLayout(sizes: number[], w: number, h: number) {
  const cols = sizes.length;
  const padX = 34;
  const padY = 22;
  const maxN = Math.max(...sizes);
  const spacing = Math.min(56, (h - padY * 2) / Math.max(1, maxN));
  const size = Math.max(16, Math.min(40, spacing - 8, (w - padX * 2) / (cols * 1.9)));
  const nodes: { layer: number; idx: number; x: number; y: number }[] = [];
  sizes.forEach((n, l) => {
    const x = cols === 1 ? w / 2 : padX + (l * (w - padX * 2)) / (cols - 1);
    for (let k = 0; k < n; k++) {
      const y = h / 2 + (k - (n - 1) / 2) * spacing;
      nodes.push({ layer: l, idx: k, x, y });
    }
  });
  return { nodes, size };
}

function drawNetwork(cv: HTMLCanvasElement | null, net: Net, sel: Sel, c: VizColors) {
  const p = prepCanvas(cv);
  if (!p) return;
  const { ctx, w, h } = p;
  const { nodes, size } = networkLayout(net.sizes, w, h);
  const at = (l: number, k: number) => nodes.find((n) => n.layer === l && n.idx === k)!;
  // edges
  for (let l = 0; l < net.sizes.length - 1; l++) {
    const inN = net.sizes[l];
    const outN = net.sizes[l + 1];
    for (let o = 0; o < outN; o++) {
      for (let i = 0; i < inN; i++) {
        const wv = net.W[l][o * inN + i];
        const a = at(l, i);
        const b = at(l + 1, o);
        ctx.beginPath();
        ctx.moveTo(a.x + size / 2, a.y);
        ctx.lineTo(b.x - size / 2, b.y);
        ctx.strokeStyle = wv >= 0 ? c.series[2] : c.series[0];
        ctx.globalAlpha = Math.min(0.9, 0.18 + Math.abs(wv) * 0.25);
        ctx.lineWidth = Math.min(6, 0.6 + Math.abs(wv) * 1.4);
        ctx.stroke();
      }
    }
  }
  ctx.globalAlpha = 1;
  // node thumbnails
  const maps = hiddenMaps(net, THUMB_RES);
  const bg = parseColor(c.surface);
  const pos = parseColor(c.series[2]);
  const neg = parseColor(c.series[0]);
  const img = offscreen(THUMB_RES);
  const ictx = img.getContext('2d')!;
  ctx.imageSmoothingEnabled = true;
  for (const n of nodes) {
    const m = maps[n.layer][n.idx];
    let mx = 1e-6;
    for (const v of m) mx = Math.max(mx, Math.abs(v));
    const id = ictx.createImageData(THUMB_RES, THUMB_RES);
    for (let k = 0; k < m.length; k++) {
      let v: number;
      if (n.layer === 0) v = m[k] / DOMAIN;
      else if (n.layer === net.sizes.length - 1) v = (m[k] - 0.5) * 2;
      else v = net.act === 'relu' ? m[k] / mx : m[k];
      const [r, g, b] = heatRGB(v, bg, neg, pos, 0.9);
      id.data.set([r, g, b, 255], k * 4);
    }
    ictx.putImageData(id, 0, 0);
    const x0 = n.x - size / 2;
    const y0 = n.y - size / 2;
    ctx.save();
    roundRect(ctx, x0, y0, size, size, 7);
    ctx.clip();
    ctx.drawImage(img, x0, y0, size, size);
    ctx.restore();
    const isSel = sel && sel.layer === n.layer && sel.idx === n.idx;
    roundRect(ctx, x0, y0, size, size, 7);
    ctx.strokeStyle = isSel ? c.brand : c.border;
    ctx.lineWidth = isSel ? 3 : 1.5;
    ctx.stroke();
  }
  // labels
  ctx.fillStyle = c.text;
  ctx.font = '700 11px Pretendard Variable, Pretendard, system-ui, sans-serif';
  ctx.textAlign = 'center';
  const first = at(0, 0);
  const second = at(0, 1);
  ctx.fillText('x₁', first.x, first.y - size / 2 - 5);
  ctx.fillText('x₂', second.x, second.y + size / 2 + 13);
  const outN = at(net.sizes.length - 1, 0);
  ctx.fillText('출력', outN.x, outN.y - size / 2 - 5);
  for (let l = 1; l < net.sizes.length - 1; l++) {
    const top = at(l, 0);
    ctx.fillText(`${l}층`, top.x, Math.max(12, top.y - size / 2 - 5));
  }
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
