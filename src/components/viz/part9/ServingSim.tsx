import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Slider, Stat } from '../core/ui';
import { MiniChart } from '../core/plot';
import { useElementSize, useInView, useRafLoop, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { clamp, rng } from '../core/math';

/*
 * Real-time queueing simulation of a model server with dynamic batching.
 * Requests arrive as a Poisson process; each GPU replica serves a batch in
 * BASE + PER × batchSize ms (GPUs love big batches).
 */
const BASE_MS = 18;
const PER_MS = 4;
const WON_PER_GPU_HOUR = 3000;
const MAX_REPLICAS = 8;
/** Requests waiting longer than this give up (time-out), like a real gateway. */
const TIMEOUT_MS = 3000;

interface Req {
  t0: number;
}
interface Replica {
  busyUntil: number;
  start: number;
  batch: Req[];
}
interface Particle {
  x: number;
  rep: number;
  lat: number;
}

const PRESETS = {
  chat: { label: '💬 채팅 (지연 중요)', lambda: 40, replicas: 2, batch: 4, wait: 2 },
  night: { label: '🌙 야간 분석 (처리량 중요)', lambda: 400, replicas: 2, batch: 32, wait: 30 },
  storm: { label: '🔥 트래픽 폭주', lambda: 220, replicas: 2, batch: 2, wait: 5 }
};
type PresetKey = keyof typeof PRESETS;

function percentile(sorted: number[], q: number) {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

export default function ServingSim({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [lambda, setLambda] = useState(60);
  const [replicas, setReplicas] = useState(2);
  const [maxBatch, setMaxBatch] = useState(8);
  const [maxWait, setMaxWait] = useState(10);
  const [preset, setPreset] = useState<PresetKey | ''>('');
  const [running, setRunning] = useState(!reduce);
  const [stats, setStats] = useState({ thr: 0, p50: 0, p95: 0, p99: 0, util: 0, queue: 0, avgBatch: 0, dropped: 0, warm: false, history: [] as number[] });

  const [box, { width }] = useElementSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inView = useInView(box, '50px');
  const sim = useRef({
    t: 0,
    nextArrival: 0,
    queue: [] as Req[],
    reps: Array.from({ length: MAX_REPLICAS }, () => ({ busyUntil: 0, start: 0, batch: [] as Req[] })) as Replica[],
    lats: [] as { t: number; lat: number }[],
    done: [] as number[],
    dropped: 0,
    util: 0,
    batches: [] as number[],
    particles: [] as Particle[],
    incoming: [] as { t0: number; y: number }[],
    lastStat: 0,
    lastHist: 0,
    history: [] as number[],
    rand: rng(11)
  });

  useEffect(() => setRunning(!reduce), [reduce]);

  const reset = () => {
    const s = sim.current;
    s.t = 0;
    s.nextArrival = 0;
    s.queue = [];
    s.reps.forEach((r) => Object.assign(r, { busyUntil: 0, start: 0, batch: [] }));
    s.lats = [];
    s.done = [];
    s.dropped = 0;
    s.util = 0;
    s.batches = [];
    s.particles = [];
    s.incoming = [];
    s.history = [];
    s.lastStat = 0;
    s.lastHist = 0;
    setStats({ thr: 0, p50: 0, p95: 0, p99: 0, util: 0, queue: 0, avgBatch: 0, dropped: 0, warm: false, history: [] });
  };

  const applyPreset = (k: PresetKey) => {
    const p = PRESETS[k];
    setPreset(k);
    setLambda(p.lambda);
    setReplicas(p.replicas);
    setMaxBatch(p.batch);
    setMaxWait(p.wait);
    reset();
  };

  const advance = (dtMs: number) => {
    const s = sim.current;
    const end = s.t + dtMs;
    while (s.t < end) {
      s.t += 1;
      const t = s.t;
      while (s.nextArrival <= t) {
        s.queue.push({ t0: s.nextArrival });
        s.incoming.push({ t0: s.nextArrival, y: s.rand.next() });
        s.nextArrival += -Math.log(1 - s.rand.next()) * (1000 / lambda);
      }
      while (s.queue.length && t - s.queue[0].t0 > TIMEOUT_MS) {
        s.queue.shift();
        s.dropped++;
      }
      let busyNow = 0;
      for (let i = 0; i < replicas; i++) {
        const r = s.reps[i];
        if (r.batch.length && t >= r.busyUntil) {
          for (const q of r.batch) {
            const lat = t - q.t0;
            s.lats.push({ t, lat });
            s.done.push(t);
            s.particles.push({ x: 0, rep: i, lat });
          }
          r.batch = [];
        }
        if (!r.batch.length && s.queue.length && (s.queue.length >= maxBatch || t - s.queue[0].t0 >= maxWait)) {
          r.batch = s.queue.splice(0, maxBatch);
          r.start = t;
          const dur = BASE_MS + PER_MS * r.batch.length;
          r.busyUntil = t + dur;
          s.batches.push(r.batch.length);
        }
        if (r.batch.length) busyNow++;
      }
      // utilisation as a ~3 s moving average, so it reacts when you move a slider
      s.util += (busyNow / replicas - s.util) / 3000;
    }
    while (s.lats.length && (s.lats[0].t < s.t - 5000 || s.lats.length > 4000)) s.lats.shift();
    if (s.batches.length > 200) s.batches.splice(0, s.batches.length - 200);
    while (s.done.length && s.done[0] < s.t - 5000) s.done.shift();
    if (s.particles.length > 160) s.particles.splice(0, s.particles.length - 160);
    if (s.incoming.length > 80) s.incoming.splice(0, s.incoming.length - 80);
  };

  const draw = () => {
    const cv = canvasRef.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = cv.clientWidth;
    const H = cv.clientHeight;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr)) {
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(H * dpr);
    }
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const s = sim.current;
    const xQ0 = W * 0.2;
    const xQ1 = W * 0.4;
    const xR0 = W * 0.46;
    const xR1 = W * 0.68;
    const xOut = W * 0.72;
    const top = 26;
    const bottom = H - 12;
    const gap = 6;
    const rh = Math.min(60, (bottom - top - gap * (replicas - 1)) / replicas);
    const stackTop = top + Math.max(0, (bottom - top - (replicas * rh + gap * (replicas - 1))) / 2);
    const repY = (i: number) => stackTop + i * (rh + gap);
    const latColor = (lat: number) => (lat <= 60 ? c.pos : lat <= 250 ? c.series[4] : c.neg);
    ctx.font = '700 12px Pretendard Variable, Pretendard, sans-serif';
    ctx.fillStyle = c.text;
    ctx.textAlign = 'center';
    ctx.fillText('🙋 사용자', W * 0.08, 16);
    ctx.fillText(`대기열 ${s.queue.length}`, (xQ0 + xQ1) / 2, 16);
    ctx.fillText(`GPU 서버 ${replicas}대`, (xR0 + xR1) / 2, 16);
    ctx.fillText('응답 →', W * 0.86, 16);
    // incoming requests travelling to the queue (they take ~400ms visually)
    for (const p of s.incoming) {
      const age = s.t - p.t0;
      if (age > 400) continue;
      const k = age / 400;
      ctx.fillStyle = withAlpha(c.series[3], 0.85);
      ctx.beginPath();
      ctx.arc(W * 0.04 + (xQ0 - W * 0.04) * k, top + p.y * (bottom - top), 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
    // queue as a grid of dots
    const dot = W < 500 ? 5 : 7;
    const qCols = Math.max(4, Math.floor((xQ1 - xQ0 + 3) / (dot + 3)));
    const maxDots = Math.floor((bottom - top) / (dot + 3)) * qCols;
    ctx.fillStyle = withAlpha(c.text.startsWith('#') ? c.text : '#888888', 0.06);
    ctx.fillRect(xQ0 - 6, top - 4, xQ1 - xQ0 + 12, bottom - top + 8);
    s.queue.slice(0, maxDots).forEach((q, i) => {
      const col = i % qCols;
      const row = Math.floor(i / qCols);
      const waited = s.t - q.t0;
      ctx.fillStyle = latColor(waited);
      ctx.beginPath();
      ctx.arc(xQ0 + col * (dot + 3) + dot / 2, bottom - row * (dot + 3) - dot / 2, dot / 2, 0, Math.PI * 2);
      ctx.fill();
    });
    if (s.queue.length > maxDots) {
      const label = `+${(s.queue.length - maxDots).toLocaleString('ko-KR')}개 더`;
      const tw = ctx.measureText(label).width + 14;
      ctx.fillStyle = c.neg;
      ctx.beginPath();
      ctx.roundRect((xQ0 + xQ1) / 2 - tw / 2, top + 2, tw, 20, 10);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, (xQ0 + xQ1) / 2, top + 16);
    }
    // replicas
    for (let i = 0; i < replicas; i++) {
      const r = s.reps[i];
      const y = repY(i);
      const busy = r.batch.length > 0;
      ctx.fillStyle = busy ? withAlpha(c.brand, 0.2) : withAlpha(c.brand, 0.06);
      ctx.strokeStyle = busy ? c.brand : withAlpha(c.brand, 0.4);
      ctx.lineWidth = busy ? 2 : 1;
      ctx.beginPath();
      ctx.roundRect(xR0, y, xR1 - xR0, rh, 8);
      ctx.fill();
      ctx.stroke();
      if (busy) {
        const k = Math.min(1, (s.t - r.start) / Math.max(1, r.busyUntil - r.start));
        ctx.fillStyle = withAlpha(c.brand, 0.35);
        ctx.beginPath();
        ctx.roundRect(xR0 + 2, y + rh - 6, (xR1 - xR0 - 4) * k, 4, 2);
        ctx.fill();
        const n = r.batch.length;
        const rows = rh >= 30 ? 2 : 1;
        const per = Math.min(n, 8 * rows);
        const step = (xR1 - xR0 - 46) / 8;
        for (let j = 0; j < per; j++) {
          ctx.fillStyle = c.series[0];
          ctx.beginPath();
          ctx.arc(xR0 + 10 + (j % 8) * step, y + 10 + Math.floor(j / 8) * 10, 3.2, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = c.text;
        ctx.textAlign = 'right';
        ctx.fillText(`${n}개`, xR1 - 8, y + Math.min(rh / 2 + 4, 22));
        ctx.textAlign = 'center';
      } else {
        ctx.fillStyle = withAlpha(c.text.startsWith('#') ? c.text : '#888888', 0.6);
        ctx.fillText('쉬는 중', (xR0 + xR1) / 2, y + rh / 2 + 4);
      }
    }
    // responses flying out
    for (const p of s.particles) {
      p.x += 0.012;
      const x = xOut + (W - xOut - 6) * p.x;
      if (p.x > 1) continue;
      ctx.fillStyle = latColor(p.lat);
      ctx.beginPath();
      ctx.arc(x, repY(Math.min(p.rep, replicas - 1)) + rh / 2 + Math.sin(p.x * 8) * 3, 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
    sim.current.particles = s.particles.filter((p) => p.x <= 1);
  };

  useRafLoop(
    (dt) => {
      advance(dt * 1000);
      draw();
      const s = sim.current;
      if (s.t - s.lastStat > 250) {
        s.lastStat = s.t;
        const sorted = s.lats.map((d) => d.lat).sort((a, b) => a - b);
        const windowMs = Math.min(5000, s.t);
        if (s.t - s.lastHist > 1000) {
          s.lastHist = s.t;
          s.history.push(percentile(sorted, 0.95));
          if (s.history.length > 60) s.history.shift();
        }
        setStats({
          thr: windowMs > 0 ? (s.done.length * 1000) / windowMs : 0,
          p50: percentile(sorted, 0.5),
          p95: percentile(sorted, 0.95),
          p99: percentile(sorted, 0.99),
          util: clamp(s.util, 0, 1),
          queue: s.queue.length,
          dropped: s.dropped,
          warm: s.t > 4000,
          avgBatch: s.batches.length ? s.batches.reduce((a, b) => a + b, 0) / s.batches.length : 0,
          history: [...s.history]
        });
      }
    },
    running && inView
  );

  // static frame when paused
  useEffect(() => {
    if (!running) draw();
  }); // eslint-disable-line react-hooks/exhaustive-deps

  const capacity = replicas * (maxBatch / ((BASE_MS + PER_MS * maxBatch) / 1000));
  const overloaded = lambda > capacity * 1.02;
  const costHour = replicas * WON_PER_GPU_HOUR;
  const costPer1k = stats.thr > 0.5 ? (costHour / (stats.thr * 3600)) * 1000 : 0;
  const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)}초` : `${Math.round(v)}ms`);
  const latTone = (v: number) => (v > 500 ? 'bad' : v > 150 ? 'warn' : 'good');

  let note: { t: ReactNode; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' | 'bad' };
  if (overloaded) note = { t: <>과부하! 😵 들어오는 요청(초당 {lambda})이 처리 능력(초당 약 {Math.round(capacity)})보다 많아서 <strong>대기열이 끝없이 길어지고</strong> 지연이 폭발해요. {stats.dropped > 0 && <> 벌써 {stats.dropped.toLocaleString('ko-KR')}건은 3초 넘게 기다리다 <strong>포기(시간 초과)</strong>했어요.</>} 서버를 늘리거나 배치를 키워 보세요.</>, mood: 'wow', tone: 'bad' };
  else if (stats.queue > 120) note = { t: <>처리 능력(초당 약 {Math.round(capacity)})이 이제 충분해요. 밀려 있던 요청 {stats.queue}건을 따라잡는 중이니 <strong>대기열이 줄어드는지</strong> 지켜보세요.</>, mood: 'think', tone: 'warn' };
  else if (stats.p99 > 3 * Math.max(1, stats.p50) && stats.p99 > 80) note = { t: <>평균적인 요청은 {ms(stats.p50)}인데, <strong>가장 느린 1%는 {ms(stats.p99)}</strong>나 기다려요. 그래서 평균이 아니라 p95·p99를 봐야 해요.</>, mood: 'think', tone: 'warn' };
  else if (maxBatch >= 16 && stats.avgBatch > 8) note = { t: <>큰 배치로 묶으니 GPU가 효율적으로 일해서 <strong>처리량</strong>이 크게 늘었어요. 대신 묶일 때까지 기다리느라 <strong>지연</strong>은 조금 늘어요 — 트레이드오프!</>, mood: 'happy' };
  else if (stats.warm && stats.util < 0.25 && stats.thr > 0) note = { t: <>GPU가 대부분 쉬고 있어요({Math.round(stats.util * 100)}%). 이 정도 요청이면 <strong>서버를 줄여서 비용</strong>을 아낄 수 있어요.</>, mood: 'think' };
  else note = { t: <><span style={{ color: c.series[3] }}>●</span> 들어오는 요청이 줄을 서면, <span style={{ color: c.brand }}>■</span> GPU 서버가 가져가 처리해요. 짧게 기다린 요청을 <strong>한 묶음(배치)</strong>으로 처리하면 GPU가 훨씬 효율적이에요. 응답 색: <span style={{ color: c.pos }}>●</span> 빠름 <span style={{ color: c.series[4] }}>●</span> 보통 <span style={{ color: c.neg }}>●</span> 느림</>, mood: 'smile' };

  return (
    <div ref={box}>
      <div style={{ padding: '12px 14px 0' }}>
        <canvas ref={canvasRef} className="vz-canvas" style={{ height: expanded ? 360 : Math.max(220, Math.min(300, width * 0.42)) }} aria-label="요청 대기열과 GPU 서버 시뮬레이션" role="img" />
      </div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <Note mood={note.mood} tone={note.tone}>
          {note.t}
        </Note>
        <div className="vz-stats">
          <Stat label="처리량" value={`${Math.round(stats.thr)}건/초`} />
          <Stat label="지연 p50 (보통)" value={ms(stats.p50)} />
          <Stat label="지연 p95" value={ms(stats.p95)} tone={latTone(stats.p95)} />
          <Stat label="지연 p99 (가장 느린 1%)" value={ms(stats.p99)} tone={latTone(stats.p99)} />
          <Stat label="GPU 사용률" value={`${Math.round(stats.util * 100)}%`} />
          <Stat label="비용" value={`시간당 ${costHour.toLocaleString('ko-KR')}원`} sub={costPer1k ? `요청 1,000건당 ${costPer1k.toFixed(1)}원` : ' '} />
        </div>
        <div>
          <div className="vz-panel-title">지연 p95의 변화 (세로: ms · 가로: 초)</div>
          <MiniChart series={[stats.history.length > 1 ? stats.history : [0, 0]]} colors={[c.series[2]]} height={110} yMin={0} />
        </div>
      </div>
      <Controls>
        <Segmented label="상황 프리셋" options={(Object.keys(PRESETS) as PresetKey[]).map((k) => ({ value: k, label: PRESETS[k].label }))} value={preset as PresetKey} onChange={(v) => applyPreset(v as PresetKey)} />
        <Slider label="초당 요청 수" value={lambda} min={5} max={500} step={5} onChange={(v) => { setLambda(v); setPreset(''); }} format={(v) => `${v}건`} accent="var(--sky)" />
        <Slider label="GPU 서버 수" value={replicas} min={1} max={MAX_REPLICAS} step={1} onChange={(v) => { setReplicas(v); setPreset(''); }} format={(v) => `${v}대`} />
        <Slider label="최대 배치 크기" value={maxBatch} min={1} max={32} step={1} onChange={(v) => { setMaxBatch(v); setPreset(''); }} format={(v) => `${v}개`} accent="var(--mint)" />
        <Slider label="배치 대기 시간" value={maxWait} min={0} max={40} step={1} onChange={(v) => { setMaxWait(v); setPreset(''); }} format={(v) => `${v}ms`} accent="var(--peach)" />
        <Button variant="primary" onClick={() => setRunning((r) => !r)}>
          {running ? '❚❚ 멈춤' : '▶ 시작'}
        </Button>
        <Button variant="ghost" onClick={reset}>
          ↺ 처음으로
        </Button>
      </Controls>
    </div>
  );
}
