import { useId, useMemo, useRef, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Legend, Note, PlayBar, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { useInView, usePrefersReducedMotion, useRafLoop } from '../core/hooks';
import { mixHex, useVizColors, withAlpha } from '../core/colors';
import { rng } from '../core/math';
import type { AlgoMood } from '@/components/mascot/Algo';
import { ALPHA_BAR, T } from './_pictures';
import { COMPONENTS, S_DATA, ddpm, denoise, onShape, scoreDir, uniformSteps } from './_gmm2d';

type Tab = 'forward' | 'reverse';
const N = 500;
const PAD = 300;
const UNIT = 50;
const PX = (v: number) => PAD / 2 + v * UNIT;
const PY = (v: number) => PAD / 2 - v * UNIT;
const TRACK = [7, 123, 311];
const ARROWS = Array.from({ length: 12 }, (_, i) => -2.75 + i * 0.5);
const STEP_OPTIONS = [1, 3, 10, 30, 100];

const DATA = (() => {
  const R = rng(20);
  const x0 = new Float32Array(2 * N);
  const eps = new Float32Array(2 * N);
  const group = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const comp = COMPONENTS[Math.floor(R.next() * COMPONENTS.length)];
    x0[2 * i] = comp.m[0] + S_DATA * R.normal();
    x0[2 * i + 1] = comp.m[1] + S_DATA * R.normal();
    eps[2 * i] = R.normal();
    eps[2 * i + 1] = R.normal();
    group[i] = comp.g;
  }
  return { x0, eps, group };
})();

const noiseCloud = (seed: number) => {
  const R = rng(100 + seed);
  return Float32Array.from({ length: 2 * N }, () => R.normal());
};

interface Rev {
  from: Float32Array;
  pts: Float32Array;
  idx: number;
  trails: [number, number][][];
}
const initRev = (seed: number): Rev => {
  const pts = noiseCloud(seed);
  return { from: pts, pts, idx: 0, trails: TRACK.map((j) => [[pts[2 * j], pts[2 * j + 1]]]) };
};

/** 2D diffusion on a smiley-shaped Gaussian mixture, with the exact score (what a perfect model would learn). */
export default function Diffusion2D({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const rootRef = useRef<HTMLDivElement>(null);
  const inView = useInView(rootRef);
  const [tab, setTab] = useState<Tab>('forward');
  const [arrows, setArrows] = useState(true);
  const headId = `d2-head-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const groupColor = [c.series[0], c.series[2], c.series[1]];

  // forward
  const [t, setT] = useState(0);
  const [sweeping, setSweeping] = useState(false);
  useRafLoop((dt) => {
    const next = Math.min(T, Math.round(t + Math.max(1, dt * 300)));
    setT(next);
    if (next >= T) setSweeping(false);
  }, sweeping && inView && tab === 'forward');

  // reverse
  const [nSteps, setNSteps] = useState(100);
  const steps = useMemo(() => uniformSteps(nSteps), [nSteps]);
  const [seed, setSeed] = useState(1);
  const [rev, setRev] = useState<Rev>(() => initRev(1));
  const [phase, setPhase] = useState(1);
  const [playing, setPlaying] = useState(false);
  const rngRef = useRef(rng(7));
  const done = rev.idx >= steps.length;
  const dur = Math.min(0.7, Math.max(0.045, 4 / nSteps));

  const doStep = () => {
    if (rev.idx >= steps.length) return false;
    const tNow = steps[rev.idx];
    const tPrev = rev.idx + 1 < steps.length ? steps[rev.idx + 1] : 0;
    const next = new Float32Array(2 * N);
    for (let i = 0; i < N; i++) {
      const x = rev.pts[2 * i];
      const y = rev.pts[2 * i + 1];
      const { mx, my } = denoise(x, y, tNow);
      next[2 * i] = ddpm(x, mx, tNow, tPrev, rngRef.current.normal());
      next[2 * i + 1] = ddpm(y, my, tNow, tPrev, rngRef.current.normal());
    }
    setRev({ from: rev.pts, pts: next, idx: rev.idx + 1, trails: rev.trails.map((tr, k) => [...tr, [next[2 * TRACK[k]], next[2 * TRACK[k] + 1]]]) });
    setPhase(reduce ? 1 : 0);
    return true;
  };

  useRafLoop((dt) => {
    if (phase < 1) {
      setPhase(Math.min(1, phase + dt / dur));
      return;
    }
    if (!playing) return;
    if (!doStep()) setPlaying(false);
  }, inView && tab === 'reverse' && (playing || phase < 1));

  const restart = (n = nSteps, s = seed + 1) => {
    setPlaying(false);
    setSeed(s);
    setRev(initRev(s));
    setPhase(1);
    rngRef.current = rng(7 + s);
    setNSteps(n);
  };

  // where to draw every point right now
  const ease = phase * phase * (3 - 2 * phase);
  const tFrom = rev.idx === 0 ? T : steps[rev.idx - 1];
  const tTo = rev.idx === 0 ? T : rev.idx < steps.length ? steps[rev.idx] : 0;
  const tRev = Math.round(tFrom + (tTo - tFrom) * ease);
  const tShow = tab === 'forward' ? t : tRev;
  const sa = Math.sqrt(ALPHA_BAR[tShow]);
  const sb = Math.sqrt(1 - ALPHA_BAR[tShow]);

  const pos = useMemo(() => {
    const out = new Float32Array(2 * N);
    if (tab === 'forward') {
      for (let i = 0; i < 2 * N; i++) out[i] = sa * DATA.x0[i] + sb * DATA.eps[i];
    } else {
      for (let i = 0; i < 2 * N; i++) out[i] = rev.from[i] + (rev.pts[i] - rev.from[i]) * ease;
    }
    return out;
  }, [tab, sa, sb, rev, ease]);

  const colors = useMemo(() => {
    const out: string[] = new Array(N);
    for (let i = 0; i < N; i++) {
      if (tab === 'forward') out[i] = groupColor[DATA.group[i]];
      else {
        const d = denoise(pos[2 * i], pos[2 * i + 1], Math.max(1, tRev));
        out[i] = mixHex(c.text, groupColor[d.group], Math.min(1, d.conf * 1.15));
      }
    }
    return out;
  }, [tab, pos, tRev, c.theme]); // eslint-disable-line react-hooks/exhaustive-deps

  const field = useMemo(() => {
    if (!arrows) return [];
    const out: { x: number; y: number; dx: number; dy: number }[] = [];
    for (const gy of ARROWS)
      for (const gx of ARROWS) {
        const [dx, dy] = scoreDir(gx, gy, Math.max(1, tShow));
        const m = Math.hypot(dx, dy) || 1;
        out.push({ x: gx, y: gy, dx: dx / m, dy: dy / m });
      }
    return out;
  }, [arrows, tShow]);

  const quality = useMemo(() => (tab === 'reverse' && done && phase >= 1 ? onShape(rev.pts) : null), [tab, done, phase, rev]);

  let note: { text: ReactNode; mood: AlgoMood; tone?: 'good' | 'bad' | 'warn' };
  if (tab === 'forward') {
    if (t === 0) note = { text: <>점 {N}개가 <strong>웃는 얼굴</strong> 모양으로 모여 있어요 — 이게 "진짜 데이터의 분포"예요. ▶ 흩뜨리기를 누르거나 t를 밀어 보세요.</>, mood: 'smile' };
    else if (t < 500) note = { text: <>점마다 노이즈가 조금씩 섞이며 얼굴이 <strong>번져요</strong>. 화살표는 각 자리에서 "데이터는 이쪽!"을 가리키는데, 확산 모델이 배우는 게 바로 이 <strong>화살표</strong>(섞인 노이즈의 반대 방향)예요.</>, mood: 'think' };
    else if (t < 950) note = { text: <>얼굴이 거의 사라졌어요. t가 크면 화살표는 대충 <strong>가운데</strong>를, t가 작으면 눈·입 같은 <strong>세부</strong>를 가리켜요. 거친 방향 → 섬세한 방향 순서로 청소하는 셈이에요.</>, mood: 'happy' };
    else note = { text: <>완전히 흩어져 동그란 <strong>노이즈 구름</strong>(표준 정규분포)이 됐어요. 2번 탭에서 이걸 거꾸로 되돌려 봐요! 🎬</>, mood: 'wow', tone: 'warn' };
  } else if (quality !== null) {
    note =
      quality > 0.9
        ? { text: <>웃는 얼굴이 다시 나타났어요! 점마다 출발한 노이즈가 달라서, 점들이 얼굴 곳곳을 <strong>골고루</strong> 채워요. 제자리 찾은 점 {Math.round(quality * 100)}%. 🎉 ↺ 처음으로 → 걸음 수를 1로 바꿔서도 해 보세요.</>, mood: 'cheer', tone: 'good' }
        : nSteps === 1
          ? { text: <>한 걸음에 끝내려니 모든 점이 <strong>한가운데</strong>(데이터의 평균)로 뛰어 버렸어요! 19장의 "흐린 평균" 문제예요. 그래서 확산은 <strong>작은 걸음을 여러 번</strong> 걸어요 — 대신 느리죠.</>, mood: 'wow', tone: 'bad' }
          : { text: <>걸음이 적으니 큰 걸음마다 "평균" 쪽으로 뛰어서 얼굴이 <strong>뭉개졌어요</strong>(제자리 찾은 점 {Math.round(quality * 100)}%). 걸음 수를 늘리면 좋아지지만 그만큼 느려져요 — 품질과 속도의 줄다리기!</>, mood: 'think', tone: 'warn' };
  } else if (rev.idx === 0) {
    note = { text: <>노이즈 구름에서 출발해요. 매 걸음 점들이 화살표(완벽한 모델의 짐작)를 따라 조금 움직이고, 새 노이즈도 살짝 섞여요. ▶ 시작! ({nSteps}걸음)</>, mood: 'smile' };
  } else {
    note = { text: <>점들이 조금씩 모양을 찾아가요. 색이 진해진 점은 "나는 어느 부분"인지 정해진 점이에요. 선은 점 세 개가 최근에 지나온 <strong>길</strong>이에요 — 처음엔 갈팡질팡, 나중엔 똑바로!</>, mood: 'happy' };
  }

  const svgMax = expanded ? 560 : 430;

  return (
    <div ref={rootRef}>
      <div className="vz-pad" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 280px), 1fr))', gap: 16, alignItems: 'start' }}>
        <div style={{ minWidth: 0, display: 'flex', justifyContent: 'center' }}>
          <svg
            viewBox={`0 0 ${PAD} ${PAD}`}
            role="img"
            aria-label="점 구름이 노이즈로 흩어졌다가 웃는 얼굴로 돌아오는 그림"
            style={{ width: '100%', maxWidth: svgMax, aspectRatio: '1', display: 'block', borderRadius: 14, border: `1px solid ${c.border}`, background: withAlpha(c.series[0], 0.03) }}
          >
            <defs>
              <marker id={headId} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill={c.axis} />
              </marker>
            </defs>
            {[-2, -1, 0, 1, 2].map((v) => (
              <g key={v} stroke={c.grid} strokeWidth={v === 0 ? 1.6 : 1}>
                <line x1={PX(v)} x2={PX(v)} y1={0} y2={PAD} />
                <line y1={PY(v)} y2={PY(v)} x1={0} x2={PAD} />
              </g>
            ))}
            {field.map((a, i) => (
              <line
                key={i}
                x1={PX(a.x) - a.dx * 6}
                y1={PY(a.y) + a.dy * 6}
                x2={PX(a.x) + a.dx * 7}
                y2={PY(a.y) - a.dy * 7}
                stroke={c.axis}
                strokeWidth={1.4}
                markerEnd={`url(#${headId})`}
              />
            ))}
            {Array.from({ length: N }, (_, i) => (
              <circle key={i} cx={PX(pos[2 * i])} cy={PY(pos[2 * i + 1])} r={2.3} fill={colors[i]} opacity={0.85} />
            ))}
            {tab === 'reverse' &&
              rev.trails.map((tr, k) => {
                const cur: [number, number] = [pos[2 * TRACK[k]], pos[2 * TRACK[k] + 1]];
                const pts = [...tr.slice(Math.max(0, rev.idx - 24), Math.max(1, rev.idx)), cur].map(([x, y]) => `${PX(x).toFixed(1)},${PY(y).toFixed(1)}`).join(' ');
                return (
                  <g key={`tr${k}`}>
                    <polyline points={pts} fill="none" stroke={c.ink} strokeWidth={1.4} strokeOpacity={0.5} strokeLinejoin="round" />
                    <circle cx={PX(cur[0])} cy={PY(cur[1])} r={4.6} fill={c.ink} stroke={c.theme === 'dark' ? '#0b1030' : '#ffffff'} strokeWidth={1.5} />
                  </g>
                );
              })}
            <rect x={5} y={5} width={70} height={20} rx={10} fill={c.theme === 'dark' ? 'rgba(11,16,48,0.88)' : 'rgba(255,253,249,0.9)'} stroke={c.border} />
            <text x={40} y={19.5} textAnchor="middle" style={{ fontSize: 12.5, fontWeight: 850, fill: c.ink, fontVariantNumeric: 'tabular-nums' }}>
              t = {tShow}
            </text>
          </svg>
        </div>

        <div style={{ minWidth: 0, display: 'grid', gap: 12 }}>
          <div className="vz-stats">
            {tab === 'forward' ? (
              <>
                <Stat label="원본 비율 √ᾱ" value={sa.toFixed(2)} />
                <Stat label="노이즈 비율 √(1-ᾱ)" value={sb.toFixed(2)} />
              </>
            ) : (
              <>
                <Stat label="남은 걸음" value={`${steps.length - rev.idx} / ${steps.length}`} />
                <Stat label="제자리 찾은 점" value={quality === null ? '…' : `${Math.round(quality * 100)}%`} tone={quality === null ? undefined : quality > 0.9 ? 'good' : quality > 0.5 ? 'warn' : 'bad'} />
              </>
            )}
          </div>
          <Note mood={note.mood} tone={note.tone}>
            {note.text}
          </Note>
          <Legend
            items={[
              { color: groupColor[0], label: '눈' },
              { color: groupColor[1], label: '입' },
              { color: groupColor[2], label: '얼굴 테두리' },
              ...(arrows ? [{ color: c.axis, label: '화살표 = "데이터는 이쪽!" (점수)' }] : [])
            ]}
          />
        </div>
      </div>

      <Controls>
        <Segmented
          options={[
            { value: 'forward', label: '1. 흩뜨리기 (앞 방향)' },
            { value: 'reverse', label: '2. 되돌리기 (역재생)' }
          ]}
          value={tab}
          onChange={(v) => {
            setTab(v as Tab);
            setSweeping(false);
            setPlaying(false);
          }}
        />
        {tab === 'forward' ? (
          <>
            <Slider label="t (노이즈 단계)" value={t} min={0} max={T} step={1} onChange={(v) => { setSweeping(false); setT(v); }} />
            <Button
              variant="primary"
              onClick={() => {
                if (!sweeping && t >= T) setT(0);
                setSweeping((s) => !s);
              }}
            >
              {sweeping ? '❚❚ 멈춤' : '▶ 흩뜨리기'}
            </Button>
          </>
        ) : (
          <>
            <PlayBar
              playing={playing}
              onToggle={() => {
                if (done && phase >= 1) {
                  restart();
                  setPlaying(true);
                  return;
                }
                setPlaying((p) => !p);
              }}
              onStep={() => {
                if (phase >= 1) doStep();
              }}
              onReset={() => restart()}
            />
            <Segmented
              label="걸음 수"
              options={STEP_OPTIONS.map((n) => ({ value: n, label: n === 1 ? '1걸음' : String(n) }))}
              value={nSteps}
              onChange={(n) => restart(n)}
            />
          </>
        )}
        <Toggle label="화살표 보기" checked={arrows} onChange={setArrows} />
      </Controls>
    </div>
  );
}
