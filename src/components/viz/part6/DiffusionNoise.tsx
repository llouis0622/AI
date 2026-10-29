import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Legend, Note, PlayBar, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { Plot, linePath } from '../core/plot';
import { useInView, useInterval, usePrefersReducedMotion, useRafLoop } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { rng } from '../core/math';
import type { AlgoMood } from '@/components/mascot/Algo';
import { ALPHA_BAR, SIZE, T, guess, noisy, pictures, reverseStep, reverseSteps } from './_pictures';

type Tab = 'forward' | 'reverse';
const STEPS = reverseSteps(80);
const D = SIZE * SIZE * 3;
const CURVE = Array.from({ length: 101 }, (_, i) => i * 10);

const Num = ({ children }: { children: ReactNode }) => <span style={{ fontVariantNumeric: 'tabular-nums' }}>{children}</span>;

const gaussian = (seed: number) => {
  const r = rng(seed);
  return Float32Array.from({ length: D }, () => r.normal());
};

/** Box blur — "squinting" averages the noise away and reveals the hidden picture. */
function squintBlur(src: Float32Array, r = 2) {
  const out = new Float32Array(src.length);
  for (let y = 0; y < SIZE; y++)
    for (let x = 0; x < SIZE; x++)
      for (let k = 0; k < 3; k++) {
        let sum = 0;
        let n = 0;
        for (let dy = -r; dy <= r; dy++)
          for (let dx = -r; dx <= r; dx++) {
            const yy = y + dy;
            const xx = x + dx;
            if (yy < 0 || yy >= SIZE || xx < 0 || xx >= SIZE) continue;
            sum += src[(yy * SIZE + xx) * 3 + k];
            n++;
          }
        out[(y * SIZE + x) * 3 + k] = (sum / n) * 1.6;
      }
  return out;
}

/** A 32×32 picture drawn on a canvas, scaled up with crisp pixels. */
function Pixels({ data, label, border }: { data: Float32Array; label: string; border: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(SIZE, SIZE);
    for (let p = 0; p < SIZE * SIZE; p++) {
      for (let k = 0; k < 3; k++) img.data[p * 4 + k] = Math.max(0, Math.min(255, (data[p * 3 + k] + 1) * 127.5));
      img.data[p * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data]);
  return (
    <canvas
      ref={ref}
      width={SIZE}
      height={SIZE}
      role="img"
      aria-label={label}
      style={{ width: '100%', aspectRatio: '1', imageRendering: 'pixelated', borderRadius: 10, display: 'block', border: `1.5px solid ${border}` }}
    />
  );
}

function Caption({ children, color }: { children: ReactNode; color: string }) {
  return <div style={{ fontSize: 12, fontWeight: 750, color, textAlign: 'center', marginTop: 4, lineHeight: 1.3 }}>{children}</div>;
}

/** Forward noising with a t slider, then the reverse (denoising) process run with an exact noise predictor. */
export default function DiffusionNoise({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const pics = useMemo(() => pictures(), []);
  const picData = useMemo(() => pics.map((p) => p.data), [pics]);
  const [tab, setTab] = useState<Tab>('forward');
  const [squint, setSquint] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const inView = useInView(rootRef);

  // ---- forward ----
  const [pic, setPic] = useState(0);
  const [t, setT] = useState(0);
  const [epsSeed, setEpsSeed] = useState(7);
  const [sweeping, setSweeping] = useState(false);
  const eps = useMemo(() => gaussian(epsSeed), [epsSeed]);
  const xt = useMemo(() => noisy(pics[pic].data, eps, t), [pics, pic, eps, t]);
  const xtShown = useMemo(() => (squint ? squintBlur(xt) : xt), [xt, squint]);
  const a = Math.sqrt(ALPHA_BAR[t]);
  const b = Math.sqrt(1 - ALPHA_BAR[t]);
  useRafLoop((dt) => {
    const next = Math.min(T, Math.round(t + Math.max(1, dt * 260)));
    setT(next);
    if (next >= T) setSweeping(false);
  }, sweeping && inView && tab === 'forward');

  // ---- reverse ----
  const [seed, setSeed] = useState(1);
  const [freshNoise, setFreshNoise] = useState(true);
  const [step, setStep] = useState(0);
  const [x, setX] = useState<Float32Array>(() => gaussian(1001));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [tally, setTally] = useState<number[]>([]);
  const noiseRng = useRef(rng(51));
  const done = step >= STEPS.length;
  const curT = done ? 0 : STEPS[step];
  const g = useMemo(() => guess(x, Math.max(1, curT), picData), [x, curT, picData]);
  const best = g.probs.indexOf(Math.max(...g.probs));
  const xShown = useMemo(() => (squint && !done ? squintBlur(x) : x), [x, squint, done]);

  const restart = (nextSeed = seed + 1) => {
    setPlaying(false);
    setSeed(nextSeed);
    setStep(0);
    setX(gaussian(1000 + nextSeed));
    noiseRng.current = rng(50 + nextSeed);
  };
  const advance = () => {
    if (step >= STEPS.length) return;
    const tNow = STEPS[step];
    const tPrev = step + 1 < STEPS.length ? STEPS[step + 1] : 0;
    const z = freshNoise ? Float32Array.from({ length: D }, () => noiseRng.current.normal()) : null;
    const nx = reverseStep(x, g.x0, tNow, tPrev, z);
    setX(nx);
    setStep(step + 1);
    if (step + 1 >= STEPS.length) {
      setPlaying(false);
      const fin = guess(nx, 1, picData);
      setTally((prev) => [...prev, fin.probs.indexOf(Math.max(...fin.probs))]);
    }
  };
  useInterval(advance, playing && inView && tab === 'reverse' ? (speed < 1 ? 130 : speed > 1 ? 25 : 65) : null);

  const tallyCounts = pics.map((_, k) => tally.filter((v) => v === k).length);
  const recent = tally.slice(-4);
  const allSame = !freshNoise && recent.length >= 3 && recent.every((v) => v === recent[0]);

  let note: { text: ReactNode; mood: AlgoMood; tone?: 'good' | 'bad' | 'warn' };
  if (tab === 'forward') {
    if (t === 0) note = { text: <>지금은 <strong>원본 그대로</strong>예요(t = 0). 아래 t 슬라이더를 오른쪽으로 밀거나 ▶ 망가뜨리기를 눌러 보세요.</>, mood: 'smile' };
    else if (t < 250) note = { text: <>원본 비율이 아직 <Num>{a.toFixed(2)}</Num>라 그림이 잘 보여요. 노이즈가 살짝 끼었을 뿐이에요.</>, mood: 'happy' };
    else if (t < 700) note = { text: <>원본은 ×<Num>{a.toFixed(2)}</Num>로 <strong>희미해지고</strong>, 노이즈는 ×<Num>{b.toFixed(2)}</Num>로 <strong>커졌어요</strong>. 어느 t든 이렇게 공식 한 줄로 <strong>바로</strong> 만들 수 있어요 — 앞 방향엔 배울 게 없어요.</>, mood: 'think' };
    else if (t < 850 && !squint) note = { text: <>우리 눈엔 거의 노이즈지만, 원본이 ×<Num>{a.toFixed(2)}</Num>만큼 <strong>숨어 있어요</strong>. 「😑 눈 가늘게 뜨고 보기」를 켜 보세요 — 노이즈가 뭉개지며 그림이 드러나요!</>, mood: 'wink' };
    else note = { text: <>원본 비율이 거의 0 — 이제 그냥 <strong>노이즈</strong>예요. 이 과정을 <strong>거꾸로</strong> 되돌리는 법을 배우는 게 확산 모델이에요. 2번 탭으로 가 보세요! 🎬</>, mood: 'wow', tone: 'warn' };
  } else if (allSame) {
    note = { text: <>새 노이즈를 끄니 <strong>매번 같은 그림</strong>만 나와요 — 다양성이 사라졌어요! 걸음마다 섞는 작은 무작위성이 바로 <strong>다양성의 원천</strong>이에요.</>, mood: 'wow', tone: 'bad' };
  } else if (done) {
    note = { text: <>완성! 소음 속에서 {pics[best].emoji} <strong>{pics[best].label}</strong>가 떠올랐어요. ↺ 처음으로 → ▶ 시작을 반복하면 다른 그림이 나오기도 해요. (그림이 4장뿐이라 그중 하나가 그대로 나와요. 진짜 모델은 수많은 그림으로 배워 <strong>새 그림</strong>을 만들어요.)</>, mood: 'cheer', tone: 'good' };
  } else if (step === 0) {
    note = { text: <>출발점은 순수한 노이즈 x<sub>T</sub>예요. ▶ 시작을 누르면 모델이 "섞인 노이즈"를 짐작해서 <strong>조금씩</strong> 걷어 내요. 쉬운 청소를 {STEPS.length}번 반복!</>, mood: 'smile' };
  } else if (g.probs[best] < 0.6) {
    note = { text: <>아직 무엇이 될지 몰라서, 모델이 짐작한 원본은 그림 네 장의 <strong>흐릿한 평균</strong>이에요. 노이즈가 걷힐수록 점점 정해져요.</>, mood: 'think' };
  } else if (curT > 450 && !squint) {
    note = { text: <>우리 눈엔 아직 노이즈인데, 모델은 벌써 {pics[best].emoji} 쪽이라고 짐작해요(<Num>{Math.round(g.probs[best] * 100)}</Num>%). 노이즈 속 희미한 흔적을 읽거든요. 「😑 눈 가늘게 뜨고 보기」를 켜면 우리도 보여요!</>, mood: 'wow' };
  } else {
    note = { text: <>점점 {pics[best].emoji} 쪽으로 기울어요(<Num>{Math.round(g.probs[best] * 100)}</Num>%). 이제부턴 그 그림을 또렷하게 다듬는 <strong>작은 교정</strong>만 남았어요.</>, mood: 'happy' };
  }

  const colA = c.series[1];
  const colB = c.series[2];
  const maxRow = expanded ? 720 : 600;

  return (
    <div ref={rootRef}>
      {tab === 'forward' ? (
        <div className="vz-pad" style={{ display: 'grid', gap: 14 }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(0, 1fr) clamp(44px, 9vw, 60px) minmax(0, 1fr) clamp(44px, 9vw, 60px) minmax(0, 1.6fr)',
              gap: 6,
              alignItems: 'center',
              width: '100%',
              maxWidth: maxRow,
              margin: '0 auto'
            }}
          >
            <div>
              <Pixels data={pics[pic].data} label={`원본 그림: ${pics[pic].label}`} border={c.border} />
              <Caption color={c.text}>원본 x<sub>0</sub></Caption>
            </div>
            <div style={{ textAlign: 'center', fontWeight: 850, color: colA, fontSize: 'clamp(12px, 2.6vw, 16px)', lineHeight: 1.2, fontVariantNumeric: 'tabular-nums' }}>
              ×{a.toFixed(2)}
              <div style={{ color: c.ink, fontSize: '1.3em' }}>+</div>
            </div>
            <div>
              <Pixels data={eps} label="표준 정규 노이즈" border={c.border} />
              <Caption color={c.text}>노이즈 ε</Caption>
            </div>
            <div style={{ textAlign: 'center', fontWeight: 850, color: colB, fontSize: 'clamp(12px, 2.6vw, 16px)', lineHeight: 1.2, fontVariantNumeric: 'tabular-nums' }}>
              ×{b.toFixed(2)}
              <div style={{ color: c.ink, fontSize: '1.3em' }}>=</div>
            </div>
            <div>
              <Pixels data={xtShown} label={`t=${t} 시점의 망가진 그림`} border={c.ink} />
              <Caption color={c.ink}>t = {t}의 그림 x<sub>t</sub>{squint ? ' · 흐리게 본 모습' : ''}</Caption>
            </div>
          </div>
          <div style={{ width: '100%', maxWidth: maxRow, margin: '0 auto' }}>
            <Plot xDomain={[0, T]} yDomain={[0, 1]} height={expanded ? 200 : 160} xLabel="t (망가뜨린 단계)" xTicks={5} yTicks={2} ariaLabel="t에 따른 원본 비율과 노이즈 비율">
              {({ x: X, y: Y, inner }) => (
                <g>
                  <path d={linePath(CURVE.map((s) => [X(s), Y(Math.sqrt(ALPHA_BAR[s]))] as [number, number]))} fill="none" stroke={colA} strokeWidth={3} />
                  <path d={linePath(CURVE.map((s) => [X(s), Y(Math.sqrt(1 - ALPHA_BAR[s]))] as [number, number]))} fill="none" stroke={colB} strokeWidth={3} />
                  <line x1={X(t)} x2={X(t)} y1={inner.top} y2={inner.top + inner.height} stroke={c.ink} strokeWidth={1.6} strokeDasharray="4 4" />
                  <circle cx={X(t)} cy={Y(a)} r={5.5} fill={colA} stroke={c.ink} strokeWidth={1.2} />
                  <circle cx={X(t)} cy={Y(b)} r={5.5} fill={colB} stroke={c.ink} strokeWidth={1.2} />
                </g>
              )}
            </Plot>
            <Legend
              items={[
                { color: colA, label: '원본 비율 √ᾱ' },
                { color: colB, label: '노이즈 비율 √(1-ᾱ)' }
              ]}
            />
          </div>
          <Note mood={note.mood} tone={note.tone}>
            {note.text}
          </Note>
        </div>
      ) : (
        <div className="vz-pad" style={{ display: 'grid', gap: 14 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto minmax(0, 1fr)', gap: 10, alignItems: 'center', width: '100%', maxWidth: expanded ? 640 : 520, margin: '0 auto' }}>
            <div>
              <Pixels data={xShown} label={`t=${curT}의 그림`} border={c.ink} />
              <Caption color={c.ink}>지금 그림 (t = {curT}){squint && !done ? ' · 흐리게 본 모습' : ''}</Caption>
            </div>
            <div style={{ fontSize: 22, color: c.text, textAlign: 'center' }} aria-hidden="true">
              🔍
            </div>
            <div>
              <Pixels data={done ? x : g.x0} label="모델이 짐작한 원본" border={withAlpha(c.series[0], 0.8)} />
              <Caption color={c.series[0]}>모델이 짐작한 원본</Caption>
            </div>
          </div>
          <div style={{ display: 'grid', gap: 6, width: '100%', maxWidth: 520, margin: '0 auto' }}>
            <div className="vz-panel-title">어떤 그림이 될까? (모델의 짐작)</div>
            {pics.map((p, k) => (
              <div key={p.id} style={{ display: 'grid', gridTemplateColumns: '70px 1fr 44px', gap: 8, alignItems: 'center', fontSize: 13 }}>
                <span style={{ fontWeight: 750, color: c.ink }}>
                  {p.emoji} {p.label}
                </span>
                <div style={{ height: 12, borderRadius: 99, background: withAlpha(c.series[0], 0.12), overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${g.probs[k] * 100}%`,
                      height: '100%',
                      borderRadius: 99,
                      background: k === best ? c.series[0] : withAlpha(c.series[0], 0.45),
                      transition: reduce ? 'none' : 'width 0.15s linear'
                    }}
                  />
                </div>
                <span style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums', color: c.text }}>{Math.round(g.probs[k] * 100)}%</span>
              </div>
            ))}
          </div>
          <div className="vz-stats">
            <Stat label="남은 걸음" value={`${STEPS.length - step} / ${STEPS.length}`} />
            <Stat
              label="지금까지 만든 그림"
              value={tally.length ? pics.map((p, k) => (tallyCounts[k] ? `${p.emoji}${tallyCounts[k]}` : '')).filter(Boolean).join(' ') : '-'}
              sub={`${tally.length}번 생성`}
              tone={allSame ? 'bad' : undefined}
            />
          </div>
          <Note mood={note.mood} tone={note.tone}>
            {note.text}
          </Note>
        </div>
      )}

      <Controls>
        <Segmented
          options={[
            { value: 'forward', label: '1. 망가뜨리기 (앞 방향)' },
            { value: 'reverse', label: '2. 되살리기 (역재생)' }
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
            <Segmented
              label="원본 그림"
              options={pics.map((p, k) => ({ value: k, label: p.emoji, title: p.label }))}
              value={pic}
              onChange={(v) => setPic(v)}
            />
            <Slider label="t (망가뜨린 단계)" value={t} min={0} max={T} step={1} onChange={(v) => { setSweeping(false); setT(v); }} format={(v) => String(v)} />
            <Button
              variant="primary"
              onClick={() => {
                if (!sweeping && t >= T) setT(0);
                setSweeping((s) => !s);
              }}
            >
              {sweeping ? '❚❚ 멈춤' : '▶ 망가뜨리기'}
            </Button>
            <Button onClick={() => setEpsSeed((s) => s + 1)}>🎲 노이즈 바꾸기</Button>
            <Toggle label="😑 눈 가늘게 뜨고 보기" checked={squint} onChange={setSquint} />
          </>
        ) : (
          <>
            <PlayBar
              playing={playing}
              onToggle={() => {
                if (done) {
                  restart();
                  setPlaying(true);
                  return;
                }
                setPlaying((p) => !p);
              }}
              onStep={advance}
              onReset={() => restart()}
              speed={speed}
              onSpeed={setSpeed}
            />
            <Toggle label="걸음마다 새 노이즈 조금 섞기" checked={freshNoise} onChange={(v) => { setFreshNoise(v); restart(); }} />
            <Toggle label="😑 눈 가늘게 뜨고 보기" checked={squint} onChange={setSquint} />
          </>
        )}
      </Controls>
    </div>
  );
}
