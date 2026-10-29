import { memo, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Slider, Stat } from '../core/ui';
import { useInView, usePrefersReducedMotion, useRafLoop } from '../core/hooks';
import { useVizColors, withAlpha, type VizColors } from '../core/colors';
import { clamp, lerp, rng } from '../core/math';
import type { AlgoMood } from '@/components/mascot/Algo';
import { CLEAN, CLOUD_SIGMA, TRAIN, decodeAe, decodeVae, vnoise, type FaceParams, type Z } from './_faces';

type Kind = 'vae' | 'ae';
type Mode = 'explore' | 'walk';

const PAD = 300;
const UNIT = 50; // px per latent unit → z in [-3, 3]
const toPx = (z: Z): [number, number] => [PAD / 2 + z[0] * UNIT, PAD / 2 - z[1] * UNIT];
const toZ = (px: number, py: number): Z => [clamp((px - PAD / 2) / UNIT, -3, 3), clamp((PAD / 2 - py) / UNIT, -3, 3)];
const GRID = [-2.6, -1.73, -0.87, 0, 0.87, 1.73, 2.6];
const r1 = (v: number) => (Math.round(v * 10) / 10).toFixed(1).replace('-0.0', '0.0');

interface Pal {
  skin: string;
  ink: string;
  blush: string;
  hair: (h: number) => string;
}

function palette(c: VizColors): Pal {
  return c.theme === 'dark'
    ? { skin: '#ffd9a6', ink: '#22183b', blush: '#ff6fa5', hair: (h) => `hsl(${h} 95% 62%)` }
    : { skin: '#ffe2bd', ink: '#3b2d4a', blush: '#ff8fa3', hair: (h) => `hsl(${h} 68% 50%)` };
}

/** Draws one decoded face in a 100×100 box. `detail` adds the TV-static overlay (big face only). */
const FaceGlyph = memo(function FaceGlyph({ f, pal, detail = false }: { f: FaceParams; pal: Pal; detail?: boolean }) {
  const g = f.glitch;
  const [za, zb] = f.z;
  const n = (k: number) => (g > 0.01 ? vnoise(za * 2.3 + k * 17.1, zb * 2.3 - k * 9.7, k) : 0);
  const rx = 26 + 8 * f.round;
  const ry = 34 - 5 * f.round;
  const cy = 55;
  const hair = pal.hair(f.hue + g * 140 * n(11));
  const hx = g * 10 * n(12);

  let face: ReactNode = <ellipse cx={50} cy={cy} rx={rx} ry={ry} fill={pal.skin} stroke={pal.ink} strokeWidth={2} />;
  if (g > 0.01) {
    const pts: string[] = [];
    for (let k = 0; k < 28; k++) {
      const th = (2 * Math.PI * k) / 28;
      const m = 1 + 0.4 * g * vnoise(Math.cos(th) * 1.6 + za * 2, Math.sin(th) * 1.6 + zb * 2, 5);
      pts.push(`${(50 + rx * m * Math.cos(th) + g * 7 * n(13)).toFixed(1)},${(cy + ry * m * Math.sin(th)).toFixed(1)}`);
    }
    face = <path d={`M${pts.join('L')}Z`} fill={pal.skin} stroke={pal.ink} strokeWidth={2} strokeLinejoin="round" />;
  }

  const eyeY = 51 - 2 * f.round;
  const eyeDX = 11 + 2 * f.round;
  const eyeR = 4.2 + 2.6 * f.eye;
  const eyes = [
    { x: 50 - eyeDX + g * 16 * n(1), y: eyeY + g * 14 * n(2), r: eyeR * (1 + 0.9 * g * n(3)) },
    { x: 50 + eyeDX + g * 16 * n(4), y: eyeY + g * 14 * n(5), r: eyeR * (1 + 0.9 * g * n(6)) }
  ];
  const tilt = -3.2 * f.mood;
  const browY = eyeY - eyeR - 4 + g * 10 * n(14);
  const mouthY = 69 + (1 - f.round) * 3;
  const curve = 13 * f.mood * (1 - g) + g * 16 * n(9);
  const filled = f.mood > 0.45 && g < 0.3;
  const mouth = `M-9,0 Q0,${curve.toFixed(1)} 9,0${filled ? ` Q0,${(curve * 0.35).toFixed(1)} -9,0Z` : ''}`;
  const scribbles = Math.floor(g * 5.5);

  return (
    <g>
      <ellipse cx={50 + hx} cy={cy - 3 + 8 * f.hair} rx={rx + 5 + 5 * f.hair} ry={ry + 4 + 9 * f.hair} fill={hair} />
      {face}
      <path
        d={`M${50 - rx + hx},${cy - 5} A${rx} ${ry + 3} 0 0 1 ${50 + rx + hx},${cy - 5} Q${50 + hx},${cy - ry * 0.45 + 4 * f.hair} ${50 - rx + hx},${cy - 5}Z`}
        fill={hair}
      />
      {f.mood > 0 && g < 0.6 && (
        <g fill={pal.blush} opacity={0.55 * f.mood * (1 - g)}>
          <circle cx={50 - rx + 8} cy={mouthY - 6} r={4.2} />
          <circle cx={50 + rx - 8} cy={mouthY - 6} r={4.2} />
        </g>
      )}
      {eyes.map((e, i) => (
        <g key={i}>
          <ellipse cx={e.x} cy={e.y} rx={Math.abs(e.r)} ry={Math.abs(e.r) * 0.92} fill="#ffffff" stroke={pal.ink} strokeWidth={1.4} />
          <circle cx={e.x + 0.6} cy={e.y + 0.6} r={Math.abs(e.r) * 0.48} fill={pal.ink} />
        </g>
      ))}
      <g stroke={pal.ink} strokeWidth={2.4} strokeLinecap="round">
        <line x1={eyes[0].x - 5} y1={browY - tilt} x2={eyes[0].x + 5} y2={browY + tilt} />
        <line x1={eyes[1].x - 5} y1={browY + tilt} x2={eyes[1].x + 5} y2={browY - tilt} />
      </g>
      <path
        d={mouth}
        transform={`translate(${50 + g * 14 * n(8)} ${mouthY + g * 10 * n(10)}) rotate(${g * 75 * n(7)})`}
        fill={filled ? pal.ink : 'none'}
        stroke={pal.ink}
        strokeWidth={2.4}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {Array.from({ length: scribbles }, (_, i) => (
        <path
          key={`s${i}`}
          d={`M${50 + 40 * n(20 + i)},${50 + 40 * n(30 + i)} Q${50 + 55 * n(40 + i)},${50 + 55 * n(45 + i)} ${50 + 40 * n(50 + i)},${50 + 40 * n(55 + i)}`}
          fill="none"
          stroke={pal.hair(i * 67 + 90 * n(60 + i))}
          strokeWidth={2.4}
          strokeLinecap="round"
          opacity={0.8 * g}
        />
      ))}
      {detail && g > 0.05 && (
        <g fill={pal.ink}>
          {Array.from({ length: 100 }, (_, k) => {
            const i = k % 10;
            const j = Math.floor(k / 10);
            const o = g * 0.6 * Math.max(0, vnoise(i * 1.7 + za * 5, j * 1.7 + zb * 5, 99));
            return o > 0.04 ? <rect key={k} x={i * 10} y={j * 10} width={10} height={10} opacity={o} /> : null;
          })}
        </g>
      )}
    </g>
  );
});

/** VAE vs plain autoencoder: drag a point around a 2D latent space and watch a (toy) decoder draw it. */
export default function LatentExplorer({ expanded }: VizProps) {
  const c = useVizColors();
  const pal = useMemo(() => palette(c), [c.theme]); // eslint-disable-line react-hooks/exhaustive-deps
  const reduce = usePrefersReducedMotion();
  const [kind, setKind] = useState<Kind>('vae');
  const [mode, setMode] = useState<Mode>('explore');
  const [cursor, setCursor] = useState<Z>([0.4, 0.6]);
  const [touched, setTouched] = useState(false);
  const [ends, setEnds] = useState<[Z, Z]>([TRAIN[0].z, TRAIN[3].z]);
  const [t, setT] = useState(0.5);
  const [walking, setWalking] = useState(false);
  const dir = useRef(1);
  const drag = useRef<number | null>(null);
  const [samples, setSamples] = useState<{ z: Z; ok: boolean }[]>([]);
  const dice = useRef(rng(19));
  const rootRef = useRef<HTMLDivElement>(null);
  const inView = useInView(rootRef);
  const svgRef = useRef<SVGSVGElement>(null);

  const decode = kind === 'vae' ? decodeVae : decodeAe;
  const pos = (i: number): Z => (kind === 'vae' ? TRAIN[i].z : TRAIN[i].ae);
  const walkZ: Z = [lerp(ends[0][0], ends[1][0], t), lerp(ends[0][1], ends[1][1], t)];
  const z = mode === 'walk' ? walkZ : cursor;
  const face = decode(z);
  const ok = face.glitch < CLEAN;
  const lastSample = samples.length ? samples[samples.length - 1] : null;
  const fromDice = mode === 'explore' && lastSample !== null && lastSample.z === cursor;

  useRafLoop((dt) => {
    setT((prev) => {
      let next = prev + dir.current * dt * 0.28;
      if (next >= 1) {
        next = 1;
        dir.current = -1;
      } else if (next <= 0) {
        next = 0;
        dir.current = 1;
      }
      return next;
    });
  }, walking && inView && mode === 'walk');

  const switchKind = (k: Kind) => {
    setKind(k);
    setSamples([]);
    setTouched(false);
    setCursor([0.4, 0.6]);
    setEnds(k === 'vae' ? [TRAIN[0].z, TRAIN[3].z] : [TRAIN[0].ae, TRAIN[3].ae]);
  };

  const roll = () => {
    const s: Z = [clamp(dice.current.normal(), -3, 3), clamp(dice.current.normal(), -3, 3)];
    setMode('explore');
    setWalking(false);
    setCursor(s);
    setTouched(true);
    setSamples((prev) => [...prev.slice(-14), { z: s, ok: decode(s).glitch < CLEAN }]);
  };

  const pointerZ = (e: PointerEvent<SVGSVGElement>): Z => {
    const r = svgRef.current!.getBoundingClientRect();
    return toZ(((e.clientX - r.left) / r.width) * PAD, ((e.clientY - r.top) / r.height) * PAD);
  };
  const onDown = (e: PointerEvent<SVGSVGElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = pointerZ(e);
    if (mode === 'walk') {
      const dA = Math.hypot(p[0] - ends[0][0], p[1] - ends[0][1]);
      const dB = Math.hypot(p[0] - ends[1][0], p[1] - ends[1][1]);
      drag.current = dA <= dB ? 0 : 1;
      setWalking(false);
      setEnds((prev) => (drag.current === 0 ? [p, prev[1]] : [prev[0], p]));
    } else {
      drag.current = 0;
      setCursor(p);
      setTouched(true);
    }
  };
  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    if (drag.current === null) return;
    const p = pointerZ(e);
    if (mode === 'walk') setEnds((prev) => (drag.current === 0 ? [p, prev[1]] : [prev[0], p]));
    else setCursor(p);
  };
  const onUp = () => {
    drag.current = null;
  };
  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    const step = 0.2;
    const d: Record<string, Z> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
    const v = d[e.key];
    if (!v) return;
    e.preventDefault();
    if (mode === 'walk') setEnds((prev) => [prev[0], [clamp(prev[1][0] + v[0], -3, 3), clamp(prev[1][1] + v[1], -3, 3)]]);
    else {
      setCursor((prev) => [clamp(prev[0] + v[0], -3, 3), clamp(prev[1] + v[1], -3, 3)]);
      setTouched(true);
    }
  };

  // The background "map": what the decoder draws at each grid point.
  const thumbs = useMemo(
    () =>
      GRID.flatMap((gy) =>
        GRID.map((gx) => {
          const [px, py] = toPx([gx, gy]);
          return { key: `${gx},${gy}`, px, py, f: decode([gx, gy]) };
        })
      ),
    [kind] // eslint-disable-line react-hooks/exhaustive-deps
  );

  const strip = useMemo(
    () => Array.from({ length: 7 }, (_, i) => decode([lerp(ends[0][0], ends[1][0], i / 6), lerp(ends[0][1], ends[1][1], i / 6)])),
    [kind, ends] // eslint-disable-line react-hooks/exhaustive-deps
  );
  const stripBad = strip.filter((f) => f.glitch >= CLEAN).length;

  const okCount = samples.filter((s) => s.ok).length;
  const [cx, cy] = toPx(z);
  const ringColor = c.ink;
  const dotStroke = c.theme === 'dark' ? '#0b1030' : '#ffffff';
  const far = Math.hypot(z[0], z[1]) > 3.3;
  const halo = { paintOrder: 'stroke' as const, stroke: c.theme === 'dark' ? '#0b1030' : '#fffdf9', strokeWidth: 3.5, strokeLinejoin: 'round' as const };

  let note: { text: ReactNode; mood: AlgoMood; tone?: 'good' | 'bad' | 'warn' };
  if (mode === 'walk') {
    note =
      stripBad === 0
        ? { text: <>A에서 B까지 <strong>직선</strong>을 따라 걸으면 한 얼굴이 다른 얼굴로 <strong>부드럽게</strong> 바뀌어요. "VAE의 보간이 아름답다"는 말이 바로 이거예요. 손잡이 A·B를 끌어 다른 두 얼굴도 이어 보세요!</>, mood: 'happy', tone: 'good' }
        : kind === 'ae'
          ? { text: <>중간 그림들이 엉망이죠? A와 B 사이에 학습 사진이 없는 <strong>빈 공간</strong>을 지나가기 때문이에요. 위에서 <strong>VAE</strong>로 바꿔 같은 걸 해 보세요.</>, mood: 'wow', tone: 'bad' }
          : { text: <>길이 지도 <strong>먼 구석</strong>을 지나서 그림이 흔들려요. 정규분포 구역 안쪽으로 손잡이를 옮겨 보세요.</>, mood: 'think', tone: 'warn' };
  } else if (fromDice) {
    note =
      kind === 'vae'
        ? { text: <>주사위로 뽑은 z → 디코더 → <strong>새 얼굴</strong>! 학습 사진에 없던 얼굴인데도 그럴듯하죠? 이게 VAE로 <strong>생성</strong>하는 방법이에요. 🎉 (지금까지 {samples.length}번 중 {okCount}번 멀쩡)</>, mood: 'cheer', tone: 'good' }
        : ok
          ? { text: <>운 좋게 학습 사진 바로 옆에 떨어졌어요! 하지만 여러 번 뽑아 보면… 지금까지 {samples.length}번 중 <strong>{okCount}번</strong>만 멀쩡해요.</>, mood: 'wink' }
          : { text: <>주사위로 뽑은 z가 <strong>빈 곳</strong>에 떨어져서 쓰레기가 나왔어요({samples.length}번 중 {okCount}번만 멀쩡). 그래서 VAE는 <strong>벌점(KL)</strong>으로 구름들을 정규분포 구역에 빈틈없이 모아요.</>, mood: 'wow', tone: 'bad' };
  } else if (!touched) {
    note =
      kind === 'vae'
        ? { text: <>지도 위 <strong>점 하나(z)가 얼굴 하나</strong>예요. 아무 데나 누르거나 끌어 보세요 — 오른쪽으로 가면 웃고, 위로 가면 얼굴이 동그래져요. 색 구름은 학습 사진 8장이 차지한 자리예요.</>, mood: 'smile' }
        : { text: <>이번엔 <strong>그냥 오토인코더</strong>예요. 사진마다 구름 대신 <strong>점 하나</strong>만 차지하고, 서로 멀리 흩어져 있어요. 점과 점 사이 빈 곳을 눌러 보세요!</>, mood: 'think' };
  } else if (kind === 'vae') {
    note = far
      ? { text: <>정규분포 구역에서 아주 <strong>먼 구석</strong>이에요. 학습할 때 여긴 거의 안 와 봐서 VAE도 조금 헷갈려요.</>, mood: 'think', tone: 'warn' }
      : { text: <>어디를 찍어도 그럴듯한 얼굴! 구름들이 겹치며 공간을 <strong>빈틈없이</strong> 채워서, 사진이 없던 자리도 뜻이 있어요. 이제 🎲를 눌러 새 얼굴을 <strong>생성</strong>해 보세요.</>, mood: 'happy', tone: 'good' };
  } else {
    note =
      face.glitch > 0.6
        ? { text: <>학습 사진이 없던 <strong>빈 곳</strong>이라 디코더가 쓰레기를 내놨어요. 이 빈틈 때문에 그냥 오토인코더는 생성기가 못 돼요.</>, mood: 'wow', tone: 'bad' }
        : face.glitch < 0.1
          ? { text: <>학습 사진 바로 그 자리라 복원은 완벽해요. 그런데 조금만 벗어나 보세요…</>, mood: 'wink' }
          : { text: <>사진에서 멀어질수록 그림이 망가져요. 점 사이가 텅 비어 있거든요.</>, mood: 'think', tone: 'warn' };
  }

  const padMax = expanded ? 520 : 400;
  const faceMax = mode === 'walk' ? (expanded ? 220 : 160) : expanded ? 280 : 210;
  const statusColor = ok ? c.pos : c.neg;

  return (
    <div ref={rootRef}>
      <style>{`@keyframes le-ping{from{transform:scale(1);opacity:.9}to{transform:scale(5);opacity:0}}`}</style>
      <div
        className="vz-pad"
        style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 270px), 1fr))', gap: 16, alignItems: 'start' }}
      >
        <div style={{ minWidth: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
          <div className="vz-panel-title" style={{ alignSelf: 'stretch', textAlign: 'center' }}>
            잠재 공간 지도 (z)
          </div>
          <svg
            ref={svgRef}
            viewBox={`0 0 ${PAD} ${PAD}`}
            role="img"
            aria-label="2차원 잠재 공간 지도. 점을 끌어 z를 바꾸면 오른쪽에 디코더가 그린 얼굴이 보여요."
            tabIndex={0}
            onPointerDown={onDown}
            onPointerMove={onMove}
            onPointerUp={onUp}
            onPointerCancel={onUp}
            onKeyDown={onKey}
            style={{ width: '100%', maxWidth: padMax, aspectRatio: '1', touchAction: 'none', cursor: 'crosshair', borderRadius: 14, border: `1px solid ${c.border}`, background: withAlpha(c.series[0], 0.03), display: 'block' }}
          >
            {[-2, -1, 0, 1, 2].map((v) => (
              <g key={v} stroke={c.grid} strokeWidth={v === 0 ? 1.6 : 1}>
                <line x1={toPx([v, 0])[0]} x2={toPx([v, 0])[0]} y1={0} y2={PAD} />
                <line y1={toPx([0, v])[1]} y2={toPx([0, v])[1]} x1={0} x2={PAD} />
              </g>
            ))}
            <g opacity={0.42}>
              {thumbs.map((th) => (
                <g key={th.key} transform={`translate(${th.px - 18} ${th.py - 18}) scale(0.36)`}>
                  <FaceGlyph f={th.f} pal={pal} />
                </g>
              ))}
            </g>
            <circle cx={PAD / 2} cy={PAD / 2} r={2 * UNIT} fill="none" stroke={c.axis} strokeWidth={1.6} strokeDasharray="5 5" />
            <text x={PAD / 2} y={PAD / 2 - 2 * UNIT - 5} textAnchor="middle" style={{ fontSize: 10.5, fontWeight: 750, fill: c.text, ...halo }}>
              정규분포 구역 (주사위가 주로 떨어지는 곳)
            </text>
            {kind === 'vae' &&
              TRAIN.map((tr, i) => {
                const [px, py] = toPx(tr.z);
                const col = c.series[i % 6];
                return <circle key={`c${i}`} cx={px} cy={py} r={2 * CLOUD_SIGMA * UNIT} fill={withAlpha(col, 0.13)} stroke={withAlpha(col, 0.55)} strokeWidth={1.2} />;
              })}
            {TRAIN.map((_, i) => {
              const [px, py] = toPx(pos(i));
              return <circle key={`p${i}`} cx={px} cy={py} r={5.5} fill={c.series[i % 6]} stroke={dotStroke} strokeWidth={2} />;
            })}
            {samples.map((s, i) => {
              const [px, py] = toPx(s.z);
              const col = s.ok ? c.pos : c.neg;
              return (
                <g key={`d${i}`} stroke={col} strokeWidth={2.4} strokeLinecap="round" opacity={0.45 + (0.55 * (i + 1)) / samples.length}>
                  <line x1={px - 4} y1={py - 4} x2={px + 4} y2={py + 4} />
                  <line x1={px - 4} y1={py + 4} x2={px + 4} y2={py - 4} />
                </g>
              );
            })}
            {mode === 'walk' ? (
              <g>
                <line x1={toPx(ends[0])[0]} y1={toPx(ends[0])[1]} x2={toPx(ends[1])[0]} y2={toPx(ends[1])[1]} stroke={ringColor} strokeWidth={2.4} strokeDasharray="6 5" />
                {strip.map((f, i) => {
                  const [px, py] = toPx([lerp(ends[0][0], ends[1][0], i / 6), lerp(ends[0][1], ends[1][1], i / 6)]);
                  return <circle key={`w${i}`} cx={px} cy={py} r={3.2} fill={f.glitch < CLEAN ? c.pos : c.neg} />;
                })}
                <circle cx={cx} cy={cy} r={8} fill={withAlpha(ringColor, 0.15)} stroke={ringColor} strokeWidth={2.6} />
                {ends.map((e, i) => {
                  const [px, py] = toPx(e);
                  return (
                    <g key={`h${i}`}>
                      <circle cx={px} cy={py} r={12} fill={i === 0 ? c.series[0] : c.series[2]} stroke={dotStroke} strokeWidth={2.4} />
                      <text x={px} y={py + 4.5} textAnchor="middle" style={{ fontSize: 13, fontWeight: 850, fill: dotStroke }}>
                        {i === 0 ? 'A' : 'B'}
                      </text>
                    </g>
                  );
                })}
              </g>
            ) : (
              <g>
                {fromDice && !reduce && (
                  <circle key={samples.length} cx={cx} cy={cy} r={6} fill="none" stroke={statusColor} strokeWidth={2.5} vectorEffect="non-scaling-stroke" style={{ animation: 'le-ping 0.9s ease-out forwards', transformBox: 'fill-box', transformOrigin: 'center' }} />
                )}
                <circle cx={cx} cy={cy} r={11} fill={withAlpha(ringColor, 0.1)} stroke={ringColor} strokeWidth={2.6} />
                <circle cx={cx} cy={cy} r={2.6} fill={ringColor} />
                <text x={cx + 14} y={cy - 10} style={{ fontSize: 13, fontWeight: 850, fill: ringColor, ...halo }}>
                  z
                </text>
              </g>
            )}
            <text x={PAD - 6} y={PAD / 2 - 6} textAnchor="end" style={{ fontSize: 11, fontWeight: 750, fill: c.text, ...halo }}>
              z1 →
            </text>
            <text x={PAD / 2 + 6} y={14} style={{ fontSize: 11, fontWeight: 750, fill: c.text, ...halo }}>
              ↑ z2
            </text>
          </svg>
        </div>

        <div style={{ minWidth: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
          <div className="vz-panel-title" style={{ alignSelf: 'stretch', textAlign: 'center' }}>
            디코더가 그린 얼굴
          </div>
          <div style={{ position: 'relative', width: '100%', maxWidth: faceMax }}>
            <svg viewBox="-4 -4 108 108" role="img" aria-label={ok ? '그럴듯한 얼굴' : '망가진 그림'} style={{ width: '100%', display: 'block', aspectRatio: '1' }}>
              <rect x={-3} y={-3} width={106} height={106} rx={14} fill={withAlpha(statusColor, 0.07)} stroke={withAlpha(statusColor, 0.6)} strokeWidth={1.4} />
              <FaceGlyph f={face} pal={pal} detail />
            </svg>
            <span
              style={{
                position: 'absolute',
                top: 8,
                right: 8,
                fontSize: 12,
                fontWeight: 800,
                padding: '2px 8px',
                borderRadius: 999,
                color: c.theme === 'dark' ? '#0b1030' : '#ffffff',
                background: statusColor
              }}
            >
              {ok ? '멀쩡한 얼굴' : '쓰레기 그림'}
            </span>
          </div>
          <div style={{ fontSize: 12, color: c.text, textAlign: 'center' }}>
            z = ({r1(z[0])}, {r1(z[1])}) · 장난감 디코더가 그렸어요
          </div>
          {mode === 'walk' && (
            <div style={{ width: '100%', display: 'grid', gridTemplateColumns: 'repeat(7, minmax(0, 1fr))', gap: 4, alignItems: 'center' }}>
              {strip.map((f, i) => {
                const near = Math.round(t * 6) === i;
                return (
                  <div key={i} style={{ textAlign: 'center' }}>
                    <svg
                      viewBox="-4 -4 108 108"
                      style={{ width: '100%', display: 'block', borderRadius: 8, outline: near ? `2.5px solid ${c.ink}` : 'none', outlineOffset: 1 }}
                      aria-hidden="true"
                    >
                      <FaceGlyph f={f} pal={pal} />
                    </svg>
                    <span style={{ fontSize: 11, fontWeight: 800, color: i === 0 ? c.series[0] : i === 6 ? c.series[2] : c.text }}>
                      {i === 0 ? 'A' : i === 6 ? 'B' : '·'}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
          <div style={{ display: 'grid', gap: 10, width: '100%', marginTop: 4 }}>
            <div className="vz-stats">
              <Stat label="그림 상태" value={ok ? '멀쩡' : '쓰레기'} tone={ok ? 'good' : 'bad'} />
              <Stat
                label="🎲 생성 결과"
                value={samples.length ? `${okCount} / ${samples.length}` : '-'}
                sub="멀쩡한 얼굴 / 뽑은 수"
                tone={samples.length === 0 ? undefined : okCount / samples.length > 0.8 ? 'good' : 'bad'}
              />
              {mode === 'walk' && <Stat label="걷는 길의 그림" value={`${7 - stripBad} / 7 멀쩡`} tone={stripBad === 0 ? 'good' : 'bad'} />}
            </div>
            <Note mood={note.mood} tone={note.tone}>
              {note.text}
            </Note>
          </div>
        </div>
      </div>

      <Controls>
        <Segmented
          label="잠재 공간 만드는 법"
          options={[
            { value: 'vae', label: '☁️ VAE (구름 + 정돈)' },
            { value: 'ae', label: '📍 그냥 오토인코더 (점만)' }
          ]}
          value={kind}
          onChange={(v) => switchKind(v as Kind)}
        />
        <Segmented
          label="해 보기"
          options={[
            { value: 'explore', label: '🧭 자유 탐험' },
            { value: 'walk', label: '🚶 두 얼굴 사이 걷기' }
          ]}
          value={mode}
          onChange={(v) => {
            setMode(v as Mode);
            setWalking(false);
          }}
        />
        <Button variant="primary" onClick={roll}>
          🎲 z 뽑아서 생성하기
        </Button>
        {mode === 'walk' && (
          <>
            <Slider label="A → B 걷기" value={t} min={0} max={1} step={0.01} onChange={(v) => { setWalking(false); setT(v); }} format={(v) => `${Math.round(v * 100)}%`} />
            <Button onClick={() => setWalking((w) => !w)} active={walking}>
              {walking ? '❚❚ 멈춤' : '▶ 걸어가기'}
            </Button>
          </>
        )}
      </Controls>
    </div>
  );
}
