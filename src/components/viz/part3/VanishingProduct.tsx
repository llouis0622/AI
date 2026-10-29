import { useEffect, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Slider, Stat } from '../core/ui';
import { useRafLoop, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { PowTen, Sci } from './_sci';

const PRESETS = [
  { value: 0.25, label: '시그모이드 0.25' },
  { value: 0.5, label: '0.5' },
  { value: 0.9, label: '0.9' },
  { value: 1, label: 'ReLU 1.0' },
  { value: 1.1, label: '1.1' }
];
const LOG_MIN = -12;
const LOG_MAX = 6;

/** Gradient reaching layer k is the product of the local sensitivities of every layer after it. */
export default function VanishingProduct({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [depth, setDepth] = useState(30);
  const [f, setF] = useState(0.9);
  const [t, setT] = useState(1);
  const [playing, setPlaying] = useState(false);

  useRafLoop((dt) => setT((v) => Math.min(1, v + dt / 2.6)), playing);
  useEffect(() => {
    if (playing && t >= 1) setPlaying(false);
  }, [playing, t]);
  useEffect(() => {
    setPlaying(false);
    setT(1);
  }, [depth, f]);

  const W = 720;
  const H = expanded ? 340 : 290;
  const left = 54;
  const right = 92;
  const base = H - 46;
  const top = 18;
  const usable = W - left - right;
  const slot = usable / depth;
  const logY = (g: number) => {
    const l = Math.max(LOG_MIN, Math.min(LOG_MAX, Math.log10(Math.max(g, 1e-300))));
    return base - ((l - LOG_MIN) / (LOG_MAX - LOG_MIN)) * (base - top);
  };
  const oneY = logY(1);
  const first = f ** depth;
  // pulse travels from the loss (right) to layer 1 (left); revealed layers are those it has passed
  const revealedFrom = depth - Math.floor(t * depth); // layers k > revealedFrom are visible
  const pulseX = left + usable - t * usable;
  const verdict = first < 1e-3 ? 'vanish' : first > 1e3 ? 'explode' : 'ok';
  const color = (g: number) => (g < 1e-3 ? c.neg : g > 1e3 ? c.series[4] : c.pos);

  let note: { text: ReactNode; mood: 'think' | 'wow' | 'happy' | 'cheer'; tone?: 'good' | 'bad' | 'warn' };
  if (verdict === 'vanish')
    note = {
      text: (
        <>
          기울기 소실! {depth}층을 지나는 동안 {f}를 {depth}번 곱해서 맨 앞 층엔 <Sci v={first} />만 도착해요. 앞쪽 층들은 거의 아무것도 못 배워요.
        </>
      ),
      mood: 'think',
      tone: 'bad'
    };
  else if (verdict === 'explode')
    note = {
      text: (
        <>
          기울기 폭발! {f}를 {depth}번 곱하니 <Sci v={first} />배로 불어났어요. 학습이 널뛰다 NaN이 돼요.
        </>
      ),
      mood: 'wow',
      tone: 'bad'
    };
  else if (f === 1)
    note = { text: 'ReLU의 통과 구간처럼 민감도가 정확히 1이면, 몇 층을 지나도 기울기가 그대로 살아 있어요. ReLU가 시그모이드를 이긴 이유예요!', mood: 'cheer', tone: 'good' };
  else note = { text: '아직은 괜찮지만, 층을 더 쌓으면 어떻게 될까요? 층 수를 늘려 보세요.', mood: 'happy', tone: 'warn' };

  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <svg className="vz-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="층마다 도착하는 기울기 크기(로그 눈금)" style={{ touchAction: 'pan-y' }}>
          {/* log grid */}
          {[-12, -9, -6, -3, 0, 3, 6].map((e) => (
            <g key={e}>
              <line x1={left} x2={W - right + 10} y1={logY(10 ** e)} y2={logY(10 ** e)} stroke={e === 0 ? c.axis : c.grid} strokeWidth={e === 0 ? 1.6 : 1} strokeDasharray={e === 0 ? '6 4' : undefined} />
              <PowTen e={e} x={left - 8} y={logY(10 ** e) + 4} />
            </g>
          ))}
          <text x={W - right + 14} y={oneY + 4} style={{ fontSize: 11.5, fontWeight: 800, fill: 'var(--text-2)' }}>
            원래 크기
          </text>
          <rect x={left} y={logY(1e-3)} width={usable} height={base - logY(1e-3)} fill={withAlpha(c.neg, 0.06)} />
          <text x={left + usable - 6} y={logY(1e-3) + 15} textAnchor="end" style={{ fontSize: 11, fontWeight: 800, fill: c.neg, paintOrder: 'stroke', stroke: 'var(--viz-bg)', strokeWidth: 4 }}>
            ↓ 소실 구역 (거의 0)
          </text>
          {first > 1e3 || f > 1 ? (
            <>
              <rect x={left} y={top} width={usable} height={logY(1e3) - top} fill={withAlpha(c.series[4], 0.08)} />
              <text x={left + 6} y={top + 14} style={{ fontSize: 11, fontWeight: 800, fill: c.series[4] }}>
                폭발 구역
              </text>
            </>
          ) : null}
          {/* layer bars: layer k (1 = input side) receives f^(depth-k+1) */}
          {Array.from({ length: depth }, (_, i) => {
            const k = i + 1;
            const g = f ** (depth - k + 1);
            const visible = k > revealedFrom || t >= 1;
            const x = left + i * slot;
            const y = logY(g);
            return (
              <g key={k} opacity={visible ? 1 : 0.12}>
                <rect x={x + slot * 0.18} y={Math.min(y, oneY)} width={Math.max(1, slot * 0.64)} height={Math.max(1.5, Math.abs(oneY - y))} rx={Math.min(3, slot * 0.2)} fill={color(g)} opacity={0.85} />
                <rect x={x + slot * 0.1} y={base + 6} width={Math.max(1, slot * 0.8)} height={14} rx={3} fill={withAlpha(c.series[0], 0.22)} stroke={c.series[0]} strokeWidth={slot > 8 ? 1 : 0} />
              </g>
            );
          })}
          {/* loss node */}
          <circle cx={W - right + 46} cy={base + 13} r={17} fill={withAlpha(c.series[2], 0.2)} stroke={c.series[2]} strokeWidth={2.5} />
          <text x={W - right + 46} y={base + 18} textAnchor="middle" style={{ fontSize: 14, fontWeight: 900, fill: 'var(--text)' }}>
            L
          </text>
          <text x={left} y={H - 4} style={{ fontSize: 11.5, fontWeight: 800, fill: 'var(--text-2)' }}>
            ← 1층 (입력 쪽)
          </text>
          <text x={W - right} y={H - 4} textAnchor="end" style={{ fontSize: 11.5, fontWeight: 800, fill: 'var(--text-2)' }}>
            {depth}층 (손실 쪽)
          </text>
          {playing && (
            <g>
              <circle cx={pulseX} cy={base + 13} r={9} fill={c.series[2]} />
              <text x={pulseX} y={base - 4} textAnchor="middle" style={{ fontSize: 12, fontWeight: 900, fill: c.series[2] }}>
                × {f}
              </text>
            </g>
          )}
        </svg>
      </div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div
          style={{
            padding: '10px 14px',
            borderRadius: 14,
            background: 'var(--surface-solid)',
            border: '1px solid var(--border)',
            fontFamily: 'var(--font-mono)',
            fontSize: 14,
            fontWeight: 700,
            textAlign: 'center',
            overflowX: 'auto',
            whiteSpace: 'nowrap'
          }}
        >
          {f} × {f} × {f} × … × {f} <span style={{ color: 'var(--text-3)' }}>({depth}번)</span> = <span style={{ color: color(first) }}><Sci v={first} /></span>
        </div>
        <div className="vz-stats">
          <Stat label="맨 앞 층에 도착한 기울기" value={<Sci v={first} />} tone={verdict === 'ok' ? 'good' : 'bad'} />
          <Stat label="층마다 곱해지는 민감도" value={f} />
          <Stat label="판정" value={verdict === 'vanish' ? '소실 😵' : verdict === 'explode' ? '폭발 💥' : '건강 💚'} tone={verdict === 'ok' ? 'good' : 'bad'} />
        </div>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>
      <Controls>
        <Segmented label="층마다 곱해지는 민감도" options={PRESETS} value={PRESETS.find((p) => Math.abs(p.value - f) < 1e-9)?.value ?? -1} onChange={(v) => setF(v)} />
        <Slider label="민감도 직접 조절" value={f} min={0.1} max={1.5} step={0.01} onChange={setF} format={(v) => v.toFixed(2)} />
        <Slider label="층 수" value={depth} min={5} max={100} step={1} onChange={setDepth} format={(v) => `${v}층`} accent="var(--mint)" />
        <Button
          variant="primary"
          onClick={() => {
            if (reduce) return setT(1);
            setT(0);
            setPlaying(true);
          }}
        >
          ▶ 역전파 보내기
        </Button>
      </Controls>
    </div>
  );
}
