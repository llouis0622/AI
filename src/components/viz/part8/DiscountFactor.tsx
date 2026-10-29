import { useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Controls, Note, Slider, Stat } from '../core/ui';
import { useVizColors, withAlpha } from '../core/colors';
import { useElementSize } from '../core/hooks';
import { fmt } from '../core/math';

const T_MAX = 20;

/** γ — how much a reward t steps away is worth today (γ^t). */
export default function DiscountFactor(_: VizProps) {
  const c = useVizColors();
  const dark = c.theme === 'dark';
  const [gamma, setGamma] = useState(0.9);
  const [delay, setDelay] = useState(10);
  const [amount, setAmount] = useState(5);

  const valueNow = 1;
  const valueLater = amount * Math.pow(gamma, delay);
  const waits = valueLater > valueNow + 1e-9;
  const tie = Math.abs(valueLater - valueNow) < 0.02;
  const threshold = Math.pow(1 / amount, 1 / delay); // γ above which waiting wins
  const forever = gamma < 1 ? 1 / (1 - gamma) : Infinity;
  const persona = gamma < 0.5 ? '😋 지금 당장파' : gamma < 0.9 ? '🙂 적당히 참는 편' : '🧘 참을성 왕';

  // a narrower drawing on phones keeps every label at a readable size
  const [box, { width: boxW }] = useElementSize<HTMLDivElement>();
  const narrow = boxW > 0 && boxW < 520;
  const W = narrow ? 360 : 600;
  const H = 244;
  const x0 = narrow ? 24 : 40;
  const x1 = W - (narrow ? 14 : 24);
  const xt = (t: number) => x0 + ((x1 - x0) * t) / T_MAX;
  const barTop = 40;
  const barH = 84;
  const bw = narrow ? 11 : 20;
  const clampX = (x: number, half: number) => Math.min(W - half, Math.max(half, x));
  const candies = Math.min(amount, narrow ? 3 : 5);
  const candySize = 14 + 10 * Math.pow(gamma, delay);
  const bRow = xt(delay) - xt(0) < 80 ? 236 : 216;

  let note: { t: ReactNode; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' };
  if (tie) note = { t: '두 선택의 가치가 거의 같아요! γ를 아주 조금만 바꿔도 마음이 바뀌어요.', mood: 'wow' };
  else if (waits) note = { t: <><strong>기다리기</strong>를 골라요. {delay}걸음 뒤 사탕 {amount}개의 지금 가치({fmt(valueLater, 2)})가 지금 당장 사탕 1개보다 커요. γ가 {fmt(threshold, 3)}보다 크면 기다려요.</>, mood: 'happy', tone: 'good' };
  else note = { t: <><strong>지금 당장</strong> 사탕 1개를 골라요. {delay}걸음 뒤의 사탕은 할인돼서 {fmt(valueLater, 2)}개 값어치밖에 안 되거든요. γ를 {fmt(threshold, 3)}보다 크게 올려 보세요.</>, mood: 'think' };

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div ref={box}>
        <svg viewBox={`0 0 ${W} ${H}`} className="vz-svg" role="img" aria-label={`할인율 ${gamma}일 때 시간에 따른 보상의 무게`} style={{ touchAction: 'pan-y' }}>
          <text x={x1} y={14} textAnchor="end" style={{ fontSize: 12, fontWeight: 800 }}>
            t걸음 뒤 보상의 무게 = γᵗ
          </text>
          {Array.from({ length: T_MAX + 1 }, (_, t) => {
            const wgt = Math.pow(gamma, t);
            const h = barH * wgt;
            const special = t === 0 || t === delay;
            return (
              <g key={t}>
                <rect
                  x={xt(t) - bw / 2}
                  y={barTop + barH - h}
                  width={bw}
                  height={Math.max(1, h)}
                  rx={narrow ? 3 : 5}
                  fill={special ? (t === 0 ? c.series[3] : c.series[0]) : withAlpha(c.series[1], dark ? 0.6 : 0.45)}
                  style={{ transition: 'all .25s' }}
                />
                {(t === delay || (t % 5 === 0 && Math.abs(t - delay) > 1)) && (
                  <text x={clampX(xt(t), 14)} y={barTop + barH - h - 5} textAnchor="middle" style={{ fontSize: narrow ? 11.5 : 10.5, fontWeight: 800 }}>
                    {wgt >= 0.995 ? '1' : fmt(wgt, 2)}
                  </text>
                )}
              </g>
            );
          })}
          {/* the road */}
          <line x1={x0 - 14} x2={x1 + 8} y1={162} y2={162} stroke="var(--viz-axis)" strokeWidth={3} strokeLinecap="round" />
          {Array.from({ length: T_MAX + 1 }, (_, t) => (
            <g key={`r${t}`}>
              <line x1={xt(t)} x2={xt(t)} y1={158} y2={166} stroke="var(--viz-axis)" strokeWidth={1.5} />
              {t % 5 === 0 && (
                <text x={xt(t)} y={182} textAnchor="middle" className="vz-tick">
                  {t}
                </text>
              )}
            </g>
          ))}
          <text x={x1 + 6} y={200} textAnchor="end" className="vz-axis-label">
            걸음 (시간) →
          </text>
          <text x={xt(0)} y={156} textAnchor="middle" style={{ fontSize: 22 }}>
            🤖
          </text>
          {/* option A: 1 candy now, option B: amount candies later — size shrinks with the discount */}
          <g transform={`translate(${clampX(xt(0), 34)} 216)`}>
            <text textAnchor="middle" style={{ fontSize: 13, fontWeight: 850, fill: c.series[3] }}>
              A: 🍬×1
            </text>
          </g>
          <g transform={`translate(${clampX(xt(delay), (candies * candySize) / 2 + 4)} 150)`} style={{ transition: 'transform .3s' }}>
            <text textAnchor="middle" y={-2} style={{ fontSize: candySize }}>
              {'🍬'.repeat(candies)}
            </text>
          </g>
          <g transform={`translate(${clampX(xt(delay), 36)} ${bRow})`}>
            <text textAnchor="middle" style={{ fontSize: 13, fontWeight: 850, fill: c.series[0] }}>
              B: 🍬×{amount}
            </text>
          </g>
        </svg>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10 }}>
          {[
            { key: 'A', label: `A. 지금 당장 사탕 1개`, v: valueNow, color: c.series[3], win: !waits },
            { key: 'B', label: `B. ${delay}걸음 뒤 사탕 ${amount}개`, v: valueLater, color: c.series[0], win: waits }
          ].map((o) => (
            <div
              key={o.key}
              style={{
                padding: '12px 14px',
                borderRadius: 16,
                border: `2px solid ${o.win ? o.color : 'var(--border)'}`,
                background: o.win ? withAlpha(o.color, 0.12) : 'var(--surface-solid)',
                transition: 'all .3s'
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 800, color: 'var(--text-2)' }}>{o.label}</div>
              <div style={{ fontSize: 22, fontWeight: 900, fontVariantNumeric: 'tabular-nums' }}>
                지금 가치 {fmt(o.v, 2)} {o.win ? '👈 선택!' : ''}
              </div>
            </div>
          ))}
        </div>

        <Note mood={note.mood} tone={note.tone}>
          {note.t}
        </Note>
        <div className="vz-stats">
          <Stat label="성격" value={persona} />
          <Stat label="100걸음 뒤 보상의 무게" value={Math.pow(gamma, 100) >= 0.001 ? fmt(Math.pow(gamma, 100), 3) : '거의 0'} sub="γ¹⁰⁰ · γ가 0.99면 약 0.37" />
          <Stat label="매 걸음 1개씩 영원히 받으면" value={Number.isFinite(forever) ? fmt(forever, 1) : '∞'} sub="리턴 = 1 / (1 - γ)" tone="brand" />
        </div>
      </div>
      <Controls>
        <Slider label="할인율 γ (참을성)" value={gamma} min={0} max={0.999} step={0.001} onChange={setGamma} format={(v) => v.toFixed(3)} accent="var(--mint)" />
        <Slider label="B를 받기까지 걸음 수" value={delay} min={1} max={T_MAX} step={1} onChange={setDelay} format={(v) => `${v}걸음`} />
        <Slider label="B의 사탕 개수" value={amount} min={2} max={10} step={1} onChange={setAmount} format={(v) => `${v}개`} accent="var(--peach)" />
      </Controls>
    </div>
  );
}
