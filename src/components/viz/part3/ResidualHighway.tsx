import { useEffect, useMemo, useState } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Slider, Stat } from '../core/ui';
import { useRafLoop, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { rng } from '../core/math';
import { Sci } from './_sci';

/**
 * Backward signal through a plain stack (gradient × F'(x) per layer) versus a residual
 * stack (gradient × (1 + F'(x)) per layer). Local sensitivities are random but seeded.
 */
export default function ResidualHighway({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [depth, setDepth] = useState(30);
  const [sens, setSens] = useState(0.7);
  const [seed, setSeed] = useState(5);
  const [t, setT] = useState(1);
  const [playing, setPlaying] = useState(false);

  useRafLoop((dt) => setT((v) => Math.min(1, v + dt / 2.8)), playing);
  useEffect(() => {
    if (playing && t >= 1) setPlaying(false);
  }, [playing, t]);

  const { plain, resid } = useMemo(() => {
    const r = rng(seed);
    const local = Array.from({ length: depth }, () => sens * (1 + 0.35 * (r.next() * 2 - 1)));
    // gradient arriving at layer k (1-based) = product of factors of layers k..depth
    const plainG: number[] = new Array(depth);
    const residG: number[] = new Array(depth);
    let gp = 1;
    let gr = 1;
    for (let k = depth - 1; k >= 0; k--) {
      gp *= local[k];
      gr *= 1 + 0.22 * (local[k] - sens);
      plainG[k] = gp;
      residG[k] = gr;
    }
    return { plain: plainG, resid: residG };
  }, [depth, sens, seed]);

  const W = 720;
  const H = expanded ? 380 : 330;
  const left = 70;
  const right = 70;
  const usable = W - left - right;
  const slot = usable / depth;
  const laneY = [H * 0.36, H * 0.84];
  const barH = H * 0.2;
  const level = (g: number) => Math.max(0, Math.min(1, (Math.log10(Math.max(g, 1e-12)) + 10) / 10.5));
  const revealed = (k: number) => t >= 1 || k >= depth - t * depth;
  const pulseX = left + usable * (1 - t);

  const lane = (gs: number[], y: number, residual: boolean) => (
    <g>
      <line x1={left} x2={left + usable} y1={y} y2={y} stroke={c.axis} strokeWidth={1.5} />
      {residual &&
        (slot > 7 ? (
          gs.map((_, i) => {
            const x0 = left + i * slot;
            return <path key={`a${i}`} d={`M${x0 + 1} ${y - 6} Q${x0 + slot / 2} ${y - 6 - Math.min(16, slot * 0.9)} ${x0 + slot - 1} ${y - 6}`} fill="none" stroke={c.pos} strokeWidth={1.6} opacity={0.8} />;
          })
        ) : (
          <rect x={left} y={y - 14} width={usable} height={5} rx={2.5} fill={withAlpha(c.pos, 0.6)} />
        ))}
      {gs.map((g, i) => {
        const k = i;
        const x0 = left + i * slot;
        const lv = level(g);
        const col = g < 1e-3 ? c.neg : c.pos;
        return (
          <g key={i} opacity={revealed(k) ? 1 : 0.15}>
            <rect x={x0 + slot * 0.15} y={y - 22 - lv * barH} width={Math.max(1, slot * 0.7)} height={Math.max(1, lv * barH)} rx={Math.min(3, slot * 0.25)} fill={col} opacity={0.35 + 0.65 * lv} />
            <rect x={x0 + slot * 0.12} y={y - 5} width={Math.max(1, slot * 0.76)} height={10} rx={3} fill={withAlpha(c.series[0], 0.25)} stroke={c.series[0]} strokeWidth={slot > 8 ? 1 : 0} />
          </g>
        );
      })}
      <circle cx={left + usable + 30} cy={y} r={14} fill={withAlpha(c.series[2], 0.2)} stroke={c.series[2]} strokeWidth={2.2} />
      <text x={left + usable + 30} y={y + 5} textAnchor="middle" style={{ fontSize: 13, fontWeight: 900, fill: 'var(--text)' }}>
        L
      </text>
      {playing && <circle cx={pulseX} cy={y} r={7} fill={c.series[2]} />}
    </g>
  );

  const p1 = plain[0];
  const r1 = resid[0];
  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <svg className="vz-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="일반 망과 잔차 망의 기울기 비교" style={{ touchAction: 'pan-y' }}>
          <text x={left} y={laneY[0] - barH - 34} style={{ fontSize: 14, fontWeight: 850, fill: 'var(--text)' }}>
            일반 망: 기울기 × F'(x)
          </text>
          <text x={left} y={laneY[1] - barH - 34} style={{ fontSize: 14, fontWeight: 850, fill: 'var(--text)' }}>
            잔차 망: 기울기 × (1 + F'(x))
            <tspan style={{ fill: c.pos }}> ← 초록 고속도로 = "+1"</tspan>
          </text>
          {lane(plain, laneY[0], false)}
          {lane(resid, laneY[1], true)}
          <text x={left - 10} y={laneY[0] + 4} textAnchor="end" style={{ fontSize: 11, fontWeight: 800, fill: 'var(--text-2)' }}>
            1층
          </text>
          <text x={left - 10} y={laneY[1] + 4} textAnchor="end" style={{ fontSize: 11, fontWeight: 800, fill: 'var(--text-2)' }}>
            1층
          </text>
        </svg>
      </div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div className="vz-stats">
          <Stat label="1층에 도착 (일반 망)" value={<Sci v={p1} />} tone={p1 < 1e-3 ? 'bad' : 'good'} />
          <Stat label="1층에 도착 (잔차 망)" value={<Sci v={r1} />} tone={r1 < 1e-3 ? 'bad' : 'good'} />
        </div>
        <Note mood={p1 < 1e-3 ? 'wow' : 'happy'} tone={p1 < 1e-3 ? 'good' : undefined}>
          {p1 < 1e-3 ? (
            <>
              일반 망은 {depth}층을 거슬러 오는 동안 기울기가 <Sci v={p1} />로 사라졌지만, 잔차 망은 <Sci v={r1} /> — 거의 그대로 살아 돌아왔어요! "1을 더하는 지름길" 덕분에 곱해도 줄지 않아요.
            </>
          ) : (
            '층이 얕거나 민감도가 1에 가까우면 일반 망도 버텨요. 층 수를 늘리거나 민감도를 낮춰 보세요.'
          )}
        </Note>
      </div>
      <Controls>
        <Slider label="층 수" value={depth} min={10} max={100} step={1} onChange={setDepth} format={(v) => `${v}층`} />
        <Slider label="층의 민감도 F'(x)" value={sens} min={0.3} max={1} step={0.01} onChange={setSens} format={(v) => v.toFixed(2)} accent="var(--peach)" />
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
        <Button size="sm" variant="ghost" onClick={() => setSeed((s) => s + 1)}>
          🎲 다시 뽑기
        </Button>
      </Controls>
    </div>
  );
}
