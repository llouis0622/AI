import { useRef, useState } from 'react';
import type { VizProps } from '../Viz';
import { Plot } from '../core/plot';
import { Button, Controls, Note, Stat, Toggle } from '../core/ui';
import { useVizColors, withAlpha } from '../core/colors';
import { clamp, fmt } from '../core/math';

type V = [number, number];
const D = 3;

const PRESETS: { label: string; a: V; b: V }[] = [
  { label: '같은 방향 (1,0)·(1,0)', a: [1, 0], b: [1, 0] },
  { label: '직각 (1,0)·(0,1)', a: [1, 0], b: [0, 1] },
  { label: '반대 (1,0)·(-1,0)', a: [1, 0], b: [-1, 0] },
  { label: '비슷한 방향', a: [2, 1], b: [1.5, 1.5] }
];

function mood(cos: number): { label: string; emoji: string } {
  if (cos > 0.7) return { label: '아주 닮았어요', emoji: '😍' };
  if (cos > 0.2) return { label: '조금 닮았어요', emoji: '🙂' };
  if (cos > -0.2) return { label: '서로 상관없어요', emoji: '😐' };
  if (cos > -0.7) return { label: '조금 반대예요', emoji: '🙁' };
  return { label: '정반대예요', emoji: '😠' };
}

/** Drag two arrows; the dot product, angle and cosine similarity update live. */
export default function DotProduct({ expanded }: VizProps) {
  const c = useVizColors();
  const [a, setA] = useState<V>([2, 1]);
  const [b, setB] = useState<V>([1, 2]);
  const [snap, setSnap] = useState(true);
  const [proj, setProj] = useState(true);
  const drag = useRef<null | 'a' | 'b'>(null);

  const dot = a[0] * b[0] + a[1] * b[1];
  const na = Math.hypot(...a);
  const nb = Math.hypot(...b);
  const cos = na > 1e-9 && nb > 1e-9 ? clamp(dot / (na * nb), -1, 1) : 0;
  const ang = (Math.acos(cos) * 180) / Math.PI;
  const m = mood(cos);
  // projection of b onto a
  const t = na > 1e-9 ? dot / (na * na) : 0;
  const pb: V = [a[0] * t, a[1] * t];
  const q = (v: number) => (snap ? Math.round(v * 2) / 2 : Math.round(v * 10) / 10);
  const f1 = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1));

  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <div style={{ maxWidth: expanded ? 640 : 460, margin: '0 auto' }}>
          <Plot
            xDomain={[-D, D]}
            yDomain={[-D, D]}
            aspect={1}
            maxHeight={expanded ? 620 : 440}
            margin={{ top: 12, right: 12, bottom: 30, left: 34 }}
            xTicks={6}
            yTicks={6}
            ariaLabel="원점에서 시작하는 두 화살표 a와 b"
            onPointerMove={(e, s) => {
              if (!drag.current) return;
              const [x, y] = s.toData(e);
              const v: V = [q(clamp(x, -D, D)), q(clamp(y, -D, D))];
              if (drag.current === 'a') setA(v);
              else setB(v);
            }}
            onPointerUp={() => (drag.current = null)}
          >
            {({ x, y }) => {
              const arrow = (v: V, color: string, name: 'a' | 'b') => {
                const x0 = x(0);
                const y0 = y(0);
                const x1 = x(v[0]);
                const y1 = y(v[1]);
                const L = Math.hypot(x1 - x0, y1 - y0);
                const ux = L > 0 ? (x1 - x0) / L : 0;
                const uy = L > 0 ? (y1 - y0) / L : 0;
                return (
                  <g>
                    <line x1={x0} y1={y0} x2={x1 - ux * 10} y2={y1 - uy * 10} stroke={color} strokeWidth={5} strokeLinecap="round" />
                    {L > 4 && <path d={`M${x1} ${y1} L${x1 - ux * 16 - uy * 8} ${y1 - uy * 16 + ux * 8} L${x1 - ux * 16 + uy * 8} ${y1 - uy * 16 - ux * 8} Z`} fill={color} />}
                    <g
                      style={{ cursor: 'grab' }}
                      onPointerDown={(e) => {
                        drag.current = name;
                        (e.currentTarget.ownerSVGElement as SVGSVGElement).setPointerCapture(e.pointerId);
                      }}
                    >
                      <circle cx={x1} cy={y1} r={22} fill="transparent" />
                      <circle cx={x1} cy={y1} r={9} fill={c.surface} stroke={color} strokeWidth={3.5} />
                    </g>
                    <text x={x1 + (ux >= 0 ? 14 : -14)} y={y1 - 12} textAnchor={ux >= 0 ? 'start' : 'end'} style={{ fontSize: 14, fontWeight: 900, fill: color }}>
                      {name} = ({f1(v[0])}, {f1(v[1])})
                    </text>
                  </g>
                );
              };
              // angle arc
              const a1 = Math.atan2(a[1], a[0]);
              const b1 = Math.atan2(b[1], b[0]);
              let d = b1 - a1;
              while (d > Math.PI) d -= 2 * Math.PI;
              while (d < -Math.PI) d += 2 * Math.PI;
              const r = 0.55;
              const arcPts = Array.from({ length: 24 }, (_, i) => {
                const th = a1 + (d * i) / 23;
                return `${x(r * Math.cos(th))},${y(r * Math.sin(th))}`;
              });
              return (
                <g>
                  <line x1={x(-D)} x2={x(D)} y1={y(0)} y2={y(0)} stroke="var(--viz-axis)" strokeWidth={1.5} />
                  <line x1={x(0)} x2={x(0)} y1={y(-D)} y2={y(D)} stroke="var(--viz-axis)" strokeWidth={1.5} />
                  {na > 1e-9 && (
                    <line
                      x1={x(-a[0] * (D / na) * 1.5)}
                      y1={y(-a[1] * (D / na) * 1.5)}
                      x2={x(a[0] * (D / na) * 1.5)}
                      y2={y(a[1] * (D / na) * 1.5)}
                      stroke={withAlpha(c.series[0], 0.25)}
                      strokeWidth={1.5}
                      strokeDasharray="4 5"
                    />
                  )}
                  {proj && na > 1e-9 && nb > 1e-9 && (
                    <g>
                      <line x1={x(b[0])} y1={y(b[1])} x2={x(pb[0])} y2={y(pb[1])} strokeWidth={1.8} strokeDasharray="4 4" style={{ stroke: 'var(--lemon)' }} />
                      <line x1={x(0)} y1={y(0)} x2={x(pb[0])} y2={y(pb[1])} strokeWidth={12} strokeLinecap="round" opacity={0.6} style={{ stroke: 'var(--lemon)' }} />
                    </g>
                  )}
                  {na > 1e-9 && nb > 1e-9 && <polyline points={arcPts.join(' ')} fill="none" stroke={c.series[4]} strokeWidth={2.5} />}
                  {arrow(a, c.series[0], 'a')}
                  {arrow(b, c.series[1], 'b')}
                </g>
              );
            }}
          </Plot>
        </div>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ padding: '10px 14px', borderRadius: 14, background: 'var(--surface-2)', border: '1px solid var(--border)', fontSize: 15, fontWeight: 700, textAlign: 'center', lineHeight: 1.8 }}>
          a · b = <span style={{ color: c.series[0] }}>{f1(a[0])}</span>×<span style={{ color: c.series[1] }}>{f1(b[0])}</span> + <span style={{ color: c.series[0] }}>{f1(a[1])}</span>×<span style={{ color: c.series[1] }}>{f1(b[1])}</span> ={' '}
          <strong style={{ fontSize: 20 }}>{fmt(dot, 2)}</strong>
          <div style={{ fontSize: 12.5, fontWeight: 650, color: 'var(--text-2)' }}>같은 자리 숫자끼리 곱해서 모두 더하면 내적!</div>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontWeight: 750, color: 'var(--text-3)', marginBottom: 4 }}>
            <span>😠 정반대 (-1)</span>
            <span>😐 상관없음 (0)</span>
            <span>😍 같은 방향 (+1)</span>
          </div>
          <div style={{ position: 'relative', height: 16, borderRadius: 99, background: `linear-gradient(90deg, ${c.neg}, var(--surface-3), ${c.pos})` }}>
            <div
              style={{
                position: 'absolute',
                top: -7,
                left: `calc(${((cos + 1) / 2) * 100}% - 15px)`,
                width: 30,
                height: 30,
                borderRadius: '50%',
                background: 'var(--surface-solid)',
                border: '2px solid var(--border-strong)',
                display: 'grid',
                placeItems: 'center',
                fontSize: 17,
                transition: 'left .2s'
              }}
              aria-hidden="true"
            >
              {m.emoji}
            </div>
          </div>
        </div>

        <div className="vz-stats">
          <Stat label="내적 a·b" value={fmt(dot, 2)} tone="brand" />
          <Stat label="사이 각도" value={`${ang.toFixed(0)}°`} />
          <Stat label="코사인 유사도" value={fmt(cos, 2)} tone={cos > 0.2 ? 'good' : cos < -0.2 ? 'bad' : undefined} sub={m.label} />
        </div>
        <Note mood={cos > 0.7 ? 'happy' : cos < -0.7 ? 'wow' : 'think'}>
          {Math.abs(cos) < 0.05
            ? '직각이면 내적이 0! 두 화살표는 서로 "상관없는" 방향이에요.'
            : cos > 0
              ? '같은 쪽을 가리킬수록 내적이 커져요. 그래서 내적으로 "두 단어의 뜻이 비슷한가"를 잴 수 있어요.'
              : '반대 쪽을 가리키면 내적이 음수가 돼요.'}{' '}
          노란 막대는 b가 a 방향으로 드리운 "그림자"예요 — 내적 = |a| × 그림자 길이.
        </Note>
      </div>

      <Controls>
        {PRESETS.map((p) => (
          <Button key={p.label} size="sm" onClick={() => { setA(p.a); setB(p.b); }}>
            {p.label}
          </Button>
        ))}
        <Toggle label="0.5칸씩 맞추기" checked={snap} onChange={setSnap} />
        <Toggle label="그림자(투영) 보기" checked={proj} onChange={setProj} />
      </Controls>
    </div>
  );
}
