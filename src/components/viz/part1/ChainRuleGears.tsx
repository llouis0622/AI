import { useRef, useState } from 'react';
import type { VizProps } from '../Viz';
import { Plot } from '../core/plot';
import { Button, Controls, Note, Segmented, Slider, Stat } from '../core/ui';
import { useRafLoop, usePrefersReducedMotion, useInView } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';

/** Chain rule as gears: sensitivities multiply. Second tab: multiply 30 times → vanishing / exploding. */
export default function ChainRuleGears({ expanded }: VizProps) {
  const [tab, setTab] = useState<'gears' | 'deep'>('gears');
  return (
    <div>
      <div style={{ padding: '12px 14px 0' }}>
        <Segmented
          options={[
            { value: 'gears', label: '⚙️ 톱니바퀴로 보기' },
            { value: 'deep', label: '🔁 30번 곱하면?' }
          ]}
          value={tab}
          onChange={(v) => setTab(v as 'gears' | 'deep')}
        />
      </div>
      {tab === 'gears' ? <Gears expanded={expanded} /> : <Deep expanded={expanded} />}
    </div>
  );
}

function gearPath(r: number, teeth: number) {
  const pts: string[] = [];
  const inner = r * 0.84;
  for (let i = 0; i < teeth * 4; i++) {
    const a = (i / (teeth * 4)) * Math.PI * 2;
    const rr = i % 4 === 1 || i % 4 === 2 ? r : inner;
    pts.push(`${(Math.cos(a) * rr).toFixed(2)},${(Math.sin(a) * rr).toFixed(2)}`);
  }
  return `M${pts.join('L')}Z`;
}

function Gears({ expanded }: { expanded?: boolean }) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [r1, setR1] = useState(3);
  const [r2, setR2] = useState(2);
  const [turns, setTurns] = useState(0); // pedal revolutions done
  const [spinning, setSpinning] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const inView = useInView(box, '0px');
  const target = useRef(0);
  const turnsRef = useRef(0);
  const setT = (v: number) => {
    turnsRef.current = v;
    setTurns(v);
  };

  useRafLoop((dt) => {
    const nt = Math.min(target.current, turnsRef.current + dt * 0.45);
    setT(nt);
    if (nt >= target.current) setSpinning(false);
  }, spinning && inView);

  const spinOnce = () => {
    target.current = Math.floor(turnsRef.current + 1e-6) + 1;
    if (reduce) setT(target.current);
    else setSpinning(true);
  };

  const wheels = [
    { name: '페달 (w)', sub: '파라미터', turns, color: c.series[0], x: 110 },
    { name: '체인 (u)', sub: '중간값', turns: turns * r1, color: c.series[3], x: 320 },
    { name: '바퀴 (L)', sub: '손실', turns: turns * r1 * r2, color: c.series[2], x: 530 }
  ];
  const total = r1 * r2;
  const f1 = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1));

  return (
    <div ref={box}>
      <div style={{ padding: '10px 14px 0' }}>
        <svg className="vz-svg" viewBox="0 0 640 250" role="img" aria-label="페달, 체인, 바퀴 세 개의 톱니바퀴" style={{ touchAction: 'pan-y' }}>
          <line x1={110} y1={110} x2={320} y2={110} stroke="var(--border-strong)" strokeWidth={10} strokeLinecap="round" opacity={0.6} />
          <line x1={320} y1={110} x2={530} y2={110} stroke="var(--border-strong)" strokeWidth={10} strokeLinecap="round" opacity={0.6} />
          {[
            { x: 215, txt: `×${f1(r1)}` },
            { x: 425, txt: `×${f1(r2)}` }
          ].map((b) => (
            <g key={b.x}>
              <rect x={b.x - 30} y={92} width={60} height={36} rx={18} fill="var(--surface-solid)" stroke={c.series[1]} strokeWidth={2.5} />
              <text x={b.x} y={116} textAnchor="middle" style={{ fontSize: 16, fontWeight: 900, fill: c.series[1] }}>
                {b.txt}
              </text>
            </g>
          ))}
          {wheels.map((w) => (
            <g key={w.name}>
              <g transform={`translate(${w.x} 110) rotate(${(w.turns * 360) % 360})`}>
                <path d={gearPath(56, 12)} fill={withAlpha(w.color, 0.22)} stroke={w.color} strokeWidth={3} />
                <circle r={14} fill={w.color} />
                <line x1={0} y1={0} x2={0} y2={-44} stroke={w.color} strokeWidth={6} strokeLinecap="round" />
              </g>
              <text x={w.x} y={196} textAnchor="middle" style={{ fontSize: 14, fontWeight: 850, fill: 'var(--text)' }}>
                {w.name}
              </text>
              <text x={w.x} y={216} textAnchor="middle" style={{ fontSize: 12, fontWeight: 700, fill: 'var(--text-2)' }}>
                {w.sub}
              </text>
              <text x={w.x} y={240} textAnchor="middle" style={{ fontSize: 13, fontWeight: 850, fill: w.color }}>
                {w.turns.toFixed(1)}바퀴
              </text>
            </g>
          ))}
        </svg>
      </div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ padding: '10px 14px', borderRadius: 14, background: 'var(--surface-2)', border: '1px solid var(--border)', textAlign: 'center', fontSize: 15, fontWeight: 750, lineHeight: 1.8 }}>
          페달 1바퀴 → 바퀴 <span style={{ color: c.series[3] }}>{f1(r1)}</span> × <span style={{ color: c.series[2] }}>{f1(r2)}</span> = <strong style={{ fontSize: 20 }}>{f1(total)}</strong>바퀴
          <div style={{ fontSize: 13, fontWeight: 650, color: 'var(--text-2)' }}>
            ∂L/∂w = ∂L/∂u × ∂u/∂w — 겹친 함수의 민감도는 단계별 민감도의 <strong>곱</strong>!
          </div>
        </div>
        <div className="vz-stats">
          <Stat label="∂u/∂w (체인 ÷ 페달)" value={f1(r1)} />
          <Stat label="∂L/∂u (바퀴 ÷ 체인)" value={f1(r2)} />
          <Stat label="∂L/∂w (전체 민감도)" value={f1(total)} tone="brand" />
        </div>
        <Note mood={turns >= 1 ? 'happy' : 'smile'}>
          {turns >= 1
            ? `페달을 ${Math.round(turns)}바퀴 돌렸더니 바퀴는 ${f1(Math.round(turns) * total)}바퀴! 신경망이 층을 수십 개 겹쳐도, 층마다의 민감도를 곱하기만 하면 전체 기울기가 나와요. 이게 역전파의 비밀이에요.`
            : '"페달 1바퀴 돌리기"를 눌러 보세요. 아래 슬라이더로 톱니 비율을 바꿔도 돼요.'}
        </Note>
      </div>
      <Controls>
        <Button variant="primary" onClick={spinOnce} disabled={spinning}>
          🚲 페달 1바퀴 돌리기
        </Button>
        <Slider label="체인 비율 (∂u/∂w)" value={r1} min={0.5} max={4} step={0.5} onChange={setR1} format={(v) => `×${f1(v)}`} accent={c.series[3]} />
        <Slider label="바퀴 비율 (∂L/∂u)" value={r2} min={0.5} max={4} step={0.5} onChange={setR2} format={(v) => `×${f1(v)}`} accent={c.series[2]} />
        <Button variant="ghost" onClick={() => { target.current = 0; setSpinning(false); setT(0); }}>
          ↺ 처음으로
        </Button>
      </Controls>
      {expanded && <div style={{ height: 4 }} />}
    </div>
  );
}

/** Number in scientific notation with a real superscript exponent (HTML). */
function Sci({ v }: { v: number }) {
  if (v === 0) return <>0</>;
  if (v >= 1e4 || v < 1e-3) {
    const e = Math.floor(Math.log10(v));
    const m = v / 10 ** e;
    return (
      <span style={{ whiteSpace: 'nowrap' }}>
        {m.toFixed(1)}×10<sup style={{ fontSize: '0.62em' }}>{e}</sup>
      </span>
    );
  }
  return <>{v.toFixed(v < 1 ? 4 : 2)}</>;
}

function Deep({ expanded }: { expanded?: boolean }) {
  const c = useVizColors();
  const [r, setR] = useState(0.5);
  const [n, setN] = useState(30);
  const vals = Array.from({ length: n }, (_, k) => r ** (k + 1));
  const final = vals[vals.length - 1];
  const lo = -12;
  const hi = 6;
  const lg = (v: number) => Math.max(lo, Math.min(hi, Math.log10(v)));
  const tone = r < 0.999 ? 'vanish' : r > 1.001 ? 'explode' : 'keep';
  const col = tone === 'vanish' ? c.series[3] : tone === 'explode' ? c.neg : c.pos;
  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <Plot
          xDomain={[0.5, n + 0.5]}
          yDomain={[lo, hi]}
          xLabel="층 (몇 번 곱했나)"
          yLabel="민감도 (10의 몇 제곱)"
          aspect={expanded ? 2.2 : 1.7}
          maxHeight={expanded ? 520 : 360}
          xTicks={Math.min(10, n)}
          yTicks={6}
          yFormat={(v) => (v === 0 ? '1' : `10의 ${v}승`)}
          margin={{ top: 16, right: 14, bottom: 44, left: 74 }}
          xFormat={(v) => String(v)}
          ariaLabel="층을 지날 때마다 곱해지는 민감도의 크기"
        >
          {({ x, y, inner }) => {
            const bw = Math.max(2, (inner.width / n) * 0.7);
            return (
              <g>
                <line x1={inner.left} x2={inner.left + inner.width} y1={y(0)} y2={y(0)} stroke={c.pos} strokeWidth={1.5} strokeDasharray="5 4" />
                {vals.map((v, k) => {
                  const top = y(Math.max(lg(v), 0));
                  const bottom = y(Math.min(lg(v), 0));
                  return <rect key={k} x={x(k + 1) - bw / 2} y={top} width={bw} height={Math.max(1.5, bottom - top)} rx={Math.min(3, bw / 2)} fill={col} opacity={0.35 + 0.65 * ((k + 1) / n)} />;
                })}
                <text x={inner.left + 6} y={y(0) - 6} style={{ fontSize: 11.5, fontWeight: 800, fill: c.pos }}>
                  1 (그대로)
                </text>
              </g>
            );
          }}
        </Plot>
      </div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div className="vz-stats">
          <Stat label="층마다 곱하는 수" value={r.toFixed(2)} />
          <Stat label="층 수" value={n} />
          <Stat label={`${r.toFixed(2)}을 ${n}번 곱하면`} value={<Sci v={final} />} tone={tone === 'keep' ? 'good' : 'bad'} />
        </div>
        <Note mood={tone === 'keep' ? 'happy' : 'wow'} tone={tone === 'keep' ? 'good' : 'warn'}>
          {tone === 'vanish' ? (
            <>
              1보다 작은 수를 계속 곱하면 순식간에 0에 가까워져요. {r.toFixed(2)}을 {n}번 곱하면 <Sci v={final} />! 앞쪽 층에는 기울기가 거의 전달되지 않아요 — "기울기 소실"이에요 (11장).
            </>
          ) : tone === 'explode' ? (
            <>
              1보다 큰 수를 계속 곱하면 폭발해요. {r.toFixed(2)}을 {n}번 곱하면 <Sci v={final} />! 이게 "기울기 폭발"이에요 (11장).
            </>
          ) : (
            '딱 1이면 몇 번을 곱해도 그대로예요. 깊은 신경망이 잘 배우려면 층마다 민감도를 1 근처로 맞춰 주는 게 중요해요.'
          )}
        </Note>
      </div>
      <Controls>
        <Slider label="층마다 곱하는 수" value={r} min={0.5} max={1.5} step={0.05} onChange={setR} format={(v) => v.toFixed(2)} accent={col} />
        <Slider label="층 수" value={n} min={1} max={40} step={1} onChange={setN} format={(v) => `${v}층`} />
        <Button size="sm" onClick={() => { setR(0.5); setN(30); }}>
          책의 예: 0.5를 30번
        </Button>
      </Controls>
    </div>
  );
}
