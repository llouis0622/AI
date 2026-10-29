import { useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import type { VizProps } from '../Viz';
import { Plot, linePath } from '../core/plot';
import { Button, Controls, Note, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { useElementSize, useRafLoop, usePrefersReducedMotion } from '../core/hooks';
import { mixHex, useVizColors } from '../core/colors';
import { clamp, linspace, sigmoid } from '../core/math';

type Tab = 'squash' | 'tumor';
const Z_MAX = 8;
const SAMPLES = [-8, -6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6, 8];

/* Tumor sizes (cm) — a made-up toy set with a little overlap in the middle. */
const BENIGN = [1.2, 1.8, 2.3, 2.9, 3.4, 3.8, 4.3, 5.1];
const MALIGNANT = [4.6, 5.6, 6.0, 6.7, 7.3, 7.9, 8.5, 9.2];
const TUMORS = [...BENIGN.map((x) => ({ x, y: 0 as 0 | 1 })), ...MALIGNANT.map((x) => ({ x, y: 1 as 0 | 1 }))];

function ce(w: number, b: number) {
  let L = 0;
  for (const t of TUMORS) {
    const p = sigmoid(w * t.x + b);
    L -= t.y ? Math.log(Math.max(p, 1e-12)) : Math.log(Math.max(1 - p, 1e-12));
  }
  return L / TUMORS.length;
}

/** Newton's method (IRLS) for the 1D logistic regression above. */
function bestFit(): [number, number] {
  let w = 0.5;
  let b = -2.5;
  for (let it = 0; it < 40; it++) {
    let gw = 0;
    let gb = 0;
    let hww = 0;
    let hwb = 0;
    let hbb = 0;
    for (const t of TUMORS) {
      const p = sigmoid(w * t.x + b);
      const r = p * (1 - p);
      gw += (p - t.y) * t.x;
      gb += p - t.y;
      hww += r * t.x * t.x;
      hwb += r * t.x;
      hbb += r;
    }
    const det = hww * hbb - hwb * hwb;
    if (Math.abs(det) < 1e-12) break;
    w -= (hbb * gw - hwb * gb) / det;
    b -= (-hwb * gw + hww * gb) / det;
  }
  return [w, b];
}
const [W_BEST, B_BEST] = bestFit();
const L_BEST = ce(W_BEST, B_BEST);

/** Least-squares straight line through the 0/1 labels ("just use the line from chapter 2"). */
const LS = (() => {
  const n = TUMORS.length;
  const mx = TUMORS.reduce((s, t) => s + t.x, 0) / n;
  const my = TUMORS.reduce((s, t) => s + t.y, 0) / n;
  const sxy = TUMORS.reduce((s, t) => s + (t.x - mx) * (t.y - my), 0);
  const sxx = TUMORS.reduce((s, t) => s + (t.x - mx) ** 2, 0);
  const a = sxy / sxx;
  return { a, d: my - a * mx };
})();

/** Sigmoid as a "squashing machine", then used on tumor sizes. */
export default function SigmoidSquash({ expanded }: VizProps) {
  const [tab, setTab] = useState<Tab>('squash');
  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <Segmented
          options={[
            { value: 'squash', label: '🎚️ 구겨 넣기 기계' },
            { value: 'tumor', label: '🩺 종양 판별에 써 보기' }
          ]}
          value={tab}
          onChange={setTab}
        />
      </div>
      {tab === 'squash' ? <Squash expanded={!!expanded} /> : <Tumor expanded={!!expanded} />}
    </div>
  );
}

/* ---------------- tab 1: squash ---------------- */

function Squash({ expanded }: { expanded: boolean }) {
  const c = useVizColors();
  const [z, setZ] = useState(1.8);
  const [box, { width }] = useElementSize<HTMLDivElement>();
  const dragging = useRef(false);

  const W = Math.max(280, width);
  const H = expanded ? 280 : 236;
  const L = 22;
  const R = W - 22;
  const yTop = 52;
  const yBot = H - 54;
  const xz = (v: number) => L + ((clamp(v, -Z_MAX, Z_MAX) + Z_MAX) / (2 * Z_MAX)) * (R - L);
  const xp = (p: number) => L + p * (R - L);
  const p = sigmoid(z);
  const off = Math.abs(z) > Z_MAX;

  const setFromEvent = (e: ReactPointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    setZ(Math.round(clamp(((px - L) / (R - L)) * 2 * Z_MAX - Z_MAX, -Z_MAX, Z_MAX) * 10) / 10);
  };

  const curve = (v: number) => {
    const x0 = xz(v);
    const x1 = xp(sigmoid(v));
    const m = (yBot - yTop) * 0.5;
    return `M${x0},${yTop} C${x0},${yTop + m} ${x1},${yBot - m} ${x1},${yBot}`;
  };

  const zLabel = off ? (z > 0 ? `${z} →` : `← ${z}`) : z.toFixed(1);
  const pText = p > 0.9999 ? '0.9999…' : p < 0.0001 ? '0.0000…' : p.toFixed(4);

  let note: { text: string; mood: 'smile' | 'happy' | 'think' | 'wow'; tone?: 'good' | 'warn' };
  if (off)
    note = {
      text: `점수 ${z}처럼 엄청난 숫자를 넣어도 확률은 ${z > 0 ? '1을 넘지' : '0 밑으로 내려가지'} 못해요. 시그모이드는 어떤 숫자든 0과 1 사이로 구겨 넣어요!`,
      mood: 'wow'
    };
  else if (Math.abs(z) < 0.15) note = { text: '점수 0 → 확률 정확히 0.5, 반반이에요. "악성일까 양성일까" 가장 헷갈리는 지점 — 여기가 결정 경계가 돼요.', mood: 'think' };
  else if (z <= -3) note = { text: '큰 음수 점수는 0 쪽에 착 붙어요. "악성일 리 없다"는 강한 확신이에요.', mood: 'smile' };
  else if (z >= 3) note = { text: '큰 양수 점수는 1 쪽에 착 붙어요. 점수를 더 올려도 확률은 거의 안 변해요.', mood: 'smile' };
  else note = { text: '가운데(점수 0 근처)에서는 점수가 조금만 변해도 확률이 크게 움직여요. 위쪽 손잡이를 좌우로 끌어 보세요!', mood: 'happy' };

  return (
    <div>
      <div ref={box} className="vz-pad" style={{ paddingBottom: 0 }}>
        {width > 0 && (
          <svg
            className="vz-svg"
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="위쪽 점수 눈금의 값들이 시그모이드를 지나 아래쪽 0~1 확률 막대로 모이는 깔때기 그림"
            style={{ touchAction: 'pan-y', cursor: 'ew-resize', userSelect: 'none' }}
            onPointerDown={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              if (e.clientY - r.top > yTop + 40) return;
              dragging.current = true;
              e.currentTarget.setPointerCapture(e.pointerId);
              setFromEvent(e);
            }}
            onPointerMove={(e) => dragging.current && setFromEvent(e)}
            onPointerUp={() => (dragging.current = false)}
            onPointerCancel={() => (dragging.current = false)}
          >
            <defs>
              <linearGradient id="sq-bar" x1="0" x2="1" y1="0" y2="0">
                <stop offset="0%" stopColor={c.series[3]} />
                <stop offset="50%" stopColor={mixHex(c.series[3], c.series[2], 0.5)} />
                <stop offset="100%" stopColor={c.series[2]} />
              </linearGradient>
            </defs>
            <text x={L} y={20} style={{ fontSize: 12.5, fontWeight: 800, fill: c.ink }}>
              ① 직선이 내놓는 점수 z (아무 숫자나 OK)
            </text>
            {/* z axis */}
            <line x1={L} x2={R} y1={yTop} y2={yTop} stroke={c.axis} strokeWidth={2} />
            {[-8, -4, 0, 4, 8].map((t) => (
              <g key={t}>
                <line x1={xz(t)} x2={xz(t)} y1={yTop - 5} y2={yTop + 5} stroke={c.axis} strokeWidth={1.5} />
                <text x={xz(t)} y={yTop - 10} textAnchor="middle" className="vz-tick" style={{ fill: c.text }}>
                  {t}
                </text>
              </g>
            ))}
            {/* funnel */}
            {SAMPLES.map((v) => (
              <path key={v} d={curve(v)} fill="none" stroke={mixHex(c.series[3], c.series[2], sigmoid(v))} strokeWidth={1.6} opacity={0.5} />
            ))}
            {SAMPLES.map((v) => (
              <g key={`d${v}`}>
                <circle cx={xz(v)} cy={yTop} r={3.5} fill={mixHex(c.series[3], c.series[2], sigmoid(v))} />
                <circle cx={xp(sigmoid(v))} cy={yBot} r={3.5} fill={mixHex(c.series[3], c.series[2], sigmoid(v))} />
              </g>
            ))}
            <text x={W / 2} y={(yTop + yBot) / 2 + 5} textAnchor="middle" style={{ fontSize: 15, fontWeight: 900, fill: c.ink, opacity: 0.55 }}>
              σ 시그모이드
            </text>
            {/* p bar */}
            <rect x={L} y={yBot - 6} width={R - L} height={12} rx={6} fill="url(#sq-bar)" opacity={0.85} />
            {[0, 0.5, 1].map((t) => (
              <text key={t} x={xp(t)} y={yBot + 24} textAnchor={t === 0 ? 'start' : t === 1 ? 'end' : 'middle'} className="vz-tick" style={{ fill: c.text }}>
                {t === 0 ? '0 (양성)' : t === 1 ? '1 (악성)' : '0.5'}
              </text>
            ))}
            <text x={L} y={H - 6} style={{ fontSize: 12.5, fontWeight: 800, fill: c.ink }}>
              ② 시그모이드를 지난 확률 p (0 ~ 1)
            </text>
            {/* current value */}
            <path d={curve(z)} fill="none" strokeWidth={3.5} style={{ stroke: 'var(--lemon)' }} />
            <circle cx={xz(z)} cy={yTop} r={11} stroke={c.surface} strokeWidth={3} style={{ fill: 'var(--lemon)', cursor: 'grab' }} />
            {off && (
              <text x={xz(z)} y={yTop + 28} textAnchor={z > 0 ? 'end' : 'start'} style={{ fontSize: 12.5, fontWeight: 900, fill: c.ink }}>
                {zLabel}
              </text>
            )}
            <circle cx={xp(p)} cy={yBot} r={9} stroke={c.surface} strokeWidth={3} style={{ fill: 'var(--lemon)' }} />
          </svg>
        )}
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12, alignItems: 'start' }}>
          <div style={{ display: 'grid', gap: 10 }}>
            <div className="vz-stats">
              <Stat label="점수 z" value={zLabel} tone="brand" />
              <Stat label="확률 σ(z)" value={pText} sub={`= ${Math.round(p * 100)}%`} />
            </div>
            <Note mood={note.mood} tone={note.tone}>
              {note.text}
            </Note>
          </div>
          <div>
            <div className="vz-panel-title">같은 기계를 그래프로 그리면: S자 곡선</div>
            <Plot
              xDomain={[-Z_MAX, Z_MAX]}
              yDomain={[0, 1]}
              height={expanded ? 240 : 190}
              xLabel="점수 z"
              yLabel="확률 p"
              margin={{ top: 10, right: 12, bottom: 40, left: 44 }}
              xTicks={4}
              yTicks={2}
              ariaLabel="시그모이드 S자 곡선"
            >
              {({ x, y }) => {
                const zc = clamp(z, -Z_MAX, Z_MAX);
                return (
                  <g>
                    <line x1={x(-Z_MAX)} x2={x(Z_MAX)} y1={y(0.5)} y2={y(0.5)} stroke={c.axis} strokeDasharray="4 4" />
                    <path d={linePath(linspace(-Z_MAX, Z_MAX, 120).map((v) => [x(v), y(sigmoid(v))]))} fill="none" stroke={c.series[0]} strokeWidth={3.5} />
                    <line x1={x(zc)} x2={x(zc)} y1={y(0)} y2={y(p)} strokeDasharray="3 3" strokeWidth={1.5} style={{ stroke: 'var(--lemon)' }} />
                    <line x1={x(-Z_MAX)} x2={x(zc)} y1={y(p)} y2={y(p)} strokeDasharray="3 3" strokeWidth={1.5} style={{ stroke: 'var(--lemon)' }} />
                    <circle cx={x(zc)} cy={y(p)} r={7} stroke={c.surface} strokeWidth={2.5} style={{ fill: 'var(--lemon)' }} />
                  </g>
                );
              }}
            </Plot>
          </div>
        </div>
      </div>

      <Controls>
        <Slider label="점수 z" value={clamp(z, -Z_MAX, Z_MAX)} min={-Z_MAX} max={Z_MAX} step={0.1} onChange={setZ} format={(v) => v.toFixed(1)} accent="var(--lemon)" />
        <Button size="sm" onClick={() => setZ(42)}>
          42 넣어 보기
        </Button>
        <Button size="sm" onClick={() => setZ(-3.7)}>
          -3.7 넣어 보기
        </Button>
        <Button size="sm" variant="ghost" onClick={() => setZ(0)}>
          0 넣어 보기
        </Button>
      </Controls>
    </div>
  );
}

/* ---------------- tab 2: tumor ---------------- */

function Tumor({ expanded }: { expanded: boolean }) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [w, setW] = useState(0.7);
  const [cut, setCut] = useState(7.2);
  const [probe, setProbe] = useState(6.2);
  const [showLine, setShowLine] = useState(false);
  const anim = useRef<{ t: number; from: [number, number]; to: [number, number] } | null>(null);
  const [animating, setAnimating] = useState(false);

  const b = -w * cut;
  const loss = ce(w, b);
  const correct = TUMORS.filter((t) => (sigmoid(w * t.x + b) >= 0.5 ? 1 : 0) === t.y).length;
  const zProbe = w * probe + b;
  const pProbe = sigmoid(zProbe);
  const nearBest = loss <= L_BEST * 1.03 + 0.003;

  useRafLoop((dt) => {
    const a = anim.current;
    if (!a) return;
    a.t = Math.min(1, a.t + dt * 1.4);
    const e = 1 - (1 - a.t) ** 3;
    setW(a.from[0] + (a.to[0] - a.from[0]) * e);
    setCut(a.from[1] + (a.to[1] - a.from[1]) * e);
    if (a.t >= 1) {
      anim.current = null;
      setAnimating(false);
    }
  }, animating);

  const autoFit = () => {
    const to: [number, number] = [W_BEST, -B_BEST / W_BEST];
    if (reduce) {
      setW(to[0]);
      setCut(to[1]);
      return;
    }
    anim.current = { t: 0, from: [w, cut], to };
    setAnimating(true);
  };

  let note: { text: string; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' };
  if (showLine)
    note = {
      text: `2장의 직선을 그대로 쓰면(점선) 큰 종양에서 ${(LS.a * 10 + LS.d).toFixed(2)}, 작은 종양에서 ${LS.d.toFixed(2)} 같은 값이 나와요. "악성일 확률 ${Math.round((LS.a * 10 + LS.d) * 100)}%"는 말이 안 되죠? 그래서 S자로 구겨 넣어요.`,
      mood: 'wow',
      tone: 'warn'
    };
  else if (nearBest)
    note = { text: `🎉 거의 최적이에요! 교차 엔트로피가 가장 작은 S자예요. 가운데 섞인 종양 ${TUMORS.length - correct}개는 어떤 S자로도 다 맞힐 수 없어요.`, mood: 'cheer', tone: 'good' };
  else if (w > 3.2)
    note = { text: '아주 가파르면 "무조건 확신"하는 모델이 돼요. 틀린 종양(빨간 테두리)에 매기는 벌점(손실)이 확 커지는 걸 보세요!', mood: 'think', tone: 'warn' };
  else note = { text: '두 슬라이더로 S자의 위치(경계)와 가파름을 바꿔 손실을 줄여 보세요. 막히면 🤖 자동으로 맞추기!', mood: 'smile' };

  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <Plot
          xDomain={[0, 10]}
          yDomain={[-0.35, 1.35]}
          xLabel="종양 크기 (cm)"
          yLabel="악성일 확률"
          aspect={expanded ? 2.2 : 1.65}
          maxHeight={expanded ? 520 : 380}
          yFormat={(v) => (v === 0 || v === 1 || v === 0.5 ? String(v) : '')}
          yTicks={6}
          ariaLabel="종양 크기에 따른 악성 확률 S자 곡선과 데이터"
        >
          {({ x, y, clip, inner }) => {
            const curve = linspace(0, 10, 160).map((v) => [x(v), y(sigmoid(w * v + b))] as [number, number]);
            return (
              <g>
                {showLine && (
                  <g>
                    <rect x={inner.left} y={inner.top} width={inner.width} height={y(1) - inner.top} fill={c.neg} opacity={0.1} />
                    <rect x={inner.left} y={y(0)} width={inner.width} height={inner.top + inner.height - y(0)} fill={c.neg} opacity={0.1} />
                    <text x={inner.left + 8} y={inner.top + 14} style={{ fontSize: 11.5, fontWeight: 800, fill: c.neg }}>
                      확률이 될 수 없는 구역 (1보다 큼)
                    </text>
                    <text x={inner.left + inner.width - 8} y={inner.top + inner.height - 6} textAnchor="end" style={{ fontSize: 11.5, fontWeight: 800, fill: c.neg }}>
                      확률이 될 수 없는 구역 (0보다 작음)
                    </text>
                  </g>
                )}
                <line x1={x(0)} x2={x(10)} y1={y(0.5)} y2={y(0.5)} stroke={c.axis} strokeDasharray="4 4" />
                <g clipPath={clip}>
                  {showLine && <path d={linePath([[x(0), y(LS.d)], [x(10), y(LS.a * 10 + LS.d)]])} fill="none" stroke={c.neg} strokeWidth={3} strokeDasharray="7 5" />}
                  <line x1={x(cut)} x2={x(cut)} y1={y(1.35)} y2={y(-0.35)} stroke={c.ink} strokeWidth={2} strokeDasharray="5 4" opacity={0.6} />
                  <path d={linePath(curve)} fill="none" stroke={c.series[0]} strokeWidth={4} />
                </g>
                <text x={x(cut)} y={y(1.35) + 13} textAnchor={cut > 7.5 ? 'end' : 'start'} dx={cut > 7.5 ? -6 : 6} style={{ fontSize: 11.5, fontWeight: 800, fill: c.ink }}>
                  결정 경계 (점수 0 = 50%)
                </text>
                {TUMORS.map((t, i) => {
                  const px = x(t.x);
                  const py = y(t.y);
                  const wrong = (sigmoid(w * t.x + b) >= 0.5 ? 1 : 0) !== t.y;
                  return (
                    <g key={i}>
                      {wrong && <circle cx={px} cy={py} r={11} fill="none" stroke={c.neg} strokeWidth={2.2} />}
                      {t.y ? (
                        <path d={`M${px},${py - 7.5} L${px + 7},${py + 5} L${px - 7},${py + 5} Z`} fill={c.series[2]} stroke={c.surface} strokeWidth={1.5} />
                      ) : (
                        <circle cx={px} cy={py} r={6} fill={c.series[3]} stroke={c.surface} strokeWidth={1.5} />
                      )}
                    </g>
                  );
                })}
                {/* probe */}
                <line x1={x(probe)} x2={x(probe)} y1={y(-0.35)} y2={y(pProbe)} strokeWidth={2} strokeDasharray="3 3" style={{ stroke: 'var(--lemon)' }} />
                <circle cx={x(probe)} cy={y(pProbe)} r={8} stroke={c.surface} strokeWidth={3} style={{ fill: 'var(--lemon)' }} />
              </g>
            );
          }}
        </Plot>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div
          style={{
            padding: '10px 14px',
            borderRadius: 14,
            background: 'var(--surface-2)',
            border: '1px solid var(--border)',
            fontSize: 14,
            lineHeight: 1.7,
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            gap: '4px 10px'
          }}
        >
          <span style={{ fontWeight: 800 }}>노란 점 종양 {probe.toFixed(1)}cm</span>
          <span aria-hidden="true">→</span>
          <span>
            점수 z = {w.toFixed(2)}×{probe.toFixed(1)} {b >= 0 ? '+' : '-'} {Math.abs(b).toFixed(2)} = <b>{zProbe.toFixed(2)}</b>
          </span>
          <span aria-hidden="true">→</span>
          <span>
            σ(z) = <b>{Math.round(pProbe * 100)}%</b>
          </span>
          <span aria-hidden="true">→</span>
          <b style={{ color: pProbe >= 0.5 ? c.series[2] : c.series[3] }}>{pProbe >= 0.5 ? '▲ 악성으로 판정' : '● 양성으로 판정'}</b>
        </div>
        <div className="vz-stats">
          <Stat label="정확도 (50%에서 자르기)" value={`${correct}/${TUMORS.length}`} tone={correct >= 15 ? 'good' : undefined} />
          <Stat label="손실 (교차 엔트로피)" value={loss.toFixed(3)} tone={nearBest ? 'good' : 'brand'} sub={`최선 ${L_BEST.toFixed(3)}`} />
        </div>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        <Slider label="경계 위치 (cm)" value={cut} min={0.5} max={9.5} step={0.05} onChange={setCut} format={(v) => v.toFixed(1)} disabled={animating} />
        <Slider label="가파름 w" value={w} min={0.2} max={6} step={0.05} onChange={setW} format={(v) => v.toFixed(2)} disabled={animating} />
        <Slider label="노란 점: 검사할 종양 크기" value={probe} min={0.5} max={9.5} step={0.1} onChange={setProbe} format={(v) => `${v.toFixed(1)}cm`} accent="var(--lemon)" />
        <Button size="sm" variant="primary" onClick={autoFit} disabled={animating}>
          🤖 자동으로 맞추기
        </Button>
        <Toggle label="직선을 그대로 쓰면?" checked={showLine} onChange={setShowLine} />
      </Controls>
    </div>
  );
}
