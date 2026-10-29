import { useMemo, useRef, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Plot, linePath } from '../core/plot';
import { Button, Controls, Legend, Note, Segmented, Slider, Toggle } from '../core/ui';
import { useVizColors, withAlpha } from '../core/colors';
import { clamp, range, sigmoid } from '../core/math';

const N = 1000;
const BINS = 20;
const PREVALENCE = [
  { value: 500, label: '반반 (50%)' },
  { value: 100, label: '10%' },
  { value: 20, label: '2%' },
  { value: 2, label: '0.2% (책 예시)' }
];

/** Inverse standard normal CDF (Acklam's rational approximation). */
function invNorm(p: number) {
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924];
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857];
  const cc = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878];
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
  const lo = 0.02425;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - lo) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** n evenly-spread scores from a logit-normal distribution (smooth and deterministic). */
const scores = (n: number, mu: number, s: number) => range(n).map((i) => sigmoid(mu + s * invNorm((i + 0.5) / n)));

/** Score histograms of healthy vs sick people, a threshold dial, the confusion matrix and the ROC point. */
export default function ConfusionThreshold({ expanded }: VizProps) {
  const c = useVizColors();
  const [sick, setSick] = useState(100);
  const [thr, setThr] = useState(0.5);
  const [norm, setNorm] = useState(false);
  const dragging = useRef(false);

  const data = useMemo(() => {
    const pos = scores(sick, 1.3, 1.05);
    const neg = scores(N - sick, -1.7, 1.05);
    const hist = (xs: number[]) => {
      const h = new Array(BINS).fill(0);
      for (const v of xs) h[Math.min(BINS - 1, Math.floor(v * BINS))]++;
      return h;
    };
    const roc = range(101).map((i) => {
      const t = i / 100;
      return [neg.filter((v) => v >= t).length / neg.length, pos.filter((v) => v >= t).length / pos.length] as [number, number];
    });
    roc.push([0, 0]);
    let auc = 0;
    for (let i = 1; i < roc.length; i++) auc += (roc[i - 1][0] - roc[i][0]) * (roc[i - 1][1] + roc[i][1]) * 0.5;
    return { pos, neg, hPos: hist(pos), hNeg: hist(neg), roc, auc };
  }, [sick]);

  const TP = data.pos.filter((v) => v >= thr).length;
  const FN = sick - TP;
  const FP = data.neg.filter((v) => v >= thr).length;
  const TN = N - sick - FP;
  const acc = (TP + TN) / N;
  const prec = TP + FP ? TP / (TP + FP) : NaN;
  const rec = TP / sick;
  const f1 = Number.isFinite(prec) && prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  const base = (N - sick) / N;
  const fpr = FP / (N - sick);

  const maxPos = Math.max(...data.hPos);
  const maxNeg = Math.max(...data.hNeg);
  const yMax = norm ? 1.08 : Math.max(maxPos, maxNeg) * 1.1;
  const hv = (v: number, max: number) => (norm ? v / max : v);
  const pct = (v: number) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : '—');
  // accuracy needs a decimal: 99.8% must not round up to a perfect-looking 100%
  const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;

  let note: { text: string; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' | 'bad' };
  if (TP + FP === 0)
    note = {
      text: `아무도 "병"이라고 판정하지 않았어요. 정확도는 ${pct1(acc)}로 그럴듯하지만 재현율은 0% — 병자를 한 명도 못 잡는 "정확도 사기꾼"이에요! 🤡`,
      mood: 'wow',
      tone: 'bad'
    };
  else if (rec >= 0.9 && prec < 0.5)
    note = { text: `병자의 ${pct(rec)}를 잡아냈지만, "병"이라고 한 사람 중 진짜 병자는 ${pct(prec)}뿐이에요 (괜한 걱정 ${FP}명). 놓치면 큰일 나는 암 검진에 맞는 설정이에요.`, mood: 'think', tone: 'warn' };
  else if (prec >= 0.85 && rec < 0.7)
    note = { text: `"병"이라고 하면 ${pct(prec)} 맞지만, 병자 ${FN}명을 놓쳤어요. 정상 메일을 막으면 안 되는 스팸 필터처럼, 오탐이 비싼 일에 어울려요.`, mood: 'think', tone: 'warn' };
  else note = { text: '노란 선(임계값)을 끌어 보세요. 낮추면 재현율↑ 정밀도↓, 높이면 그 반대 — 둘은 줄다리기를 해요. 어느 실수가 더 비싼지가 답을 정해요.', mood: 'smile' };

  const setFromPointer = (px: number, s: { x: { invert: (v: number) => number } }) => setThr(clamp(Math.round(s.x.invert(px) * 100) / 100, 0, 1));

  return (
    <div>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <Plot
          xDomain={[0, 1]}
          yDomain={[0, yMax]}
          xLabel="모델이 매긴 점수 (병일 확률)"
          yLabel={norm ? '높이 맞춤' : '사람 수'}
          aspect={expanded ? 2.4 : 1.9}
          maxHeight={expanded ? 460 : 330}
          xFormat={(v) => v.toFixed(1)}
          yFormat={(v) => (norm ? '' : String(Math.round(v)))}
          ariaLabel="정상인과 환자의 점수 분포 히스토그램과 판정 임계값"
          onPointerDown={(e, s) => {
            if (e.pointerType === 'touch') return; // fingers scroll the page; the slider below handles touch
            dragging.current = true;
            (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
            const r = e.currentTarget.getBoundingClientRect();
            setFromPointer(e.clientX - r.left, s);
          }}
          onPointerMove={(e, s) => {
            if (!dragging.current) return;
            const r = e.currentTarget.getBoundingClientRect();
            setFromPointer(e.clientX - r.left, s);
          }}
          onPointerUp={() => (dragging.current = false)}
        >
          {({ x, y, inner }) => {
            const bw = (x(1 / BINS) - x(0)) * 0.86;
            return (
              <g style={{ cursor: 'ew-resize' }}>
                <rect x={x(thr)} y={inner.top} width={Math.max(0, x(1) - x(thr))} height={inner.height} fill={c.neg} opacity={0.07} />
                {data.hNeg.map((v, i) => (
                  <rect key={`n${i}`} x={x(i / BINS) + (x(1 / BINS) - x(0) - bw) / 2} y={y(hv(v, maxNeg))} width={bw} height={Math.max(0, y(0) - y(hv(v, maxNeg)))} rx={2} fill={c.series[3]} opacity={0.6} />
                ))}
                {data.hPos.map((v, i) => (
                  <rect
                    key={`p${i}`}
                    x={x(i / BINS) + (x(1 / BINS) - x(0) - bw) / 2 + bw * 0.18}
                    y={y(hv(v, maxPos))}
                    width={bw * 0.64}
                    height={Math.max(v > 0 ? 2 : 0, y(0) - y(hv(v, maxPos)))}
                    rx={2}
                    fill={c.series[2]}
                    opacity={0.9}
                  />
                ))}
                <line x1={x(thr)} x2={x(thr)} y1={inner.top} y2={inner.top + inner.height} strokeWidth={3.5} style={{ stroke: 'var(--lemon)' }} />
                <circle cx={x(thr)} cy={inner.top + 12} r={10} stroke={c.surface} strokeWidth={3} style={{ fill: 'var(--lemon)' }} />
                <text x={x(thr) + 14} y={inner.top + 34} style={{ fontSize: 11.5, fontWeight: 850, fill: c.neg, display: inner.left + inner.width - x(thr) < 112 ? 'none' : undefined }}>
                  “병”이라고 판정 →
                </text>
                <text x={x(thr) - 14} y={inner.top + 34} textAnchor="end" style={{ fontSize: 11.5, fontWeight: 850, fill: c.text, display: x(thr) - inner.left < 124 ? 'none' : undefined }}>
                  ← “정상”이라고 판정
                </text>
              </g>
            );
          }}
        </Plot>
        <Legend
          items={[
            { color: c.series[3], label: `정상인 ${N - sick}명` },
            { color: c.series[2], label: `병 있는 사람 ${sick}명` },
            { color: 'var(--lemon)', label: `임계값 ${thr.toFixed(2)} (노란 선을 끌어 보세요)` }
          ]}
        />
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 14, alignItems: 'start' }}>
          <div>
            <div className="vz-panel-title" style={{ marginBottom: 6 }}>
              혼동 행렬 (네 칸 성적표)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr', gap: 6, fontSize: 13 }}>
              <span />
              <span style={headCell}>판정: 병</span>
              <span style={headCell}>판정: 정상</span>
              <span style={sideCell}>실제 병</span>
              <Cell bg={withAlpha(c.pos, 0.2)} emoji="✅" name="진양성 TP" n={TP} hint="잡아냄" />
              <Cell bg={withAlpha(c.neg, 0.2)} emoji="😱" name="위음성 FN" n={FN} hint="놓침" />
              <span style={sideCell}>실제 정상</span>
              <Cell bg="color-mix(in srgb, var(--lemon) 22%, transparent)" emoji="😅" name="위양성 FP" n={FP} hint="괜한 걱정" />
              <Cell bg="var(--surface-3)" emoji="👍" name="진음성 TN" n={TN} hint="정상 확인" />
            </div>
          </div>
          <div style={{ display: 'grid', gap: 8 }}>
            <div className="vz-panel-title">지표</div>
            <Bar label="정확도" sub="전체 중 맞힌 비율" v={acc} color={c.series[0]} marker={base} markerLabel={`🤡 무조건 "정상" ${pct1(base)}`} />
            <Bar label="정밀도" sub='"병"이라고 한 사람 중 진짜' v={prec} color={c.series[1]} />
            <Bar label="재현율" sub="진짜 병자 중 잡아낸 비율" v={rec} color={c.series[2]} />
            <Bar label="F1" sub="정밀도·재현율의 조화평균" v={f1} color={c.series[4]} />
          </div>
          <div>
            <div className="vz-panel-title">ROC 곡선 (임계값을 쭉 훑은 궤적)</div>
            <Plot
              xDomain={[0, 1]}
              yDomain={[0, 1]}
              height={190}
              margin={{ top: 10, right: 12, bottom: 38, left: 44 }}
              xTicks={2}
              yTicks={2}
              xLabel="위양성 비율 (괜한 걱정)"
              yLabel="재현율"
              ariaLabel="ROC 곡선과 현재 임계값의 위치"
            >
              {({ x, y }) => (
                <g>
                  <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke={c.axis} strokeDasharray="4 4" />
                  <path d={`${linePath(data.roc.map(([a, b]) => [x(a), y(b)]))}L${x(1)},${y(0)}Z`} fill={c.series[0]} opacity={0.12} />
                  <path d={linePath(data.roc.map(([a, b]) => [x(a), y(b)]))} fill="none" stroke={c.series[0]} strokeWidth={3} />
                  <circle cx={x(fpr)} cy={y(rec)} r={7} stroke={c.surface} strokeWidth={2.5} style={{ fill: 'var(--lemon)' }} />
                  <text x={x(0.96)} y={y(0.08)} textAnchor="end" style={{ fontSize: 12.5, fontWeight: 850, fill: c.ink }}>
                    AUC = {data.auc.toFixed(2)}
                  </text>
                </g>
              )}
            </Plot>
            <div style={{ fontSize: 12, color: 'var(--text-2)' }}>AUC(면적)는 임계값과 상관없는 "판별력" 점수 — 1이면 완벽, 0.5면 동전 던지기예요.</div>
          </div>
        </div>
      </div>

      <Controls>
        <Slider label="판정 임계값" value={thr} min={0} max={1} step={0.01} onChange={setThr} format={(v) => v.toFixed(2)} accent="var(--lemon)" />
        <Segmented label="병 있는 사람의 비율" options={PREVALENCE} value={sick} onChange={setSick} />
        <Toggle label="막대 높이 맞춰 보기" checked={norm} onChange={setNorm} />
        <Button size="sm" onClick={() => setThr(0.2)}>
          🩺 암 검진 (놓치면 안 돼)
        </Button>
        <Button size="sm" onClick={() => setThr(0.8)}>
          📧 스팸 필터 (오탐은 안 돼)
        </Button>
        <Button size="sm" variant="ghost" onClick={() => setThr(1)}>
          🤡 무조건 정상
        </Button>
      </Controls>
    </div>
  );
}

const headCell = { fontWeight: 800, textAlign: 'center' as const, color: 'var(--text-2)', fontSize: 12.5 };
const sideCell = { fontWeight: 800, color: 'var(--text-2)', fontSize: 12.5, alignSelf: 'center' as const, whiteSpace: 'nowrap' as const };

function Cell({ bg, emoji, name, n, hint }: { bg: string; emoji: string; name: string; n: number; hint: string }) {
  return (
    <div style={{ background: bg, borderRadius: 12, padding: '8px 6px', textAlign: 'center', border: '1px solid var(--border)', display: 'grid', gap: 1 }}>
      <span style={{ fontSize: 18, lineHeight: 1.1 }}>{emoji}</span>
      <span style={{ fontSize: 20, fontWeight: 900, fontVariantNumeric: 'tabular-nums' }}>{n}</span>
      <span style={{ fontSize: 11.5, fontWeight: 750, color: 'var(--text-2)' }}>
        {hint} · {name}
      </span>
    </div>
  );
}

function Bar({ label, sub, v, color, marker, markerLabel }: { label: string; sub: string; v: number; color: string; marker?: number; markerLabel?: ReactNode }) {
  const ok = Number.isFinite(v);
  return (
    <div style={{ display: 'grid', gap: 3 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 13 }}>
        <span>
          <b>{label}</b> <span style={{ color: 'var(--text-3)', fontSize: 12 }}>{sub}</span>
        </span>
        <b style={{ fontVariantNumeric: 'tabular-nums' }}>{ok ? `${(v * 100).toFixed(1)}%` : '—'}</b>
      </div>
      <div style={{ position: 'relative', height: 14, borderRadius: 7, background: 'var(--surface-3)', border: '1px solid var(--border)' }}>
        <div style={{ width: `${ok ? v * 100 : 0}%`, height: '100%', borderRadius: 7, background: color, transition: 'width 0.2s' }} />
        {marker !== undefined && (
          <div style={{ position: 'absolute', left: `${marker * 100}%`, top: -3, bottom: -3, borderLeft: '2.5px dashed var(--text)' }} title={typeof markerLabel === 'string' ? markerLabel : undefined} />
        )}
      </div>
      {markerLabel && <div style={{ fontSize: 11.5, color: 'var(--text-2)', textAlign: 'right' }}>{markerLabel} (점선)</div>}
    </div>
  );
}
