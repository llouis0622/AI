import { useState } from 'react';
import type { VizProps } from '../Viz';
import { MiniChart } from '../core/plot';
import { Button, Controls, Note, PlayBar, Segmented, Stat } from '../core/ui';
import { useElementSize, useInterval, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors } from '../core/colors';
import { sigmoid } from '../core/math';
import { FieldCanvas, ClassLegend } from './_Field';
import { makeDataset, type Pt } from './_data';

type DS = 'blobs' | 'overlap' | 'xor';
interface Params {
  w1: number;
  w2: number;
  b: number;
}

/** Features are centred and stretched: u = (x - 0.5) * K, so weights stay in a friendly range. */
const K = 4;
const START: Params = { w1: -1.1, w2: 1.4, b: 0.5 }; // a deliberately wrong line to begin with
const L2 = 0.004; // tiny weight penalty so separable data still settles
const LR = 1.4;
const SEEDS: Record<DS, number> = { blobs: 5, overlap: 11, xor: 4 };

function lossAndGrad(p: Params, pts: Pt[]) {
  let L = 0;
  let g1 = 0;
  let g2 = 0;
  let gb = 0;
  for (const q of pts) {
    const u = (q.x - 0.5) * K;
    const v = (q.y - 0.5) * K;
    const pr = sigmoid(p.w1 * u + p.w2 * v + p.b);
    const e = pr - q.label;
    g1 += e * u;
    g2 += e * v;
    gb += e;
    L -= q.label ? Math.log(Math.max(pr, 1e-12)) : Math.log(Math.max(1 - pr, 1e-12));
  }
  const n = Math.max(1, pts.length);
  return { L: L / n, g: [g1 / n + L2 * p.w1, g2 / n + L2 * p.w2, gb / n] as [number, number, number] };
}

const prob = (p: Params, x: number, y: number) => sigmoid(p.w1 * (x - 0.5) * K + p.w2 * (y - 0.5) * K + p.b);

/** Segment of a·x + b·y + c = 0 inside the unit square, or null. */
function lineInSquare(a: number, b: number, c: number): [[number, number], [number, number]] | null {
  const out: [number, number][] = [];
  const push = (x: number, y: number) => {
    if (x >= -1e-9 && x <= 1 + 1e-9 && y >= -1e-9 && y <= 1 + 1e-9 && !out.some((q) => Math.hypot(q[0] - x, q[1] - y) < 1e-6)) out.push([x, y]);
  };
  if (Math.abs(b) > 1e-9) {
    push(0, -c / b);
    push(1, -(a + c) / b);
  }
  if (Math.abs(a) > 1e-9) {
    push(-c / a, 0);
    push(-(b + c) / a, 1);
  }
  return out.length >= 2 ? [out[0], out[1]] : null;
}

/** 2D logistic regression that trains live: the straight decision boundary finds its place. */
export default function DecisionBoundary({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [ds, setDs] = useState<DS>('blobs');
  const [points, setPoints] = useState<Pt[]>(() => makeDataset('blobs', 40, SEEDS.blobs));
  const [params, setParams] = useState<Params>(START);
  const [losses, setLosses] = useState<number[]>([]);
  const [steps, setSteps] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [addClass, setAddClass] = useState<0 | 1>(1);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [added, setAdded] = useState(0);
  const [box, { width }] = useElementSize<HTMLDivElement>();

  const { L, g } = lossAndGrad(params, points);
  const gnorm = Math.hypot(...g);
  const converged = steps > 0 && gnorm < 2.5e-3;
  const correct = points.filter((q) => (prob(params, q.x, q.y) >= 0.5 ? 1 : 0) === q.label).length;
  const acc = points.length ? correct / points.length : 0;

  const train = (n: number) => {
    let p = params;
    let last = L;
    for (let i = 0; i < n; i++) {
      const r = lossAndGrad(p, points);
      last = r.L;
      p = { w1: p.w1 - LR * r.g[0], w2: p.w2 - LR * r.g[1], b: p.b - LR * r.g[2] };
    }
    setParams(p);
    setSteps((s) => s + n);
    setLosses((ls) => [...ls, last].slice(-300));
  };

  useInterval(
    () => {
      if (converged || steps > 4000) setPlaying(false);
      else train(speed >= 3 ? 8 : speed >= 1 ? 3 : 1);
    },
    playing ? 50 : null
  );

  const reset = (next: DS = ds) => {
    setPlaying(false);
    setPoints(makeDataset(next, next === 'xor' ? 44 : 40, SEEDS[next]));
    setParams(START);
    setLosses([]);
    setSteps(0);
    setAdded(0);
  };

  const onPick = (x: number, y: number) => {
    setPoints((ps) => [...ps, { x, y, label: addClass }]);
    setAdded((a) => a + 1);
    if (!reduce) setPlaying(true);
  };

  const height = Math.round(expanded ? Math.min(620, Math.max(340, width * 0.56)) : Math.min(430, Math.max(270, width * 0.62)));

  // boundary z = 0  ⇔  a·x + b·y + c = 0 in unit coordinates
  const A = params.w1 * K;
  const B = params.w2 * K;
  const C = params.b - 0.5 * K * (params.w1 + params.w2);
  const seg = lineInSquare(A, B, C);
  const band = Math.log(9); // p = 0.1 / 0.9

  const isXor = ds === 'xor';
  let note: { text: string; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'bad' | 'warn' };
  if (steps === 0 && added === 0)
    note = { text: '지금 경계선은 엉터리예요. ▶ 시작을 누르면 교차 엔트로피(손실)를 줄이는 쪽으로 w와 b가 고쳐지면서, 직선이 스스로 자리를 찾아가요.', mood: 'smile' };
  else if (converged && isXor)
    note = { text: `학습이 멈췄는데 정확도가 ${Math.round(acc * 100)}%뿐이에요. XOR은 직선 하나로는 절대 못 갈라요 — 로지스틱 회귀의 경계는 언제나 직선이니까요. 9장의 신경망이 이 문제를 풀어요!`, mood: 'think', tone: 'warn' };
  else if (converged && acc === 1)
    note = { text: '완벽하게 갈랐어요! 🎉 빈 곳을 눌러 "반대편"에 점을 하나 심어 보세요. 경계가 어떻게 움직일까요?', mood: 'cheer', tone: 'good' };
  else if (converged)
    note = { text: `더 줄일 수 없는 곳에 도착했어요. 섞인 점 ${points.length - correct}개는 어떤 직선으로도 다 맞힐 수 없어서, 손실이 가장 작은 타협점을 고른 거예요.`, mood: 'happy', tone: 'good' };
  else if (playing)
    note = { text: '한 걸음마다 "틀린 만큼(p - 정답)"을 보고 w와 b를 조금씩 고쳐요. 배경색이 진할수록 모델이 확신하는 곳이에요.', mood: 'happy' };
  else note = { text: '빈 곳을 누르면 아래에서 고른 종류의 점이 생겨요. ▶ 시작으로 학습을 이어 가 보세요.', mood: 'smile' };

  const hp = hover ? prob(params, hover[0], hover[1]) : 0;

  return (
    <div ref={box}>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <FieldCanvas
          height={height}
          points={points}
          predict={(x, y) => prob(params, x, y)}
          boundary={false}
          res={56}
          wrong={(q) => steps > 0 && (prob(params, q.x, q.y) >= 0.5 ? 1 : 0) !== q.label}
          onPick={onPick}
          onHover={setHover}
          deps={[params.w1, params.w2, params.b, steps]}
          ariaLabel="두 종류의 점과, 로지스틱 회귀가 그은 직선 결정 경계"
          xLabel="크기 →"
          yLabel="↑ 울퉁불퉁한 정도"
          under={(ctx, w, h, col) => {
            const px = (q: [number, number]): [number, number] => [q[0] * w, (1 - q[1]) * h];
            ctx.save();
            // confidence band (10% / 90%)
            ctx.setLineDash([5, 5]);
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = col.theme === 'dark' ? 'rgba(235,240,255,0.45)' : 'rgba(42,37,64,0.35)';
            for (const s of [band, -band]) {
              const l = lineInSquare(A, B, C - s);
              if (!l) continue;
              ctx.beginPath();
              ctx.moveTo(...px(l[0]));
              ctx.lineTo(...px(l[1]));
              ctx.stroke();
            }
            ctx.setLineDash([]);
            if (seg) {
              ctx.lineWidth = 3.5;
              ctx.strokeStyle = col.theme === 'dark' ? '#f4f6ff' : '#2a2540';
              ctx.beginPath();
              ctx.moveTo(...px(seg[0]));
              ctx.lineTo(...px(seg[1]));
              ctx.stroke();
              // w arrow: the direction in which p(악성) grows
              const mid: [number, number] = [(seg[0][0] + seg[1][0]) / 2, (seg[0][1] + seg[1][1]) / 2];
              const [mx, my] = px(mid);
              let dx = A / w;
              let dy = -B / h;
              const n = Math.hypot(dx, dy) || 1;
              dx /= n;
              dy /= n;
              const len = Math.min(56, Math.max(30, w * 0.08));
              const tx = mx + dx * len;
              const ty = my + dy * len;
              const lemon = getComputedStyle(document.documentElement).getPropertyValue('--lemon').trim() || '#e7a90f';
              ctx.strokeStyle = lemon;
              ctx.fillStyle = lemon;
              ctx.lineWidth = 3.5;
              ctx.beginPath();
              ctx.moveTo(mx, my);
              ctx.lineTo(tx - dx * 8, ty - dy * 8);
              ctx.stroke();
              ctx.beginPath();
              ctx.moveTo(tx, ty);
              ctx.lineTo(tx - dx * 12 - dy * 7, ty - dy * 12 + dx * 7);
              ctx.lineTo(tx - dx * 12 + dy * 7, ty - dy * 12 - dx * 7);
              ctx.closePath();
              ctx.fill();
            }
            ctx.restore();
          }}
        >
          {hover && (
            <span
              style={{
                position: 'absolute',
                left: `${hover[0] * 100}%`,
                top: `${(1 - hover[1]) * 100}%`,
                transform: `translate(${hover[0] > 0.7 ? 'calc(-100% - 12px)' : '12px'}, -50%)`,
                padding: '3px 9px',
                borderRadius: 10,
                fontSize: 12.5,
                fontWeight: 800,
                background: 'var(--surface-solid)',
                border: '1px solid var(--border-strong)',
                boxShadow: 'var(--shadow-sm)',
                pointerEvents: 'none',
                whiteSpace: 'nowrap',
                color: hp >= 0.5 ? c.series[2] : c.series[3]
              }}
            >
              악성 확률 {Math.round(hp * 100)}%
            </span>
          )}
        </FieldCanvas>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 16px', justifyContent: 'space-between', alignItems: 'center', marginTop: 8 }}>
          <ClassLegend
            extra={
              <>
                <span className="vz-legend-item">
                  <span style={{ width: 18, height: 0, borderTop: `3px solid ${c.ink}`, display: 'inline-block' }} />
                  결정 경계 (확률 50%)
                </span>
                <span className="vz-legend-item">
                  <span style={{ width: 18, height: 0, borderTop: '3px solid var(--lemon)', display: 'inline-block' }} />
                  확률이 커지는 방향
                </span>
              </>
            }
          />
        </div>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <PlayBar
          playing={playing}
          onToggle={() => setPlaying((p) => !p)}
          onStep={() => train(5)}
          onReset={() => reset()}
          speed={speed}
          onSpeed={setSpeed}
        />
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: 12, alignItems: 'start' }}>
          <div style={{ display: 'grid', gap: 8 }}>
            <div className="vz-stats">
              <Stat label="학습 걸음" value={steps} />
              <Stat label="손실 (교차 엔트로피)" value={L.toFixed(3)} tone={converged ? 'good' : 'brand'} />
              <Stat label="정확도" value={`${Math.round(acc * 100)}%`} sub={`${correct} / ${points.length}`} tone={acc === 1 ? 'good' : isXor && converged ? 'warn' : undefined} />
            </div>
            <div style={{ fontSize: 12.5, color: 'var(--text-2)', fontFamily: 'var(--font-mono)', overflowWrap: 'anywhere' }}>
              점수 = {params.w1.toFixed(2)}×크기 {params.w2 >= 0 ? '+' : '-'} {Math.abs(params.w2).toFixed(2)}×울퉁불퉁 {params.b >= 0 ? '+' : '-'} {Math.abs(params.b).toFixed(2)}
            </div>
          </div>
          <div>
            <div className="vz-panel-title">학습할수록 손실이…</div>
            <MiniChart series={[losses]} colors={[c.series[0]]} height={110} xLabel="시간" yMin={0} />
          </div>
        </div>
      </div>

      <Controls>
        <Segmented
          label="데이터"
          options={[
            { value: 'blobs', label: '두 무리' },
            { value: 'overlap', label: '섞인 무리' },
            { value: 'xor', label: 'XOR 함정' }
          ]}
          value={ds}
          onChange={(v) => {
            setDs(v);
            reset(v);
          }}
        />
        <Segmented
          label="빈 곳을 누르면 생기는 점"
          options={[
            { value: 1, label: '▲ 악성' },
            { value: 0, label: '● 양성' }
          ]}
          value={addClass}
          onChange={(v) => setAddClass(v as 0 | 1)}
        />
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            setPoints((ps) => ps.slice(0, -1));
            setAdded((a) => a - 1);
          }}
          disabled={added <= 0}
        >
          ↶ 마지막 점 빼기
        </Button>
      </Controls>
    </div>
  );
}
