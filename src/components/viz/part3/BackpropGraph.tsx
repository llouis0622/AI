import { useEffect, useMemo, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Slider, Stat } from '../core/ui';
import { useElementSize, useInterval, useRafLoop, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';

/* z = w·x + b, a = ReLU(z), L = (a - y)² — the exact example from chapter 10. */
type NodeId = 'w' | 'x' | 'mul' | 'b' | 'add' | 'relu' | 'y' | 'loss' | 'L';
type EdgeId = 'w-mul' | 'x-mul' | 'mul-add' | 'b-add' | 'add-relu' | 'relu-loss' | 'y-loss' | 'loss-L';

const EDGES: { id: EdgeId; from: NodeId; to: NodeId }[] = [
  { id: 'w-mul', from: 'w', to: 'mul' },
  { id: 'x-mul', from: 'x', to: 'mul' },
  { id: 'mul-add', from: 'mul', to: 'add' },
  { id: 'b-add', from: 'b', to: 'add' },
  { id: 'add-relu', from: 'add', to: 'relu' },
  { id: 'relu-loss', from: 'relu', to: 'loss' },
  { id: 'y-loss', from: 'y', to: 'loss' },
  { id: 'loss-L', from: 'loss', to: 'L' }
];

const POS_H: Record<NodeId, [number, number]> = {
  w: [55, 70],
  x: [55, 195],
  mul: [190, 132],
  b: [190, 272],
  add: [318, 178],
  relu: [440, 178],
  y: [440, 292],
  loss: [562, 178],
  L: [650, 178]
};
const POS_V: Record<NodeId, [number, number]> = {
  w: [95, 48],
  x: [235, 48],
  mul: [165, 150],
  b: [300, 222],
  add: [165, 268],
  relu: [165, 382],
  y: [300, 462],
  loss: [165, 500],
  L: [165, 606]
};

/** Which edges carry a value forward / a gradient backward after each step. */
const STEPS = [
  { kind: 'start', fwd: [] as EdgeId[], bwd: [] as EdgeId[], active: ['w', 'x', 'b', 'y'] as NodeId[] },
  { kind: 'f', fwd: ['w-mul', 'x-mul'], bwd: [], active: ['mul'] },
  { kind: 'f', fwd: ['mul-add', 'b-add'], bwd: [], active: ['add'] },
  { kind: 'f', fwd: ['add-relu'], bwd: [], active: ['relu'] },
  { kind: 'f', fwd: ['relu-loss', 'y-loss', 'loss-L'], bwd: [], active: ['loss', 'L'] },
  { kind: 'b', fwd: [], bwd: ['loss-L'], active: ['L'] },
  { kind: 'b', fwd: [], bwd: ['relu-loss'], active: ['loss'] },
  { kind: 'b', fwd: [], bwd: ['add-relu'], active: ['relu'] },
  { kind: 'b', fwd: [], bwd: ['mul-add', 'b-add'], active: ['add'] },
  { kind: 'b', fwd: [], bwd: ['w-mul', 'x-mul'], active: ['mul', 'w'] }
] as const;
const LAST = STEPS.length - 1;

function compute(x: number, w: number, b: number, y: number) {
  const z1 = w * x;
  const z = z1 + b;
  const a = Math.max(0, z);
  const L = (a - y) ** 2;
  const dL = 1;
  const da = 2 * (a - y);
  const reluLocal = z > 0 ? 1 : 0;
  const dz = da * reluLocal;
  const dz1 = dz;
  const db = dz;
  const dw = dz1 * x;
  const dx = dz1 * w;
  return { z1, z, a, L, dL, da, reluLocal, dz, dz1, db, dw, dx };
}

const n = (v: number) => {
  const r = Math.round(v * 1000) / 1000;
  return (Object.is(r, -0) ? 0 : r).toString();
};

export default function BackpropGraph({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [x, setX] = useState(2);
  const [w, setW] = useState(3);
  const [b, setB] = useState(1);
  const [y, setY] = useState(10);
  const [step, setStep] = useState(0);
  const [auto, setAuto] = useState(false);
  const [pulse, setPulse] = useState(1);
  const [extra, setExtra] = useState<null | { kind: 'check'; L0: number; L1: number; dw: number } | { kind: 'learn'; L0: number; L1: number }>(null);
  const [box, { width }] = useElementSize<HTMLDivElement>();
  const vertical = width > 0 && width < 560;
  const POS = vertical ? POS_V : POS_H;
  const v = useMemo(() => compute(x, w, b, y), [x, w, b, y]);

  useEffect(() => setPulse(reduce ? 1 : 0), [step, reduce]);
  useRafLoop((dt) => setPulse((p) => Math.min(1, p + dt / 0.7)), pulse < 1);
  useInterval(
    () => {
      if (step >= LAST) setAuto(false);
      else setStep((s) => s + 1);
    },
    auto && pulse >= 1 ? 900 : null
  );

  const go = (s: number) => {
    setExtra(null);
    setStep(Math.max(0, Math.min(LAST, s)));
  };

  const fwdDone = (e: EdgeId) => STEPS.slice(0, step + 1).some((s) => (s.fwd as readonly EdgeId[]).includes(e));
  const bwdDone = (e: EdgeId) => STEPS.slice(0, step + 1).some((s) => (s.bwd as readonly EdgeId[]).includes(e));
  const nodeDone = (id: NodeId) => STEPS.slice(0, step + 1).some((s) => (s.active as readonly NodeId[]).includes(id));
  const cur = STEPS[step];
  const blocked = v.reluLocal === 0;

  const fwdVal: Record<EdgeId, string> = {
    'w-mul': `w=${n(w)}`,
    'x-mul': `x=${n(x)}`,
    'mul-add': `z₁=${n(v.z1)}`,
    'b-add': `b=${n(b)}`,
    'add-relu': `z=${n(v.z)}`,
    'relu-loss': `a=${n(v.a)}`,
    'y-loss': `y=${n(y)}`,
    'loss-L': `L=${n(v.L)}`
  };
  const bwdVal: Record<EdgeId, string> = {
    'loss-L': `${n(v.dL)}`,
    'relu-loss': `${n(v.da)}`,
    'add-relu': `${n(v.dz)}`,
    'mul-add': `${n(v.dz1)}`,
    'b-add': `${n(v.db)}`,
    'w-mul': `${n(v.dw)}`,
    'x-mul': `${n(v.dx)}`,
    'y-loss': ''
  };

  const explain: ReactNode[] = [
    <>입력 <b>x={n(x)}</b>, 정답 <b>y={n(y)}</b>, 지금의 파라미터 <b>w={n(w)}, b={n(b)}</b>. "다음 →"을 눌러 계산을 왼쪽부터 굴려 봐요.</>,
    <>곱셈 노드: <b>z₁ = w × x = {n(w)} × {n(x)} = {n(v.z1)}</b>. 들어온 값(w, x)을 <b>기억</b>해 둬요 — 곧 쓸 거예요.</>,
    <>덧셈 노드: <b>z = z₁ + b = {n(v.z1)} + {n(b)} = {n(v.z)}</b>.</>,
    blocked ? (
      <>ReLU 노드: z = {n(v.z)}는 음수라 <b>0으로 막혀요</b> → a = 0.</>
    ) : (
      <>ReLU 노드: z = {n(v.z)}는 양수라 <b>그대로 통과</b> → a = {n(v.a)}.</>
    ),
    <>손실 노드: <b>L = (a - y)² = ({n(v.a)} - {n(y)})² = {n(v.L)}</b>. 순전파 끝! 이제 오른쪽에서 왼쪽으로 거슬러 가요.</>,
    <>역전파 시작: L의 L 자신에 대한 민감도는 당연히 <b>1</b>.</>,
    <>(a-y)²의 국소 민감도는 2(a-y) = <b>{n(v.da)}</b>. 받은 1 × {n(v.da)} = <b>{n(v.da)}</b>을 a로 넘겨요. "a를 1 키우면 손실이 {n(Math.abs(v.da))}만큼 {v.da < 0 ? '줄어요' : '늘어요'}."</>,
    blocked ? (
      <>ReLU가 <b>차단 중</b>이라 국소 민감도 0 → {n(v.da)} × 0 = <b>0</b>. 민감도가 여기서 끊겨서 앞쪽(w, b)은 아무것도 못 배워요!</>
    ) : (
      <>ReLU는 통과 중이라 국소 민감도 1 → {n(v.da)} × 1 = <b>{n(v.dz)}</b>.</>
    ),
    <>덧셈 노드는 받은 민감도를 <b>그대로 나눠 줘요</b> → z₁에도 {n(v.dz)}, b에도 <b>{n(v.db)}</b>.</>,
    <>곱셈 노드는 <b>"반대편 입력값"</b>을 곱해요 → w 쪽: {n(v.dz1)} × x({n(x)}) = <b>{n(v.dw)}</b>, x 쪽: {n(v.dz1)} × w({n(w)}) = {n(v.dx)}. 끝! 모든 기울기를 한 번에 얻었어요.</>
  ];

  const pos = (id: NodeId) => POS[id];
  const vb = vertical ? '0 0 360 650' : '0 0 700 350';
  const fwdColor = c.series[0];
  const bwdColor = c.series[2];

  const nodeEl = (id: NodeId) => {
    const [cx, cy] = pos(id);
    const isParam = id === 'w' || id === 'b';
    const isData = id === 'x' || id === 'y';
    const isOp = id === 'mul' || id === 'add' || id === 'relu' || id === 'loss';
    const on = nodeDone(id);
    const now = (cur.active as readonly NodeId[]).includes(id) && step > 0;
    const label: Record<NodeId, string> = { w: 'w', x: 'x', b: 'b', y: 'y', mul: '×', add: '+', relu: 'ReLU', loss: '(a-y)²', L: 'L' };
    const sub: Partial<Record<NodeId, string>> = {
      w: `= ${n(w)}`,
      x: `= ${n(x)}`,
      b: `= ${n(b)}`,
      y: `= ${n(y)} (정답)`,
      L: step >= 4 ? `= ${n(v.L)}` : ''
    };
    const grad: Partial<Record<NodeId, string>> = {
      w: step >= 9 ? `∂L/∂w = ${n(v.dw)}` : '',
      b: step >= 8 ? `∂L/∂b = ${n(v.db)}` : ''
    };
    const r = isOp ? (id === 'loss' ? 32 : id === 'relu' ? 30 : 24) : 26;
    const stroke = now ? (cur.kind === 'b' ? bwdColor : fwdColor) : isParam ? c.series[1] : c.border;
    const fill = isParam ? withAlpha(c.series[1], 0.14) : isData ? 'var(--surface-3)' : id === 'L' ? withAlpha(bwdColor, 0.16) : 'var(--surface-solid)';
    const blockedRelu = id === 'relu' && blocked && step >= 3;
    return (
      <g key={id}>
        {now && <circle cx={cx} cy={cy} r={r + 9} fill={withAlpha(stroke, 0.18)} />}
        {isParam ? (
          <rect x={cx - r} y={cy - r} width={r * 2} height={r * 2} rx={12} fill={fill} stroke={stroke} strokeWidth={now ? 3.5 : 2.5} />
        ) : (
          <circle cx={cx} cy={cy} r={r} fill={blockedRelu ? withAlpha(c.neg, 0.18) : fill} stroke={blockedRelu ? c.neg : stroke} strokeWidth={now ? 3.5 : 2.2} />
        )}
        <text x={cx} y={cy + (id === 'loss' || id === 'relu' ? 4.5 : 6)} textAnchor="middle" style={{ fontSize: id === 'loss' ? 12.5 : id === 'relu' ? 13 : 17, fontWeight: 850, fill: on || isParam || isData ? 'var(--text)' : 'var(--text-3)' }}>
          {label[id]}
        </text>
        {sub[id] && (
          <text x={cx} y={cy + r + 16} textAnchor="middle" style={{ fontSize: 12, fontWeight: 750, fill: 'var(--text-2)' }}>
            {sub[id]}
          </text>
        )}
        {grad[id] && (
          <text x={cx} y={cy - r - 9} textAnchor="middle" style={{ fontSize: 12.5, fontWeight: 850, fill: bwdColor }}>
            {grad[id]}
          </text>
        )}
        {blockedRelu && (
          <text x={cx} y={cy - r - 8} textAnchor="middle" style={{ fontSize: 12, fontWeight: 850, fill: c.neg }}>
            🚫 차단
          </text>
        )}
      </g>
    );
  };

  const edgeEl = (e: (typeof EDGES)[number]) => {
    const [x1, y1] = pos(e.from);
    const [x2, y2] = pos(e.to);
    const f = fwdDone(e.id);
    const g = bwdDone(e.id) && bwdVal[e.id] !== '';
    const nowF = (cur.fwd as readonly EdgeId[]).includes(e.id);
    const nowB = (cur.bwd as readonly EdgeId[]).includes(e.id);
    const ang = Math.atan2(y2 - y1, x2 - x1);
    const nx = -Math.sin(ang);
    const ny = Math.cos(ang);
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const off = 15;
    // moving pulse: forward goes from → to, backward to → from
    const tP = nowB ? 1 - pulse : pulse;
    const px = x1 + (x2 - x1) * tP;
    const py = y1 + (y2 - y1) * tP;
    return (
      <g key={e.id}>
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={g ? bwdColor : f ? fwdColor : c.border} strokeWidth={g || f ? 3 : 2} strokeOpacity={g || f ? 0.85 : 1} />
        {f && (
          <text x={mx + nx * off} y={my + ny * off + 4} textAnchor="middle" style={{ fontSize: 12, fontWeight: 800, fill: fwdColor, paintOrder: 'stroke', stroke: 'var(--viz-bg)', strokeWidth: 4 }}>
            {fwdVal[e.id]}
          </text>
        )}
        {g && (
          <text x={mx - nx * off} y={my - ny * off + 4} textAnchor="middle" style={{ fontSize: 12.5, fontWeight: 850, fill: bwdColor, paintOrder: 'stroke', stroke: 'var(--viz-bg)', strokeWidth: 4 }}>
            ◀ {bwdVal[e.id]}
          </text>
        )}
        {(nowF || nowB) && pulse < 1 && <circle cx={px} cy={py} r={7} fill={nowB ? bwdColor : fwdColor} />}
      </g>
    );
  };

  const doCheck = () => {
    const L1 = compute(x, w + 0.001, b, y).L;
    setExtra({ kind: 'check', L0: v.L, L1, dw: v.dw });
  };
  const doLearn = () => {
    const lr = 0.01;
    const L0 = v.L;
    const nw = Math.round((w - lr * v.dw) * 1000) / 1000;
    const nb = Math.round((b - lr * v.db) * 1000) / 1000;
    setW(nw);
    setB(nb);
    setExtra({ kind: 'learn', L0, L1: compute(x, nw, nb, y).L });
    setStep(4);
  };

  return (
    <div ref={box}>
      <div className="vz-pad" style={{ paddingBottom: 4 }}>
        <svg className="vz-svg" viewBox={vb} role="img" aria-label="계산 그래프: 순전파와 역전파" style={{ maxHeight: expanded ? '62vh' : vertical ? 560 : 360, touchAction: 'pan-y' }}>
          {EDGES.map(edgeEl)}
          {(Object.keys(POS) as NodeId[]).map(nodeEl)}
        </svg>
      </div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12, paddingTop: 0 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 12.5, fontWeight: 750, color: 'var(--text-2)' }}>
          <span style={{ color: fwdColor }}>━ 순전파: 값 (왼쪽→오른쪽)</span>
          <span style={{ color: bwdColor }}>━ 역전파: 민감도 (오른쪽→왼쪽)</span>
          <span style={{ color: c.series[1] }}>■ 학습되는 파라미터</span>
        </div>
        <div className="vz-note" role="status" style={{ display: 'block', fontSize: 14.5 }}>
          <strong style={{ display: 'block', marginBottom: 2, fontSize: 13, color: cur.kind === 'b' ? bwdColor : fwdColor }}>
            {step === 0 ? '준비' : cur.kind === 'f' ? `순전파 ${step}/4` : `역전파 ${step - 4}/5`}
          </strong>
          {explain[step]}
        </div>
        {step === LAST && (
          <div className="vz-stats">
            <Stat label="∂L/∂w" value={n(v.dw)} tone="brand" sub="w를 키우면 손실이 이만큼 변해요" />
            <Stat label="∂L/∂b" value={n(v.db)} tone="brand" />
            <Stat label="걸린 계산" value="순전파 1번 + 역전파 1번" />
          </div>
        )}
        {extra?.kind === 'check' && (
          <Note mood="wow" tone="good">
            w를 0.001 키워 봤어요: 손실 {n(extra.L0)} → {n(extra.L1)}. 변화량 {n(extra.L1 - extra.L0)} ≈ 기울기 {n(extra.dw)} × 0.001 = {n(extra.dw * 0.001)}. 역전파가 정확했어요! ✅
          </Note>
        )}
        {extra?.kind === 'learn' && (
          <Note mood="cheer" tone="good">
            기울기 반대 방향으로 한 걸음(학습률 0.01)! 손실이 {n(extra.L0)} → <b>{n(extra.L1)}</b>로 {extra.L1 < extra.L0 ? '줄었어요 🎉' : '그대로예요 (ReLU가 막혀 기울기가 0이에요)'}. 역전파를 다시 굴려 보세요.
          </Note>
        )}
      </div>
      <Controls>
        <div className="vz-playbar">
          <Button onClick={() => go(step - 1)} disabled={step === 0}>
            ← 이전
          </Button>
          <Button variant="primary" onClick={() => go(step + 1)} disabled={step === LAST}>
            {step < 4 ? '순전파 →' : '역전파 ←'}
          </Button>
          <Button variant="ghost" onClick={() => { setExtra(null); setAuto((a) => !a); if (step === LAST) setStep(0); }}>
            {auto ? '❚❚ 멈춤' : '▶ 자동 재생'}
          </Button>
          <Button variant="ghost" onClick={() => go(0)}>
            ↺ 처음으로
          </Button>
          {step === LAST && (
            <>
              <Button onClick={doCheck}>🔍 검산하기</Button>
              <Button onClick={doLearn}>👣 한 걸음 학습</Button>
            </>
          )}
        </div>
        <Slider label="입력 x" value={x} min={-3} max={4} step={0.5} onChange={(val) => { setExtra(null); setX(val); }} format={(val) => n(val)} />
        <Slider label="w (가중치)" value={w} min={-3} max={5} step={0.5} onChange={(val) => { setExtra(null); setW(val); }} format={(val) => n(val)} accent="var(--mint)" />
        <Slider label="b (편향)" value={b} min={-4} max={4} step={0.5} onChange={(val) => { setExtra(null); setB(val); }} format={(val) => n(val)} accent="var(--mint)" />
        <Slider label="y (정답)" value={y} min={0} max={12} step={1} onChange={(val) => { setExtra(null); setY(val); }} format={(val) => n(val)} accent="var(--peach)" />
      </Controls>
    </div>
  );
}
