import { useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Slider, Stat } from '../core/ui';
import { useElementSize, useInterval, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, mixHex } from '../core/colors';
import type { AlgoMood } from '@/components/mascot/Algo';
import { useWide } from './_layout';

type Part = 'in' | 'ln1' | 'att' | 'add1' | 'ln2' | 'ffn' | 'add2' | 'out';

const TOKENS = ['나는', '강가', '은행에', '앉아'];
const XS = [130, 197, 264, 331];
const Y: Record<Part, number> = { in: 52, ln1: 128, att: 212, add1: 292, ln2: 352, ffn: 436, add2: 516, out: 584 };
const ORDER: Part[] = ['in', 'ln1', 'att', 'add1', 'ln2', 'ffn', 'add2', 'out'];

/** who-looks-at-whom inside the attention box (rows sum to 1) */
const ATT = [
  [0.55, 0.1, 0.15, 0.2],
  [0.1, 0.5, 0.3, 0.1],
  [0.06, 0.52, 0.22, 0.2],
  [0.4, 0.08, 0.27, 0.25]
];

const INFO: Record<Part, { emoji: string; title: string; text: ReactNode }> = {
  in: { emoji: '📥', title: '입력 벡터들', text: <>문장의 토큰마다 벡터(숫자 묶음)가 하나씩 들어와요. 16장의 <strong>임베딩</strong>에 17장의 <strong>위치 정보</strong>를 더한 것이에요.</> },
  ln1: { emoji: '📏', title: '층 정규화', text: <>숫자들의 크기를 <strong>고르게</strong> 맞춰요. 너무 크거나 작은 값 때문에 학습이 흔들리지 않게 하는 정리 정돈이에요 (11장).</> },
  att: { emoji: '💬', title: '멀티헤드 어텐션 — 섞는 단계', text: <>토큰끼리 서로 바라보며 정보를 주고받아요. 블록 안에서 토큰 <strong>사이</strong>의 대화는 <strong>오직 여기서만</strong> 일어나요 (17장).</> },
  add1: { emoji: '➕', title: '잔차 연결', text: <>원래 값에 <strong>고친 만큼만</strong> 더해요. 이 지름길 덕분에 기울기가 막힘없이 흘러서 수십 층을 쌓아도 학습이 돼요 (11장).</> },
  ln2: { emoji: '📏', title: '층 정규화 (한 번 더)', text: <>피드포워드에 들어가기 전에 다시 크기를 맞춰요.</> },
  ffn: { emoji: '🧠', title: '피드포워드 망 — 소화하는 단계', text: <>토큰마다 <strong>각자 자기 자리에서</strong> 모아 온 정보를 가공해요. 서로 대화하지 않아요. 파라미터의 약 <strong>3분의 2</strong>가 여기 있고, 배운 지식이 많이 저장돼요.</> },
  add2: { emoji: '➕', title: '잔차 연결 (한 번 더)', text: <>피드포워드가 만든 수정분을 다시 원래 값에 더해요.</> },
  out: { emoji: '📤', title: '출력 벡터들', text: <>들어올 때와 <strong>같은 개수</strong>의 벡터가 나와요. 하지만 이제 문맥이 스며든 벡터예요! 이걸 <strong>다음 블록에 또</strong> 넣어요.</> }
};

/** One Transformer block: watch four token vectors flow through it, step by step. */
export default function TransformerBlock2D({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [phase, setPhase] = useState(0);
  const [auto, setAuto] = useState(false);
  const [picked, setPicked] = useState<Part | null>(null);
  const [blocks, setBlocks] = useState(12);
  const [box, { width }] = useElementSize<HTMLDivElement>();
  const wide = useWide(width, 700);

  useInterval(
    () => {
      if (phase >= ORDER.length - 1) setAuto(false);
      else setPhase((p) => p + 1);
    },
    auto ? 1400 : null
  );

  const dark = c.theme === 'dark';
  const base = (dark ? ['#3db8ff', '#ff7a59', '#2ef2c0', '#ff4fd8'] : ['#4f8ff0', '#ff8a65', '#2fbf98', '#c86bd8']);
  // token vectors as 4 color cells each, evolving through the block
  const cellsAt = (p: number) =>
    TOKENS.map((_, i) => {
      const orig = [0, 1, 2, 3].map((k) => mixHex(base[i], dark ? '#ffffff' : '#ffffff', k * 0.14));
      if (p < 2) return orig;
      const mixed = orig.map((_, k) => ATT[i].reduce((acc, w, j) => mixHex(acc, mixHex(base[j], '#ffffff', k * 0.14), w / (ATT[i].slice(0, j + 1).reduce((a, b) => a + b, 0) || 1)), orig[k]));
      const afterAdd = p >= 3 ? mixed.map((m, k) => mixHex(m, orig[k], 0.45)) : mixed;
      if (p < 5) return afterAdd;
      const processed = afterAdd.map((m, k) => mixHex(m, dark ? '#fff7a8' : '#ffd23f', 0.18 + 0.12 * ((k + i) % 3)));
      return p >= 6 ? processed.map((m, k) => mixHex(m, afterAdd[k], 0.35)) : processed;
    });
  const cells = cellsAt(phase);
  const cur = ORDER[phase];
  const show = picked ?? cur;
  const active = (part: Part) => cur === part || picked === part;
  const params = blocks * 12 * 768 * 768;

  const glyph = (i: number) => (
    <g
      key={i}
      style={{
        transform: `translate(${XS[i]}px, ${Y[cur] - (cur === 'att' || cur === 'ffn' ? 0 : 0)}px)`,
        transition: reduce ? 'none' : 'transform 0.9s cubic-bezier(.22,1,.36,1)'
      }}
    >
      {cells[i].map((col, k) => (
        <rect key={k} x={-9} y={-20 + k * 10} width={18} height={9} rx={2.5} fill={col} stroke={dark ? 'rgba(0,0,0,.35)' : 'rgba(42,37,64,.25)'} strokeWidth={0.8} style={{ transition: 'fill .6s' }} />
      ))}
    </g>
  );

  const boxProps = (part: Part) => ({
    onClick: () => setPicked((p) => (p === part ? null : part)),
    style: { cursor: 'pointer' },
    role: 'button',
    'aria-label': INFO[part].title
  });

  const boxFill = (part: Part, tint: string) => (active(part) ? `color-mix(in srgb, ${tint} 26%, var(--surface-solid))` : 'var(--surface-solid)');
  const boxStroke = (part: Part, tint: string) => (active(part) ? tint : 'var(--border-strong)');
  const lineColor = dark ? '#b4c3ff' : '#5b5570';

  const diagram = (
    <svg className="vz-svg" viewBox="0 0 440 630" style={{ maxWidth: 470, margin: '0 auto', touchAction: 'pan-y' }} role="img" aria-label="Transformer 블록 하나의 구조와 데이터 흐름">
      {/* main spine */}
      <line x1={225} y1={70} x2={225} y2={560} stroke={lineColor} strokeOpacity={0.3} strokeWidth={2} />
      {/* residual paths */}
      <path d={`M100 ${Y.in} H34 V${Y.add1} H207`} fill="none" stroke={c.series[1]} strokeWidth={3} strokeDasharray={active('add1') ? '8 6' : undefined} strokeOpacity={active('add1') ? 1 : 0.5} className={active('add1') ? 'tb-flow' : undefined} />
      <path d={`M243 ${Y.add1} H412 V${Y.add2} H243`} fill="none" stroke={c.series[1]} strokeWidth={3} strokeDasharray={active('add2') ? '8 6' : undefined} strokeOpacity={active('add2') ? 1 : 0.5} className={active('add2') ? 'tb-flow' : undefined} />
      <text x={20} y={(Y.in + Y.add1) / 2} transform={`rotate(-90 20 ${(Y.in + Y.add1) / 2})`} textAnchor="middle" style={{ fontSize: 11.5, fontWeight: 800, fill: c.series[1] }}>
        지름길 (잔차)
      </text>
      <text x={428} y={(Y.add1 + Y.add2) / 2} transform={`rotate(90 428 ${(Y.add1 + Y.add2) / 2})`} textAnchor="middle" style={{ fontSize: 11.5, fontWeight: 800, fill: c.series[1] }}>
        지름길 (잔차)
      </text>

      {/* input / output labels */}
      <g {...boxProps('in')}>
        <rect x={98} y={Y.in - 34} width={256} height={48} rx={14} fill={boxFill('in', c.series[3])} stroke={boxStroke('in', c.series[3])} strokeDasharray="5 4" />
        <text x={225} y={Y.in - 40} textAnchor="middle" style={{ fontSize: 12.5, fontWeight: 800, fill: 'var(--text-2)' }}>
          입력 벡터들 (토큰당 하나)
        </text>
      </g>
      {TOKENS.map((t, i) => (
        <text key={t} x={XS[i]} y={Y.in + 30} textAnchor="middle" style={{ fontSize: 11, fontWeight: 800, fill: 'var(--text-2)' }}>
          {t}
        </text>
      ))}

      <g {...boxProps('ln1')}>
        <rect x={98} y={Y.ln1 - 18} width={256} height={36} rx={12} fill={boxFill('ln1', c.series[4])} stroke={boxStroke('ln1', c.series[4])} strokeWidth={2} />
        <text x={225} y={Y.ln1 + 5} textAnchor="middle" style={{ fontSize: 13.5, fontWeight: 850, fill: 'var(--text)' }}>
          📏 층 정규화
        </text>
      </g>

      <g {...boxProps('att')}>
        <rect x={98} y={Y.att - 40} width={256} height={80} rx={14} fill={boxFill('att', c.series[0])} stroke={boxStroke('att', c.series[0])} strokeWidth={2.5} />
        {XS.flatMap((x1, i) =>
          XS.map((x2, j) =>
            i < j ? (
              <path
                key={`${i}-${j}`}
                d={`M${x1} ${Y.att + 22} Q${(x1 + x2) / 2} ${Y.att - 14 - (j - i) * 6} ${x2} ${Y.att + 22}`}
                fill="none"
                stroke={c.series[0]}
                strokeWidth={1 + 5 * (ATT[i][j] + ATT[j][i])}
                strokeOpacity={active('att') ? 0.85 : 0.25}
                className={active('att') ? 'tb-pulse' : undefined}
              />
            ) : null
          )
        )}
        <text x={225} y={Y.att - 22} textAnchor="middle" style={{ fontSize: 13.5, fontWeight: 850, fill: 'var(--text)' }}>
          💬 멀티헤드 어텐션
        </text>
        <text x={225} y={Y.att + 36} textAnchor="middle" style={{ fontSize: 11, fontWeight: 700, fill: 'var(--text-2)' }}>
          토큰들 사이의 정보 교환
        </text>
      </g>

      <g {...boxProps('add1')}>
        <circle cx={225} cy={Y.add1} r={17} fill={boxFill('add1', c.series[1])} stroke={boxStroke('add1', c.series[1])} strokeWidth={2.5} />
        <text x={225} y={Y.add1 + 7} textAnchor="middle" style={{ fontSize: 20, fontWeight: 900, fill: 'var(--text)' }}>
          +
        </text>
      </g>

      <g {...boxProps('ln2')}>
        <rect x={98} y={Y.ln2 - 18} width={256} height={36} rx={12} fill={boxFill('ln2', c.series[4])} stroke={boxStroke('ln2', c.series[4])} strokeWidth={2} />
        <text x={225} y={Y.ln2 + 5} textAnchor="middle" style={{ fontSize: 13.5, fontWeight: 850, fill: 'var(--text)' }}>
          📏 층 정규화
        </text>
      </g>

      <g {...boxProps('ffn')}>
        <rect x={98} y={Y.ffn - 40} width={256} height={80} rx={14} fill={boxFill('ffn', c.series[2])} stroke={boxStroke('ffn', c.series[2])} strokeWidth={2.5} />
        {XS.map((x) => (
          <rect key={x} x={x - 22} y={Y.ffn - 26} width={44} height={52} rx={9} fill="none" stroke={c.series[2]} strokeWidth={1.5} strokeDasharray={active('ffn') ? undefined : '4 3'} strokeOpacity={0.9} />
        ))}
        <text x={225} y={Y.ffn - 48} textAnchor="middle" style={{ fontSize: 13.5, fontWeight: 850, fill: 'var(--text)' }}>
          🧠 피드포워드 망
        </text>
        <text x={225} y={Y.ffn + 56} textAnchor="middle" style={{ fontSize: 11, fontWeight: 700, fill: 'var(--text-2)' }}>
          토큰 각자의 정보 가공 (칸막이!)
        </text>
      </g>

      <g {...boxProps('add2')}>
        <circle cx={225} cy={Y.add2} r={17} fill={boxFill('add2', c.series[1])} stroke={boxStroke('add2', c.series[1])} strokeWidth={2.5} />
        <text x={225} y={Y.add2 + 7} textAnchor="middle" style={{ fontSize: 20, fontWeight: 900, fill: 'var(--text)' }}>
          +
        </text>
      </g>

      <g {...boxProps('out')}>
        <rect x={98} y={Y.out - 30} width={256} height={48} rx={14} fill={boxFill('out', c.series[3])} stroke={boxStroke('out', c.series[3])} strokeDasharray="5 4" />
        <text x={225} y={Y.out + 36} textAnchor="middle" style={{ fontSize: 12.5, fontWeight: 800, fill: 'var(--text-2)' }}>
          출력 벡터들 (같은 개수) → 다음 블록으로
        </text>
      </g>

      {TOKENS.map((_, i) => glyph(i))}
    </svg>
  );

  const info = INFO[show];
  const step = phase + 1;
  const mood: AlgoMood = cur === 'out' ? 'cheer' : cur === 'att' ? 'wow' : 'happy';

  return (
    <div ref={box}>
      <style>{`
        @keyframes tb-dash { to { stroke-dashoffset: -28; } }
        .tb-flow { animation: tb-dash 0.9s linear infinite; }
        @keyframes tb-pulse { 0%,100% { stroke-opacity: .45; } 50% { stroke-opacity: 1; } }
        .tb-pulse { animation: tb-pulse 1.1s ease-in-out infinite; }
        @media (prefers-reduced-motion: reduce) { .tb-flow, .tb-pulse { animation: none; } }
      `}</style>
      <div style={{ display: 'grid', gridTemplateColumns: wide ? 'minmax(0, 1.1fr) minmax(0, 1fr)' : '1fr', alignItems: 'start', minHeight: expanded ? '62vh' : undefined }}>
        <div style={{ padding: '14px 10px 4px' }}>{diagram}</div>
        <div className="vz-pad" style={{ display: 'grid', gap: 12, alignContent: 'start' }}>
          <div className="vz-panel-title">
            단계 {step} / {ORDER.length} {picked ? '· 누른 부품 설명' : ''}
          </div>
          <div style={{ padding: '14px 16px', borderRadius: 16, background: 'var(--surface-solid)', border: '1px solid var(--border)' }}>
            <div style={{ fontSize: 26, lineHeight: 1 }}>{info.emoji}</div>
            <div style={{ fontWeight: 850, fontSize: 16.5, margin: '8px 0 4px', letterSpacing: '-0.02em' }}>{info.title}</div>
            <div style={{ fontSize: 14.5, lineHeight: 1.7, color: 'var(--text-2)' }}>{info.text}</div>
          </div>
          <Note mood={mood}>
            {cur === 'in' && '▶ 다음 단계를 눌러 네 토큰이 블록을 통과하는 모습을 따라가 보세요. 그림 속 부품을 눌러도 설명이 나와요.'}
            {cur === 'att' && '색이 서로 섞이는 걸 보세요 — 토큰끼리 정보를 나눈 거예요.'}
            {cur === 'ffn' && '이번엔 칸막이 안에서 각자 따로 변해요. 서로 섞이지 않아요!'}
            {(cur === 'add1' || cur === 'add2') && '옆쪽 지름길(잔차)로 원래 값이 와서 더해져요. 원래 색이 조금 남아 있죠?'}
            {(cur === 'ln1' || cur === 'ln2') && '크기를 고르게 다듬는 중이에요.'}
            {cur === 'out' && `한 블록 통과! 이 블록을 ${blocks}번 쌓으면 GPT의 몸통이 돼요. 새 부품은 하나도 없어요 — 배운 부품들의 조립이에요.`}
          </Note>
          <div className="vz-panel-title" style={{ marginTop: 4 }}>
            블록을 몇 층 쌓을까?
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 64, padding: '0 2px' }} aria-hidden="true">
            {Array.from({ length: 48 }, (_, k) => (
              <span
                key={k}
                style={{
                  flex: 1,
                  height: `${18 + (k % 4) * 3}px`,
                  borderRadius: 3,
                  background: k < blocks ? (k % 2 ? c.series[0] : c.series[2]) : 'var(--surface-3)',
                  opacity: k < blocks ? 1 : 0.6,
                  transition: 'background-color .25s'
                }}
              />
            ))}
          </div>
          <div className="vz-stats">
            <Stat label="블록 수" value={`${blocks}층`} />
            <Stat label="파라미터 (대략)" value={params >= 1e8 ? `${(params / 1e8).toFixed(1)}억` : `${Math.round(params / 1e4).toLocaleString()}만`} tone="brand" sub="벡터 768차원 기준" />
          </div>
        </div>
      </div>

      <Controls>
        <Button onClick={() => { setPicked(null); setPhase((p) => Math.max(0, p - 1)); }} disabled={phase === 0}>
          ← 이전
        </Button>
        <Button variant="primary" onClick={() => { setPicked(null); setPhase((p) => Math.min(ORDER.length - 1, p + 1)); }} disabled={phase === ORDER.length - 1}>
          다음 단계 ▶
        </Button>
        <Button onClick={() => { setPicked(null); if (phase >= ORDER.length - 1) setPhase(0); setAuto((a) => !a); }}>
          {auto ? '❚❚ 멈춤' : '▶ 자동 재생'}
        </Button>
        <Button variant="ghost" onClick={() => { setAuto(false); setPicked(null); setPhase(0); }}>
          ↺ 처음으로
        </Button>
        <Slider label="쌓는 블록 수" value={blocks} min={1} max={48} step={1} onChange={setBlocks} format={(v) => `${v}층`} />
      </Controls>
    </div>
  );
}
