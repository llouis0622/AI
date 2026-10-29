import { useRef, useState } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Legend, Note, Toggle } from '../core/ui';
import { useElementSize, useInView, usePrefersReducedMotion, useRafLoop } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';

type Tool = 'vec' | 'prob' | 'diff';
type NodeId = 'data' | 'model' | 'loss' | 'grad' | 'update';

const NODES: { id: NodeId; emoji: string; title: string; sub: string; tools: Tool[]; where: string; text: string }[] = [
  { id: 'data', emoji: '📦', title: '데이터', sub: '벡터·행렬', tools: ['vec'], where: '이 장 1절', text: '세상의 모든 것을 숫자 묶음(벡터)으로 바꿔요. 집 한 채 = (평수, 방 개수, 역까지 거리, 층). 여러 채를 쌓으면 행렬이 돼요.' },
  { id: 'model', emoji: '🧮', title: '모델', sub: '행렬 곱 → 확률', tools: ['vec', 'prob'], where: '이 장 1절·3절', text: '행렬 곱(내적을 한꺼번에)으로 점수를 계산하고, 그 점수를 확률로 바꿔 답해요. 예: "고양이 85%, 개 12%".' },
  { id: 'loss', emoji: '📉', title: '손실', sub: '최대가능도로 채점', tools: ['prob'], where: '이 장 3절', text: '정답에 얼마나 높은 확률을 줬는지로 채점해요. 이 채점법은 아무렇게나 정한 게 아니라 "최대가능도" 원리에서 나와요.' },
  { id: 'grad', emoji: '🔍', title: '기울기', sub: '연쇄법칙으로 민감도', tools: ['diff'], where: '이 장 2절', text: '파라미터를 살짝 바꾸면 손실이 얼마나 변하는지(민감도)를 전부 계산해요. 층이 여러 겹이어도 연쇄법칙(곱셈)으로 척척!' },
  { id: 'update', emoji: '👣', title: '갱신', sub: '기울기 반대로 한 걸음', tools: ['diff'], where: '2장', text: '기울기의 반대 방향으로 학습률만큼 파라미터를 옮겨요. 그리고 다시 모델로 — 이렇게 한 바퀴 돌 때마다 조금씩 똑똑해져요.' }
];

const LOOP: NodeId[] = ['model', 'loss', 'grad', 'update'];

export default function LearningPipeline({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [box, { width }] = useElementSize<HTMLDivElement>();
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, '0px');
  const [playing, setPlaying] = useState(true);
  const [phase, setPhase] = useState(0); // 0..4, one unit per loop edge
  const [laps, setLaps] = useState(0);
  const [sel, setSel] = useState<NodeId>('model');
  const [tools, setTools] = useState(true);
  const narrow = width > 0 && width < 520;

  const phaseRef = useRef(0);
  const advance = (to: number) => {
    let np = to;
    if (np >= 4) {
      np -= 4;
      setLaps((l) => l + 1);
    }
    phaseRef.current = np;
    setPhase(np);
  };
  useRafLoop((dt) => advance(phaseRef.current + dt * 0.55), playing && inView && !reduce);

  const toolColor: Record<Tool, string> = { vec: c.series[0], prob: c.series[2], diff: c.series[1] };
  const pos: Record<NodeId, [number, number]> = narrow
    ? { data: [180, 44], model: [180, 140], loss: [180, 236], grad: [180, 332], update: [180, 428] }
    : { data: [86, 136], model: [250, 66], loss: [500, 66], grad: [500, 206], update: [250, 206] };
  const vb = narrow ? '0 0 360 480' : '0 0 640 270';
  const NW = narrow ? 200 : 168;
  const NH = 64;

  // packet position on the loop
  const seg = Math.floor(phase) % 4;
  const fr = phase - Math.floor(phase);
  const from = pos[LOOP[seg]];
  const to = pos[LOOP[(seg + 1) % 4]];
  let px: number;
  let py: number;
  if (narrow && seg === 3) {
    // update → model: along the cubic curve drawn on the right side
    const u = 1 - fr;
    px = u ** 3 * 280 + 3 * u * u * fr * 340 + 3 * u * fr * fr * 340 + fr ** 3 * 280;
    py = u ** 3 * 428 + 3 * u * u * fr * 428 + 3 * u * fr * fr * 140 + fr ** 3 * 140;
  } else {
    px = from[0] + (to[0] - from[0]) * fr;
    py = from[1] + (to[1] - from[1]) * fr;
  }
  const lit: NodeId | null = fr < 0.18 ? LOOP[seg] : fr > 0.82 ? LOOP[(seg + 1) % 4] : null;
  const lossVal = 2.4 * 0.72 ** laps + 0.12;
  const node = NODES.find((n) => n.id === sel)!;

  const edge = (a: NodeId, b: NodeId, key: string) => {
    const [x1, y1] = pos[a];
    const [x2, y2] = pos[b];
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    const ux = dx / len;
    const uy = dy / len;
    // trim so arrows start/end at box edges
    const trimX = Math.abs(ux) > 0.01 ? NW / 2 / Math.abs(ux) : Infinity;
    const trimY = Math.abs(uy) > 0.01 ? NH / 2 / Math.abs(uy) : Infinity;
    const trim = Math.min(trimX, trimY) + 6;
    const sx = x1 + ux * trim;
    const sy = y1 + uy * trim;
    const ex = x2 - ux * trim;
    const ey = y2 - uy * trim;
    return (
      <g key={key}>
        <line x1={sx} y1={sy} x2={ex} y2={ey} stroke="var(--border-strong)" strokeWidth={3} strokeLinecap="round" />
        <path d={`M${ex} ${ey} l${-ux * 11 - uy * 6} ${-uy * 11 + ux * 6} l${uy * 12} ${-ux * 12} Z`} fill="var(--border-strong)" />
      </g>
    );
  };

  return (
    <div ref={ref}>
      <div ref={box} style={{ padding: '12px 14px 0' }}>
        <svg className="vz-svg" viewBox={vb} role="img" aria-label="데이터 → 모델 → 손실 → 기울기 → 갱신을 반복하는 학습 루프" style={{ touchAction: 'pan-y' }}>
          {edge('data', 'model', 'e0')}
          {edge('model', 'loss', 'e1')}
          {edge('loss', 'grad', 'e2')}
          {edge('grad', 'update', 'e3')}
          {narrow ? (
            <g>
              <path d="M280 428 C 340 428, 340 140, 280 140" fill="none" stroke="var(--border-strong)" strokeWidth={3} />
              <path d="M284 140 l11 -6 l0 12 Z" fill="var(--border-strong)" />
            </g>
          ) : (
            edge('update', 'model', 'e4')
          )}
          <text x={narrow ? 330 : 375} y={narrow ? 290 : 146} textAnchor="middle" style={{ fontSize: 12.5, fontWeight: 850, fill: 'var(--text-2)' }}>
            🔁 {laps}바퀴
          </text>
          {NODES.map((n) => {
            const [x, y] = pos[n.id];
            const active = lit === n.id || sel === n.id;
            const stroke = tools ? toolColor[n.tools[0]] : c.brand;
            return (
              <g
                key={n.id}
                transform={`translate(${x - NW / 2} ${y - NH / 2})`}
                style={{ cursor: 'pointer' }}
                onClick={() => setSel(n.id)}
                role="button"
                tabIndex={0}
                aria-label={`${n.title}: ${n.sub}`}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setSel(n.id)}
              >
                <rect
                  width={NW}
                  height={NH}
                  rx={16}
                  fill={active ? withAlpha(stroke, 0.22) : 'var(--surface-solid)'}
                  stroke={stroke}
                  strokeWidth={sel === n.id ? 3.5 : 2}
                  style={{ transition: 'fill .25s', filter: active && c.theme === 'dark' ? `drop-shadow(0 0 8px ${stroke})` : undefined }}
                />
                {tools && n.tools.length > 1 && <rect x={NW - 30} y={8} width={20} height={6} rx={3} fill={toolColor[n.tools[1]]} />}
                <text x={18} y={39} style={{ fontSize: 24 }}>
                  {n.emoji}
                </text>
                <text x={54} y={28} style={{ fontSize: 15, fontWeight: 850, fill: 'var(--text)' }}>
                  {n.title}
                </text>
                <text x={54} y={48} style={{ fontSize: 11.5, fontWeight: 700, fill: 'var(--text-2)' }}>
                  {n.sub}
                </text>
              </g>
            );
          })}
          <circle cx={px} cy={py} r={9} stroke={c.surface} strokeWidth={3} style={{ fill: 'var(--lemon)', filter: c.theme === 'dark' ? 'drop-shadow(0 0 8px var(--lemon))' : undefined }} />
        </svg>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ padding: '12px 14px', borderRadius: 16, background: 'var(--surface-2)', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', marginBottom: 4 }}>
            <strong style={{ fontSize: 16 }}>
              {node.emoji} {node.title}
            </strong>
            {node.tools.map((t) => (
              <span key={t} style={{ fontSize: 11.5, fontWeight: 800, padding: '1px 8px', borderRadius: 99, background: withAlpha(toolColor[t], 0.18), color: 'var(--text)' }}>
                {t === 'vec' ? '벡터·행렬' : t === 'prob' ? '확률' : '미분'}
              </span>
            ))}
            <span style={{ fontSize: 12, color: 'var(--text-3)', marginLeft: 'auto' }}>{node.where}에서 배워요</span>
          </div>
          <div style={{ fontSize: 14.5, lineHeight: 1.7 }}>{node.text}</div>
        </div>
        {tools && (
          <Legend
            items={[
              { color: toolColor.vec, label: '벡터·행렬 (1절)' },
              { color: toolColor.diff, label: '미분 (2절)' },
              { color: toolColor.prob, label: '확률 (3절)' }
            ]}
          />
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13, fontWeight: 750 }}>
          <span style={{ flex: 'none', color: 'var(--text-2)' }}>손실</span>
          <span style={{ flex: 1, height: 12, borderRadius: 99, background: 'var(--surface-3)', overflow: 'hidden' }}>
            <span style={{ display: 'block', height: '100%', width: `${Math.min(100, (lossVal / 2.6) * 100)}%`, borderRadius: 99, background: lossVal < 0.4 ? c.pos : c.series[2], transition: 'width .6s' }} />
          </span>
          <span style={{ flex: 'none', minWidth: 44, textAlign: 'right' }}>{lossVal.toFixed(2)}</span>
        </div>
        <Note mood={laps >= 5 ? 'cheer' : 'smile'} tone={laps >= 5 ? 'good' : undefined}>
          {laps >= 5
            ? `벌써 ${laps}바퀴! 바퀴를 돌 때마다 손실이 줄어요. 세 가지 수학 도구가 이 루프 곳곳에 꽂혀 있다는 것, 기억해 두세요.`
            : '노란 공이 루프를 도는 게 "학습 한 번"이에요. 상자를 누르면 그 단계에서 쓰는 수학이 나와요.'}
        </Note>
      </div>

      <Controls>
        <Button variant="primary" onClick={() => setPlaying((p) => !p)}>
          {playing ? '❚❚ 멈춤' : '▶ 돌리기'}
        </Button>
        <Button
          onClick={() => {
            setPlaying(false);
            advance(Math.floor(phaseRef.current) + 1);
          }}
        >
          다음 단계 ⟶
        </Button>
        <Button variant="ghost" onClick={() => { setLaps(0); phaseRef.current = 0; setPhase(0); }}>
          ↺ 처음으로
        </Button>
        <Toggle label="어떤 수학이 쓰이나 색칠하기" checked={tools} onChange={setTools} />
      </Controls>
      {expanded && <div style={{ height: 8 }} />}
    </div>
  );
}
