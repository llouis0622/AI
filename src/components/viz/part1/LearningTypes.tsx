import { useMemo, useRef, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Note, Segmented } from '../core/ui';
import { useInView, usePrefersReducedMotion, useRafLoop } from '../core/hooks';
import { useVizColors, withAlpha, type VizColors } from '../core/colors';
import { clamp, rng } from '../core/math';

type Kind = 'sup' | 'unsup' | 'rl' | 'self';

const INFO: Record<Kind, { name: string; signal: string; emoji: string; text: string; examples: string }> = {
  sup: { name: '지도학습', signal: '정답', emoji: '🏷️', text: '문제와 정답을 쌍으로 보여 줘요. "이건 고양이, 이건 개" 하고 이름표를 붙여 주면, 처음 보는 사진에도 이름표를 붙일 수 있게 돼요.', examples: '스팸 필터 · 집값 예측 · 사진 분류' },
  unsup: { name: '비지도학습', signal: '데이터만', emoji: '🧺', text: '정답 없이 데이터만 줘요. 기계가 알아서 비슷한 것끼리 묶거나, 데이터의 모양을 찾아내요.', examples: '고객 무리 나누기 · 뉴스 주제 묶기' },
  rl: { name: '강화학습', signal: '보상', emoji: '🍬', text: '정답 대신 결과에 대한 보상(당근)과 벌(채찍)만 줘요. 여러 번 해 보면서 보상을 많이 받는 행동을 스스로 익혀요.', examples: '바둑·게임 AI · 로봇 걷기 · 청소 경로' },
  self: { name: '자기지도학습', signal: '가린 부분', emoji: '🙈', text: '데이터의 일부를 가리고 맞히게 해요. 정답이 원래 데이터 안에 있으니 사람이 이름표를 달 필요가 없어요. GPT의 학습법!', examples: '다음 단어 맞히기 · 사진 조각 채우기' }
};

const QUIZ: { task: string; answer: Kind; why: string }[] = [
  { task: '부동산 가격 예측', answer: 'sup', why: '과거 집들의 실제 가격(정답)을 보고 배우니까 지도학습 — 그중에서도 숫자를 맞히는 회귀예요.' },
  { task: '뉴스 기사를 비슷한 주제끼리 묶기', answer: 'unsup', why: '"이 기사는 몇 번 주제"라는 정답 없이 비슷한 것끼리 모으니까 비지도학습이에요.' },
  { task: '로봇 청소기가 청소 경로를 개선하기', answer: 'rl', why: '정답 경로는 아무도 몰라요. 잘 청소하면 보상을 받으며 시행착오로 익히니 강화학습이에요.' },
  { task: '문장의 가려진 단어 맞추기', answer: 'self', why: '가린 단어가 곧 정답이라 사람이 정답을 달 필요가 없어요. 자기지도학습!' }
];

/** Each scene animates for ~6 s, then holds its final frame before looping. */
const PERIOD = 9;

export default function LearningTypes(_: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [kind, setKind] = useState<Kind>('sup');
  const [t, setT] = useState(0);
  const [answers, setAnswers] = useState<Record<number, Kind>>({});
  const box = useRef<HTMLDivElement>(null);
  const inView = useInView(box, '0px');
  useRafLoop((dt) => setT((v) => (v + dt) % PERIOD), inView && !reduce);
  const time = reduce ? PERIOD : t;
  const info = INFO[kind];
  const score = QUIZ.filter((q, i) => answers[i] === q.answer).length;

  return (
    <div ref={box}>
      <div style={{ padding: '12px 14px 0' }}>
        <Segmented
          options={(Object.keys(INFO) as Kind[]).map((k) => ({ value: k, label: `${INFO[k].emoji} ${INFO[k].name}` }))}
          value={kind}
          onChange={(k) => {
            setKind(k);
            setT(0);
          }}
        />
      </div>
      <div style={{ padding: '8px 14px 0' }}>
        <svg
          className="vz-svg"
          viewBox="0 0 480 260"
          role="img"
          aria-label={`${info.name} 장면`}
          style={{ touchAction: 'pan-y', borderRadius: 16, background: 'var(--surface-2)', maxWidth: 640, margin: '0 auto' }}
        >
          {kind === 'sup' && <SupScene t={time} c={c} />}
          {kind === 'unsup' && <UnsupScene t={time} c={c} />}
          {kind === 'rl' && <RlScene t={time} c={c} />}
          {kind === 'self' && <SelfScene t={time} c={c} />}
        </svg>
      </div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
          <strong style={{ fontSize: 17 }}>{info.name}</strong>
          <span style={{ padding: '3px 10px', borderRadius: 99, background: withAlpha(c.series[0], 0.15), color: 'var(--text)', fontSize: 13, fontWeight: 800 }}>
            주어지는 신호: {info.emoji} {info.signal}
          </span>
        </div>
        <div style={{ fontSize: 14.5, lineHeight: 1.7 }}>{info.text}</div>
        <div style={{ fontSize: 13, color: 'var(--text-2)' }}>예: {info.examples}</div>

        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12 }}>
          <div className="vz-panel-title" style={{ marginBottom: 8 }}>
            맞혀 보기 — 어떤 배움일까요? ({score}/4)
          </div>
          <div style={{ display: 'grid', gap: 8 }}>
            {QUIZ.map((q, i) => {
              const a = answers[i];
              return (
                <div key={q.task} style={{ display: 'grid', gap: 8, padding: '10px 12px', borderRadius: 14, background: 'var(--surface-solid)', border: `1.5px solid ${a ? (a === q.answer ? c.pos : c.neg) : 'var(--border)'}` }}>
                  <strong style={{ fontSize: 14 }}>
                    {i + 1}. {q.task}
                  </strong>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(96px, 1fr))', gap: 6 }}>
                    {(Object.keys(INFO) as Kind[]).map((k) => (
                      <button key={k} type="button" className={`vz-btn vz-btn-sm${a === k ? ' is-active' : ''}`} onClick={() => setAnswers((s) => ({ ...s, [i]: k }))}>
                        {INFO[k].name}
                      </button>
                    ))}
                  </div>
                  {a && (
                    <span style={{ fontSize: 13, color: a === q.answer ? 'var(--text-2)' : c.neg, lineHeight: 1.55 }}>
                      {a === q.answer ? `✓ ${q.why}` : '✗ 다시 생각해 볼까요? 무엇이 주어지는지(정답·데이터만·보상·가린 부분) 떠올려 보세요.'}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
        {score === 4 && (
          <Note mood="cheer" tone="good">
            네 가지를 모두 구분했어요! 🎉 이 구분은 책 전체의 목차이기도 해요.
          </Note>
        )}
      </div>
    </div>
  );
}

/* ---------- scenes (viewBox 480×260) ---------- */

const ease = (x: number) => 1 - (1 - clamp(x, 0, 1)) ** 3;

function Title({ children }: { children: ReactNode }) {
  return (
    <text x={16} y={30} style={{ fontSize: 15, fontWeight: 850, fill: 'var(--text-2)' }}>
      {children}
    </text>
  );
}

function Card({ x, y, emoji, tag, tagColor, show = 1, tagShow = 1 }: { x: number; y: number; emoji: string; tag?: string; tagColor: string; show?: number; tagShow?: number }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${0.7 + 0.3 * ease(show)})`} opacity={ease(show)}>
      <rect x={-28} y={-28} width={56} height={56} rx={13} fill="var(--surface-solid)" stroke="var(--border-strong)" strokeWidth={1.5} />
      <text y={10} textAnchor="middle" style={{ fontSize: 28 }}>
        {emoji}
      </text>
      {tag && (
        <g opacity={ease(tagShow)} transform={`translate(0 ${42 - 10 * (1 - ease(tagShow))})`}>
          <rect x={-30} y={-12} width={60} height={24} rx={12} fill={tagColor} />
          <text y={5} textAnchor="middle" style={{ fontSize: 13, fontWeight: 900, fill: '#fff' }}>
            {tag}
          </text>
        </g>
      )}
    </g>
  );
}

function SupScene({ t, c }: { t: number; c: VizColors }) {
  const ex = [
    { e: '🐱', tag: '고양이' },
    { e: '🐶', tag: '개' },
    { e: '🐱', tag: '고양이' },
    { e: '🐶', tag: '개' }
  ];
  const col = (tag: string) => (tag === '고양이' ? c.series[0] : c.series[2]);
  const guessed = t > 5;
  return (
    <g>
      <Title>👩‍🏫 선생님이 정답(이름표)을 붙여 줘요</Title>
      {ex.map((d, i) => (
        <Card key={i} x={46 + i * 64} y={104} emoji={d.e} tag={d.tag} tagColor={col(d.tag)} show={(t - i * 0.55) / 0.4} tagShow={(t - i * 0.55 - 0.35) / 0.4} />
      ))}
      <g opacity={ease((t - 2.7) / 0.5)}>
        <line x1={284} y1={104} x2={302} y2={104} stroke={c.series[1]} strokeWidth={3} />
        <path d="M302 97 L313 104 L302 111 Z" fill={c.series[1]} />
        <rect x={318} y={74} width={62} height={60} rx={15} fill={withAlpha(c.series[1], 0.18)} stroke={c.series[1]} strokeWidth={2} />
        <text x={349} y={113} textAnchor="middle" style={{ fontSize: 28 }}>
          🤖
        </text>
        <text x={349} y={154} textAnchor="middle" style={{ fontSize: 12.5, fontWeight: 850, fill: 'var(--text-2)' }}>
          규칙 찾는 중
        </text>
      </g>
      <Card x={432} y={104} emoji="🐱" tag={guessed ? '고양이!' : '?'} tagColor={guessed ? c.pos : c.theme === 'dark' ? '#6f7aa6' : '#8c86a0'} show={(t - 3.8) / 0.5} tagShow={(t - 4.1) / 0.4} />
      <text x={432} y={190} textAnchor="middle" opacity={ease((t - 5) / 0.4)} style={{ fontSize: 13, fontWeight: 850, fill: c.pos }}>
        처음 본 사진도 척척
      </text>
      <text x={240} y={236} textAnchor="middle" opacity={ease((t - 5.4) / 0.5)} style={{ fontSize: 14, fontWeight: 800, fill: 'var(--text)' }}>
        정답 달린 예시 → 규칙 → 새 문제에 답!
      </text>
    </g>
  );
}

const CENTERS = [
  [100, 130],
  [240, 104],
  [370, 168]
];

function UnsupScene({ t, c }: { t: number; c: VizColors }) {
  const centers = CENTERS;
  const pts = useMemo(() => {
    const r = rng(11);
    return Array.from({ length: 27 }, (_, i) => {
      const k = i % 3;
      return {
        k,
        sx: 30 + r.next() * 420,
        sy: 56 + r.next() * 180,
        tx: CENTERS[k][0] + r.normal() * 26,
        ty: CENTERS[k][1] + r.normal() * 18
      };
    });
  }, []);
  const p = ease((t - 1.2) / 2.4);
  const colorOn = ease((t - 3.4) / 0.9);
  const base = c.theme === 'dark' ? '#6f7aa6' : '#b9b2c9';
  return (
    <g>
      <Title>🧺 정답 없이, 비슷한 것끼리 모아요</Title>
      {pts.map((d, i) => (
        <circle
          key={i}
          cx={d.sx + (d.tx - d.sx) * p}
          cy={d.sy + (d.ty - d.sy) * p}
          r={8}
          fill={colorOn > 0.01 ? c.series[d.k] : base}
          opacity={0.55 + 0.45 * colorOn}
          stroke="var(--surface-solid)"
          strokeWidth={2}
        />
      ))}
      {['무리 1', '무리 2', '무리 3'].map((name, k) => (
        <text key={name} x={centers[k][0]} y={centers[k][1] + 62} textAnchor="middle" opacity={colorOn} style={{ fontSize: 14, fontWeight: 900, fill: c.series[k] }}>
          {name}
        </text>
      ))}
    </g>
  );
}

function RlScene({ t, c }: { t: number; c: VizColors }) {
  const cell = 54;
  const ox = 140;
  const oy = 58;
  const goal = [5, 1];
  const trap = [3, 1];
  // three scripted tries: hit the trap, wander to the goal, go straight
  const paths: [number, number][][] = [
    [[0, 1], [1, 1], [2, 1], [3, 1]],
    [[0, 1], [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 1], [5, 1]],
    [[0, 1], [1, 1], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [5, 1]]
  ];
  const slot = PERIOD / 3;
  const ep = Math.min(2, Math.floor(t / slot));
  const local = (t - ep * slot) / slot;
  const path = paths[ep];
  const f = clamp(local * 1.35, 0, 1) * (path.length - 1);
  const i0 = Math.floor(f);
  const i1 = Math.min(path.length - 1, i0 + 1);
  const fr = f - i0;
  const rx = path[i0][0] + (path[i1][0] - path[i0][0]) * fr;
  const ry = path[i0][1] + (path[i1][1] - path[i0][1]) * fr;
  const done = f >= path.length - 1;
  const result = ep === 0 ? { txt: '-1 아야!', col: c.neg } : { txt: '+1 보상!', col: c.pos };
  const rewards = [-1, 1, 1].slice(0, ep + (done ? 1 : 0)).reduce((a, b) => a + b, 0);
  const mid = (gx: number, gy: number) => [ox + gx * cell + (cell - 4) / 2, oy + gy * cell + (cell - 4) / 2];
  return (
    <g>
      <Title>🍬 결과에 보상 — 해 볼수록 길을 익혀요</Title>
      {Array.from({ length: 3 }, (_, y) =>
        Array.from({ length: 6 }, (_, x) => (
          <rect key={`${x}-${y}`} x={ox + x * cell} y={oy + y * cell} width={cell - 4} height={cell - 4} rx={10} fill="var(--surface-solid)" stroke="var(--border)" />
        ))
      )}
      <text x={mid(goal[0], goal[1])[0]} y={mid(goal[0], goal[1])[1] + 9} textAnchor="middle" style={{ fontSize: 26 }}>
        ⭐
      </text>
      <text x={mid(trap[0], trap[1])[0]} y={mid(trap[0], trap[1])[1] + 9} textAnchor="middle" style={{ fontSize: 24 }}>
        💥
      </text>
      {path.slice(0, i1 + 1).map((p, k) => (
        <circle key={k} cx={mid(p[0], p[1])[0]} cy={mid(p[0], p[1])[1]} r={4} fill={withAlpha(c.series[0], 0.7)} />
      ))}
      <text x={ox + rx * cell + (cell - 4) / 2} y={oy + ry * cell + (cell - 4) / 2 + 9} textAnchor="middle" style={{ fontSize: 26 }}>
        🤖
      </text>
      {done && (
        <text x={mid(path[path.length - 1][0], path[path.length - 1][1])[0]} y={oy - 8} textAnchor="middle" style={{ fontSize: 15, fontWeight: 900, fill: result.col }}>
          {result.txt}
        </text>
      )}
      <text x={18} y={92} style={{ fontSize: 16, fontWeight: 900, fill: 'var(--text)' }}>
        도전 {ep + 1}회
      </text>
      <text x={18} y={124} style={{ fontSize: 13.5, fontWeight: 750, fill: 'var(--text-2)' }}>
        보상 합계
      </text>
      <text x={18} y={156} style={{ fontSize: 26, fontWeight: 900, fill: rewards >= 0 ? c.pos : c.neg }}>
        {rewards > 0 ? `+${rewards}` : rewards}
      </text>
      <text x={18} y={204} style={{ fontSize: 12.5, fontWeight: 750, fill: 'var(--text-3)' }}>
        ⭐ +1 · 💥 -1
      </text>
    </g>
  );
}

function SelfScene({ t, c }: { t: number; c: VizColors }) {
  const words = ['어제', '비가', '와서', '우산을'];
  const cands = [
    { w: '챙겼다', p: 0.72 },
    { w: '샀다', p: 0.15 },
    { w: '잃어버렸다', p: 0.09 },
    { w: '먹었다', p: 0.01 }
  ];
  const masked = t > 1.2;
  const grow = ease((t - 2) / 1.6);
  const reveal = t > 4.6;
  return (
    <g>
      <Title>🙈 문장 일부를 가리고 맞혀요</Title>
      {words.map((w, i) => (
        <g key={w} transform={`translate(${12 + i * 82} 50)`}>
          <rect width={76} height={42} rx={12} fill="var(--surface-solid)" stroke="var(--border-strong)" />
          <text x={38} y={27} textAnchor="middle" style={{ fontSize: 16, fontWeight: 800, fill: 'var(--text)' }}>
            {w}
          </text>
        </g>
      ))}
      <g transform="translate(342 50)">
        <rect width={126} height={42} rx={12} fill={masked && !reveal ? withAlpha(c.series[0], 0.25) : 'var(--surface-solid)'} stroke={reveal ? c.pos : c.series[0]} strokeWidth={2.5} strokeDasharray={masked && !reveal ? '6 4' : undefined} />
        <text x={63} y={27} textAnchor="middle" style={{ fontSize: 16, fontWeight: 900, fill: reveal ? c.pos : 'var(--text)' }}>
          {!masked || reveal ? '챙겼다' : '[ ? ]'}
        </text>
      </g>
      {reveal && (
        <text x={405} y={112} textAnchor="middle" style={{ fontSize: 13.5, fontWeight: 900, fill: c.pos }}>
          정답! 🎉
        </text>
      )}
      {cands.map((d, i) => {
        const y = 128 + i * 30;
        return (
          <g key={d.w} opacity={masked ? 1 : 0.15}>
            <text x={112} y={y + 15} textAnchor="end" style={{ fontSize: 14, fontWeight: 800, fill: 'var(--text)' }}>
              {d.w}
            </text>
            <rect x={122} y={y} width={300} height={20} rx={10} fill="var(--surface-3)" />
            <rect x={122} y={y} width={300 * d.p * grow} height={20} rx={10} fill={i === 0 ? c.series[1] : c.series[3]} />
            <text x={128 + 300 * d.p * grow} y={y + 15} style={{ fontSize: 13, fontWeight: 850, fill: 'var(--text-2)' }}>
              {Math.round(d.p * 100 * grow)}%
            </text>
          </g>
        );
      })}
    </g>
  );
}
