import { useMemo, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Slider, Toggle } from '../core/ui';
import { useVizColors, withAlpha } from '../core/colors';

/*
 * RAG on a tiny "도서관 이용 안내" document. Chunks live on a 2D "meaning map"
 * (stand-in for a high-dimensional embedding space); retrieval is a real
 * nearest-neighbour search on that map.
 */
type Topic = 'loan' | 'late' | 'room' | 'device' | 'food';
interface Chunk {
  id: string;
  text: string;
  x: number;
  y: number;
  topic: Topic;
}

const TOPIC_COLOR: Record<Topic, number> = { loan: 0, late: 2, room: 3, device: 1, food: 4 };

const RIGHT: Chunk[] = [
  { id: '1', text: '대출: 학생증이 있으면 한 번에 책을 5권까지 빌릴 수 있어요.', x: 0.22, y: 0.74, topic: 'loan' },
  { id: '2', text: '대출 기간은 2주이고, 한 번 연장하면 1주 더 빌릴 수 있어요.', x: 0.3, y: 0.84, topic: 'loan' },
  { id: '3', text: '연체: 반납이 늦으면 늦은 날짜만큼 대출이 정지돼요.', x: 0.28, y: 0.46, topic: 'late' },
  { id: '4', text: '책을 잃어버리면 같은 책으로 사 오거나 책값을 내야 해요.', x: 0.15, y: 0.36, topic: 'late' },
  { id: '5', text: '열람실은 평일 오전 8시부터 오후 9시까지 열어요.', x: 0.7, y: 0.8, topic: 'room' },
  { id: '6', text: '주말에는 열람실을 오전 10시부터 오후 5시까지 열어요.', x: 0.81, y: 0.7, topic: 'room' },
  { id: '7', text: '열람실에서는 노트북을 쓸 수 있고, 창가 자리에 콘센트가 있어요.', x: 0.78, y: 0.36, topic: 'device' },
  { id: '8', text: '음식은 1층 휴게 공간에서만 먹을 수 있어요.', x: 0.55, y: 0.15, topic: 'food' }
];

const SMALL: Chunk[] = [
  { id: '1a', text: '대출: 학생증이 있으면', x: 0.2, y: 0.7, topic: 'loan' },
  { id: '1b', text: '한 번에 책을 5권까지 빌릴 수 있어요.', x: 0.25, y: 0.78, topic: 'loan' },
  { id: '2a', text: '대출 기간은 2주이고,', x: 0.33, y: 0.8, topic: 'loan' },
  { id: '2b', text: '한 번 연장하면 1주 더 빌릴 수 있어요.', x: 0.29, y: 0.89, topic: 'loan' },
  { id: '3a', text: '연체: 반납이 늦으면', x: 0.29, y: 0.5, topic: 'late' },
  { id: '3b', text: '늦은 날짜만큼 대출이 정지돼요.', x: 0.24, y: 0.41, topic: 'late' },
  { id: '4a', text: '책을 잃어버리면', x: 0.13, y: 0.4, topic: 'late' },
  { id: '4b', text: '같은 책으로 사 오거나 책값을 내야 해요.', x: 0.17, y: 0.31, topic: 'late' },
  { id: '5a', text: '열람실은 평일', x: 0.66, y: 0.78, topic: 'room' },
  { id: '5b', text: '오전 8시부터 오후 9시까지 열어요.', x: 0.73, y: 0.85, topic: 'room' },
  { id: '6a', text: '주말에는 열람실을', x: 0.83, y: 0.76, topic: 'room' },
  { id: '6b', text: '오전 10시부터 오후 5시까지 열어요.', x: 0.87, y: 0.66, topic: 'room' },
  { id: '7a', text: '열람실에서는 노트북을 쓸 수 있고,', x: 0.74, y: 0.41, topic: 'device' },
  { id: '7b', text: '창가 자리에 콘센트가 있어요.', x: 0.82, y: 0.32, topic: 'device' },
  { id: '8a', text: '음식은 1층 휴게 공간에서만', x: 0.52, y: 0.19, topic: 'food' },
  { id: '8b', text: '먹을 수 있어요.', x: 0.58, y: 0.11, topic: 'food' }
];

const LARGE: Chunk[] = [
  { id: '1+2', text: `${RIGHT[0].text} ${RIGHT[1].text}`, x: 0.26, y: 0.79, topic: 'loan' },
  { id: '3+4', text: `${RIGHT[2].text} ${RIGHT[3].text}`, x: 0.215, y: 0.41, topic: 'late' },
  { id: '5+6', text: `${RIGHT[4].text} ${RIGHT[5].text}`, x: 0.755, y: 0.75, topic: 'room' },
  { id: '7+8', text: `${RIGHT[6].text} ${RIGHT[7].text}`, x: 0.665, y: 0.255, topic: 'device' }
];

const SETS = { small: SMALL, right: RIGHT, large: LARGE };
type SizeKey = keyof typeof SETS;

interface Question {
  q: string;
  x: number;
  y: number;
  need?: string;
  answer?: string;
  topicWord?: string;
  noRag: string;
}
const QUESTIONS: Question[] = [
  {
    q: '책을 늦게 반납하면 어떻게 돼요?',
    x: 0.3,
    y: 0.53,
    need: '대출이 정지',
    answer: '반납이 늦으면 늦은 날짜만큼 대출이 정지돼요.',
    topicWord: '반납',
    noRag: '보통 하루에 100원씩 연체료를 내요. 😊'
  },
  {
    q: '토요일에 열람실은 몇 시까지 해요?',
    x: 0.84,
    y: 0.74,
    need: '오후 5시',
    answer: '주말(토요일 포함)에는 오전 10시부터 오후 5시까지 열어요.',
    topicWord: '열람실',
    noRag: '열람실은 보통 밤 10시까지 열어요!'
  },
  {
    q: '열람실에서 노트북 충전할 수 있어요?',
    x: 0.74,
    y: 0.44,
    need: '콘센트',
    answer: '네! 열람실에서 노트북을 쓸 수 있고, 창가 자리에 콘센트가 있어요.',
    topicWord: '노트북',
    noRag: '도서관에서는 전자기기를 쓸 수 없어요.'
  },
  {
    q: '학교 축제는 언제예요?',
    x: 0.07,
    y: 0.08,
    noRag: '학교 축제는 10월 넷째 주 금요일이에요! 🎉'
  }
];

const RELEVANT = 0.25;
const sim = (a: { x: number; y: number }, b: { x: number; y: number }) => Math.exp(-((Math.hypot(a.x - b.x, a.y - b.y) / 0.18) ** 2));

const STEPS = [
  { icon: '✂️', label: '조각내기' },
  { icon: '🧭', label: '임베딩' },
  { icon: '❓', label: '질문' },
  { icon: '🔎', label: '검색' },
  { icon: '🧩', label: '프롬프트' },
  { icon: '✅', label: '답변' }
];

export default function RAGPipeline(_: VizProps) {
  const c = useVizColors();
  const dark = c.theme === 'dark';
  const [step, setStep] = useState(0);
  const [size, setSize] = useState<SizeKey>('right');
  const [qi, setQi] = useState(0);
  const [k, setK] = useState(2);
  const [noRag, setNoRag] = useState(false);
  const [hover, setHover] = useState<string | null>(null);

  const chunks = SETS[size];
  const question = QUESTIONS[qi];
  const ranked = useMemo(
    () => chunks.map((ch) => ({ ch, s: sim(ch, question) })).sort((p, q) => q.s - p.s),
    [chunks, question]
  );
  const topK = ranked.slice(0, k);
  const used = topK.filter((r) => r.s >= RELEVANT);
  const retrievedText = used.map((r) => r.ch.text).join(' ');

  // compose the grounded answer from what was actually retrieved
  let answer: ReactNode;
  let answerKind: 'good' | 'partial' | 'refuse' | 'hallucinate';
  if (noRag) {
    answer = question.noRag;
    answerKind = 'hallucinate';
  } else if (!question.need || used.length === 0) {
    answer = '제가 받은 문서 발췌에는 그 내용이 없어요. 학교 홈페이지나 선생님께 확인해 보세요.';
    answerKind = 'refuse';
  } else if (retrievedText.includes(question.need)) {
    const cites = used.filter((r) => r.ch.text.includes(question.need!) || r.ch.text.includes(question.topicWord ?? '§')).map((r) => r.ch.id);
    answer = (
      <>
        {question.answer}{' '}
        {cites.map((id) => (
          <sup key={id} style={{ fontWeight: 850, color: c.brand }}>
            [{id}]
          </sup>
        ))}
      </>
    );
    answerKind = 'good';
  } else {
    answer = `발췌에 "${used[0].ch.text}"까지만 있어서, 정확히 어떻게 되는지는 답하기 어려워요.`;
    answerKind = 'partial';
  }

  const W = 400;
  const H = 260;
  const px = (v: number) => 20 + v * (W - 40);
  const py = (v: number) => H - 18 - v * (H - 36);
  const showMap = step >= 1 && step <= 3;
  const showQ = step >= 2;
  const showLines = step >= 3;

  const active = (node: string) => {
    const on: Record<number, string[]> = {
      0: ['doc', 'split'],
      1: ['split', 'store'],
      2: ['question', 'qembed'],
      3: ['store', 'search'],
      4: ['prompt'],
      5: ['llm', 'answer']
    };
    return on[step].includes(node);
  };
  const Node = ({ id, children }: { id: string; children: ReactNode }) => (
    <span
      style={{
        padding: '5px 10px',
        borderRadius: 10,
        fontSize: 12.5,
        fontWeight: 800,
        whiteSpace: 'nowrap',
        border: `1.5px solid ${active(id) ? c.brand : 'var(--border)'}`,
        background: active(id) ? withAlpha(c.brand, dark ? 0.2 : 0.12) : 'var(--surface-solid)',
        color: active(id) ? 'var(--text)' : 'var(--text-3)',
        boxShadow: active(id) && dark ? '0 0 14px rgba(34,229,255,.35)' : undefined,
        transition: 'all .3s'
      }}
    >
      {children}
    </span>
  );
  const Arrow = ({ label }: { label?: string }) => (
    <span style={{ color: 'var(--text-3)', fontSize: 12, fontWeight: 700, whiteSpace: 'nowrap' }}>
      {label ? `─${label}→` : '→'}
    </span>
  );

  let note: { t: ReactNode; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'bad' | 'warn' };
  switch (step) {
    case 0:
      note =
        size === 'small'
          ? { t: '조각이 너무 작으면 한 문장이 반으로 잘려서, 조각만 봐서는 뜻을 알기 어려워요.', mood: 'think', tone: 'warn' }
          : size === 'large'
            ? { t: '조각이 너무 크면 여러 주제가 한 조각에 섞여서, 질문과 딱 맞는 조각을 찾기 어려워져요.', mood: 'think', tone: 'warn' }
            : { t: <>긴 문서를 <strong>적당한 크기의 조각</strong>으로 잘라요. 3천 쪽을 통째로 프롬프트에 넣을 수는 없으니까요!</>, mood: 'smile' };
      break;
    case 1:
      note = { t: <>각 조각을 <strong>의미를 담은 숫자(임베딩)</strong>로 바꿔 지도 위에 놓아요(16장). 뜻이 비슷한 조각끼리 가까이 모여요. 점에 마우스를 올려 보세요.</>, mood: 'happy' };
      break;
    case 2:
      note = { t: '질문도 똑같이 임베딩으로 바꿔요. 별(★)이 질문의 위치예요. 아래에서 다른 질문도 골라 보세요.', mood: 'smile' };
      break;
    case 3:
      note =
        used.length === 0
          ? { t: '질문 가까이에 조각이 하나도 없어요! 관련 있는 문서가 없다는 뜻이에요.', mood: 'wow', tone: 'warn' }
          : { t: <>별에서 <strong>가장 가까운 조각 {k}개</strong>를 꺼내요. RAG가 실패하는 원인의 대부분은 생성이 아니라 바로 이 <strong>검색</strong>이에요.</>, mood: 'think' };
      break;
    case 4:
      note = { t: <>꺼낸 조각을 "이 발췌만 근거로 답하고, 없으면 없다고 하라"는 지시와 함께 <strong>프롬프트</strong>에 붙여요.</>, mood: 'smile' };
      break;
    default:
      note =
        answerKind === 'hallucinate'
          ? { t: <>RAG 없이 물으면 모델이 <strong>그럴듯하지만 틀린 답(환각)</strong>을 자신 있게 지어내요. 이 도서관의 실제 규칙을 모르니까요!</>, mood: 'wow', tone: 'bad' }
          : answerKind === 'refuse'
            ? { t: <>근거가 없으니 <strong>모른다고 솔직하게</strong> 답했어요. 지어내지 않는 것 — RAG가 환각을 줄이는 방법이에요. 🎉</>, mood: 'cheer', tone: 'good' }
            : answerKind === 'partial'
              ? { t: '조각이 잘려서 핵심 내용이 검색되지 않았어요. 조각 크기를 "알맞게"로 바꾸거나 k를 늘려 보세요.', mood: 'think', tone: 'warn' }
              : { t: <>근거 있는 답 + <strong>출처 번호</strong>까지! 사람이 확인할 수 있고, 문서가 바뀌면 다시 학습할 필요 없이 저장소만 바꾸면 돼요.</>, mood: 'cheer', tone: 'good' };
  }

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 14 }}>
        {/* pipeline strip */}
        <div style={{ display: 'grid', gap: 8 }} aria-label="RAG 파이프라인">
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6 }}>
            <Node id="doc">📄 문서들</Node>
            <Arrow />
            <Node id="split">✂️ 조각내기 + 임베딩</Node>
            <Arrow />
            <Node id="store">🗄️ 벡터 저장소</Node>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6 }}>
            <Node id="question">❓ 질문</Node>
            <Arrow />
            <Node id="qembed">🧭 임베딩</Node>
            <Arrow />
            <Node id="search">🔎 유사도 검색</Node>
            <Arrow label={`조각 ${k}개`} />
            <Node id="prompt">🧩 프롬프트</Node>
            <Arrow />
            <Node id="llm">🤖 LLM</Node>
            <Arrow />
            <Node id="answer">✅ 답변 + 출처</Node>
          </div>
        </div>

        {/* step pills */}
        <div role="tablist" aria-label="단계" style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {STEPS.map((s, i) => (
            <button
              key={s.label}
              type="button"
              role="tab"
              aria-selected={step === i}
              onClick={() => setStep(i)}
              style={{
                padding: '6px 12px',
                borderRadius: 99,
                fontSize: 13,
                fontWeight: 800,
                border: `1.5px solid ${step === i ? c.brand : 'var(--border)'}`,
                background: step === i ? 'var(--grad-brand)' : 'var(--surface-solid)',
                color: step === i ? (dark ? '#041018' : '#fff') : i < step ? 'var(--text)' : 'var(--text-3)'
              }}
            >
              {i + 1}. {s.icon} {s.label}
            </button>
          ))}
        </div>

        {/* stage */}
        {step === 0 && (
          <div style={{ display: 'grid', gap: 10 }}>
            <div style={{ padding: '10px 12px', borderRadius: 14, background: 'var(--surface-solid)', border: '1px solid var(--border)', fontSize: 13.5, lineHeight: 1.7, color: 'var(--text-2)' }}>
              <strong style={{ color: 'var(--text)' }}>📄 도서관 이용 안내</strong> — {RIGHT.map((ch) => ch.text).join(' ')}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 6 }}>
              {chunks.map((ch, i) => (
                <div
                  key={ch.id}
                  style={{
                    padding: '8px 10px',
                    borderRadius: 12,
                    fontSize: 12.5,
                    lineHeight: 1.5,
                    border: `1.5px solid ${c.series[TOPIC_COLOR[ch.topic]]}`,
                    background: withAlpha(c.series[TOPIC_COLOR[ch.topic]], dark ? 0.16 : 0.1),
                    animation: 'vz-note-in .4s cubic-bezier(.34,1.56,.64,1) both',
                    animationDelay: `${i * 40}ms`
                  }}
                >
                  <strong>[{ch.id}]</strong> {ch.text}
                </div>
              ))}
            </div>
          </div>
        )}

        {showMap && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12, alignItems: 'start' }}>
            <svg viewBox={`0 0 ${W} ${H}`} className="vz-svg" role="img" aria-label="문서 조각과 질문의 의미 지도" style={{ borderRadius: 14, background: 'var(--surface-solid)', border: '1px solid var(--border)', touchAction: 'pan-y' }}>
              <text x={12} y={18} style={{ fontSize: 11, fontWeight: 750 }}>
                의미 지도 (실제로는 수백 차원이에요)
              </text>
              {showLines &&
                topK.map((r) => (
                  <line
                    key={`l${r.ch.id}`}
                    x1={px(question.x)}
                    y1={py(question.y)}
                    x2={px(r.ch.x)}
                    y2={py(r.ch.y)}
                    stroke={r.s >= RELEVANT ? c.pos : c.neg}
                    strokeWidth={2.5}
                    strokeDasharray={r.s >= RELEVANT ? undefined : '5 4'}
                  />
                ))}
              {chunks.map((ch) => {
                const col = c.series[TOPIC_COLOR[ch.topic]];
                const picked = showLines && topK.some((r) => r.ch.id === ch.id);
                return (
                  <g key={ch.id} onPointerEnter={() => setHover(ch.id)} onPointerLeave={() => setHover(null)} style={{ cursor: 'pointer' }}>
                    <circle cx={px(ch.x)} cy={py(ch.y)} r={picked ? 11 : 8} fill={withAlpha(col, 0.35)} stroke={col} strokeWidth={2} />
                    <text x={px(ch.x)} y={py(ch.y) - 13} textAnchor="middle" style={{ fontSize: 10.5, fontWeight: 800 }}>
                      {ch.id}
                    </text>
                  </g>
                );
              })}
              {showQ && (
                <g>
                  <text x={px(question.x)} y={py(question.y) + 7} textAnchor="middle" style={{ fontSize: 22, fill: c.series[2] }}>
                    ★
                  </text>
                  <text x={px(question.x) + (question.x > 0.7 ? -12 : 12)} y={py(question.y) - 8} textAnchor={question.x > 0.7 ? 'end' : 'start'} style={{ fontSize: 11, fontWeight: 850, fill: c.series[2] }}>
                    질문
                  </text>
                </g>
              )}
            </svg>
            <div style={{ display: 'grid', gap: 8 }}>
              {hover && (
                <div style={{ padding: '8px 10px', borderRadius: 12, background: 'var(--surface-solid)', border: '1px solid var(--border)', fontSize: 13 }}>
                  <strong>[{hover}]</strong> {chunks.find((ch) => ch.id === hover)?.text}
                </div>
              )}
              {step >= 2 && (
                <div style={{ display: 'grid', gap: 6 }}>
                  <div className="vz-panel-title">질문 고르기</div>
                  {QUESTIONS.map((qq, i) => (
                    <button
                      key={qq.q}
                      type="button"
                      onClick={() => setQi(i)}
                      style={{
                        textAlign: 'left',
                        padding: '7px 10px',
                        borderRadius: 12,
                        fontSize: 13.5,
                        fontWeight: 700,
                        border: `1.5px solid ${i === qi ? c.series[2] : 'var(--border)'}`,
                        background: i === qi ? withAlpha(c.series[2], 0.12) : 'var(--surface-solid)'
                      }}
                    >
                      ★ {qq.q}
                    </button>
                  ))}
                </div>
              )}
              {step === 3 && (
                <div style={{ display: 'grid', gap: 5 }}>
                  <div className="vz-panel-title">가까운 조각 {k}개 (닮음 점수)</div>
                  {topK.map((r) => (
                    <div key={r.ch.id} style={{ display: 'grid', gridTemplateColumns: '1fr 44px', gap: 6, alignItems: 'center', fontSize: 12.5 }}>
                      <span style={{ opacity: r.s >= RELEVANT ? 1 : 0.55 }}>
                        <strong>[{r.ch.id}]</strong> {r.ch.text}
                      </span>
                      <strong style={{ textAlign: 'right', color: r.s >= RELEVANT ? c.pos : c.neg }}>{Math.round(r.s * 100)}%</strong>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {step === 4 && (
          <div style={{ display: 'grid', gap: 6, fontSize: 13.5, lineHeight: 1.6 }}>
            <PromptBlock color={c.series[4]} label="지시">
              아래 발췌만 근거로 답하고, 발췌에 없으면 "없다"고 말하세요. 답에는 발췌 번호를 붙이세요.
            </PromptBlock>
            {used.length ? (
              used.map((r) => (
                <PromptBlock key={r.ch.id} color={c.series[TOPIC_COLOR[r.ch.topic]]} label={`발췌 [${r.ch.id}]`}>
                  {r.ch.text}
                </PromptBlock>
              ))
            ) : (
              <PromptBlock color={c.neg} label="발췌">
                (관련 있는 조각을 찾지 못했어요)
              </PromptBlock>
            )}
            <PromptBlock color={c.series[2]} label="질문">
              {question.q}
            </PromptBlock>
          </div>
        )}

        {step === 5 && (
          <div style={{ display: 'grid', gap: 10 }}>
            <Bubble side="right" color={c.series[2]} who="🙋 질문">
              {question.q}
            </Bubble>
            <Bubble side="left" color={answerKind === 'hallucinate' ? c.neg : answerKind === 'partial' ? c.series[4] : c.pos} who={noRag ? '🤖 LLM (RAG 없이)' : '🤖 LLM (RAG)'}>
              {answer}
            </Bubble>
            {!noRag && answerKind === 'good' && (
              <div style={{ fontSize: 12.5, color: 'var(--text-2)' }}>
                출처: {used.map((r) => `[${r.ch.id}] ${r.ch.text}`).join(' · ')}
              </div>
            )}
          </div>
        )}

        <Note mood={note.mood} tone={note.tone}>
          {note.t}
        </Note>
      </div>

      <Controls>
        <Button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}>
          ← 이전
        </Button>
        <Button variant="primary" onClick={() => setStep((s) => Math.min(STEPS.length - 1, s + 1))} disabled={step === STEPS.length - 1}>
          다음 단계 →
        </Button>
        <Segmented
          label="조각 크기"
          options={[
            { value: 'small', label: '작게' },
            { value: 'right', label: '알맞게' },
            { value: 'large', label: '크게' }
          ]}
          value={size}
          onChange={(v) => setSize(v as SizeKey)}
        />
        <Slider label="꺼낼 조각 수 (k)" value={k} min={1} max={4} step={1} onChange={setK} format={(v) => `${v}개`} />
        <Toggle label="RAG 끄고 물어보기" checked={noRag} onChange={(v) => { setNoRag(v); if (v) setStep(5); }} />
      </Controls>
    </div>
  );
}

function PromptBlock({ color, label, children }: { color: string; label: string; children: ReactNode }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '78px 1fr', gap: 8, alignItems: 'start', padding: '8px 10px', borderRadius: 12, borderLeft: `4px solid ${color}`, background: withAlpha(color, 0.1) }}>
      <strong style={{ fontSize: 12, color: 'var(--text-2)' }}>{label}</strong>
      <span>{children}</span>
    </div>
  );
}

function Bubble({ side, color, who, children }: { side: 'left' | 'right'; color: string; who: string; children: ReactNode }) {
  return (
    <div style={{ display: 'flex', justifyContent: side === 'right' ? 'flex-end' : 'flex-start' }}>
      <div
        style={{
          maxWidth: '90%',
          padding: '10px 14px',
          borderRadius: 18,
          borderTopRightRadius: side === 'right' ? 6 : 18,
          borderTopLeftRadius: side === 'left' ? 6 : 18,
          background: withAlpha(color, 0.13),
          border: `1.5px solid ${withAlpha(color, 0.6)}`,
          fontSize: 14.5,
          lineHeight: 1.6
        }}
      >
        <div style={{ fontSize: 12, fontWeight: 800, color: 'var(--text-2)', marginBottom: 2 }}>{who}</div>
        {children}
      </div>
    </div>
  );
}
