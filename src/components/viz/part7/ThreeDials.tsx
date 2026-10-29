import { useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note } from '../core/ui';
import { useVizColors, withAlpha } from '../core/colors';
import { usePrefersReducedMotion } from '../core/hooks';

type Dial = 'prompt' | 'rag' | 'tune';

const DIALS: { id: Dial; icon: string; name: string; what: string; cost: number }[] = [
  { id: 'prompt', icon: '💬', name: '프롬프트', what: '입력을 바꾸기', cost: 1 },
  { id: 'rag', icon: '🔎', name: 'RAG', what: '입력에 정보 더하기', cost: 2 },
  { id: 'tune', icon: '🛠️', name: '파인튜닝', what: '모델 자체를 바꾸기', cost: 3 }
];

const SCENES: { icon: string; text: string; answer: Dial; extra?: Dial; why: ReactNode }[] = [
  {
    icon: '🗣️',
    text: '답은 맞는데, 말투가 너무 딱딱하고 길어요.',
    answer: 'prompt',
    why: <>답은 이미 알고 있어요. <strong>형식과 태도</strong>만 바꾸면 되니 프롬프트로 충분해요 — 역할을 정해 주고 원하는 답 예시 몇 개(few-shot)를 보여 주세요.</>
  },
  {
    icon: '🍱',
    text: '"우리 학교 이번 주 급식 메뉴"를 물으면 모른대요.',
    answer: 'rag',
    why: <>모델이 <strong>본 적 없는 최신·내부 정보</strong>예요. 지식은 모델에 심는 게 아니라 <strong>찾아서 넣어 주는 것</strong> — RAG가 딱이에요.</>
  },
  {
    icon: '🧾',
    text: '답을 항상 똑같은 JSON 형식으로 내야 해요. 프롬프트가 너무 길어졌어요.',
    answer: 'tune',
    why: <>형식과 문체를 <strong>매번 일관되게</strong> 지키게 하려면 행동을 몸에 익히는 파인튜닝이 맞아요. LoRA 덕분에 비용도 크게 줄었어요.</>
  },
  {
    icon: '💸',
    text: '비용과 속도를 줄이고 싶어요. 딱 한 가지 일만 잘하면 돼요.',
    answer: 'tune',
    extra: 'rag',
    why: <>범용성을 조금 팔아 효율을 사요 — <strong>작은 모델을 그 일에 맞게 파인튜닝</strong>하고, 필요한 지식은 RAG로 붙여요.</>
  },
  {
    icon: '🧐',
    text: '엉뚱한 사실을 지어내면 안 되고, 출처도 보여 줘야 해요.',
    answer: 'rag',
    extra: 'prompt',
    why: <>근거 문서를 주고 프롬프트로 <strong>"근거 밖은 말하지 마"</strong>라고 하면 환각이 크게 줄고, 출처 번호도 붙일 수 있어요.</>
  },
  {
    icon: '📚',
    text: '규정집으로 파인튜닝했더니, 자꾸 틀린 조항을 대요.',
    answer: 'rag',
    why: <>파인튜닝은 <strong>지식을 넣는 데 약해요</strong>. 규정은 RAG로 찾아서 주고, 파인튜닝은 말투·형식에 쓰세요. 가장 흔한 실수예요!</>
  }
];

export default function ThreeDials(_: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const dark = c.theme === 'dark';
  const [i, setI] = useState(0);
  const [picked, setPicked] = useState<Dial | null>(null);
  const [score, setScore] = useState<boolean[]>([]);
  const finished = score.length === SCENES.length && picked !== null && i === SCENES.length - 1;
  const scene = SCENES[i];
  const correct = picked !== null && (picked === scene.answer || picked === scene.extra);

  const pick = (d: Dial, el: HTMLElement) => {
    if (picked) return;
    const ok = d === scene.answer || d === scene.extra;
    setPicked(d);
    setScore((s) => [...s, ok]);
    if (!ok && !reduce) {
      el.animate(
        [{ transform: 'translateX(0)' }, { transform: 'translateX(-7px)' }, { transform: 'translateX(7px)' }, { transform: 'translateX(-4px)' }, { transform: 'translateX(0)' }],
        { duration: 420, easing: 'ease-in-out' }
      );
    }
  };
  const next = () => {
    if (i < SCENES.length - 1) {
      setI(i + 1);
      setPicked(null);
    }
  };
  const restart = () => {
    setI(0);
    setPicked(null);
    setScore([]);
  };

  const points = score.filter(Boolean).length;

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 14 }}>
        {/* the three dials */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 10 }}>
          {DIALS.map((d) => {
            const isAnswer = picked && (d.id === scene.answer || d.id === scene.extra);
            const wrongPick = picked === d.id && !isAnswer;
            const angle = isAnswer ? 120 : picked === d.id ? 40 : -120;
            const ring = isAnswer ? c.pos : wrongPick ? c.neg : 'var(--border-strong)';
            return (
              <button
                key={d.id}
                type="button"
                onClick={(e) => pick(d.id, e.currentTarget)}
                disabled={!!picked}
                aria-label={`${d.name}: ${d.what}`}
                style={{
                  display: 'grid',
                  justifyItems: 'center',
                  gap: 4,
                  padding: '12px 6px 10px',
                  borderRadius: 18,
                  border: `2px solid ${ring}`,
                  background: isAnswer ? withAlpha(c.pos, dark ? 0.16 : 0.1) : 'var(--surface-solid)',
                  cursor: picked ? 'default' : 'pointer',
                  boxShadow: isAnswer && dark ? `0 0 18px ${withAlpha(c.pos, 0.45)}` : undefined,
                  transition: 'border-color .3s, background-color .3s'
                }}
              >
                <svg viewBox="0 0 100 100" width="76" height="76" aria-hidden="true">
                  {Array.from({ length: 9 }, (_, t) => {
                    const a = ((-120 + t * 30) * Math.PI) / 180;
                    return (
                      <line
                        key={t}
                        x1={50 + Math.sin(a) * 40}
                        y1={50 - Math.cos(a) * 40}
                        x2={50 + Math.sin(a) * 46}
                        y2={50 - Math.cos(a) * 46}
                        stroke={t <= ((angle + 120) / 30) ? (isAnswer ? c.pos : c.brand) : 'var(--viz-grid)'}
                        strokeWidth={4}
                        strokeLinecap="round"
                      />
                    );
                  })}
                  <circle cx={50} cy={50} r={32} fill="var(--surface-3)" stroke={ring} strokeWidth={2} />
                  <g style={{ transform: `rotate(${angle}deg)`, transformOrigin: '50px 50px', transition: reduce ? undefined : 'transform .7s cubic-bezier(.34,1.56,.64,1)' }}>
                    <line x1={50} y1={50} x2={50} y2={24} stroke={isAnswer ? c.pos : c.ink} strokeWidth={6} strokeLinecap="round" />
                  </g>
                  <text x={50} y={58} textAnchor="middle" style={{ fontSize: 20 }}>
                    {d.icon}
                  </text>
                </svg>
                <strong style={{ fontSize: 15 }}>{d.name}</strong>
                <span style={{ fontSize: 12, color: 'var(--text-2)', textAlign: 'center' }}>{d.what}</span>
                <span style={{ fontSize: 12 }} aria-label={`비용 ${d.cost}단계`}>
                  {'💰'.repeat(d.cost)}
                  <span style={{ opacity: 0.25 }}>{'💰'.repeat(3 - d.cost)}</span>
                </span>
              </button>
            );
          })}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12.5, fontWeight: 750, color: 'var(--text-2)' }}>
          <span>쉽고 싸요</span>
          <span style={{ flex: 1, height: 4, borderRadius: 99, background: `linear-gradient(90deg, ${c.pos}, ${c.series[4]}, ${c.neg})` }} />
          <span>어렵고 비싸요</span>
          <span style={{ marginLeft: 6 }}>→ 왼쪽부터 시도!</span>
        </div>

        {/* scenario */}
        <div
          key={i}
          style={{
            display: 'flex',
            gap: 12,
            alignItems: 'center',
            padding: '14px 16px',
            borderRadius: 18,
            background: 'var(--surface-solid)',
            border: '1px solid var(--border)',
            animation: reduce ? undefined : 'vz-note-in .4s cubic-bezier(.34,1.56,.64,1)'
          }}
        >
          <span style={{ fontSize: 36 }} aria-hidden="true">
            {scene.icon}
          </span>
          <div>
            <div style={{ fontSize: 12, fontWeight: 800, color: 'var(--text-3)' }}>
              상황 {i + 1} / {SCENES.length} · 어떤 다이얼을 돌릴까요?
            </div>
            <div style={{ fontSize: 16.5, fontWeight: 800, lineHeight: 1.5 }}>{scene.text}</div>
          </div>
        </div>

        {picked ? (
          <Note mood={correct ? 'cheer' : 'think'} tone={correct ? 'good' : 'warn'}>
            <strong>{correct ? '정답! 🎉 ' : `아쉬워요 — 정답은 ${DIALS.find((d) => d.id === scene.answer)!.name}${scene.extra ? `(+${DIALS.find((d) => d.id === scene.extra)!.name})` : ''}예요. `}</strong>
            {scene.why}
          </Note>
        ) : (
          <Note mood="smile">위의 다이얼 셋 중 하나를 눌러 보세요. 실무의 철칙은 <strong>왼쪽(싸고 쉬운 것)부터 시도</strong>하는 거예요.</Note>
        )}

        {finished && (
          <div style={{ padding: '14px 16px', borderRadius: 18, border: `2px solid ${c.pos}`, background: withAlpha(c.pos, 0.08), display: 'grid', gap: 6 }}>
            <strong style={{ fontSize: 17 }}>
              {points} / {SCENES.length} 맞힘 {points === SCENES.length ? '🏆 완벽해요!' : points >= 4 ? '👏 훌륭해요!' : '💪 다시 도전해 봐요!'}
            </strong>
            <span style={{ fontSize: 14, color: 'var(--text-2)', lineHeight: 1.65 }}>
              기억할 한 줄: <strong>지식은 RAG로, 행동과 형식은 파인튜닝으로.</strong> 셋은 함께 쓸 수 있고, 어떤 경우든 <strong>내 과제의 평가 세트로 재 보는 것</strong>이 먼저예요.
            </span>
          </div>
        )}

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }} aria-label="진행 상황">
          {SCENES.map((s, k) => (
            <span
              key={s.text}
              title={s.text}
              style={{
                width: 28,
                height: 28,
                display: 'grid',
                placeItems: 'center',
                borderRadius: 10,
                fontSize: 13,
                fontWeight: 850,
                border: `1.5px solid ${k === i ? c.brand : 'var(--border)'}`,
                background: score[k] === undefined ? 'var(--surface-solid)' : score[k] ? withAlpha(c.pos, 0.2) : withAlpha(c.neg, 0.18)
              }}
            >
              {score[k] === undefined ? k + 1 : score[k] ? '✓' : '✗'}
            </span>
          ))}
        </div>
      </div>
      <Controls>
        <Button variant="primary" onClick={next} disabled={!picked || i === SCENES.length - 1}>
          다음 상황 →
        </Button>
        <Button variant="ghost" onClick={restart}>
          ↺ 처음부터
        </Button>
      </Controls>
    </div>
  );
}
