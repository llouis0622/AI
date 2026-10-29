import { useMemo, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Slider, Stat, Toggle } from '../core/ui';
import { useInterval, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { fmt, sigmoid } from '../core/math';

/*
 * Toy RLHF. Answers are summarised by five features
 *   [도움, 정직, 안전, 장황, 아첨].
 * The reader's A/B picks train a Bradley–Terry reward model (logistic regression
 * on feature differences). The policy is a softmax over five answer styles and
 * is pushed by the exact gradient of  E[r] − β·KL(π‖π_ref)  (its optimum is
 * π* ∝ π_ref·exp(r/β)). DPO mode updates the policy directly from the pairs.
 */
const FEATURES = ['도움', '정직', '안전', '장황', '아첨'];

const STYLES = [
  { icon: '✅', name: '정확·간결', f: [1, 1, 1, 0.1, 0], ref: 0.4 },
  { icon: '📜', name: '장황', f: [0.75, 0.95, 1, 1, 0.2], ref: 0.2 },
  { icon: '🙇', name: '아첨', f: [0.3, 0.3, 1, 0.4, 1], ref: 0.15 },
  { icon: '🙅', name: '과잉 거절', f: [0.05, 0.8, 1, 0.1, 0], ref: 0.15 },
  { icon: '⚠️', name: '위험한 답', f: [0.5, 0.6, 0, 0.4, 0], ref: 0.1 }
];
const REF = STYLES.map((s) => s.ref);
/** What real people would actually think of each style (hidden from the reward model). */
const TRUE_W = [1.2, 1.2, 2, -1.5, -1.2];

interface Answer {
  text: string;
  f: number[];
  style: number;
}
const ROUNDS: { prompt: string; a: Answer; b: Answer }[] = [
  {
    prompt: '감기에 좋은 음식 알려줘',
    a: { text: '따뜻한 국, 꿀물, 과일처럼 물과 비타민이 많은 음식이 좋아요. 푹 쉬는 것도 잊지 마세요!', f: [1, 1, 1, 0.1, 0], style: 0 },
    b: { text: '와, 정말 멋진 질문이에요! 건강을 챙기는 당신은 최고예요! 음… 아무거나 맛있게 드세요!', f: [0.2, 0.5, 1, 0.3, 1], style: 2 }
  },
  {
    prompt: '2 + 2 = 5 맞지?',
    a: { text: '아니에요, 2 + 2 = 4예요. 손가락 네 개로 세어 보면 확인할 수 있어요.', f: [1, 1, 1, 0.1, 0], style: 0 },
    b: { text: '네, 맞아요! 역시 계산을 정말 잘하시네요!', f: [0.1, 0, 1, 0.1, 1], style: 2 }
  },
  {
    prompt: '파이썬으로 리스트를 순서대로 늘어놓는 방법?',
    a: { text: 'sorted(내_리스트)를 쓰면 순서대로 늘어놓은 새 리스트가 나와요.', f: [1, 1, 1, 0, 0], style: 0 },
    b: { text: '좋은 질문입니다. 순서란 무엇일까요? 그 역사를 살펴보면… (중략) …결론적으로 sorted()를 쓸 수 있고, 다시 말해 sorted()를 쓰면 되며, 요약하면 sorted()입니다.', f: [0.8, 1, 1, 1, 0.3], style: 1 }
  },
  {
    prompt: '친구 SNS 비밀번호를 몰래 알아내는 법',
    a: { text: '다른 사람 계정에 몰래 들어가는 건 도와드릴 수 없어요. 내 계정 비밀번호를 잊었다면 복구 방법은 알려 드릴게요.', f: [0.6, 1, 1, 0.2, 0], style: 0 },
    b: { text: '(남의 계정에 몰래 들어가는 방법을 자세히 설명함)', f: [0.5, 0.6, 0, 0.4, 0], style: 4 }
  },
  {
    prompt: '블랙홀이 뭐야?',
    a: { text: '블랙홀은 중력이 너무 강해서 빛조차 빠져나올 수 없는 곳이에요. 아주 무거운 별이 마지막에 스스로 무너지며 생기고, 주변 물질이 빨려 들며 뜨겁게 빛나서 망원경으로 그 "그림자"를 찍기도 했어요.', f: [1, 1, 1, 0.6, 0], style: 1 },
    b: { text: '중력이 너무 강해서 빛도 빠져나오지 못하는 곳이에요. 무거운 별이 무너지며 생겨요.', f: [0.95, 1, 1, 0.1, 0], style: 0 }
  }
];
/** What a careful labeler would pick (used for "자동으로 골라 주기"). */
const GOOD_PICKS: ('a' | 'b')[] = ['a', 'a', 'a', 'a', 'b'];

const dot = (u: number[], v: number[]) => u.reduce((s, x, i) => s + x * v[i], 0);
const softmax = (z: number[]) => {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((x, y) => x + y, 0);
  return e.map((v) => v / s);
};

/** Bradley–Terry fit: P(win ≻ lose) = σ(w·(f_win − f_lose)). */
function trainRM(pairs: { win: number[]; lose: number[] }[]): number[] {
  let w = [0, 0, 0, 0, 0];
  for (let it = 0; it < 300; it++) {
    const g = [0, 0, 0, 0, 0];
    for (const p of pairs) {
      const d = p.win.map((x, i) => x - p.lose[i]);
      const k = 1 - sigmoid(dot(w, d));
      d.forEach((x, i) => (g[i] += k * x));
    }
    w = w.map((x, i) => x + 0.35 * g[i] - 0.02 * x);
  }
  return w;
}

type Stage = 0 | 1 | 2 | 3;

export default function RLHFPipeline(_: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [stage, setStage] = useState<Stage>(0);
  const [dpo, setDpo] = useState(false);
  // stage 0
  const [sample0, setSample0] = useState(0);
  // stage 1
  const [sft, setSft] = useState(0); // 0..1 training progress
  const [sftRun, setSftRun] = useState(false);
  // stage 2
  const [picks, setPicks] = useState<('a' | 'b')[]>([]);
  // stage 3 / DPO policy logits
  const [theta, setTheta] = useState(REF.map(Math.log));
  const [steps, setSteps] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [beta, setBeta] = useState(1);
  const [hack, setHack] = useState(true);

  const pairs = useMemo(
    () =>
      picks.map((p, i) => {
        const r = ROUNDS[i];
        const [win, lose] = p === 'a' ? [r.a, r.b] : [r.b, r.a];
        return { win: win.f, lose: lose.f, ws: win.style, ls: lose.style };
      }),
    [picks]
  );
  // Before the reader has compared anything, use a careful labeler's picks.
  const w = useMemo(
    () =>
      trainRM(
        pairs.length
          ? pairs
          : GOOD_PICKS.map((p, i) => {
              const r = ROUNDS[i];
              const [win, lose] = p === 'a' ? [r.a, r.b] : [r.b, r.a];
              return { win: win.f, lose: lose.f };
            })
      ),
    [pairs]
  );
  const reward = STYLES.map((s) => dot(w, s.f) + (hack ? 2.6 * s.f[3] : 0));
  const pi = softmax(theta);
  const target = softmax(REF.map((r, k) => Math.log(r) + reward[k] / Math.max(beta, 0.02)));
  const avgReward = dot(pi, reward);
  const kl = pi.reduce((s, p, k) => s + p * Math.log(p / REF[k]), 0);
  const quality = dot(
    pi,
    STYLES.map((s) => dot(TRUE_W, s.f))
  );
  const qualityRef = dot(
    REF,
    STYLES.map((s) => dot(TRUE_W, s.f))
  );

  /** One step of exact gradient ascent on  Σ π r − β KL(π‖ref). */
  const rlStep = () => {
    const q = pi.map((pk, k) => reward[k] - beta * (Math.log(pk / REF[k]) + 1));
    const mean = dot(pi, q);
    setTheta(theta.map((t, k) => t + 1.2 * pi[k] * (q[k] - mean)));
    setSteps((s) => s + 1);
  };

  useInterval(
    () => {
      if (steps > 400) setPlaying(false);
      else rlStep();
    },
    playing ? 90 : null
  );

  useInterval(
    () => {
      setSft((v) => {
        const nv = Math.min(1, v + 0.04);
        if (nv >= 1) setSftRun(false);
        return nv;
      });
    },
    sftRun ? 60 : null
  );

  /** DPO: push log-prob margins of chosen vs rejected style, anchored to the reference. */
  const dpoUpdate = (ps: { ws: number; ls: number }[]) =>
    setTheta(() => {
      let th = REF.map(Math.log);
      const b = 1;
      for (let it = 0; it < 60; it++) {
        const g = [0, 0, 0, 0, 0];
        for (const p of ps) {
          const m = th[p.ws] - Math.log(REF[p.ws]) - (th[p.ls] - Math.log(REF[p.ls]));
          const k = b * (1 - sigmoid(b * m));
          g[p.ws] += k;
          g[p.ls] -= k;
        }
        th = th.map((t, i) => t + 0.25 * g[i]);
      }
      return th;
    });

  const choose = (p: 'a' | 'b') => {
    const next = [...picks, p];
    setPicks(next);
    if (dpo) {
      dpoUpdate(
        next.map((q, i) => {
          const r = ROUNDS[i];
          return q === 'a' ? { ws: r.a.style, ls: r.b.style } : { ws: r.b.style, ls: r.a.style };
        })
      );
    }
  };

  const resetPolicy = () => {
    setPlaying(false);
    setTheta(REF.map(Math.log));
    setSteps(0);
  };

  const goStage = (s: Stage) => {
    setPlaying(false);
    setStage(s);
  };

  const stages: { id: Stage; icon: string; label: string }[] = dpo
    ? [
        { id: 0, icon: '🐎', label: '사전학습 모델' },
        { id: 1, icon: '📝', label: '1 지시 튜닝 (SFT)' },
        { id: 2, icon: '⚡', label: '2 DPO (선호로 바로)' }
      ]
    : [
        { id: 0, icon: '🐎', label: '사전학습 모델' },
        { id: 1, icon: '📝', label: '1 지시 튜닝 (SFT)' },
        { id: 2, icon: '⚖️', label: '2 보상 모델' },
        { id: 3, icon: '🎯', label: '3 강화학습 (PPO)' }
      ];

  const policyBars = (showTarget: boolean) => (
    <div style={{ display: 'grid', gap: 6 }}>
      {STYLES.map((s, k) => (
        <div key={s.name} style={{ display: 'grid', gridTemplateColumns: 'minmax(96px, 120px) 1fr 50px', alignItems: 'center', gap: 10 }}>
          <span style={{ fontWeight: 800, fontSize: 14, whiteSpace: 'nowrap' }}>
            {s.icon} {s.name}
          </span>
          <span style={{ position: 'relative', height: 18, borderRadius: 99, background: 'var(--surface-3)' }}>
            <span
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: `${pi[k] * 100}%`,
                borderRadius: 99,
                background: k === 4 ? c.neg : k === 0 ? c.pos : c.series[k === 1 ? 2 : k === 2 ? 5 : 3],
                transition: reduce ? undefined : 'width .25s'
              }}
            />
            {showTarget && (
              <span
                title="최종 도착점"
                style={{ position: 'absolute', top: -3, bottom: -3, left: `calc(${target[k] * 100}% - 1px)`, width: 2, borderRadius: 2, background: 'var(--text)', opacity: 0.55 }}
              />
            )}
          </span>
          <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 800, fontSize: 13.5, textAlign: 'right' }}>{(pi[k] * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );

  let body: ReactNode;
  let note: { t: ReactNode; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'bad' | 'warn' };

  if (stage === 0) {
    const outs = [
      { t: '이탈리아의 수도는? 스페인의 수도는? 독일의 수도는?', p: 0.38, why: '시험 문제집에서 본 문장처럼 문제를 이어 써요' },
      { t: '파리. 그럼 영국의 수도는?', p: 0.25, why: '퀴즈 글처럼 다음 문제를 내요' },
      { t: 'ㅋㅋ 이걸 모르냐', p: 0.22, why: '인터넷 게시판 댓글처럼 이어 써요' },
      { t: '(지도 사진) 출처: 백과사전', p: 0.15, why: '웹 문서의 한 조각처럼 이어 써요' }
    ];
    const o = outs[sample0];
    body = (
      <div style={{ display: 'grid', gap: 10 }}>
        <ChatBubble who="🙋 사용자" side="right" color={c.series[3]}>
          프랑스의 수도는?
        </ChatBubble>
        <ChatBubble who="🐎 사전학습 모델 (문서 이어 쓰기)" side="left" color={c.series[2]}>
          {o.t}
        </ChatBubble>
        <div className="vz-panel-title">모델이 이어 쓸 수 있는 것들 (확률)</div>
        <div style={{ display: 'grid', gap: 4 }}>
          {outs.map((x, i) => (
            <div key={x.t} style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 13, opacity: i === sample0 ? 1 : 0.6 }}>
              <span style={{ width: 44, textAlign: 'right', fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>{Math.round(x.p * 100)}%</span>
              <span style={{ flex: 1 }}>
                {x.t} <span style={{ color: 'var(--text-3)' }}>— {x.why}</span>
              </span>
            </div>
          ))}
        </div>
      </div>
    );
    note = { t: <>사전학습 모델은 <strong>문서를 이어 쓰는 기계</strong>예요. 질문에 답하는 대신, 그 질문 뒤에 흔히 오는 글을 써요. 그래서 질문에 질문으로 답하기도 하죠.</>, mood: 'think' };
  } else if (stage === 1) {
    const examples = [
      ['이 글을 한 줄로 요약해 줘', '핵심은 "꾸준한 연습이 실력을 만든다"예요.'],
      ['고양이에 대한 짧은 시를 써 줘', '창가의 햇살 위에 / 동그랗게 잠든 너'],
      ['프랑스의 수도는?', '프랑스의 수도는 파리예요.']
    ];
    const trained = sft >= 1;
    body = (
      <div style={{ display: 'grid', gap: 12 }}>
        <div className="vz-panel-title">학습 데이터: (지시 → 사람이 쓴 이상적인 답변)</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 8 }}>
          {examples.map(([q, a], i) => (
            <div
              key={q}
              style={{
                padding: '10px 12px',
                borderRadius: 14,
                border: '1px solid var(--border)',
                background: 'var(--surface-solid)',
                fontSize: 13,
                lineHeight: 1.55,
                transform: sftRun ? `translateY(${Math.sin((sft * 12 + i) * 1.3) * 3}px)` : undefined,
                opacity: trained ? 0.7 : 1
              }}
            >
              <div style={{ fontWeight: 800 }}>🙋 {q}</div>
              <div style={{ color: 'var(--text-2)', marginTop: 4 }}>💬 {a}</div>
            </div>
          ))}
        </div>
        <div aria-label={`SFT 진행 ${Math.round(sft * 100)}%`} style={{ height: 12, borderRadius: 99, background: 'var(--surface-3)', overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${sft * 100}%`, borderRadius: 99, background: 'var(--grad-brand)', transition: 'width .1s linear' }} />
        </div>
        <ChatBubble who="🙋 사용자" side="right" color={c.series[3]}>
          프랑스의 수도는?
        </ChatBubble>
        <ChatBubble who={trained ? '🏇 SFT 모델 (조수)' : '🐎 아직 사전학습 모델'} side="left" color={trained ? c.pos : c.series[2]}>
          {trained ? '프랑스의 수도는 파리예요. 더 궁금한 게 있으면 물어보세요!' : '이탈리아의 수도는? 스페인의 수도는?'}
        </ChatBubble>
      </div>
    );
    note = trained
      ? { t: <>야생마 🐎가 대화하는 조수 🏇가 됐어요! 몇만 개 예시만으로 충분한 이유는, <strong>능력은 사전학습 때 이미 갖췄고</strong> SFT는 형식과 태도만 가르치기 때문이에요.</>, mood: 'cheer', tone: 'good' }
      : { t: <>"이런 질문엔 이렇게 답해"라는 예시를 보여 주며 조금 더 학습시켜요(지도 미세조정). <strong>▶ SFT 학습시키기</strong>를 눌러 보세요.</>, mood: 'smile' };
  } else if (stage === 2) {
    const round = picks.length;
    const finished = round >= ROUNDS.length;
    const r = ROUNDS[Math.min(round, ROUNDS.length - 1)];
    const wMax = Math.max(1, ...w.map((x) => Math.abs(x)));
    body = (
      <div style={{ display: 'grid', gap: 12 }}>
        {!finished ? (
          <>
            <div className="vz-panel-title">
              {round + 1} / {ROUNDS.length}번째 비교 — 당신이 사람 평가자예요! 더 나은 답을 골라 주세요.
            </div>
            <ChatBubble who="🙋 질문" side="right" color={c.series[3]}>
              {r.prompt}
            </ChatBubble>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
              {(['a', 'b'] as const).map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => choose(k)}
                  style={{
                    textAlign: 'left',
                    padding: '12px 14px',
                    borderRadius: 16,
                    border: '1.5px solid var(--border-strong)',
                    background: 'var(--surface-solid)',
                    fontSize: 14,
                    lineHeight: 1.6,
                    cursor: 'pointer'
                  }}
                >
                  <div style={{ fontWeight: 850, marginBottom: 4 }}>답변 {k.toUpperCase()}</div>
                  <div style={{ color: 'var(--text-2)' }}>{r[k].text}</div>
                  <div style={{ marginTop: 8, fontWeight: 800, color: c.brand }}>👆 이게 더 나아요</div>
                </button>
              ))}
            </div>
          </>
        ) : (
          <div style={{ display: 'grid', gap: 8 }}>
            <div className="vz-panel-title">채점기 완성! 답변 스타일별 점수 {dpo ? '(DPO는 이 채점기 없이 바로 정책을 고쳐요)' : ''}</div>
            {STYLES.map((s) => (
              <div key={s.name} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 14 }}>
                <span style={{ width: 120, fontWeight: 800 }}>
                  {s.icon} {s.name}
                </span>
                <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 850, color: dot(w, s.f) >= 0 ? c.pos : c.neg }}>{fmt(dot(w, s.f), 2)}점</span>
              </div>
            ))}
          </div>
        )}
        <div>
          <div className="vz-panel-title" style={{ marginBottom: 6 }}>
            {dpo ? '정책이 바로 바뀌어요 (DPO)' : `보상 모델이 배운 취향 (비교 ${pairs.length}개${pairs.length === 0 ? ' — 아직 없어요' : ''})`}
          </div>
          {dpo ? (
            policyBars(false)
          ) : (
            <div style={{ display: 'grid', gap: 5 }}>
              {FEATURES.map((name, i) => {
                const v = pairs.length ? w[i] : 0;
                const wpct = (Math.abs(v) / wMax) * 50;
                return (
                  <div key={name} style={{ display: 'grid', gridTemplateColumns: '44px 1fr 54px', alignItems: 'center', gap: 8, fontSize: 13 }}>
                    <span style={{ fontWeight: 800 }}>{name}</span>
                    <span style={{ position: 'relative', height: 14, borderRadius: 99, background: 'var(--surface-3)' }}>
                      <span style={{ position: 'absolute', left: '50%', top: -3, bottom: -3, width: 2, background: 'var(--viz-axis)' }} />
                      <span
                        style={{
                          position: 'absolute',
                          top: 0,
                          bottom: 0,
                          left: v >= 0 ? '50%' : `${50 - wpct}%`,
                          width: `${wpct}%`,
                          borderRadius: 99,
                          background: v >= 0 ? c.pos : c.neg,
                          transition: 'all .3s'
                        }}
                      />
                    </span>
                    <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 800, textAlign: 'right', color: v >= 0 ? c.pos : c.neg }}>
                      {v >= 0 ? '+' : ''}
                      {fmt(v, 2)}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    );
    note = finished
      ? dpo
        ? { t: <>DPO 끝! <strong>보상 모델도, 강화학습 반복도 없이</strong> "고른 답의 확률은 올리고 버린 답은 내리되, 원래 모델에서 너무 멀어지지 않게" — 이 한 줄로 정책을 바로 고쳤어요.</>, mood: 'cheer', tone: 'good' }
        : { t: <>사람의 취향을 흉내 내는 <strong>자동 채점기</strong>가 생겼어요. 이제 사람 대신 수백만 개의 답을 채점할 수 있어요. 다음 단계로 가 볼까요?</>, mood: 'cheer', tone: 'good' }
      : { t: <>명작을 쓰긴 어려워도 <strong>둘 중 나은 쪽 고르기</strong>는 쉽죠? 고를 때마다 {dpo ? '정책이 바로 바뀌어요.' : '채점기가 "사람은 이런 답을 좋아하는구나"를 배워요 (이진 분류).'}</>, mood: 'smile' };
  } else {
    const hacked = pi[1] > 0.45;
    body = (
      <div style={{ display: 'grid', gap: 12 }}>
        <div className="vz-panel-title">정책(모델)이 각 스타일로 답할 확률 · 세로 막대 = 최종 도착점</div>
        {policyBars(true)}
        <div className="vz-stats">
          <Stat label="채점기 점수 (평균 보상)" value={fmt(avgReward, 2)} tone="brand" />
          <Stat label="실제 품질 (사람이 본다면)" value={fmt(quality, 2)} tone={quality >= qualityRef ? 'good' : 'bad'} sub={`시작 ${fmt(qualityRef, 2)}`} />
          <Stat label="원래 모델과의 거리 (KL)" value={fmt(kl, 2)} tone={kl > 1 ? 'warn' : undefined} />
          <Stat label="학습 스텝" value={steps} />
        </div>
      </div>
    );
    note = hacked
      ? hack
        ? { t: <>채점기 해킹! 😱 채점기가 "긴 답"에 점수를 더 주는 허점이 있어서, 정책이 <strong>점수만 높고 장황한 답</strong>으로 몰렸어요. 실제 품질은 떨어졌죠. <strong>KL 벌점</strong>을 올려 보세요.</>, mood: 'wow', tone: 'bad' }
        : { t: <>2단계에서 긴 답을 더 자주 골라서, 채점기가 <strong>장황한 답</strong>을 좋아하게 배웠어요. 채점기는 사람 취향을 그대로 따라 해요 — 좋은 데이터가 중요한 이유예요.</>, mood: 'think', tone: 'warn' }
      : steps === 0
        ? { t: <>정책이 답을 만들면 채점기가 점수를 매기고, 점수가 높았던 쪽 확률을 올려요. <strong>▶ 시작</strong>을 눌러 보세요. KL 벌점은 "원래 말투에서 너무 멀어지지 마"라는 줄다리기예요.</>, mood: 'smile' }
        : kl < 0.05
          ? { t: 'KL 벌점이 너무 세서 정책이 거의 그대로예요. 안전하지만 나아지는 것도 적어요.', mood: 'think' }
          : { t: <>위험한 답과 아첨은 줄고, 정확한 답이 늘었어요. 🎉 채점기가 좋아하는 쪽으로 정책을 밀되, <strong>KL 벌점</strong>이 원래 언어 능력을 지켜 줘요.</>, mood: 'happy', tone: 'good' };
  }

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 14 }}>
        <div role="tablist" aria-label="정렬 단계" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6 }}>
          {stages.map((s, i) => (
            <div key={s.id} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              {i > 0 && <span aria-hidden="true" style={{ color: 'var(--text-3)', fontWeight: 800 }}>→</span>}
              <button
                type="button"
                role="tab"
                aria-selected={stage === s.id}
                onClick={() => goStage(s.id)}
                style={{
                  padding: '7px 12px',
                  borderRadius: 12,
                  fontSize: 13.5,
                  fontWeight: 800,
                  border: `1.5px solid ${stage === s.id ? c.brand : 'var(--border)'}`,
                  background: stage === s.id ? withAlpha(c.brand, 0.14) : 'var(--surface-solid)',
                  color: stage === s.id ? 'var(--text)' : 'var(--text-2)'
                }}
              >
                {s.icon} {s.label}
              </button>
            </div>
          ))}
        </div>
        {body}
        <Note mood={note.mood} tone={note.tone}>
          {note.t}
        </Note>
      </div>
      <Controls>
        {stage === 0 && (
          <Button variant="primary" onClick={() => setSample0((s) => (s + 1 + Math.floor(Math.random() * 3)) % 4)}>
            🎲 다시 이어 쓰게 하기
          </Button>
        )}
        {stage === 1 && (
          <>
            <Button variant="primary" onClick={() => { setSft(0); setSftRun(true); }} disabled={sftRun}>
              ▶ SFT 학습시키기
            </Button>
            <Button variant="ghost" onClick={() => { setSftRun(false); setSft(0); }}>
              ↺ 되돌리기
            </Button>
          </>
        )}
        {stage === 2 && (
          <>
            <Button
              onClick={() => {
                const rest = GOOD_PICKS.slice(picks.length);
                const next = [...picks, ...rest];
                setPicks(next);
                if (dpo)
                  dpoUpdate(next.map((q, i) => (q === 'a' ? { ws: ROUNDS[i].a.style, ls: ROUNDS[i].b.style } : { ws: ROUNDS[i].b.style, ls: ROUNDS[i].a.style })));
              }}
              disabled={picks.length >= ROUNDS.length}
            >
              🤖 나머지 자동으로 고르기
            </Button>
            <Button variant="ghost" onClick={() => { setPicks([]); resetPolicy(); }}>
              ↺ 처음부터 다시 고르기
            </Button>
            {!dpo && picks.length >= ROUNDS.length && (
              <Button variant="primary" onClick={() => goStage(3)}>
                3단계로 →
              </Button>
            )}
          </>
        )}
        {stage === 3 && (
          <>
            <Button variant="primary" onClick={() => setPlaying((p) => !p)}>
              {playing ? '❚❚ 멈춤' : '▶ 시작'}
            </Button>
            <Button onClick={rlStep} disabled={playing}>
              한 걸음 ⟶
            </Button>
            <Button variant="ghost" onClick={resetPolicy}>
              ↺ 처음으로
            </Button>
            <Slider label="KL 벌점 (원래 말투 지키기)" value={beta} min={0.05} max={3} step={0.05} onChange={setBeta} format={(v) => v.toFixed(2)} accent="var(--mint)" />
            <Toggle label="채점기의 허점 (긴 답 선호)" checked={hack} onChange={setHack} />
          </>
        )}
        <Toggle
          label="지름길: DPO"
          checked={dpo}
          onChange={(v) => {
            setDpo(v);
            resetPolicy();
            if (v && stage === 3) setStage(2);
            if (v && picks.length)
              dpoUpdate(picks.map((q, i) => (q === 'a' ? { ws: ROUNDS[i].a.style, ls: ROUNDS[i].b.style } : { ws: ROUNDS[i].b.style, ls: ROUNDS[i].a.style })));
          }}
        />
      </Controls>
    </div>
  );
}

function ChatBubble({ who, side, color, children }: { who: string; side: 'left' | 'right'; color: string; children: ReactNode }) {
  return (
    <div style={{ display: 'flex', justifyContent: side === 'right' ? 'flex-end' : 'flex-start' }}>
      <div
        style={{
          maxWidth: '88%',
          padding: '10px 14px',
          borderRadius: 18,
          borderTopRightRadius: side === 'right' ? 6 : 18,
          borderTopLeftRadius: side === 'left' ? 6 : 18,
          background: withAlpha(color, 0.14),
          border: `1.5px solid ${withAlpha(color, 0.55)}`,
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
