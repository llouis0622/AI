import { useMemo, useRef, useState } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { useElementSize, useRafLoop, usePrefersReducedMotion, useInView } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';

type Mail = { text: string; spam: boolean };

/* Labeled examples the learner may study (alternating spam / normal). */
const TRAIN: Mail[] = [
  { text: '코인 수익 보장 300%', spam: true },
  { text: '과제 제출 확인', spam: false },
  { text: '수익 보장 투자 비법', spam: true },
  { text: '주간 회의 자료 공유', spam: false },
  { text: '특가 마감 지금 클릭', spam: true },
  { text: '주문하신 책 발송 완료', spam: false },
  { text: '무료 대출 지금 클릭', spam: true },
  { text: '도서관 반납 기한 알림', spam: false },
  { text: '대출 즉시 승인 무료 상담', spam: true },
  { text: '은행 대출 상환일 안내', spam: false },
  { text: '당첨 축하 쿠폰 클릭', spam: true },
  { text: '회의실 예약 변경 안내', spam: false },
  { text: '대출 한도 조회 무료', spam: true },
  { text: '동아리 MT 일정 투표', spam: false },
  { text: '무료 이벤트 당첨 안내', spam: true },
  { text: '생일 축하해 친구야', spam: false },
  { text: '당첨금 수령 링크 클릭', spam: true },
  { text: '저녁 먹고 와?', spam: false },
  { text: '광고 무료 체험 클릭', spam: true },
  { text: '택배 도착 예정 안내', spam: false }
];

/* Mail arriving in the inbox (what we grade on). */
const TEST: Mail[] = [
  { text: '무료 쿠폰 지금 클릭', spam: true },
  { text: '팀 회의 일정 공유', spam: false },
  { text: '대출 무료 상담 신청', spam: true },
  { text: '택배 배송 완료 안내', spam: false },
  { text: '투자 수익 보장 확정', spam: true },
  { text: '은행 대출 이자 안내', spam: false },
  { text: '이벤트 당첨 확인 클릭', spam: true },
  { text: '주말에 영화 볼래?', spam: false },
  { text: '특가 마감 임박 클릭', spam: true },
  { text: '과제 마감 연장 공지', spam: false }
];

/* After spammers change their tricks. */
const TRICK_TEST: Mail[] = [
  { text: 'ㄷH출 오늘만 가능', spam: true },
  { text: '공짜 선물 받아가세요', spam: true },
  { text: '적금 대박 기회', spam: true },
  { text: '시험 대박 나세요!', spam: false },
  { text: '한정 혜택 공짜 쿠폰', spam: true }
];

/* New labeled examples a person can hand to the learner. */
const TRICK_TRAIN: Mail[] = [
  { text: 'ㄷH출 한도 조회', spam: true },
  { text: '공짜 선물 이벤트', spam: true },
  { text: '적금 오늘만 특별', spam: true },
  { text: '적금 공짜 혜택', spam: true },
  { text: '대박 축하해 친구야', spam: false },
  { text: '시험 범위 확인 부탁', spam: false }
];

type Rule = { id: string; label: string; test: (t: string) => boolean; late?: boolean };
const RULES: Rule[] = [
  { id: 'loan', label: "'대출' → 스팸", test: (t) => t.includes('대출') },
  { id: 'freeclick', label: "'무료'+'클릭' → 스팸", test: (t) => t.includes('무료') && t.includes('클릭') },
  { id: 'win', label: "'당첨' → 스팸", test: (t) => t.includes('당첨') },
  { id: 'profit', label: "'수익 보장' → 스팸", test: (t) => t.includes('수익 보장') },
  { id: 'deal', label: "'특가' → 스팸", test: (t) => t.includes('특가') },
  { id: 'free2', label: "'공짜' → 스팸", test: (t) => t.includes('공짜'), late: true },
  { id: 'dh', label: "'ㄷH출' → 스팸", test: (t) => t.includes('ㄷH출'), late: true },
  { id: 'jackpot', label: "'대박' → 스팸", test: (t) => t.includes('대박'), late: true }
];

const tokens = (t: string) => t.replace(/[?!%.,~]/g, ' ').split(/\s+/).filter(Boolean);

/** Tiny naive-Bayes spam learner over words (Laplace smoothing). The example order above
 * was chosen so accuracy never drops as more examples are shown. */
function trainNB(data: Mail[]) {
  const count = { spam: new Map<string, number>(), ham: new Map<string, number>() };
  let wSpam = 0;
  let wHam = 0;
  const vocab = new Set<string>();
  for (const m of data) {
    const bag = m.spam ? count.spam : count.ham;
    for (const w of tokens(m.text)) {
      vocab.add(w);
      bag.set(w, (bag.get(w) ?? 0) + 1);
      if (m.spam) wSpam++;
      else wHam++;
    }
  }
  const V = Math.max(1, vocab.size);
  const llr = (w: string) => Math.log(((count.spam.get(w) ?? 0) + 1) / (wSpam + V)) - Math.log(((count.ham.get(w) ?? 0) + 1) / (wHam + V));
  // No class prior: an email made only of unseen words scores 0 and stays in the inbox.
  const score = (t: string) => tokens(t).filter((w) => vocab.has(w)).reduce((s, w) => s + llr(w), 0);
  const clues = [...vocab].map((w) => ({ w, v: llr(w) })).sort((a, b) => b.v - a.v);
  return { score, clues };
}

export default function RulesVsLearning({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [mode, setMode] = useState<'rules' | 'learn'>('rules');
  const [on, setOn] = useState<Record<string, boolean>>({ loan: true, freeclick: true });
  const [tricked, setTricked] = useState(false);
  const [n, setN] = useState(4);
  const [retrained, setRetrained] = useState(false);
  const [phase, setPhase] = useState(0);
  const box = useRef<HTMLDivElement>(null);
  const inView = useInView(box, '0px');
  const [sizer, { width }] = useElementSize<HTMLDivElement>();
  useRafLoop((dt) => setPhase((p) => (p + dt * 0.45) % 1), inView && !reduce);

  const mails = tricked ? [...TEST, ...TRICK_TEST] : TEST;
  const activeRules = RULES.filter((r) => on[r.id] && (!r.late || tricked));
  const model = useMemo(() => trainNB([...TRAIN.slice(0, n), ...(retrained ? TRICK_TRAIN : [])]), [n, retrained]);

  const predict = (m: Mail) => (mode === 'rules' ? activeRules.some((r) => r.test(m.text)) : model.score(m.text) > 0);
  const results = mails.map((m) => ({ m, p: predict(m) }));
  const correct = results.filter((r) => r.p === r.m.spam).length;
  const missed = results.filter((r) => r.m.spam && !r.p).length;
  const blocked = results.filter((r) => !r.m.spam && r.p).length;
  const acc = correct / results.length;

  let note: { text: string; mood: 'smile' | 'happy' | 'wow' | 'think' | 'cheer'; tone?: 'good' | 'bad' | 'warn' };
  if (mode === 'rules') {
    if (tricked && missed > 0) note = { text: `스패머가 수법을 바꿨어요! 😈 규칙에 없는 새 단어가 빠져나가요. 규칙을 더 켜 보세요 — 그런데 정상 메일까지 막히진 않을까요?`, mood: 'wow', tone: 'warn' };
    else if (blocked > 0) note = { text: `정상 메일 ${blocked}통이 스팸함에 갇혔어요. 예를 들어 은행의 '대출' 안내도 막혀 버렸죠. 규칙 하나가 다른 메일을 망가뜨려요.`, mood: 'think', tone: 'warn' };
    else if (missed > 0) note = { text: `스팸 ${missed}통이 받은편지함으로 들어왔어요. 아래에서 규칙을 하나씩 켜 보세요.`, mood: 'think' };
    else note = { text: '지금은 완벽해요! 하지만 😈 버튼을 눌러 스패머가 수법을 바꾸면 어떻게 될까요?', mood: 'happy', tone: 'good' };
  } else {
    if (tricked && !retrained && missed > 0) note = { text: '새 수법에는 기계도 처음엔 속아요. 하지만 규칙을 고칠 필요 없이 "새 예시 보여 주기"만 누르면 스스로 고쳐요!', mood: 'think', tone: 'warn' };
    else if (acc === 1) note = { text: `예시 ${n + (retrained ? TRICK_TRAIN.length : 0)}개만 보고 모두 맞혔어요! 🎉 사람이 쓴 규칙은 0개 — 단서는 기계가 데이터에서 찾았어요.`, mood: 'cheer', tone: 'good' };
    else note = { text: '예시를 더 많이 보여 줄수록 기계가 찾는 단서가 정확해져요. 슬라이더를 오른쪽으로 밀어 보세요.', mood: 'smile' };
  }

  const inbox = results.filter((r) => !r.p);
  const spambox = results.filter((r) => r.p);
  const topSpam = model.clues.slice(0, 4);
  const topHam = model.clues.slice(-4).reverse();
  const maxClue = Math.max(1, ...model.clues.map((x) => Math.abs(x.v)));

  const chip = (r: { m: Mail; p: boolean }, i: number) => {
    const ok = r.p === r.m.spam;
    return (
      <li
        key={r.m.text + i}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '6px 9px',
          borderRadius: 10,
          background: 'var(--surface-solid)',
          border: `1.5px solid ${ok ? 'var(--border)' : c.neg}`,
          boxShadow: ok ? 'none' : `0 0 0 3px ${withAlpha(c.neg, 0.15)}`,
          fontSize: 12.5,
          lineHeight: 1.35,
          transition: 'border-color .3s'
        }}
        title={r.m.spam ? '실제로는 스팸' : '실제로는 정상 메일'}
      >
        <span aria-hidden="true">{r.m.spam ? '😈' : '🙂'}</span>
        <span style={{ flex: 1, minWidth: 0 }}>{r.m.text}</span>
        <span aria-label={ok ? '맞음' : '틀림'} style={{ fontWeight: 900, color: ok ? c.pos : c.neg }}>
          {ok ? '✓' : '✗'}
        </span>
      </li>
    );
  };

  return (
    <div ref={box}>
      <div ref={sizer} />
      <FlowDiagram mode={mode} phase={phase} narrow={width > 0 && width < 520} />

      <div className="vz-pad" style={{ display: 'grid', gap: 12, paddingTop: 4 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 10 }}>
          {[
            { title: '📥 받은편지함', items: inbox },
            { title: '🚫 스팸함', items: spambox }
          ].map((col) => (
            <div key={col.title} style={{ padding: 10, borderRadius: 16, background: 'var(--surface-2)', border: '1px solid var(--border)', minHeight: expanded ? 260 : 150 }}>
              <div className="vz-panel-title" style={{ marginBottom: 8 }}>
                {col.title} · {col.items.length}통
              </div>
              <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 6 }}>{col.items.map(chip)}</ul>
            </div>
          ))}
        </div>

        <div className="vz-stats">
          <Stat label="맞힌 메일" value={`${correct}/${results.length}`} tone={acc === 1 ? 'good' : acc >= 0.8 ? 'warn' : 'bad'} />
          <Stat label="놓친 스팸" value={missed} tone={missed ? 'bad' : 'good'} />
          <Stat label="잘못 막은 메일" value={blocked} tone={blocked ? 'bad' : 'good'} />
          <Stat label="사람이 쓴 규칙" value={mode === 'rules' ? `${activeRules.length}개` : '0개'} sub={mode === 'learn' ? `대신 예시 ${n + (retrained ? TRICK_TRAIN.length : 0)}개` : undefined} />
        </div>

        {mode === 'learn' && (
          <div>
            <div className="vz-panel-title" style={{ marginBottom: 6 }}>기계가 예시에서 스스로 찾은 단서</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 8 }}>
              {[
                { title: '스팸 쪽 단서', list: topSpam, color: c.neg },
                { title: '정상 쪽 단서', list: topHam, color: c.pos }
              ].map((g) => (
                <div key={g.title} style={{ display: 'grid', gap: 4 }}>
                  <span style={{ fontSize: 12, fontWeight: 750, color: 'var(--text-3)' }}>{g.title}</span>
                  {g.list.map((k) => (
                    <div key={k.w} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
                      <span style={{ width: 64, fontWeight: 750 }}>{k.w}</span>
                      <span style={{ flex: 1, height: 10, borderRadius: 99, background: 'var(--surface-3)', overflow: 'hidden' }}>
                        <span style={{ display: 'block', height: '100%', width: `${(Math.abs(k.v) / maxClue) * 100}%`, background: g.color, borderRadius: 99, transition: 'width .4s' }} />
                      </span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}

        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        <Segmented
          label="방법 고르기"
          options={[
            { value: 'rules', label: '✍️ 규칙 직접 짜기' },
            { value: 'learn', label: '🤖 예시로 배우기' }
          ]}
          value={mode}
          onChange={(v) => setMode(v as 'rules' | 'learn')}
        />
        {mode === 'rules' ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', flex: '1 1 320px' }}>
            {RULES.filter((r) => !r.late || tricked).map((r) => (
              <Toggle key={r.id} label={<span style={{ fontWeight: 650 }}>{r.label}</span>} checked={!!on[r.id]} onChange={(v) => setOn((o) => ({ ...o, [r.id]: v }))} />
            ))}
          </div>
        ) : (
          <Slider label="보여 준 예시 (메일 + 정답)" value={n} min={2} max={TRAIN.length} step={1} onChange={setN} format={(v) => `${v}개`} />
        )}
        <Button variant={tricked ? 'ghost' : 'danger'} onClick={() => { setTricked((t) => !t); setRetrained(false); }}>
          {tricked ? '↺ 원래 스팸으로' : '😈 스패머가 수법을 바꿨다!'}
        </Button>
        {mode === 'learn' && tricked && (
          <Button variant="primary" onClick={() => setRetrained(true)} disabled={retrained}>
            📚 새 예시 보여 주기
          </Button>
        )}
      </Controls>
    </div>
  );
}

/** 규칙+데이터→답  vs  데이터+답→규칙, with dots flowing along the arrows. */
function FlowDiagram({ mode, phase, narrow }: { mode: 'rules' | 'learn'; phase: number; narrow: boolean }) {
  const c = useVizColors();
  const rules = mode === 'rules';
  const inA = rules ? { e: '📜', t: '규칙' } : { e: '✉️', t: '데이터' };
  const inB = rules ? { e: '✉️', t: '데이터' } : { e: '✅', t: '정답' };
  const mid = rules ? { e: '⚙️', t: '프로그램' } : { e: '🤖', t: '머신러닝' };
  const out = rules ? { e: '✅', t: '답' } : { e: '📜', t: '규칙(모델)' };
  const accent = rules ? c.series[3] : c.series[0];
  // geometry: wide (640) or compact phone layout (360)
  const g = narrow
    ? { W: 360, H: 150, inX: 4, inW: 92, inY: [30, 90], midX: 126, midW: 104, outX: 262, outW: 94, bh: 52, emoji: 18, font: 11.5 }
    : { W: 640, H: 150, inX: 10, inW: 110, inY: [30, 92], midX: 254, midW: 132, outX: 516, outW: 116, bh: 56, emoji: 20, font: 12.5 };
  const midY = (g.inY[0] + g.inY[1] + g.bh) / 2 - g.bh / 2;
  const cy = midY + g.bh / 2;
  const box = (x: number, y: number, w: number, item: { e: string; t: string }, strong = false) => (
    <g transform={`translate(${x} ${y})`}>
      <rect width={w} height={g.bh} rx={15} fill={strong ? withAlpha(accent, 0.16) : 'var(--surface-solid)'} stroke={strong ? accent : 'var(--border-strong)'} strokeWidth={strong ? 2.5 : 1.5} />
      <text x={w / 2} y={g.bh * 0.44} textAnchor="middle" style={{ fontSize: g.emoji }}>
        {item.e}
      </text>
      <text x={w / 2} y={g.bh * 0.82} textAnchor="middle" style={{ fontSize: g.font, fontWeight: 800, fill: 'var(--text)' }}>
        {item.t}
      </text>
    </g>
  );
  const dot = (x1: number, y1: number, x2: number, y2: number, off: number) => {
    const t = (phase + off) % 1;
    return <circle cx={x1 + (x2 - x1) * t} cy={y1 + (y2 - y1) * t} r={4} fill={accent} opacity={Math.sin(Math.PI * t)} />;
  };
  const inR = g.inX + g.inW + 3;
  const arrowEnd = g.outX - 14;
  return (
    <div style={{ padding: '14px 14px 4px' }}>
      <svg
        className="vz-svg"
        viewBox={`0 0 ${g.W} ${g.H}`}
        role="img"
        aria-label={rules ? '규칙과 데이터를 넣으면 답이 나온다' : '데이터와 정답을 넣으면 규칙이 나온다'}
        style={{ maxWidth: 640, margin: '0 auto' }}
      >
        <text x={g.inX} y={16} style={{ fontSize: g.font, fontWeight: 800, fill: accent }}>
          {rules ? '전통적 프로그래밍: 사람이 규칙을 적어 넣어요' : '머신러닝: 기계가 규칙을 찾아내요'}
        </text>
        {box(g.inX, g.inY[0], g.inW, inA)}
        {box(g.inX, g.inY[1], g.inW, inB)}
        <line x1={inR} y1={g.inY[0] + g.bh / 2} x2={g.midX - 4} y2={cy - 3} stroke="var(--border-strong)" strokeWidth={2} />
        <line x1={inR} y1={g.inY[1] + g.bh / 2} x2={g.midX - 4} y2={cy + 3} stroke="var(--border-strong)" strokeWidth={2} />
        {dot(inR, g.inY[0] + g.bh / 2, g.midX - 4, cy - 3, 0)}
        {dot(inR, g.inY[1] + g.bh / 2, g.midX - 4, cy + 3, 0.5)}
        {box(g.midX, midY, g.midW, mid, true)}
        <line x1={g.midX + g.midW + 4} y1={cy} x2={arrowEnd} y2={cy} stroke={accent} strokeWidth={2.5} />
        <path d={`M${arrowEnd} ${cy - 7} L${arrowEnd + 12} ${cy} L${arrowEnd} ${cy + 7} Z`} fill={accent} />
        {dot(g.midX + g.midW + 4, cy, arrowEnd, cy, 0.25)}
        {box(g.outX, midY, g.outW, out, !rules)}
      </svg>
    </div>
  );
}
