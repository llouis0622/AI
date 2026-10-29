import { useEffect, useMemo, useRef, useState } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { useVizColors, withAlpha } from '../core/colors';
import { fmt } from '../core/math';
import { usePrefersReducedMotion } from '../core/hooks';

/*
 * A tiny word-level n-gram language model trained on 40 sentences.
 * It does exactly what GPT does at the interface level — output a probability
 * for every possible next token — just with counting instead of a Transformer.
 */
const CORPUS = [
  '오늘 날씨가 정말 좋다 .',
  '오늘 날씨가 너무 덥다 .',
  '오늘 날씨가 흐리고 비가 온다 .',
  '오늘 날씨가 맑아서 기분이 좋다 .',
  '내일 날씨가 맑으면 소풍을 간다 .',
  '비가 오면 우산을 챙긴다 .',
  '비가 그치면 무지개가 뜬다 .',
  '비가 와서 소풍이 취소됐다 .',
  '우리 강아지는 산책을 정말 좋아한다 .',
  '우리 강아지는 간식을 좋아한다 .',
  '우리 강아지는 공놀이를 좋아한다 .',
  '우리 고양이는 낮잠을 좋아한다 .',
  '우리 고양이는 상자를 정말 좋아한다 .',
  '강아지가 공을 물고 달려온다 .',
  '고양이가 창밖을 바라본다 .',
  '점심으로 김밥을 먹었다 .',
  '점심으로 라면을 먹었다 .',
  '점심으로 떡볶이를 먹었다 .',
  '점심으로 친구와 김밥을 먹었다 .',
  '친구와 함께 떡볶이를 먹었다 .',
  '나는 떡볶이를 정말 좋아한다 .',
  '나는 점심 시간이 제일 좋다 .',
  '나는 수학 시간이 정말 좋다 .',
  '달에 처음 발을 디딘 사람은 암스트롱이다 .',
  '달에 처음 간 우주선은 아폴로 11호다 .',
  '달은 지구 주위를 돈다 .',
  '지구는 태양 주위를 돈다 .',
  '태양은 아주 뜨거운 별이다 .',
  '인공지능은 데이터로 배운다 .',
  '인공지능은 다음 단어를 맞힌다 .',
  '인공지능은 그림도 그린다 .',
  '언어 모델은 다음 단어를 예측한다 .',
  '모델은 데이터로 배운다 .',
  '우산을 챙기면 비가 와도 괜찮다 .',
  '소풍을 가면 김밥을 먹는다 .',
  '친구와 함께 소풍을 간다 .',
  '기분이 좋으면 노래를 부른다 .',
  '무지개가 뜨면 기분이 좋다 .',
  '간식을 먹으면 기분이 좋다 .',
  '산책을 하면 기분이 좋다 .'
];
const END = '.';
const MAX_NEW = 12;
const TOP = 6;

const PROMPTS = ['오늘 날씨가', '우리 강아지는', '점심으로', '달에 처음', '인공지능은', '비가'];

type Table = Map<string, Map<string, number>>;

function buildModel() {
  const tri: Table = new Map();
  const bi: Table = new Map();
  const uni = new Map<string, number>();
  const add = (t: Table, ctx: string, w: string) => {
    const m = t.get(ctx) ?? new Map<string, number>();
    m.set(w, (m.get(w) ?? 0) + 1);
    t.set(ctx, m);
  };
  for (const s of CORPUS) {
    const t = s.split(' ');
    t.forEach((w, i) => {
      if (w !== END) uni.set(w, (uni.get(w) ?? 0) + 1);
      if (i >= 1) add(bi, t[i - 1], w);
      if (i >= 2) add(tri, `${t[i - 2]} ${t[i - 1]}`, w);
    });
  }
  return { tri, bi, uni };
}

const MODEL = buildModel();

type Source = 'trigram' | 'bigram' | 'unknown';

/** P(next | last two tokens), backing off to the last token, then to word frequency. */
function nextDist(tokens: string[]): { dist: [string, number][]; source: Source } {
  const a = tokens[tokens.length - 2];
  const b = tokens[tokens.length - 1];
  const p = new Map<string, number>();
  const mix = (m: Map<string, number>, weight: number) => {
    let tot = 0;
    m.forEach((v) => (tot += v));
    m.forEach((v, k) => p.set(k, (p.get(k) ?? 0) + (weight * v) / tot));
  };
  const triM = a !== undefined ? MODEL.tri.get(`${a} ${b}`) : undefined;
  const biM = b !== undefined ? MODEL.bi.get(b) : undefined;
  let source: Source;
  if (triM && biM) {
    mix(triM, 0.8);
    mix(biM, 0.2);
    source = 'trigram';
  } else if (biM) {
    mix(biM, 1);
    source = 'bigram';
  } else {
    mix(MODEL.uni, 1);
    source = 'unknown';
  }
  return { dist: [...p.entries()].sort((x, y) => y[1] - x[1]), source };
}

function withTemperature(dist: [string, number][], T: number): [string, number][] {
  const w = dist.map(([k, v]) => [k, Math.pow(v, 1 / Math.max(T, 0.05))] as [string, number]);
  const s = w.reduce((acc, [, v]) => acc + v, 0);
  return w.map(([k, v]) => [k, v / s]);
}

function sample(dist: [string, number][]): number {
  let r = Math.random();
  for (let i = 0; i < dist.length; i++) {
    r -= dist[i][1];
    if (r <= 0) return i;
  }
  return dist.length - 1;
}

export default function NextToken(_: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [prompt, setPrompt] = useState(PROMPTS[0]);
  const [tokens, setTokens] = useState<string[]>(PROMPTS[0].split(' '));
  const [promptLen, setPromptLen] = useState(PROMPTS[0].split(' ').length);
  const [T, setT] = useState(1);
  const [greedy, setGreedy] = useState(false);
  const [surprises, setSurprises] = useState<number[]>([]);
  const [rolling, setRolling] = useState<number | null>(null);
  const [auto, setAuto] = useState(false);
  const [showData, setShowData] = useState(false);
  const [custom, setCustom] = useState('');
  const timers = useRef<number[]>([]);

  const done = tokens[tokens.length - 1] === END || tokens.length - promptLen >= MAX_NEW;
  const { dist: raw, source } = useMemo(() => nextDist(tokens), [tokens]);
  const dist = useMemo(() => withTemperature(raw, T), [raw, T]);
  const shown = dist.slice(0, TOP);
  const restCount = dist.length - shown.length;
  const restMass = dist.slice(TOP).reduce((a, [, v]) => a + v, 0);

  const clearTimers = () => {
    timers.current.forEach((t) => window.clearTimeout(t));
    timers.current = [];
  };
  useEffect(() => clearTimers, []);

  const reset = (text: string) => {
    clearTimers();
    setAuto(false);
    setRolling(null);
    const t = text.trim().split(/\s+/).filter(Boolean);
    const safe = t.length ? t : PROMPTS[0].split(' ');
    setTokens(safe);
    setPromptLen(safe.length);
    setSurprises([]);
  };

  const append = (idx: number) => {
    const [word] = dist[idx];
    const pRaw = raw.find(([k]) => k === word)?.[1] ?? 1e-9;
    setSurprises((s) => [...s, -Math.log(pRaw)]);
    setTokens((t) => [...t, word]);
    setRolling(null);
  };

  /** Pick the next token — with a short "roulette" so the randomness is visible. */
  const generateOne = () => {
    if (done || rolling !== null) return;
    const pick = greedy ? 0 : sample(dist);
    if (reduce) {
      append(pick);
      return;
    }
    const hops = Math.min(shown.length, 6) * 2 + (pick < TOP ? pick : TOP - 1);
    for (let h = 0; h <= hops; h++) {
      timers.current.push(
        window.setTimeout(() => setRolling(h % Math.max(1, Math.min(shown.length, 6))), h * 45)
      );
    }
    timers.current.push(window.setTimeout(() => append(pick), hops * 45 + 260));
  };

  // auto-write until the sentence ends
  useEffect(() => {
    if (!auto) return;
    if (done) {
      setAuto(false);
      return;
    }
    if (rolling !== null) return;
    const id = window.setTimeout(generateOne, 380);
    return () => window.clearTimeout(id);
  }); // eslint-disable-line react-hooks/exhaustive-deps

  const avgSurprise = surprises.length ? surprises.reduce((a, b) => a + b, 0) / surprises.length : 0;
  const lastSurprise = surprises[surprises.length - 1];
  const dark = c.theme === 'dark';

  let note: { text: React.ReactNode; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' | 'bad' };
  if (source === 'unknown' && !done)
    note = {
      text: <>"{tokens[tokens.length - 1]}" 다음에 오는 단어는 훈련 문장에서 본 적이 없어요. 그래서 아무 단어나 자주 나온 순서로 골라요. <strong>데이터에 없는 건 잘 못해요</strong> — 데이터가 곧 능력의 지도예요!</>,
      mood: 'think',
      tone: 'warn'
    };
  else if (done && tokens[tokens.length - 1] === END)
    note = {
      text: <>문장 완성! 🎉 한 단어씩 "다음 단어의 확률"을 보고 뽑았을 뿐인데 그럴듯한 문장이 됐죠? 평균 놀람(손실)은 <strong>{fmt(avgSurprise, 2)}</strong>예요. 사전학습은 진짜 문장에서 이 놀람을 줄이는 훈련이에요.</>,
      mood: 'cheer',
      tone: 'good'
    };
  else if (T < 0.45)
    note = { text: '온도가 낮으면 1등 단어만 거의 독차지해요. 늘 똑같은 문장이 나오기 쉬워요.', mood: 'think' };
  else if (T > 1.6)
    note = { text: '온도가 높으면 막대가 평평해져서 엉뚱한 단어도 자주 뽑혀요. 창의적이지만 엉망이 되기 쉬워요!', mood: 'wow', tone: 'warn' };
  else if (lastSurprise !== undefined && lastSurprise > 1.6)
    note = { text: `방금 단어는 확률이 낮았어요 (놀람 ${fmt(lastSurprise, 2)}). 모델 입장에선 "의외"였던 셈이에요.`, mood: 'wow' };
  else
    note = {
      text: <>막대가 모델이 매긴 <strong>다음 단어의 확률</strong>이에요. 🎲를 누르면 확률대로 하나를 뽑고, 막대를 직접 눌러 골라도 돼요.</>,
      mood: 'smile'
    };

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 14 }}>
        {/* the sentence so far */}
        <div
          aria-live="polite"
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 6,
            alignItems: 'center',
            minHeight: 64,
            padding: '14px 14px',
            borderRadius: 18,
            background: 'var(--surface-solid)',
            border: '1px solid var(--border)'
          }}
        >
          {tokens.map((t, i) => {
            const gen = i >= promptLen;
            const isEnd = t === END;
            return (
              <span
                key={`${i}-${t}`}
                style={{
                  padding: '5px 11px',
                  borderRadius: 11,
                  fontSize: 16,
                  fontWeight: gen ? 800 : 650,
                  background: isEnd ? 'var(--surface-3)' : gen ? withAlpha(c.series[0], dark ? 0.22 : 0.14) : 'var(--surface-3)',
                  color: 'var(--text)',
                  border: `1.5px solid ${gen ? c.series[0] : 'transparent'}`,
                  animation: gen && !reduce ? 'vz-note-in .35s cubic-bezier(.34,1.56,.64,1)' : undefined
                }}
              >
                {isEnd ? '. (끝)' : t}
              </span>
            );
          })}
          {!done && (
            <span
              aria-hidden="true"
              style={{ width: 3, height: 26, borderRadius: 2, background: c.series[0], animation: reduce ? undefined : 'twinkle .8s steps(2) infinite' }}
            />
          )}
        </div>

        {/* candidate bars */}
        <div>
          <div className="vz-panel-title" style={{ marginBottom: 8 }}>
            {done ? '문장이 끝났어요' : `다음에 올 단어의 확률 (온도 ${T.toFixed(2)})`}
          </div>
          {done ? (
            <div style={{ padding: '18px 14px', borderRadius: 14, border: '1.5px dashed var(--border-strong)', textAlign: 'center', color: 'var(--text-2)', fontSize: 14.5 }}>
              ✍️ 다른 <strong>출발 문장</strong>을 고르거나 <strong>↺ 처음부터</strong>를 눌러 새 문장을 지어 보세요. 같은 출발이어도 뽑기 때문에 매번 달라져요!
            </div>
          ) : (
          <div style={{ display: 'grid', gap: 6 }}>
            {shown.map(([word, p], i) => {
              const active = rolling === i;
              return (
                <button
                  key={word}
                  type="button"
                  disabled={done || rolling !== null}
                  onClick={() => append(i)}
                  aria-label={`${word === END ? '문장 끝' : word} 고르기, 확률 ${(p * 100).toFixed(0)}퍼센트`}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'minmax(78px, 112px) 1fr 52px',
                    alignItems: 'center',
                    gap: 10,
                    padding: '6px 10px',
                    borderRadius: 12,
                    border: `1.5px solid ${active ? c.series[2] : 'var(--border)'}`,
                    background: active ? withAlpha(c.series[2], dark ? 0.2 : 0.14) : 'var(--surface-solid)',
                    textAlign: 'left',
                    cursor: done ? 'default' : 'pointer',
                    transition: 'border-color .12s, background-color .12s'
                  }}
                >
                  <span style={{ fontWeight: 800, fontSize: 15, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {word === END ? '. (문장 끝)' : word}
                  </span>
                  <span style={{ position: 'relative', height: 14, borderRadius: 99, background: 'var(--surface-3)', overflow: 'hidden' }}>
                    <span
                      style={{
                        position: 'absolute',
                        inset: 0,
                        width: `${Math.max(1.5, p * 100)}%`,
                        borderRadius: 99,
                        background: i === 0 ? c.series[0] : withAlpha(c.series[0], dark ? 0.65 : 0.55),
                        transition: 'width .35s cubic-bezier(.22,1,.36,1)'
                      }}
                    />
                  </span>
                  <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 800, fontSize: 14, textAlign: 'right' }}>{(p * 100).toFixed(p < 0.1 ? 1 : 0)}%</span>
                </button>
              );
            })}
            {restCount > 0 && (
              <div style={{ fontSize: 12.5, color: 'var(--text-3)', paddingLeft: 10 }}>
                그 밖의 단어 {restCount}개 · 합쳐서 {(restMass * 100).toFixed(1)}%
              </div>
            )}
          </div>
          )}
        </div>

        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>

        <div className="vz-stats">
          <Stat label="새로 쓴 단어" value={`${tokens.length - promptLen}개`} />
          <Stat label="방금 단어의 놀람" value={lastSurprise === undefined ? '—' : fmt(lastSurprise, 2)} sub="= -log(확률)" />
          <Stat label="평균 놀람 (손실)" value={surprises.length ? fmt(avgSurprise, 2) : '—'} tone="brand" />
        </div>

        {showData && (
          <div style={{ maxHeight: 180, overflowY: 'auto', padding: '10px 12px', borderRadius: 14, background: 'var(--surface-solid)', border: '1px solid var(--border)', fontSize: 13, lineHeight: 1.75, color: 'var(--text-2)' }}>
            <strong style={{ color: 'var(--text)' }}>이 모델이 배운 문장 {CORPUS.length}개</strong> (띄어쓰기 단위로 잘랐어요)
            <br />
            {CORPUS.map((s) => s.replace(' .', '.')).join(' · ')}
          </div>
        )}
      </div>

      <Controls>
        <Segmented label="출발 문장" options={PROMPTS.map((p) => ({ value: p, label: p }))} value={prompt} onChange={(v) => { setPrompt(v); reset(v); }} />
        <Slider label="온도 (창의성)" value={T} min={0.2} max={2.5} step={0.05} onChange={setT} format={(v) => v.toFixed(2)} accent="var(--peach)" />
        <Toggle label="항상 1등만 고르기" checked={greedy} onChange={setGreedy} />
        <Toggle label="훈련 문장 보기" checked={showData} onChange={setShowData} />
        <Button variant="primary" onClick={generateOne} disabled={done || rolling !== null || auto}>
          🎲 한 단어 뽑기
        </Button>
        <Button onClick={() => setAuto(true)} disabled={done || auto}>
          ▶ 끝까지 쓰기
        </Button>
        <Button variant="ghost" onClick={() => reset(prompt)}>
          ↺ 처음부터
        </Button>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (custom.trim()) {
              setPrompt('');
              reset(custom);
            }
          }}
          style={{ display: 'flex', gap: 6, flex: '1 1 220px' }}
        >
          <input
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
            placeholder="출발 문장 직접 쓰기 (예: 우리 고양이는)"
            aria-label="출발 문장 직접 쓰기"
            style={{ flex: 1, minWidth: 0, height: 38, padding: '0 12px', borderRadius: 12, border: '1px solid var(--border-strong)', background: 'var(--surface-solid)', fontSize: 14 }}
          />
          <Button size="sm">시작</Button>
        </form>
      </Controls>
    </div>
  );
}
