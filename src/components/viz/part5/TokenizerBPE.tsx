import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { AlgoMood } from '@/components/mascot/Algo';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Slider, Stat } from '../core/ui';
import { useInterval, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors } from '../core/colors';
import { chipColors, encodeBpe, trainBpe } from './_tokens';

type CorpusKey = 'ko' | 'ab';

const CORPORA: Record<CorpusKey, { label: string; words: [string, number][] }> = {
  ko: {
    label: '한국어 낱말 모음',
    words: [
      ['학교', 10],
      ['학교에서', 6],
      ['학생', 8],
      ['학생들', 4],
      ['탄소', 5],
      ['탄소중립', 3],
      ['탈탄소화', 1],
      ['수업에서', 3]
    ]
  },
  ab: { label: 'aaabdaaabac', words: [['aaabdaaabac', 1]] }
};

const DEFAULT_TEXT = '탈탄소화 학교에서 학생들이 수업에서 졸았다';

/** Positions (left index of each pair) that a merge of a+b would fuse, left to right. */
function pairSpots(seg: string[], a: string, b: string): Set<number> {
  const out = new Set<number>();
  for (let i = 0; i < seg.length - 1; i++) {
    if (seg[i] === a && seg[i + 1] === b) {
      out.add(i);
      i++;
    }
  }
  return out;
}

function Chip({ text, theme, glow, small, sub }: { text: string; theme: 'light' | 'dark'; glow?: boolean; small?: boolean; sub?: string }) {
  const col = chipColors(text, theme);
  return (
    <span
      style={{
        display: 'inline-flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 1,
        padding: small ? '2px 7px' : '4px 10px',
        borderRadius: 10,
        background: col.bg,
        border: `1.5px solid ${glow ? 'var(--brand)' : col.border}`,
        color: col.text,
        fontWeight: 800,
        fontSize: small ? 13 : 15,
        lineHeight: 1.3,
        boxShadow: glow ? '0 0 0 3px var(--brand-soft), 0 0 14px color-mix(in srgb, var(--brand) 45%, transparent)' : 'none',
        transform: glow ? 'translateY(-2px)' : 'none',
        transition: 'transform .25s, box-shadow .25s, border-color .25s',
        animation: 'vz-note-in .35s var(--ease-spring)',
        whiteSpace: 'nowrap'
      }}
    >
      {text}
      {sub && <span style={{ fontSize: 10, fontWeight: 700, opacity: 0.75, fontFamily: 'var(--font-mono)' }}>{sub}</span>}
    </span>
  );
}

/** Step through BPE training on a toy corpus, then tokenize your own sentence. */
export default function TokenizerBPE({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [mode, setMode] = useState<'train' | 'encode'>('train');
  const [corpus, setCorpus] = useState<CorpusKey>('ko');
  const history = useMemo(() => trainBpe(CORPORA[corpus].words), [corpus]);
  const koHistory = useMemo(() => trainBpe(CORPORA.ko.words), []);
  const koMerges = koHistory[koHistory.length - 1].merges;
  const [step, setStep] = useState(0);
  const [pending, setPending] = useState(false);
  const [auto, setAuto] = useState(false);
  const [text, setText] = useState(DEFAULT_TEXT);
  const [k, setK] = useState(koMerges.length);
  const timer = useRef(0);

  useEffect(() => {
    setStep(0);
    setPending(false);
    setAuto(false);
  }, [corpus]);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const state = history[step];
  const atEnd = step >= history.length - 1;
  const best = state.ranked[0];

  const doMerge = () => {
    if (atEnd || pending) return;
    if (reduce) {
      setStep((s) => Math.min(history.length - 1, s + 1));
      return;
    }
    setPending(true);
    timer.current = window.setTimeout(() => {
      setStep((s) => Math.min(history.length - 1, s + 1));
      setPending(false);
    }, 650);
  };

  useInterval(
    () => {
      if (atEnd) setAuto(false);
      else doMerge();
    },
    auto && mode === 'train' ? 1500 : null
  );

  const words = CORPORA[corpus].words;
  const alphabet = useMemo(() => new Set(words.flatMap(([w]) => Array.from(w))), [words]);
  const totalPieces = state.segs.reduce((a, seg, i) => a + seg.length * words[i][1], 0);
  const startPieces = history[0].segs.reduce((a, seg, i) => a + seg.length * words[i][1], 0);
  const lastMerge = state.merges[state.merges.length - 1];

  // encode mode
  const koVocab = useMemo(() => {
    const chars: string[] = [];
    CORPORA.ko.words.forEach(([w]) => Array.from(w).forEach((ch) => !chars.includes(ch) && chars.push(ch)));
    return [...chars, ...koMerges.map(([a, b]) => a + b)];
  }, [koMerges]);
  const encoded = useMemo(() => encodeBpe(text, koMerges, k), [text, koMerges, k]);
  const nChars = Array.from(text.replace(/\s+/g, '')).length;
  const nTokens = encoded.reduce((a, w) => a + w.length, 0);
  const unknownChars = Array.from(new Set(Array.from(text.replace(/\s+/g, '')).filter((ch) => !koVocab.includes(ch))));

  let note: { text: ReactNode; mood: AlgoMood; tone?: 'good' | 'warn' };
  if (mode === 'train') {
    if (step === 0)
      note = {
        text: (
          <>
            처음엔 모든 낱말이 <strong>글자 하나하나</strong>로 잘려 있어요. 가장 자주 붙어 나오는 두 조각을 찾아 하나로 합쳐 볼까요? <strong>「한 번 병합」</strong>을 눌러 보세요!
          </>
        ),
        mood: 'smile'
      };
    else if (atEnd)
      note = {
        text:
          corpus === 'ko' ? (
            <>
              더 이상 두 번 이상 붙어 나오는 쌍이 없어요. 🎉 자주 쓰는 <strong>「학교」「학생」</strong>은 한 덩어리가 되었고, 딱 한 번 나온 <strong>「탈탄소화」</strong>는 탈 + 탄소 + 화로 쪼개졌죠? 자주 쓰는 건 짧게, 드문 건 조합으로!
            </>
          ) : (
            <>
              첫 병합은 <strong>a + a → aa</strong>였어요 (4번이나 붙어 나왔죠). 스스로 점검 1번의 답이에요! 같은 쌍이 2번 이상 나오지 않으면 멈춰요.
            </>
          ),
        mood: 'cheer',
        tone: 'good'
      };
    else if (pending && best)
      note = { text: <>「{best.a}」와 「{best.b}」가 {best.count}번 붙어 나와요 → 하나로 합칩니다!</>, mood: 'wow' };
    else
      note = {
        text: (
          <>
            방금 <strong>「{lastMerge?.[0]}」+「{lastMerge?.[1]}」 → 「{lastMerge?.join('')}」</strong> 규칙을 배웠어요. 조각 종류(어휘)는 하나 늘고, 전체 조각 수는 줄었어요.
          </>
        ),
        mood: 'happy'
      };
  } else if (k === 0)
    note = { text: <>규칙을 하나도 안 쓰면 <strong>글자 하나가 토큰 하나</strong>예요. 토큰이 너무 많아져서 비효율적이죠. 슬라이더를 오른쪽으로 밀어 보세요.</>, mood: 'think' };
  else if (unknownChars.length)
    note = {
      text: (
        <>
          「{unknownChars.slice(0, 3).join('」「')}」처럼 처음 보는 글자도 <strong>글자(바이트) 단위로는 언제나 표현</strong>돼요. 그래서 BPE에는 "모르는 낱말"이 없어요! 모델은 글자가 아니라 이 덩어리들을 보기 때문에, 글자 수 세기엔 약해요.
        </>
      ),
      mood: 'wink'
    };
  else note = { text: <>규칙을 많이 쓸수록 자주 쓰는 말이 한 덩어리가 되어 토큰 수가 줄어요.</>, mood: 'happy' };

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 14, minHeight: expanded ? '60vh' : undefined }}>
        <Segmented
          options={[
            { value: 'train', label: '1. 병합 규칙 배우기' },
            { value: 'encode', label: '2. 내 문장 잘라 보기' }
          ]}
          value={mode}
          onChange={(v) => {
            setAuto(false);
            setMode(v as 'train' | 'encode');
          }}
        />

        {mode === 'train' ? (
          <>
            <div>
              <div className="vz-panel-title" style={{ marginBottom: 8 }}>
                코퍼스(학습용 글 모음) — 낱말 × 나온 횟수
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }} aria-live="polite">
                {state.segs.map((seg, wi) => {
                  const spots = pending && best ? pairSpots(seg, best.a, best.b) : new Set<number>();
                  return (
                    <div
                      key={wi}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 4,
                        padding: '8px 10px',
                        borderRadius: 14,
                        background: 'var(--surface-solid)',
                        border: '1px solid var(--border)'
                      }}
                    >
                      {seg.map((tok, ti) => (
                        <Chip key={`${tok}-${ti}`} text={tok} theme={c.theme} glow={spots.has(ti) || spots.has(ti - 1)} />
                      ))}
                      <span style={{ marginLeft: 4, fontSize: 12.5, fontWeight: 800, color: 'var(--text-3)' }}>×{words[wi][1]}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14 }}>
              <div>
                <div className="vz-panel-title" style={{ marginBottom: 8 }}>
                  지금 가장 자주 붙어 나오는 쌍
                </div>
                {state.ranked.length === 0 || (atEnd && (best?.count ?? 0) < 2) ? (
                  <div style={{ fontSize: 13.5, color: 'var(--text-3)' }}>두 번 이상 붙어 나오는 쌍이 없어요 — 학습 끝!</div>
                ) : (
                  <div style={{ display: 'grid', gap: 6 }}>
                    {state.ranked.map((p, i) => {
                      const w = (p.count / state.ranked[0].count) * 100;
                      return (
                        <div key={`${p.a}|${p.b}`} style={{ display: 'grid', gridTemplateColumns: '112px 1fr 44px', alignItems: 'center', gap: 8, fontSize: 13.5 }}>
                          <span style={{ fontWeight: 800, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {i === 0 ? '👑 ' : ''}
                            {p.a} + {p.b}
                          </span>
                          <span style={{ height: 12, borderRadius: 99, background: 'var(--surface-3)', overflow: 'hidden' }}>
                            <span
                              style={{
                                display: 'block',
                                height: '100%',
                                width: `${w}%`,
                                borderRadius: 99,
                                background: i === 0 ? c.series[0] : c.series[3],
                                transition: 'width .4s'
                              }}
                            />
                          </span>
                          <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 750, color: 'var(--text-2)' }}>{p.count}번</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
              <div>
                <div className="vz-panel-title" style={{ marginBottom: 8 }}>
                  배운 병합 규칙 (순서대로)
                </div>
                {state.merges.length === 0 ? (
                  <div style={{ fontSize: 13.5, color: 'var(--text-3)' }}>아직 없어요.</div>
                ) : (
                  <ol style={{ margin: 0, paddingLeft: 22, display: 'grid', gap: 4, fontSize: 13.5 }}>
                    {state.merges.map(([a, b], i) => (
                      <li key={i} style={{ fontWeight: i === state.merges.length - 1 ? 800 : 600 }}>
                        {a} + {b} → <Chip text={a + b} theme={c.theme} small />
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            </div>

            <div className="vz-stats">
              <Stat label="병합 횟수" value={step} />
              <Stat label="조각 종류(어휘)" value={alphabet.size + step} sub={`처음 글자 ${alphabet.size}종`} />
              <Stat label="전체 조각 수" value={totalPieces} tone="brand" sub={`처음 ${startPieces}개`} />
            </div>
          </>
        ) : (
          <>
            <label style={{ display: 'grid', gap: 6 }}>
              <span className="vz-label">잘라 볼 문장 (직접 바꿔 보세요)</span>
              <input
                type="text"
                value={text}
                onChange={(e) => setText(e.target.value.slice(0, 60))}
                style={{
                  height: 46,
                  padding: '0 14px',
                  borderRadius: 14,
                  border: '1.5px solid var(--border-strong)',
                  background: 'var(--surface-solid)',
                  fontSize: 16,
                  outline: 'none'
                }}
              />
            </label>
            <div>
              <div className="vz-panel-title" style={{ marginBottom: 8 }}>
                토큰으로 자른 결과 (아래 숫자는 토큰 번호)
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                {encoded.map((seg, wi) => (
                  <div key={wi} style={{ display: 'flex', gap: 3 }}>
                    {seg.map((tok, ti) => {
                      const id = koVocab.indexOf(tok);
                      return <Chip key={ti} text={tok} theme={c.theme} sub={id >= 0 ? `#${id}` : '바이트'} />;
                    })}
                  </div>
                ))}
              </div>
            </div>
            <div className="vz-stats">
              <Stat label="글자 수" value={nChars} />
              <Stat label="토큰 수" value={nTokens} tone="brand" />
              <Stat label="줄어든 비율" value={`${nChars ? Math.round((1 - nTokens / nChars) * 100) : 0}%`} tone={nTokens < nChars ? 'good' : undefined} />
            </div>
          </>
        )}

        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        {mode === 'train' ? (
          <>
            <Segmented
              label="코퍼스"
              options={[
                { value: 'ko', label: '한국어 낱말' },
                { value: 'ab', label: 'aaabdaaabac (점검 1번)' }
              ]}
              value={corpus}
              onChange={(v) => setCorpus(v as CorpusKey)}
            />
            <Button variant="primary" onClick={doMerge} disabled={atEnd || pending}>
              한 번 병합 ⟶
            </Button>
            <Button onClick={() => setAuto((a) => !a)} disabled={atEnd}>
              {auto ? '❚❚ 멈춤' : '▶ 끝까지 자동'}
            </Button>
            <Button variant="ghost" onClick={() => { setAuto(false); setPending(false); setStep(0); }}>
              ↺ 처음으로
            </Button>
          </>
        ) : (
          <>
            <Slider
              label="배운 규칙을 몇 개까지 쓸까?"
              value={k}
              min={0}
              max={koMerges.length}
              step={1}
              onChange={setK}
              format={(v) => `${v} / ${koMerges.length}개`}
            />
            <Button variant="ghost" onClick={() => { setText(DEFAULT_TEXT); setK(koMerges.length); }}>
              ↺ 예시 문장으로
            </Button>
          </>
        )}
      </Controls>
    </div>
  );
}
