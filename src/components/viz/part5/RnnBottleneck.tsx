import { useEffect, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, PlayBar, Segmented, Slider, Stat } from '../core/ui';
import { useInterval, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors } from '../core/colors';
import { softmax } from '../core/math';
import type { AlgoMood } from '@/components/mascot/Algo';

const WORDS = ['내가', '어릴', '적', '파리에서', '자란', '덕분에,', '어른이', '되어서도', '유창하게', '하는', '언어는'];
const PARIS = 3;
const ANSWERS = ['프랑스어', '영어', '한국어', '중국어'];
const ATTN_LOGITS = [0.2, 0.5, 0.2, 3.2, 0.9, 0.4, 0.3, 0.3, 1.2, 0.6, 1.6];
const HUES = [210, 30, 120, 330, 260, 50, 180, 0, 95, 285, 160];

const wordColor = (i: number, theme: 'light' | 'dark') =>
  theme === 'dark' ? `hsl(${HUES[i]}, 90%, 62%)` : `hsl(${HUES[i]}, 62%, 58%)`;

/** RNN squeezes everything into one fixed-size memory; attention looks back directly. */
export default function RnnBottleneck({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [mode, setMode] = useState<'rnn' | 'attn'>('rnn');
  const [read, setRead] = useState(0); // words read so far (RNN)
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [keep, setKeep] = useState(0.6);
  const [attnShown, setAttnShown] = useState(false);

  useEffect(() => {
    setPlaying(false);
    setRead(0);
    setAttnShown(false);
  }, [mode]);

  useInterval(
    () => {
      if (read >= WORDS.length) setPlaying(false);
      else setRead((r) => r + 1);
    },
    playing && mode === 'rnn' ? 850 / speed : null
  );

  // memory shares: newest word weighs most, older words fade by `keep` each step
  const t = read - 1;
  const raw = WORDS.map((_, i) => (i <= t ? keep ** (t - i) : 0));
  const total = raw.reduce((a, v) => a + v, 0) || 1;
  const shares = raw.map((v) => v / total);
  const parisShare = shares[PARIS];
  const done = mode === 'rnn' ? read >= WORDS.length : attnShown;

  const attn = softmax(ATTN_LOGITS);
  const strength = mode === 'rnn' ? parisShare * Math.max(1, read) : attn[PARIS];
  const logits = mode === 'rnn' ? [0.4 + 2.6 * strength, 1.8, 1.2, 0.6] : [1 + 6 * attn[PARIS], 1.8, 1.2, 0.6];
  const probs = softmax(logits);
  const guess = probs.indexOf(Math.max(...probs));

  let note: { text: ReactNode; mood: AlgoMood; tone?: 'good' | 'bad' | 'warn' };
  if (mode === 'rnn') {
    if (read === 0) note = { text: <>RNN은 사람처럼 <strong>한 낱말씩</strong> 읽으며, 읽은 내용을 <strong>크기가 정해진 기억 상자</strong> 하나에 요약해요. ▶ 시작을 눌러 보세요.</>, mood: 'smile' };
    else if (!done)
      note = {
        text: (
          <>
            「{WORDS[read - 1]}」를 읽었어요. 새 낱말이 들어올 때마다 상자 속 옛 기억은 조금씩 밀려나요. 「파리에서」의 몫: <strong>{Math.round(parisShare * 100)}%</strong>
          </>
        ),
        mood: 'think'
      };
    else if (guess === 0)
      note = {
        text: <>기억이 잘 남는 설정이라 「파리」를 기억해서 <strong>프랑스어</strong>를 맞혔어요! LSTM·GRU의 "게이트"가 바로 이 남기는 비율을 높여 줘요. 하지만 문장이 수백 낱말이면 결국 희미해져요.</>,
        mood: 'happy',
        tone: 'good'
      };
    else
      note = {
        text: <>빈칸에 도착했을 때 「파리」의 정보는 고작 <strong>{(parisShare * 100).toFixed(1)}%</strong>만 남았어요. 그래서 <strong>{ANSWERS[guess]}</strong>로 잘못 짐작했죠. 먼 기억이 죽는 <strong>벽 1</strong>이에요!</>,
        mood: 'wow',
        tone: 'bad'
      };
  } else if (!attnShown) note = { text: <>어텐션은 요약하지 않아요. 빈칸 자리에서 <strong>모든 낱말을 한 번에 직접</strong> 바라봐요. ⚡ 버튼을 눌러 보세요.</>, mood: 'smile' };
  else
    note = {
      text: <>빈칸이 「파리에서」를 <strong>{Math.round(attn[PARIS] * 100)}%</strong>로 가장 강하게 바라봤어요 → <strong>프랑스어</strong>! 거리가 아무리 멀어도 한 걸음, 그리고 모든 낱말을 동시에 계산해서 순서를 기다릴 필요도 없어요 (<strong>벽 1·2</strong> 해결).</>,
      mood: 'cheer',
      tone: 'good'
    };

  const chipState = (i: number) => (mode === 'attn' ? (attnShown ? 'read' : 'idle') : i < read - 1 ? 'read' : i === read - 1 ? 'current' : 'idle');

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 16, minHeight: expanded ? '58vh' : undefined }}>
        <Segmented
          options={[
            { value: 'rnn', label: '🔁 RNN — 순서대로 요약' },
            { value: 'attn', label: '👀 어텐션 — 직접 바라보기' }
          ]}
          value={mode}
          onChange={(v) => setMode(v as 'rnn' | 'attn')}
        />

        <div>
          <div className="vz-panel-title" style={{ marginBottom: 8 }}>
            문장 — 빈칸에 들어갈 언어는?
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {WORDS.map((w, i) => {
              const st = chipState(i);
              const col = wordColor(i, c.theme);
              const weight = mode === 'attn' && attnShown ? attn[i] : 0;
              return (
                <span
                  key={w}
                  style={{
                    position: 'relative',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 4,
                    padding: '6px 10px',
                    borderRadius: 12,
                    border: `2px solid ${st === 'idle' ? 'var(--border)' : col}`,
                    background: st === 'idle' ? 'var(--surface-2)' : `color-mix(in srgb, ${col} ${c.theme === 'dark' ? 16 : 20}%, var(--surface-solid))`,
                    color: st === 'idle' ? 'var(--text-3)' : 'var(--text)',
                    fontWeight: 800,
                    fontSize: 14.5,
                    transform: st === 'current' ? 'translateY(-3px) scale(1.05)' : 'none',
                    boxShadow: st === 'current' ? `0 6px 18px color-mix(in srgb, ${col} 45%, transparent)` : 'none',
                    transition: 'all .3s var(--ease-spring)'
                  }}
                >
                  {i === PARIS && '🗼'}
                  {w}
                  {mode === 'rnn' && i < read && (
                    <span style={{ fontSize: 10.5, fontWeight: 800, color: 'var(--text-3)', fontVariantNumeric: 'tabular-nums' }}>{i + 1}</span>
                  )}
                  {mode === 'attn' && attnShown && (
                    <span
                      aria-hidden="true"
                      style={{
                        position: 'absolute',
                        left: 6,
                        right: 6,
                        bottom: -7,
                        height: 4,
                        borderRadius: 4,
                        background: c.series[2],
                        transformOrigin: 'left',
                        transform: `scaleX(${Math.min(1, weight / attn[PARIS])})`,
                        transition: 'transform .6s var(--ease-out)'
                      }}
                    />
                  )}
                </span>
              );
            })}
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                padding: '6px 12px',
                borderRadius: 12,
                border: '2px dashed var(--brand)',
                fontWeight: 900,
                fontSize: 14.5,
                color: done ? 'var(--text)' : 'var(--text-3)',
                background: done ? 'var(--brand-soft)' : 'transparent'
              }}
            >
              {done ? `${ANSWERS[guess]}?` : '___'}
            </span>
          </div>
        </div>

        {mode === 'rnn' ? (
          <div>
            <div className="vz-panel-title" style={{ marginBottom: 8 }}>
              🧠 기억 상자 h — 낱말이 몇 개 들어와도 크기는 그대로
            </div>
            <div
              style={{
                position: 'relative',
                display: 'flex',
                height: 54,
                borderRadius: 16,
                border: '2.5px dashed var(--border-strong)',
                padding: 5,
                gap: 2,
                background: 'var(--surface-solid)',
                overflow: 'hidden'
              }}
              aria-label={`기억 상자: 파리의 몫 ${Math.round(parisShare * 100)}%`}
            >
              {read === 0 ? (
                <span style={{ margin: 'auto', fontSize: 13, color: 'var(--text-3)', fontWeight: 700 }}>아직 비어 있어요</span>
              ) : (
                WORDS.map((w, i) =>
                  shares[i] > 0 ? (
                    <span
                      key={w}
                      title={`${w} ${Math.round(shares[i] * 100)}%`}
                      style={{
                        flex: `${shares[i]} 1 0`,
                        minWidth: 0,
                        borderRadius: 10,
                        background: wordColor(i, c.theme),
                        outline: i === PARIS ? `3px solid ${c.ink}` : 'none',
                        outlineOffset: -3,
                        display: 'grid',
                        placeItems: 'center',
                        color: '#fff',
                        fontSize: 12,
                        fontWeight: 850,
                        overflow: 'hidden',
                        whiteSpace: 'nowrap',
                        textShadow: '0 1px 2px rgba(0,0,0,.35)',
                        transition: 'flex-grow .5s var(--ease-out)'
                      }}
                    >
                      {shares[i] > 0.09 ? w : ''}
                    </span>
                  ) : null
                )
              )}
            </div>
          </div>
        ) : (
          <div>
            <div className="vz-panel-title" style={{ marginBottom: 8 }}>
              👀 빈칸이 각 낱말을 바라보는 정도 (어텐션 가중치)
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height: 96, padding: '0 2px' }}>
              {WORDS.map((w, i) => (
                <div key={w} style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
                  <span style={{ fontSize: 10.5, fontWeight: 800, color: 'var(--text-2)', fontVariantNumeric: 'tabular-nums' }}>
                    {attnShown ? `${Math.round(attn[i] * 100)}` : ''}
                  </span>
                  <span
                    style={{
                      width: '100%',
                      height: attnShown ? `${(attn[i] / attn[PARIS]) * 64}px` : '2px',
                      borderRadius: '6px 6px 2px 2px',
                      background: wordColor(i, c.theme),
                      transition: reduce ? 'none' : `height .7s var(--ease-spring) ${i * 40}ms`
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14 }}>
          <div className="vz-stats">
            <Stat label="「파리」 정보" value={mode === 'rnn' ? `${(parisShare * 100).toFixed(1)}%` : attnShown ? `${Math.round(attn[PARIS] * 100)}%` : '—'} tone={mode === 'attn' || guess === 0 ? 'good' : 'bad'} />
            <Stat label="차례로 해야 하는 계산" value={mode === 'rnn' ? `${Math.max(read, 0)}단계` : '1단계'} sub={mode === 'rnn' ? '앞이 끝나야 다음 가능' : '모두 동시에'} />
          </div>
          <div>
            <div className="vz-panel-title" style={{ marginBottom: 6 }}>
              빈칸 예측 {done ? '' : '(다 읽으면 나와요)'}
            </div>
            <div style={{ display: 'grid', gap: 5, opacity: done ? 1 : 0.35, transition: 'opacity .3s' }}>
              {ANSWERS.map((a, i) => (
                <div key={a} style={{ display: 'grid', gridTemplateColumns: '64px 1fr 40px', gap: 8, alignItems: 'center', fontSize: 13 }}>
                  <span style={{ fontWeight: 800 }}>{a}</span>
                  <span style={{ height: 10, borderRadius: 99, background: 'var(--surface-3)', overflow: 'hidden' }}>
                    <span
                      style={{
                        display: 'block',
                        height: '100%',
                        width: `${probs[i] * 100}%`,
                        borderRadius: 99,
                        background: i === 0 ? c.pos : c.series[3],
                        transition: 'width .5s var(--ease-out)'
                      }}
                    />
                  </span>
                  <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 750, color: 'var(--text-2)' }}>{Math.round(probs[i] * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        {mode === 'rnn' ? (
          <>
            <PlayBar
              playing={playing}
              onToggle={() => {
                if (read >= WORDS.length) setRead(0);
                setPlaying((p) => !p);
              }}
              onStep={read < WORDS.length ? () => setRead((r) => r + 1) : undefined}
              onReset={() => {
                setPlaying(false);
                setRead(0);
              }}
              speed={speed}
              onSpeed={setSpeed}
            />
            <Slider
              label="한 걸음마다 기억이 남는 비율"
              value={keep}
              min={0.4}
              max={0.97}
              step={0.01}
              onChange={setKeep}
              format={(v) => `${Math.round(v * 100)}%`}
              hint="LSTM·GRU의 게이트가 이 비율을 높여요"
            />
          </>
        ) : (
          <>
            <Button variant="primary" onClick={() => setAttnShown(true)}>
              ⚡ 한 번에 바라보기
            </Button>
            <Button variant="ghost" onClick={() => setAttnShown(false)}>
              ↺ 처음으로
            </Button>
          </>
        )}
      </Controls>
    </div>
  );
}
