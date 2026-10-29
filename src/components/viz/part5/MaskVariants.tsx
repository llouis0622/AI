import { useEffect, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Stat } from '../core/ui';
import { useVizColors } from '../core/colors';
import type { AlgoMood } from '@/components/mascot/Algo';

type Mode = 'bert' | 'gpt' | 'encdec';

const GPT_TOKENS = ['오늘', '날씨가', '정말', '좋다', '.'];
const GPT_NEXT: { options: [string, number][] }[] = [
  { options: [['날씨가', 0.46], ['은', 0.21], ['하루', 0.13], ['나는', 0.1]] },
  { options: [['정말', 0.38], ['너무', 0.27], ['맑다', 0.15], ['좋다', 0.12]] },
  { options: [['좋다', 0.55], ['덥다', 0.2], ['춥다', 0.15], ['맑다', 0.06]] },
  { options: [['.', 0.72], ['!', 0.18], ['요', 0.06], ['고', 0.03]] },
  { options: [['〈끝〉', 0.9], ['그래서', 0.05], ['나는', 0.03], ['또', 0.02]] }
];
const BERT_TOKENS = ['오늘', '[MASK]', '정말', '좋다', '.'];
const BERT_GUESS: [string, number][] = [['날씨가', 0.72], ['기분이', 0.14], ['하늘이', 0.08], ['나는', 0.03]];
const SRC = ['나는', '학교에', '간다'];
const TGT = ['I', 'go', 'to', 'school'];
const CROSS = [
  [0.82, 0.08, 0.1],
  [0.12, 0.1, 0.78],
  [0.1, 0.62, 0.28],
  [0.05, 0.85, 0.1]
];

/** synthetic attention weight for display: self + nearby tokens a bit stronger */
const weightOf = (i: number, j: number, visible: (a: number, b: number) => boolean, n: number) => {
  const raw = (b: number) => (visible(i, b) ? Math.exp(-Math.abs(i - b) * 0.55) + (b === 0 ? 0.25 : 0) : 0);
  const s = Array.from({ length: n }, (_, b) => raw(b)).reduce((a, v) => a + v, 0) || 1;
  return raw(j) / s;
};

/** Same Transformer, three masks: who may look at whom decides BERT vs GPT vs translator. */
export default function MaskVariants({ expanded }: VizProps) {
  const c = useVizColors();
  const [mode, setMode] = useState<Mode>('gpt');
  const [gptView, setGptView] = useState<'train' | 'gen'>('train');
  const [genLen, setGenLen] = useState(1);
  const [hot, setHot] = useState<number | null>(null);
  const [hotKey, setHotKey] = useState<string>('');

  useEffect(() => {
    setHot(null);
    setGenLen(1);
  }, [mode, gptView]);

  const cellMax = expanded ? 56 : 46;
  const dark = c.theme === 'dark';

  const renderMatrix = (key: string, rows: string[], cols: string[], visible: (i: number, j: number) => boolean, weight: (i: number, j: number) => number, color: string, rightLabels?: (string | null)[]) => (
    <div style={{ overflowX: 'auto' }} key={key}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `62px repeat(${cols.length}, minmax(26px, ${cellMax}px))${rightLabels ? ' minmax(64px, auto)' : ''}`,
          gap: 3,
          alignItems: 'center',
          fontSize: 11.5,
          maxWidth: 62 + cols.length * (cellMax + 3) + (rightLabels ? 80 : 0)
        }}
        onMouseLeave={() => setHot(null)}
      >
        <span />
        {cols.map((t, j) => (
          <span key={`h${j}`} style={{ textAlign: 'center', fontWeight: 800, color: 'var(--text-2)', fontSize: t.length > 3 ? 10 : 11.5, whiteSpace: 'nowrap' }}>
            {t}
          </span>
        ))}
        {rightLabels && <span style={{ fontWeight: 800, color: 'var(--text-3)', fontSize: 11, paddingLeft: 8 }}>맞힐 정답</span>}
        {rows.map((r, i) => {
          const isHot = hot === i && hotKey === key;
          return [
            <button
              key={`r${i}`}
              type="button"
              onMouseEnter={() => { setHot(i); setHotKey(key); }}
              onClick={() => { setHot(i); setHotKey(key); }}
              style={{ textAlign: 'right', paddingRight: 6, fontWeight: isHot ? 900 : 750, color: isHot ? 'var(--brand)' : 'var(--text-2)', whiteSpace: 'nowrap' }}
            >
              {r}
            </button>,
            ...cols.map((_, j) => {
              const ok = visible(i, j);
              const w = ok ? weight(i, j) : 0;
              return (
                <span
                  key={`c${i}-${j}`}
                  onMouseEnter={() => { setHot(i); setHotKey(key); }}
                  onClick={() => { setHot(i); setHotKey(key); }}
                  title={ok ? `${r} → ${cols[j]} 볼 수 있음` : `${r} → ${cols[j]} 가려짐 (미래)`}
                  style={{
                    aspectRatio: '1',
                    borderRadius: 7,
                    display: 'grid',
                    placeItems: 'center',
                    fontSize: 13,
                    cursor: 'pointer',
                    background: ok
                      ? `color-mix(in srgb, ${color} ${Math.round(18 + 75 * Math.min(1, w * 1.6))}%, var(--surface-3))`
                      : `repeating-linear-gradient(45deg, var(--surface-3) 0 5px, ${dark ? 'rgba(255,255,255,.04)' : 'rgba(42,37,64,.05)'} 5px 10px)`,
                    outline: isHot ? `2px solid ${c.brand}` : 'none',
                    outlineOffset: 1,
                    opacity: ok ? 1 : 0.8,
                    transition: 'background-color .3s'
                  }}
                >
                  {ok ? '' : '🔒'}
                </span>
              );
            }),
            ...(rightLabels
              ? [
                  <span key={`t${i}`} style={{ paddingLeft: 8, fontWeight: 850, fontSize: 12.5, color: rightLabels[i] ? 'var(--text)' : 'var(--text-3)', whiteSpace: 'nowrap' }}>
                    {rightLabels[i] ? `→ ${rightLabels[i]}` : ''}
                  </span>
                ]
              : [])
          ];
        })}
      </div>
    </div>
  );

  // token strip: highlight what the hot row may see
  const strip = (tokens: string[], visibleTo: (j: number) => boolean | null, label: string) => (
    <div>
      <div className="vz-panel-title" style={{ marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {tokens.map((t, j) => {
          const v = visibleTo(j);
          return (
            <span
              key={`${t}-${j}`}
              style={{
                padding: '5px 11px',
                borderRadius: 11,
                fontWeight: 850,
                fontSize: 14,
                border: `2px solid ${v === null ? 'var(--border)' : v ? c.series[0] : 'var(--border)'}`,
                background: v ? `color-mix(in srgb, ${c.series[0]} ${dark ? 18 : 16}%, var(--surface-solid))` : 'var(--surface-2)',
                color: v === false ? 'var(--text-3)' : 'var(--text)',
                textDecoration: v === false ? 'line-through' : 'none',
                opacity: v === false ? 0.6 : 1,
                transition: 'all .25s'
              }}
            >
              {t}
            </span>
          );
        })}
      </div>
    </div>
  );

  let body: ReactNode;
  let note: { text: ReactNode; mood: AlgoMood; tone?: 'good' };
  if (mode === 'bert') {
    const vis = () => true;
    body = (
      <>
        {strip(BERT_TOKENS, () => (hot === null || hotKey !== 'bert' ? null : true), hot !== null && hotKey === 'bert' ? `「${BERT_TOKENS[hot]}」가 볼 수 있는 토큰 — 전부!` : '문장 (가운데를 가리고 맞히기)')}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 16, alignItems: 'start' }}>
          {renderMatrix('bert', BERT_TOKENS, BERT_TOKENS, vis, (i, j) => weightOf(i, j, vis, 5), c.series[3])}
          <div>
            <div className="vz-panel-title" style={{ marginBottom: 6 }}>
              [MASK] 자리 예측 — 앞뒤 문맥을 모두 써서
            </div>
            {BERT_GUESS.map(([t, p], i) => (
              <ProbRow key={t} label={t} p={p} color={i === 0 ? c.pos : c.series[3]} />
            ))}
          </div>
        </div>
      </>
    );
    note = { text: <>BERT는 <strong>가림 없이 앞뒤를 모두</strong> 봐요. 그래서 [MASK] 빈칸을 양쪽 문맥으로 맞히며 배우고, 분류·검색 같은 <strong>이해</strong> 과제에 강해요. 대신 글을 왼쪽부터 써 나가는 생성은 어색해요.</>, mood: 'happy' };
  } else if (mode === 'gpt') {
    const n = gptView === 'train' ? GPT_TOKENS.length : genLen;
    const toks = GPT_TOKENS.slice(0, n);
    const vis = (i: number, j: number) => j <= i;
    const targets = gptView === 'train' ? [...GPT_TOKENS.slice(1), '〈끝〉'] : undefined;
    const next = GPT_NEXT[Math.min(genLen - 1, GPT_NEXT.length - 1)];
    body = (
      <>
        {strip(
          toks,
          (j) => (hot === null || hotKey !== 'gpt' ? null : j <= hot),
          hot !== null && hotKey === 'gpt' && hot < toks.length ? `「${toks[hot]}」가 볼 수 있는 토큰 — 자기와 그 앞쪽만` : gptView === 'train' ? '훈련 문장' : '지금까지 생성한 문장'
        )}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16, alignItems: 'start' }}>
          {renderMatrix('gpt', toks, toks, vis, (i, j) => weightOf(i, j, vis, n), c.series[0], targets)}
          {gptView === 'gen' ? (
            <div>
              <div className="vz-panel-title" style={{ marginBottom: 6 }}>
                다음 토큰 확률 — 맨 아래 줄(마지막 토큰)이 계산해요
              </div>
              {genLen <= GPT_TOKENS.length ? (
                next.options.map(([t, p], i) => <ProbRow key={t} label={t} p={p} color={i === 0 ? c.pos : c.series[3]} />)
              ) : (
                <div style={{ fontSize: 13.5, color: 'var(--text-3)' }}>문장이 끝났어요.</div>
              )}
            </div>
          ) : (
            <div className="vz-stats">
              <Stat label="문장 하나로 푸는 문제 수" value={`${GPT_TOKENS.length}개`} tone="good" sub="모든 자리가 동시에" />
              <Stat label="가려진 칸 (미래)" value={`${(GPT_TOKENS.length * (GPT_TOKENS.length - 1)) / 2}칸`} sub="커닝 방지 🔒" />
            </div>
          )}
        </div>
      </>
    );
    note =
      gptView === 'train'
        ? { text: <>GPT는 <strong>오른쪽(미래)을 🔒 가려요</strong>. 그래서 각 자리는 자기 앞만 보고 다음 토큰을 맞혀야 해요. 문장 하나를 한 번 통과시키면 <strong>{GPT_TOKENS.length}개의 문제를 동시에</strong> 푸는 셈 — 훈련이 엄청 효율적이에요!</>, mood: 'wow' }
        : { text: <>생성은 <strong>한 토큰씩 직렬</strong>이에요. 확률이 가장 높은 토큰을 골라 끝에 붙이고, 늘어난 문장으로 다시 예측해요. 챗봇이 또박또박 타자 치듯 나오는 이유예요.</>, mood: 'think' };
  } else {
    const vEnc = () => true;
    const vDec = (i: number, j: number) => j <= i;
    const vCross = () => true;
    const srcVisible = hot !== null && hotKey === 'cross' ? (j: number) => CROSS[hot][j] > 0.3 : () => null;
    body = (
      <>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
          {strip(SRC, srcVisible, '원문 (인코더가 읽어요)')}
          {strip(TGT, (j) => (hot !== null && (hotKey === 'dec' || hotKey === 'cross') ? j <= hot : null), '번역문 (디코더가 써요)')}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, alignItems: 'start' }}>
          <div>
            <div className="vz-panel-title" style={{ marginBottom: 6 }}>1. 인코더 자기 어텐션 — 양방향</div>
            {renderMatrix('enc', SRC, SRC, vEnc, (i, j) => weightOf(i, j, vEnc, 3), c.series[3])}
          </div>
          <div>
            <div className="vz-panel-title" style={{ marginBottom: 6 }}>2. 디코더 자기 어텐션 — 인과적</div>
            {renderMatrix('dec', TGT, TGT, vDec, (i, j) => weightOf(i, j, vDec, 4), c.series[0])}
          </div>
          <div>
            <div className="vz-panel-title" style={{ marginBottom: 6 }}>3. 교차 어텐션 — 번역문이 원문을 참고</div>
            {renderMatrix('cross', TGT, SRC, vCross, (i, j) => CROSS[i][j], c.series[2])}
          </div>
        </div>
      </>
    );
    note = { text: <>인코더는 원문을 <strong>양방향</strong>으로 이해하고, 디코더는 번역문을 <strong>왼쪽부터</strong> 쓰면서(인과적) <strong>교차 어텐션</strong>으로 원문을 참고해요. 3번 지도에서 「school」이 「학교에」를 가장 많이 보는 걸 확인해 보세요! Whisper도 이 구조예요.</>, mood: 'smile' };
  }

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 14, minHeight: expanded ? '58vh' : undefined }}>
        <Segmented
          options={[
            { value: 'bert', label: 'BERT · 양방향' },
            { value: 'gpt', label: 'GPT · 인과적' },
            { value: 'encdec', label: '인코더-디코더 · 번역' }
          ]}
          value={mode}
          onChange={(v) => setMode(v as Mode)}
        />
        {body}
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>
      <Controls>
        {mode === 'gpt' ? (
          <>
            <Segmented
              label="GPT 보기"
              options={[
                { value: 'train', label: '훈련: 모든 자리 동시에' },
                { value: 'gen', label: '생성: 한 토큰씩' }
              ]}
              value={gptView}
              onChange={(v) => setGptView(v as 'train' | 'gen')}
            />
            {gptView === 'gen' && (
              <>
                <Button variant="primary" onClick={() => setGenLen((k) => Math.min(GPT_TOKENS.length + 1, k + 1))} disabled={genLen > GPT_TOKENS.length}>
                  다음 토큰 생성 ▶
                </Button>
                <Button variant="ghost" onClick={() => setGenLen(1)}>
                  ↺ 처음으로
                </Button>
                <span style={{ fontSize: 13, fontWeight: 750, color: 'var(--text-2)' }}>직렬 단계: {Math.min(genLen, GPT_TOKENS.length)}</span>
              </>
            )}
          </>
        ) : (
          <span style={{ fontSize: 13, fontWeight: 650, color: 'var(--text-2)' }}>💡 지도의 줄(행)에 마우스를 올리거나 눌러서, 그 토큰이 누구를 볼 수 있는지 확인해 보세요.</span>
        )}
      </Controls>
    </div>
  );
}

function ProbRow({ label, p, color }: { label: string; p: number; color: string }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '64px 1fr 42px', gap: 8, alignItems: 'center', fontSize: 13, marginBottom: 5 }}>
      <span style={{ fontWeight: 800 }}>{label}</span>
      <span style={{ height: 10, borderRadius: 99, background: 'var(--surface-3)', overflow: 'hidden' }}>
        <span style={{ display: 'block', height: '100%', width: `${p * 100}%`, borderRadius: 99, background: color, transition: 'width .4s var(--ease-out)' }} />
      </span>
      <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 750, color: 'var(--text-2)', textAlign: 'right' }}>{Math.round(p * 100)}%</span>
    </div>
  );
}
