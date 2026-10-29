import { Fragment, useMemo, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Controls, Note, Segmented, Slider } from '../core/ui';
import { Plot } from '../core/plot';
import { useElementSize } from '../core/hooks';
import { useVizColors } from '../core/colors';
import { softmax } from '../core/math';
import type { AlgoMood } from '@/components/mascot/Algo';

type SentKey = 'river' | 'money' | 'vec';
type HeadKey = 'meaning' | 'grammar' | 'neighbor';

const prevHead = (n: number) =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === 0 ? (j === 0 ? 2.4 : 0.1) : j === i - 1 ? 2.4 : j === i ? 1.2 : 0.1)));

const SENTENCES: Record<SentKey, { label: string; tokens: string[]; meaning?: number[]; heads?: Record<HeadKey, number[][]>; vecs?: [number, number][] }> = {
  river: {
    label: '강가 은행',
    tokens: ['나는', '강가', '은행에', '앉아', '강물을', '보았다'],
    meaning: [0, -1, 0, -0.3, -0.9, -0.2],
    heads: {
      meaning: [
        [1.2, 0.3, 0.3, 0.8, 0.2, 0.8],
        [0.2, 1.0, 1.6, 0.4, 2.2, 0.3],
        [0.6, 2.4, 1.0, 1.4, 0.9, 0.2],
        [0.9, 0.6, 1.5, 1.0, 0.3, 0.6],
        [0.2, 2.3, 0.8, 0.3, 1.0, 1.2],
        [0.8, 0.4, 0.3, 0.5, 1.6, 1.0]
      ],
      grammar: [
        [1.0, 0.2, 0.2, 1.6, 0.2, 1.6],
        [0.2, 0.8, 2.0, 0.3, 0.2, 0.2],
        [0.3, 0.6, 0.8, 2.2, 0.2, 0.4],
        [2.2, 0.2, 1.2, 0.8, 0.2, 0.4],
        [0.3, 0.2, 0.2, 0.2, 0.8, 2.3],
        [2.0, 0.2, 0.2, 0.6, 1.5, 0.8]
      ],
      neighbor: prevHead(6)
    }
  },
  money: {
    label: '돈 찾는 은행',
    tokens: ['나는', '은행에', '가서', '돈을', '찾았다'],
    meaning: [0, 0, 0.2, 1, 0.5],
    heads: {
      meaning: [
        [1.2, 0.3, 0.6, 0.2, 0.8],
        [0.4, 1.0, 1.0, 2.5, 1.4],
        [0.6, 1.5, 1.0, 0.3, 0.8],
        [0.2, 2.2, 0.3, 1.0, 1.6],
        [0.7, 1.2, 0.4, 1.8, 1.0]
      ],
      grammar: [
        [1.0, 0.2, 1.5, 0.2, 1.6],
        [0.3, 0.8, 2.2, 0.2, 0.4],
        [2.0, 1.2, 0.8, 0.2, 0.5],
        [0.3, 0.2, 0.2, 0.8, 2.3],
        [2.0, 0.3, 0.6, 1.5, 0.8]
      ],
      neighbor: prevHead(5)
    }
  },
  vec: {
    label: '벡터 a·b·c (점검 2번)',
    tokens: ['a', 'b', 'c'],
    vecs: [
      [1, 0],
      [0.9, 0.1],
      [0, 1]
    ]
  }
};

const HEADS: { value: HeadKey; label: string }[] = [
  { value: 'meaning', label: '🧠 뜻 헤드' },
  { value: 'grammar', label: '📐 문법 헤드' },
  { value: 'neighbor', label: '👈 이웃 헤드' }
];

const HUES = [210, 28, 330, 140, 265, 48];
const tokColor = (i: number, theme: 'light' | 'dark') => (theme === 'dark' ? `hsl(${HUES[i % 6]}, 92%, 62%)` : `hsl(${HUES[i % 6]}, 65%, 55%)`);

/** Tap a word: see who it looks at, then the three steps — dot product, softmax, weighted mix. */
export default function AttentionHeatmap({ expanded }: VizProps) {
  const c = useVizColors();
  const [sent, setSent] = useState<SentKey>('river');
  const [head, setHead] = useState<HeadKey>('meaning');
  const [temp, setTemp] = useState(1);
  const [q, setQ] = useState(2);
  const S = SENTENCES[sent];
  const n = S.tokens.length;
  const query = Math.min(q, n - 1);

  const scores: number[][] = useMemo(() => {
    if (S.vecs) return S.vecs.map((a) => S.vecs!.map((b) => a[0] * b[0] + a[1] * b[1]));
    return S.heads![head];
  }, [S, head]);
  const weights = useMemo(() => scores.map((row) => softmax(row, temp)), [scores, temp]);
  const row = weights[query];
  const maxW = Math.max(...row);
  const top = row.indexOf(maxW);

  const [rowBox, { width }] = useElementSize<HTMLDivElement>();
  const W = Math.max(280, width);
  const slot = W / n;
  const tokenY = 122;
  const xOf = (i: number) => slot * i + slot / 2;

  const newMeaning = S.meaning ? row.reduce((a, w, j) => a + w * S.meaning![j], 0) : 0;
  const vecs = S.vecs;
  const newVec: [number, number] | null = S.vecs ? [row.reduce((a, w, j) => a + w * S.vecs![j][0], 0), row.reduce((a, w, j) => a + w * S.vecs![j][1], 0)] : null;

  const pickSent = (k: SentKey) => {
    setSent(k);
    setQ(k === 'vec' ? 0 : 1 + (k === 'river' ? 1 : 0));
  };

  let note: { text: ReactNode; mood: AlgoMood; tone?: 'good' | 'warn' };
  const qt = S.tokens[query];
  if (sent === 'vec')
    note = {
      text: (
        <>
          쿼리 <strong>{qt}</strong>의 점수(내적): {S.tokens.map((t, j) => `${qt}·${t} = ${scores[query][j].toFixed(1)}`).join(', ')}. 소프트맥스를 지나면{' '}
          <strong>
            {S.tokens.map((t, j) => `${t} ${Math.round(row[j] * 100)}%`).join(' · ')}
          </strong>
          — 새 벡터에는 닮은 벡터의 성분이 가장 많이 섞여요.
        </>
      ),
      mood: 'happy'
    };
  else if (qt === '은행에' && head === 'meaning')
    note =
      sent === 'river'
        ? { text: <>「은행에」가 「<strong>강가</strong>」를 가장 많이 바라봐요({Math.round(row[1] * 100)}%). 그래서 새 벡터는 강가 성분을 듬뿍 머금은 <strong>'강둑'</strong> 쪽으로 움직였어요! 🏞️</>, mood: 'cheer', tone: 'good' }
        : { text: <>같은 「은행에」인데, 이번엔 「<strong>돈을</strong>」을 가장 많이 바라봐요({Math.round(row[3] * 100)}%). 새 벡터는 <strong>'돈 맡기는 은행'</strong> 쪽! 🏦 문맥이 뜻을 정했어요.</>, mood: 'cheer', tone: 'good' };
  else if (head === 'grammar') note = { text: <>문법 헤드는 동사가 주어를, 목적어가 동사를 찾는 식으로 바라봐요. 「{qt}」는 「{S.tokens[top]}」를 가장 많이 봐요.</>, mood: 'think' };
  else if (head === 'neighbor') note = { text: <>이웃 헤드는 주로 <strong>바로 앞 낱말</strong>을 봐요. 헤드마다 다른 관계를 맡는 게 멀티헤드의 핵심이에요.</>, mood: 'think' };
  else note = { text: <>「{qt}」는 「{S.tokens[top]}」를 가장 많이({Math.round(maxW * 100)}%) 바라봐요. 다른 낱말을 눌러 보세요!</>, mood: 'smile' };
  if (temp < 0.45) note = { text: <>온도가 낮아서 거의 <strong>한 낱말에만</strong> 몰렸어요. 내적이 너무 커지면 이렇게 쏠려서, 실제로는 √d로 나눠 알맞게 조절해요.</>, mood: 'wow', tone: 'warn' };
  else if (temp > 2.2) note = { text: <>온도가 높아서 <strong>모두를 거의 똑같이</strong> 바라봐요. 이러면 누가 중요한지 구별이 안 돼요!</>, mood: 'think', tone: 'warn' };


  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 14 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          <Segmented
            label="문장"
            options={(Object.keys(SENTENCES) as SentKey[]).map((k) => ({ value: k, label: SENTENCES[k].label }))}
            value={sent}
            onChange={(v) => pickSent(v as SentKey)}
          />
          {sent !== 'vec' && <Segmented label="헤드(관점)" options={HEADS} value={head} onChange={(v) => setHead(v as HeadKey)} />}
        </div>

        <div ref={rowBox} style={{ width: '100%' }}>
          <div className="vz-panel-title">낱말을 눌러 보세요 — 그 낱말이 누구를 얼마나 바라보는지</div>
          {width > 0 && (
            <svg className="vz-svg" width={W} height={150} viewBox={`0 0 ${W} 150`} role="img" aria-label={`${qt}의 어텐션 가중치`} style={{ touchAction: 'pan-y' }}>
              {row.map((w, j) => {
                if (j === query) return null;
                const x1 = xOf(query);
                const x2 = xOf(j);
                const h = 30 + Math.min(70, Math.abs(x2 - x1) * 0.38);
                const path = `M${x1} ${tokenY - 16} Q${(x1 + x2) / 2} ${tokenY - 16 - h} ${x2} ${tokenY - 16}`;
                return (
                  <g key={j}>
                    <path d={path} fill="none" stroke={c.series[0]} strokeWidth={1.5 + 11 * w} strokeOpacity={0.2 + 0.8 * (w / maxW)} strokeLinecap="round" />
                    {w > 0.06 && (
                      <text x={(x1 + x2) / 2} y={tokenY - 18 - h / 2 - 6} textAnchor="middle" style={{ fontSize: 12, fontWeight: 800, fill: 'var(--text)' }}>
                        {Math.round(w * 100)}%
                      </text>
                    )}
                  </g>
                );
              })}
              {S.tokens.map((t, j) => {
                const sel = j === query;
                const bw = Math.min(slot - 8, Math.max(40, t.length * 15 + 18));
                return (
                  <g key={t} onClick={() => setQ(j)} style={{ cursor: 'pointer' }} role="button" aria-label={`${t} 선택`}>
                    <rect
                      x={xOf(j) - bw / 2}
                      y={tokenY - 16}
                      width={bw}
                      height={32}
                      rx={10}
                      fill={sel ? tokColor(j, c.theme) : 'var(--surface-solid)'}
                      stroke={tokColor(j, c.theme)}
                      strokeWidth={2}
                    />
                    <text x={xOf(j)} y={tokenY + 5} textAnchor="middle" style={{ fontSize: 14, fontWeight: 850, fill: sel ? '#fff' : 'var(--text)' }}>
                      {t}
                    </text>
                    {j === query && (
                      <text x={xOf(j)} y={tokenY + 30} textAnchor="middle" style={{ fontSize: 11, fontWeight: 800, fill: 'var(--brand)' }}>
                        ▲ 쿼리
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          )}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16, alignItems: 'start' }}>
          <div style={{ minWidth: 0 }}>
            <div className="vz-panel-title" style={{ marginBottom: 6 }}>
              어텐션 지도 (행 = 바라보는 쪽, 열 = 바라봐지는 쪽)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: `56px repeat(${n}, minmax(24px, ${n <= 3 ? 64 : expanded ? 58 : 46}px))`, gap: 3, alignItems: 'center', fontSize: 11.5, maxWidth: 56 + n * (n <= 3 ? 67 : expanded ? 61 : 49) }}>
              <span />
              {S.tokens.map((t) => (
                <span key={t} style={{ textAlign: 'center', fontWeight: 750, color: 'var(--text-2)', overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'clip' }}>
                  {t.length > 3 ? t.slice(0, 3) : t}
                </span>
              ))}
              {weights.map((r, i) => (
                <Fragment key={i}>
                  <button
                    key={`l${i}`}
                    type="button"
                    onClick={() => setQ(i)}
                    style={{ textAlign: 'right', paddingRight: 6, fontWeight: i === query ? 900 : 700, color: i === query ? 'var(--brand)' : 'var(--text-2)' }}
                  >
                    {S.tokens[i]}
                  </button>
                  {r.map((w, j) => (
                    <button
                      key={`c${i}-${j}`}
                      type="button"
                      onClick={() => setQ(i)}
                      aria-label={`${S.tokens[i]}→${S.tokens[j]} ${Math.round(w * 100)}%`}
                      style={{
                        aspectRatio: '1',
                        borderRadius: 7,
                        background: `color-mix(in srgb, ${c.series[0]} ${Math.round(8 + 88 * w)}%, var(--surface-3))`,
                        outline: i === query ? `2px solid ${c.brand}` : 'none',
                        outlineOffset: 1,
                        color: w > 0.35 ? '#fff' : 'var(--text-2)',
                        fontSize: 10.5,
                        fontWeight: 800,
                        fontVariantNumeric: 'tabular-nums'
                      }}
                    >
                      {Math.round(w * 100)}
                    </button>
                  ))}
                </Fragment>
              ))}
            </div>
          </div>

          <div style={{ display: 'grid', gap: 12, minWidth: 0 }}>
            <Step n="1" title={sent === 'vec' ? '관련도 = 내적' : '관련도 점수 (쿼리·키 내적)'}>
              {S.tokens.map((t, j) => (
                <Bar key={t} label={t} value={scores[query][j]} max={Math.max(...scores[query].map(Math.abs), 1)} text={scores[query][j].toFixed(1)} color={c.series[3]} />
              ))}
            </Step>
            <Step n="2" title="소프트맥스 → 가중치 (합 = 100%)">
              {S.tokens.map((t, j) => (
                <Bar key={t} label={t} value={row[j]} max={1} text={`${Math.round(row[j] * 100)}%`} color={c.series[0]} />
              ))}
            </Step>
            <Step n="3" title={`가중 평균 → 「${qt}」의 새 벡터`}>
              <div style={{ display: 'flex', height: 22, borderRadius: 8, overflow: 'hidden', border: '1px solid var(--border)' }}>
                {row.map((w, j) => (
                  <span
                    key={j}
                    title={`${S.tokens[j]} ${Math.round(w * 100)}%`}
                    style={{ flex: `${w} 1 0`, background: tokColor(j, c.theme), transition: 'flex-grow .4s', display: 'grid', placeItems: 'center', color: '#fff', fontSize: 10.5, fontWeight: 800, overflow: 'hidden', whiteSpace: 'nowrap' }}
                  >
                    {w > 0.14 ? S.tokens[j] : ''}
                  </span>
                ))}
              </div>
              {S.meaning && qt === '은행에' && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontWeight: 800, color: 'var(--text-2)' }}>
                    <span>🏞️ 강둑</span>
                    <span>🏦 돈 맡기는 곳</span>
                  </div>
                  <div style={{ position: 'relative', height: 12, marginTop: 4, borderRadius: 99, background: `linear-gradient(90deg, ${c.series[1]}, var(--surface-3), ${c.series[4]})` }}>
                    <span
                      style={{
                        position: 'absolute',
                        top: -5,
                        left: `calc(${((newMeaning + 1) / 2) * 100}% - 11px)`,
                        width: 22,
                        height: 22,
                        borderRadius: 99,
                        background: 'var(--surface-solid)',
                        border: `3px solid ${c.ink}`,
                        transition: 'left .5s var(--ease-spring)'
                      }}
                    />
                  </div>
                </div>
              )}
              {newVec && vecs && (
                <Plot xDomain={[-0.2, 1.2]} yDomain={[-0.2, 1.2]} height={190} margin={{ top: 8, right: 10, bottom: 30, left: 34 }} xTicks={3} yTicks={3} ariaLabel="2차원 벡터 a, b, c와 새 벡터">
                  {({ x, y }) => (
                    <g>
                      <defs>
                        <marker id="vz-attn-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="11" markerHeight="11" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
                          <path d="M0 0 L10 5 L0 10 z" fill="currentColor" />
                        </marker>
                      </defs>
                      {vecs.map(([vx, vy], j) => (
                        <g key={j} style={{ color: tokColor(j, c.theme) }}>
                          <line x1={x(0)} y1={y(0)} x2={x(vx)} y2={y(vy)} stroke="currentColor" strokeWidth={3} markerEnd="url(#vz-attn-arrow)" />
                          <text x={x(vx) + (j === 1 ? 8 : 6)} y={y(vy) + (j === 1 ? 16 : -6)} style={{ fontSize: 13, fontWeight: 900, fill: 'currentColor' }}>
                            {S.tokens[j]}
                          </text>
                        </g>
                      ))}
                      <g style={{ color: c.ink }}>
                        <line x1={x(0)} y1={y(0)} x2={x(newVec[0])} y2={y(newVec[1])} stroke="currentColor" strokeWidth={2.5} strokeDasharray="5 4" markerEnd="url(#vz-attn-arrow)" />
                        <text x={x(newVec[0]) - 4} y={y(newVec[1]) - 12} textAnchor="end" style={{ fontSize: 12, fontWeight: 900, fill: 'currentColor' }}>
                          새 {qt} ({newVec[0].toFixed(2)}, {newVec[1].toFixed(2)})
                        </text>
                      </g>
                    </g>
                  )}
                </Plot>
              )}
            </Step>
          </div>
        </div>

        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        <Slider
          label="온도 (√d로 나누기)"
          value={temp}
          min={0.25}
          max={3}
          step={0.05}
          onChange={setTemp}
          format={(v) => v.toFixed(2)}
          hint="작을수록 한 곳에 몰리고, 클수록 고르게 퍼져요"
        />
      </Controls>
    </div>
  );
}

function Step({ n, title, children }: { n: string; title: string; children: ReactNode }) {
  return (
    <div>
      <div className="vz-panel-title" style={{ marginBottom: 6 }}>
        <span style={{ color: 'var(--brand)', marginRight: 4 }}>{n}</span>
        {title}
      </div>
      <div style={{ display: 'grid', gap: 4 }}>{children}</div>
    </div>
  );
}

function Bar({ label, value, max, text, color }: { label: string; value: number; max: number; text: string; color: string }) {
  const w = Math.max(0, value) / (max || 1);
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr 42px', gap: 8, alignItems: 'center', fontSize: 12.5 }}>
      <span style={{ fontWeight: 800, overflow: 'hidden', whiteSpace: 'nowrap' }}>{label}</span>
      <span style={{ height: 9, borderRadius: 99, background: 'var(--surface-3)', overflow: 'hidden' }}>
        <span style={{ display: 'block', height: '100%', width: `${w * 100}%`, borderRadius: 99, background: color, transition: 'width .35s var(--ease-out)' }} />
      </span>
      <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 750, color: 'var(--text-2)', textAlign: 'right' }}>{text}</span>
    </div>
  );
}
