import { useState } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { useElementSize } from '../core/hooks';
import { useVizColors } from '../core/colors';
import { rng, softmax } from '../core/math';

const CLASSES = [
  { emoji: '🐱', name: '고양이' },
  { emoji: '🐶', name: '개' },
  { emoji: '🐦', name: '새' }
];
const PRESETS: { label: string; z: number[] }[] = [
  { label: '책 예시 (2, 1, 0.1)', z: [2, 1, 0.1] },
  { label: '동점 (1, 1, 1)', z: [1, 1, 1] },
  { label: '압승 (10, 1, 1)', z: [10, 1, 1] }
];
const MAX_SURPRISE = 5;

/** Scores → softmax probabilities, with a temperature dial and a cross-entropy "surprise" meter. */
export default function SoftmaxTemperature({ expanded }: VizProps) {
  const c = useVizColors();
  const [z, setZ] = useState<number[]>([2, 1, 0.1]);
  const [logT, setLogT] = useState(0);
  const [answer, setAnswer] = useState(0);
  const [showCalc, setShowCalc] = useState(false);
  const [draws, setDraws] = useState<{ counts: number[]; n: number } | null>(null);
  const [drawSeed, setDrawSeed] = useState(1);
  const [box, { width }] = useElementSize<HTMLDivElement>();

  const T = 10 ** logT;
  const p = softmax(z, T);
  const colors = [c.series[0], c.series[1], c.series[2]];
  const surprise = -Math.log(Math.max(p[answer], 1e-12));
  const winner = p.indexOf(Math.max(...p));
  const wide = width >= 560;
  const exRaw = z.map((v) => Math.exp(v / T));
  const exSum = exRaw.reduce((s, v) => s + v, 0);

  const sample = () => {
    const r = rng(drawSeed * 7919 + 13);
    const counts = [0, 0, 0];
    for (let i = 0; i < 20; i++) {
      let u = r.next();
      let k = 0;
      while (k < p.length - 1 && u > p[k]) {
        u -= p[k];
        k++;
      }
      counts[k]++;
    }
    setDraws({ counts, n: 20 });
    setDrawSeed((s) => s + 1);
  };

  const setScore = (i: number, v: number) => {
    setZ((zs) => zs.map((x, k) => (k === i ? v : x)));
    setDraws(null);
  };

  let note: { text: string; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' | 'bad' };
  if (T < 0.45) note = { text: `온도가 낮아 "승자 독식"이에요 🧊 점수가 조금만 앞서도 ${CLASSES[winner].emoji}이(가) 확률을 거의 다 가져가요.`, mood: 'think' };
  else if (T > 2.5) note = { text: '온도가 높아 분포가 평평해졌어요 🔥 점수 차이가 흐려져서 골고루 뽑혀요. ChatGPT의 temperature 다이얼이 바로 이거예요!', mood: 'wow' };
  else if (surprise > 2.3) note = { text: `정답(${CLASSES[answer].emoji})에 겨우 ${Math.round(p[answer] * 100)}%만 줬어요. 확신에 찬 오답이라 놀람(손실)이 아주 커요 😱`, mood: 'wow', tone: 'bad' };
  else if (surprise < 0.25) note = { text: `정답(${CLASSES[answer].emoji})에 ${Math.round(p[answer] * 100)}%를 줬어요. 거의 놀라지 않았어요 — 손실이 작아요 😌`, mood: 'cheer', tone: 'good' };
  else note = { text: '점수 막대를 움직여 보세요. 점수는 아무 숫자나 되지만, 확률은 언제나 0~1이고 합이 100%예요. 이름을 누르면 정답이 바뀌어요.', mood: 'smile' };

  return (
    <div ref={box}>
      <div className="vz-pad" style={{ display: 'grid', gap: 10 }}>
        <div style={{ display: 'grid', gridTemplateColumns: wide ? '1.15fr 0.85fr' : '1fr', gap: wide ? '4px 18px' : 4 }}>
          {wide && (
            <>
              <div className="vz-panel-title">① 모델이 낸 점수 (아무 숫자) — 끌어서 바꾸기</div>
              <div className="vz-panel-title">② 소프트맥스를 지난 확률</div>
            </>
          )}
          {CLASSES.map((k, i) => (
            <div key={k.name} style={{ display: 'contents' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                <button
                  type="button"
                  onClick={() => setAnswer(i)}
                  aria-pressed={answer === i}
                  title="정답으로 정하기"
                  style={{
                    flex: '0 0 auto',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 4,
                    padding: '4px 9px',
                    borderRadius: 999,
                    border: `2px solid ${answer === i ? colors[i] : 'var(--border)'}`,
                    background: answer === i ? 'var(--surface-solid)' : 'transparent',
                    fontSize: 13.5,
                    fontWeight: 800,
                    color: 'var(--text)',
                    cursor: 'pointer',
                    minWidth: 96
                  }}
                >
                  <span style={{ fontSize: 18 }}>{k.emoji}</span>
                  {k.name}
                  {answer === i && <span style={{ fontSize: 11, color: colors[i] }}>✓정답</span>}
                </button>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <Slider label="점수" value={z[i]} min={-3} max={10} step={0.1} onChange={(v) => setScore(i, v)} format={(v) => v.toFixed(1)} accent={colors[i]} />
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingBottom: wide ? 0 : 8 }}>
                <div style={{ flex: 1, height: 22, borderRadius: 8, background: 'var(--surface-3)', overflow: 'hidden', border: '1px solid var(--border)' }}>
                  <div
                    style={{
                      width: `${p[i] * 100}%`,
                      height: '100%',
                      background: colors[i],
                      borderRadius: 8,
                      transition: 'width 0.25s var(--ease-out, ease)'
                    }}
                  />
                </div>
                <span style={{ width: 52, textAlign: 'right', fontWeight: 850, fontVariantNumeric: 'tabular-nums', fontSize: 15 }}>{(p[i] * 100).toFixed(p[i] < 0.1 ? 1 : 0)}%</span>
              </div>
            </div>
          ))}
        </div>

        {/* 100% stacked bar */}
        <div>
          <div className="vz-panel-title" style={{ marginBottom: 6 }}>
            확률을 한 줄로 이으면 언제나 딱 100%
          </div>
          <div style={{ display: 'flex', height: 34, borderRadius: 12, overflow: 'hidden', border: '1px solid var(--border)' }}>
            {p.map((v, i) => (
              <div
                key={i}
                style={{
                  width: `${v * 100}%`,
                  background: colors[i],
                  display: 'grid',
                  placeItems: 'center',
                  fontSize: 17,
                  overflow: 'hidden',
                  transition: 'width 0.25s var(--ease-out, ease)'
                }}
                title={`${CLASSES[i].name} ${(v * 100).toFixed(1)}%`}
              >
                {v > 0.07 ? CLASSES[i].emoji : ''}
              </div>
            ))}
          </div>
        </div>

        {showCalc && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13.5, fontVariantNumeric: 'tabular-nums', minWidth: 340 }}>
              <thead>
                <tr style={{ color: 'var(--text-3)', textAlign: 'right' }}>
                  <th style={{ textAlign: 'left', padding: '4px 6px' }}>선택지</th>
                  <th style={{ padding: '4px 6px' }}>점수 z</th>
                  <th style={{ padding: '4px 6px' }}>z ÷ T</th>
                  <th style={{ padding: '4px 6px' }}>e^(z÷T)</th>
                  <th style={{ padding: '4px 6px' }}>÷ 합 = 확률</th>
                </tr>
              </thead>
              <tbody>
                {CLASSES.map((k, i) => (
                  <tr key={k.name} style={{ borderTop: '1px solid var(--border)', textAlign: 'right' }}>
                    <td style={{ textAlign: 'left', padding: '4px 6px', fontWeight: 800 }}>
                      {k.emoji} {k.name}
                    </td>
                    <td style={{ padding: '4px 6px' }}>{z[i].toFixed(1)}</td>
                    <td style={{ padding: '4px 6px' }}>{(z[i] / T).toFixed(2)}</td>
                    <td style={{ padding: '4px 6px' }}>{exRaw[i] > 1e5 ? exRaw[i].toExponential(2) : exRaw[i].toFixed(2)}</td>
                    <td style={{ padding: '4px 6px', fontWeight: 850, color: colors[i] }}>{p[i].toFixed(3)}</td>
                  </tr>
                ))}
                <tr style={{ borderTop: '2px solid var(--border-strong)', textAlign: 'right', color: 'var(--text-2)' }}>
                  <td style={{ textAlign: 'left', padding: '4px 6px' }}>합</td>
                  <td />
                  <td />
                  <td style={{ padding: '4px 6px' }}>{exSum > 1e5 ? exSum.toExponential(2) : exSum.toFixed(2)}</td>
                  <td style={{ padding: '4px 6px' }}>1.000</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12, alignItems: 'center' }}>
          <SurpriseMeter value={surprise} p={p[answer]} emoji={CLASSES[answer].emoji} expanded={!!expanded} />
          <div style={{ display: 'grid', gap: 10 }}>
            <div className="vz-stats">
              <Stat label="온도 T" value={T.toFixed(T < 1 ? 2 : 1)} tone="brand" />
              <Stat label="1등" value={`${CLASSES[winner].emoji} ${Math.round(p[winner] * 100)}%`} />
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
              <Button size="sm" onClick={sample}>
                🎲 이 확률로 20번 뽑기
              </Button>
              {draws && (
                <span style={{ display: 'inline-flex', flexWrap: 'wrap', gap: 6, fontWeight: 800, fontSize: 14 }}>
                  {draws.counts.map((n, i) => (
                    <span key={i} style={{ padding: '2px 8px', borderRadius: 999, background: 'var(--surface-3)', border: `1.5px solid ${colors[i]}` }}>
                      {CLASSES[i].emoji}×{n}
                    </span>
                  ))}
                </span>
              )}
            </div>
            {draws && <div style={{ fontSize: 12.5, color: 'var(--text-2)' }}>온도를 바꾸고 다시 뽑아 보세요. 높을수록 결과가 다양해져요 (GPT가 다음 단어를 고르는 방식!).</div>}
          </div>
        </div>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        <Slider
          label="온도 T (점수를 T로 나눠요)"
          value={logT}
          min={-1}
          max={1}
          step={0.01}
          onChange={(v) => {
            setLogT(v);
            setDraws(null);
          }}
          format={() => `T = ${T.toFixed(T < 1 ? 2 : 1)}`}
        />
        <Segmented
          label="온도 빠르게"
          options={[
            { value: -0.52, label: '🧊 0.3' },
            { value: 0, label: '1' },
            { value: 1, label: '🔥 10' }
          ]}
          value={[-0.52, 0, 1].find((v) => Math.abs(v - logT) < 0.005) ?? 99}
          onChange={(v) => {
            setLogT(v);
            setDraws(null);
          }}
        />
        <Segmented
          label="점수 예시"
          options={PRESETS.map((pr, i) => ({ value: i, label: pr.label }))}
          value={PRESETS.findIndex((pr) => pr.z.every((v, i) => Math.abs(v - z[i]) < 1e-9))}
          onChange={(i) => {
            setZ([...PRESETS[i].z]);
            setDraws(null);
          }}
        />
        <Toggle label="계산 과정 보기" checked={showCalc} onChange={setShowCalc} />
      </Controls>
    </div>
  );
}

function SurpriseMeter({ value, p, emoji, expanded }: { value: number; p: number; emoji: string; expanded: boolean }) {
  const c = useVizColors();
  const W = 260;
  const H = 150;
  const cx = W / 2;
  const cy = 122;
  const r = 96;
  const v = Math.min(value, MAX_SURPRISE);
  const ang = (t: number) => Math.PI * (1 - t / MAX_SURPRISE); // 0 → left, max → right
  const pt = (t: number, rr = r): [number, number] => [cx + rr * Math.cos(ang(t)), cy - rr * Math.sin(ang(t))];
  const arc = (a: number, b: number) => {
    const [x0, y0] = pt(a);
    const [x1, y1] = pt(b);
    return `M${x0},${y0} A${r},${r} 0 0 1 ${x1},${y1}`;
  };
  const face = value < 0.3 ? '😌' : value < 0.8 ? '🙂' : value < 1.8 ? '😮' : '😱';
  const [nx, ny] = pt(v, r - 14);
  return (
    <div style={{ display: 'grid', justifyItems: 'center', gap: 2 }}>
      <div className="vz-panel-title" style={{ justifySelf: 'start' }}>
        ③ 놀람 미터 = 손실(교차 엔트로피)
      </div>
      <svg className="vz-svg" viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: expanded ? 360 : 300 }} role="img" aria-label={`놀람 정도 ${value.toFixed(2)}`}>
        <path d={arc(0, 0.7)} fill="none" stroke={c.pos} strokeWidth={16} strokeLinecap="round" />
        <path d={arc(0.8, 2.2)} fill="none" strokeWidth={16} style={{ stroke: 'var(--lemon)' }} />
        <path d={arc(2.3, MAX_SURPRISE)} fill="none" stroke={c.neg} strokeWidth={16} strokeLinecap="round" />
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={c.ink} strokeWidth={4} strokeLinecap="round" style={{ transition: 'all 0.25s' }} />
        <circle cx={cx} cy={cy} r={8} fill={c.ink} />
        <text x={cx} y={cy - 34} textAnchor="middle" style={{ fontSize: 30 }}>
          {face}
        </text>
        <text x={pt(0, r + 2)[0] + 4} y={cy + 20} textAnchor="middle" className="vz-tick" style={{ fill: c.text }}>
          0
        </text>
        <text x={pt(MAX_SURPRISE, r + 2)[0] - 4} y={cy + 20} textAnchor="middle" className="vz-tick" style={{ fill: c.text }}>
          5+
        </text>
      </svg>
      <div style={{ fontSize: 13.5, fontWeight: 750, textAlign: 'center' }}>
        놀람 = -log(정답 {emoji}의 확률) = -log({p < 0.001 ? p.toExponential(1) : p.toFixed(3)}) = <b style={{ fontSize: 16 }}>{value > 99 ? '∞' : value.toFixed(2)}</b>
      </div>
    </div>
  );
}
