import { useMemo, useState } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Note, Segmented, Slider, Stat } from '../core/ui';
import { useElementSize, useInterval } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { rng } from '../core/math';

const N = 40;
type Bin = 'all' | 'train' | 'val' | 'test';

/** Shuffle data into train / validation / test, and watch 5-fold cross-validation. */
export default function DataSplit(_: VizProps) {
  const [mode, setMode] = useState<'split' | 'cv'>('split');
  return (
    <div>
      <div style={{ padding: '12px 14px 0' }}>
        <Segmented
          options={[
            { value: 'split', label: '🗂️ 셋으로 나누기' },
            { value: 'cv', label: '🔁 교차검증' }
          ]}
          value={mode}
          onChange={(v) => setMode(v as 'split' | 'cv')}
        />
      </div>
      {mode === 'split' ? <Split /> : <CrossVal />}
    </div>
  );
}

function Split() {
  const c = useVizColors();
  const [box, { width }] = useElementSize<HTMLDivElement>();
  const [trainPct, setTrainPct] = useState(70);
  const [split, setSplit] = useState(false);
  const [seed, setSeed] = useState(3);
  const [peeked, setPeeked] = useState(false);
  const order = useMemo(() => {
    const r = rng(seed);
    const idx = Array.from({ length: N }, (_, i) => i);
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(r.next() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx;
  }, [seed]);
  const nTrain = Math.round((N * trainPct) / 100);
  const nVal = Math.round((N - nTrain) / 2);
  const binOf = (i: number): Bin => {
    if (!split) return 'all';
    const rank = order.indexOf(i);
    return rank < nTrain ? 'train' : rank < nTrain + nVal ? 'val' : 'test';
  };
  const W = Math.max(width, 300);
  const cell = W < 420 ? 20 : 26;
  const gap = W < 420 ? 4 : 6;
  // tall enough for the fullest panel, or for the unsplit 10-column grid
  const colsFor = (frac: number) => Math.max(1, Math.floor((W * frac - 12) / (cell + gap)));
  const rowsNeeded = split
    ? Math.max(Math.ceil(nTrain / colsFor(0.52)), Math.ceil(nVal / colsFor(0.22)), Math.ceil((N - nTrain - nVal) / colsFor(0.22)))
    : Math.ceil(N / 10);
  const H = Math.max(170, 54 + rowsNeeded * (cell + gap) + 18);
  // panel geometry when split
  const panels: Record<Exclude<Bin, 'all'>, { x: number; w: number; label: string; emoji: string; color: string }> = {
    train: { x: 0, w: W * 0.52, label: '훈련 — 교재', emoji: '📘', color: c.series[0] },
    val: { x: W * 0.54, w: W * 0.22, label: '검증 — 모의고사', emoji: '📝', color: c.series[3] },
    test: { x: W * 0.78, w: W * 0.22, label: '테스트 — 수능', emoji: '🎓', color: c.series[2] }
  };
  const posOf = (i: number) => {
    const b = binOf(i);
    if (b === 'all') {
      const cols = 10;
      const gx = (W - cols * (cell + gap)) / 2;
      return { x: gx + (i % cols) * (cell + gap), y: 54 + Math.floor(i / cols) * (cell + gap) };
    }
    const p = panels[b];
    const members = order.filter((j) => binOf(j) === b);
    const k = members.indexOf(i);
    const cols = Math.max(1, Math.floor((p.w - 12) / (cell + gap)));
    return { x: p.x + 8 + (k % cols) * (cell + gap), y: 54 + Math.floor(k / cols) * (cell + gap) };
  };

  return (
    <div>
      <div ref={box} style={{ position: 'relative', height: H, margin: '10px 14px 0' }} aria-label="데이터 카드 40장" role="img">
        {split &&
          (Object.keys(panels) as Exclude<Bin, 'all'>[]).map((b) => {
            const p = panels[b];
            const dirty = b === 'test' && peeked;
            return (
              <div
                key={b}
                style={{
                  position: 'absolute',
                  left: p.x,
                  top: 0,
                  width: p.w,
                  height: H,
                  borderRadius: 16,
                  border: `2px ${dirty ? 'dashed' : 'solid'} ${dirty ? c.neg : p.color}`,
                  background: withAlpha(dirty ? c.neg : p.color, 0.08),
                  transition: 'all .4s'
                }}
              >
                <div style={{ padding: '8px 10px', fontSize: W < 420 ? 11.5 : 13, fontWeight: 850, lineHeight: 1.3 }}>
                  {p.emoji} {p.label}
                  <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--text-2)' }}>{dirty ? '⚠️ 오염됨!' : `${b === 'train' ? nTrain : b === 'val' ? nVal : N - nTrain - nVal}장`}</div>
                </div>
              </div>
            );
          })}
        {!split && (
          <div style={{ position: 'absolute', left: 0, right: 0, top: 10, textAlign: 'center', fontSize: 13.5, fontWeight: 800, color: 'var(--text-2)' }}>
            가진 데이터 40장 — 아직 나누지 않았어요
          </div>
        )}
        {Array.from({ length: N }, (_, i) => {
          const { x, y } = posOf(i);
          const b = binOf(i);
          const col = b === 'all' ? c.series[4] : panels[b].color;
          return (
            <div
              key={i}
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                width: cell,
                height: cell,
                borderRadius: 7,
                background: withAlpha(col, b === 'test' && peeked ? 0.3 : 0.85),
                border: '2px solid var(--surface-solid)',
                boxShadow: 'var(--shadow-sm)',
                transform: `translate(${x}px, ${y}px)`,
                transition: `transform .7s cubic-bezier(.22,1,.36,1) ${(order.indexOf(i) % 12) * 18}ms, background-color .4s`
              }}
            />
          );
        })}
      </div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 8 }}>
          {[
            { e: '📘', t: '훈련', d: '모델이 공부하는 교재. 파라미터가 여기서 학습돼요.', col: c.series[0] },
            { e: '📝', t: '검증', d: '모의고사. 어떤 모델이 나은지, 언제 멈출지 같은 결정을 여기 성적으로 해요.', col: c.series[3] },
            { e: '🎓', t: '테스트', d: '수능. 모든 결정이 끝난 뒤 딱 한 번만 열어요.', col: c.series[2] }
          ].map((s) => (
            <div key={s.t} style={{ padding: '10px 12px', borderRadius: 14, borderLeft: `5px solid ${s.col}`, background: 'var(--surface-2)', fontSize: 13.5, lineHeight: 1.6 }}>
              <strong>
                {s.e} {s.t}
              </strong>
              <div style={{ color: 'var(--text-2)' }}>{s.d}</div>
            </div>
          ))}
        </div>
        {peeked ? (
          <Note mood="wow" tone="bad">
            테스트 점수를 보고 모델을 고쳤어요! 이제 이 테스트 세트는 더 이상 "처음 보는 문제"가 아니에요. 여기서 나온 점수는 실제보다 좋게 부풀려져요 — 진짜 실력을 재려면 새 테스트 데이터가 필요해요.
          </Note>
        ) : split ? (
          <Note mood="happy">
            나눴어요! 훈련으로 공부하고, 검증으로 고르고, 테스트는 마지막에 한 번만. 훈련 성적은 좋은데 검증 성적이 나쁘면 과적합이에요.
          </Note>
        ) : (
          <Note mood="smile">"🔀 섞어서 나누기"를 눌러 보세요. 섞지 않고 앞뒤로만 자르면 한쪽에 비슷한 데이터만 몰릴 수 있어요.</Note>
        )}
      </div>
      <Controls>
        <Button variant="primary" onClick={() => { setSeed((s) => s + 1); setSplit(true); setPeeked(false); }}>
          🔀 섞어서 나누기
        </Button>
        <Slider label="훈련 비율" value={trainPct} min={50} max={80} step={5} onChange={setTrainPct} format={(v) => `${v}% / ${Math.round((100 - v) / 2)}% / ${Math.round((100 - v) / 2)}%`} />
        <Button variant="danger" onClick={() => setPeeked(true)} disabled={!split || peeked}>
          🙈 테스트 점수 보고 모델 고치기
        </Button>
        <Button variant="ghost" onClick={() => { setSplit(false); setPeeked(false); }}>
          ↺ 처음으로
        </Button>
      </Controls>
    </div>
  );
}

const FOLD_SCORES = [0.81, 0.78, 0.84, 0.8, 0.82];

function CrossVal() {
  const c = useVizColors();
  const [round, setRound] = useState(-1);
  const [playing, setPlaying] = useState(false);
  useInterval(
    () => {
      if (round >= 4) setPlaying(false);
      else setRound(round + 1);
    },
    playing ? 1100 : null
  );
  const done = FOLD_SCORES.slice(0, round + 1);
  const avg = done.length ? done.reduce((a, b) => a + b, 0) / done.length : 0;
  const sd = done.length > 1 ? Math.sqrt(done.reduce((a, b) => a + (b - avg) ** 2, 0) / (done.length - 1)) : 0;
  return (
    <div>
      <div style={{ padding: '12px 14px 0', display: 'grid', gap: 8 }}>
        {Array.from({ length: 5 }, (_, r) => {
          const active = r === round;
          return (
            <div key={r} style={{ display: 'flex', alignItems: 'center', gap: 10, opacity: round === -1 || r <= round ? 1 : 0.45, transition: 'opacity .3s' }}>
              <span style={{ width: 54, fontSize: 13, fontWeight: 850, color: active ? 'var(--text)' : 'var(--text-2)' }}>{r + 1}번째</span>
              <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 4 }}>
                {Array.from({ length: 5 }, (_, k) => {
                  const isVal = k === r;
                  return (
                    <div
                      key={k}
                      style={{
                        height: 30,
                        borderRadius: 8,
                        display: 'grid',
                        placeItems: 'center',
                        fontSize: 11.5,
                        fontWeight: 800,
                        color: isVal ? '#fff' : 'var(--text-2)',
                        background: isVal ? c.series[3] : withAlpha(c.series[0], 0.25),
                        boxShadow: active && isVal ? `0 0 0 3px ${withAlpha(c.series[3], 0.35)}` : 'none',
                        transition: 'box-shadow .3s'
                      }}
                    >
                      {isVal ? '모의고사' : '공부'}
                    </div>
                  );
                })}
              </div>
              <span style={{ width: 54, textAlign: 'right', fontSize: 14, fontWeight: 900, color: c.series[3] }}>{r <= round ? FOLD_SCORES[r].toFixed(2) : ''}</span>
            </div>
          );
        })}
      </div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div className="vz-stats">
          <Stat label="끝낸 라운드" value={`${done.length}/5`} />
          <Stat label="평균 점수" value={done.length ? avg.toFixed(3) : '—'} tone={done.length === 5 ? 'good' : 'brand'} sub={done.length > 1 ? `± ${sd.toFixed(3)} (흔들림)` : undefined} />
        </div>
        <Note mood={done.length === 5 ? 'cheer' : 'smile'} tone={done.length === 5 ? 'good' : undefined}>
          {done.length === 5
            ? `다섯 조각이 한 번씩 모의고사가 됐어요. 평균 ${avg.toFixed(2)}점 — 데이터를 버리지 않고도 공정하게 실력을 쟀어요. 점수가 라운드마다 조금씩 다른 것도 보이죠? 그래서 평균을 내요.`
            : '데이터가 적을 땐 다섯 조각으로 나눠, 조각마다 한 번씩 모의고사로 쓰고 나머지로 공부해요. ▶를 눌러 보세요.'}
        </Note>
      </div>
      <Controls>
        <Button variant="primary" onClick={() => { if (round >= 4) setRound(-1); setPlaying((p) => !p); }}>
          {playing ? '❚❚ 멈춤' : round >= 4 ? '↺ 다시 보기' : '▶ 차례로 보기'}
        </Button>
        <Button onClick={() => setRound((r) => Math.min(4, r + 1))} disabled={playing || round >= 4}>
          한 라운드 ⟶
        </Button>
      </Controls>
    </div>
  );
}
