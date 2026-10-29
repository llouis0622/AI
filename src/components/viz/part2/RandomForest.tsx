import { useMemo, useState } from 'react';
import type { VizProps } from '../Viz';
import { Plot, linePath } from '../core/plot';
import { Button, Controls, Legend, Note, Slider, Stat, Toggle } from '../core/ui';
import { useElementSize } from '../core/hooks';
import { useVizColors } from '../core/colors';
import { range } from '../core/math';
import { FieldCanvas, ClassLegend } from './_Field';
import { makeDataset, splitTrainTest } from './_data';
import { accuracy, buildForest, buildTree, predictForest, predictTree } from './_tree';

const MAX_TREES = 60;
const DEPTH = 7;
const DATA = makeDataset('noisy', 140, 3);
const { train: TRAIN, test: TEST } = splitTrainTest(DATA);
const SINGLE = buildTree(TRAIN, { maxDepth: DEPTH });
const SINGLE_ACC = accuracy((x, y) => predictTree(SINGLE, x, y), TEST);
const MINI = 6;

/** A small random forest: deep, jumpy trees that vote — more trees, smoother boundary. */
export default function RandomForest({ expanded }: VizProps) {
  const c = useVizColors();
  const [n, setN] = useState(1);
  const [seed, setSeed] = useState(4);
  const [randomFeat, setRandomFeat] = useState(true);
  const [showTest, setShowTest] = useState(false);
  const [probe, setProbe] = useState<[number, number] | null>(null);
  const [box, { width }] = useElementSize<HTMLDivElement>();

  const forest = useMemo(() => buildForest(TRAIN, MAX_TREES, DEPTH, seed, randomFeat), [seed, randomFeat]);
  const trees = useMemo(() => forest.map((f) => f.tree), [forest]);
  const curve = useMemo(() => range(MAX_TREES).map((i) => accuracy((x, y) => predictForest(trees.slice(0, i + 1), x, y), TEST)), [trees]);
  const treeAcc = useMemo(() => trees.slice(0, MINI).map((t) => accuracy((x, y) => predictTree(t, x, y), TEST)), [trees]);
  const active = useMemo(() => trees.slice(0, n), [trees, n]);
  const acc = curve[n - 1];
  const predict = useMemo(() => (x: number, y: number) => predictForest(active, x, y), [active]);

  const height = Math.round(expanded ? Math.min(560, Math.max(320, width * 0.5)) : Math.min(400, Math.max(260, width * 0.56)));
  const miniW = Math.max(70, Math.min(150, (width - 32 - (MINI - 1) * 8) / MINI));
  const votes = probe ? active.filter((t) => predictTree(t, probe[0], probe[1]) >= 0.5).length : 0;

  let note: { text: string; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer'; tone?: 'good' | 'warn' | 'bad' };
  if (n === 1)
    note = { text: '나무 1그루는 들쭉날쭉해요. 잡음 점까지 외워서 엉뚱한 섬이 생겨요. 아래 작은 나무들도 보세요 — 나무마다 생김새가 전부 달라요! 나무 수를 늘려 볼까요?', mood: 'think' };
  else if (n < 10)
    note = { text: `${n}그루가 투표 중이에요. 나무마다 다른 표본(중복 허용 뽑기)으로 자라서 서로 "다르게" 틀려요. 투표하면 그 실수들이 서로 지워지기 시작해요.`, mood: 'happy' };
  else
    note = {
      text: `${n}그루의 다수결! 경계가 부드러워지고 시험 정확도가 ${Math.round(acc * 100)}%예요 — 혼자 자란 깊은 나무(${Math.round(SINGLE_ACC * 100)}%)보다 ${acc > SINGLE_ACC ? '훨씬 나아요 🎉' : '안정적이에요'}.${randomFeat ? '' : ' (특징 무작위 고르기를 끄면 나무들이 서로 닮아서, 투표 효과가 줄어들 수 있어요.)'}`,
      mood: 'cheer',
      tone: 'good'
    };

  return (
    <div ref={box}>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <div className="vz-panel-title" style={{ marginBottom: 6 }}>
          🌲 숲 전체의 투표 결과 (지도를 누르면 그 자리의 표가 보여요)
        </div>
        <FieldCanvas
          height={height}
          points={TRAIN}
          hollow={showTest ? TEST : undefined}
          predict={predict}
          res={60}
          onPick={(x, y) => setProbe([x, y])}
          deps={[n, seed, randomFeat, showTest]}
          ariaLabel="랜덤 포레스트의 투표로 칠해진 지도"
          xLabel="크기 →"
          yLabel="↑ 울퉁불퉁한 정도"
        >
          {probe && (
            <span
              style={{
                position: 'absolute',
                left: `${probe[0] * 100}%`,
                top: `${(1 - probe[1]) * 100}%`,
                transform: `translate(${probe[0] > 0.6 ? 'calc(-100% - 10px)' : '10px'}, -50%)`,
                padding: '6px 10px',
                borderRadius: 12,
                background: 'var(--surface-solid)',
                border: '2px solid var(--lemon)',
                boxShadow: 'var(--shadow-sm)',
                pointerEvents: 'none',
                fontSize: 13,
                fontWeight: 800,
                whiteSpace: 'nowrap',
                display: 'grid',
                gap: 4
              }}
            >
              <span>
                <span style={{ color: c.series[2] }}>▲ {votes}표</span> : <span style={{ color: c.series[3] }}>● {n - votes}표</span>
              </span>
              <span style={{ display: 'flex', height: 6, width: 110, borderRadius: 3, overflow: 'hidden' }}>
                <span style={{ width: `${(votes / n) * 100}%`, background: c.series[2] }} />
                <span style={{ flex: 1, background: c.series[3] }} />
              </span>
            </span>
          )}
          {probe && (
            <span
              style={{
                position: 'absolute',
                left: `${probe[0] * 100}%`,
                top: `${(1 - probe[1]) * 100}%`,
                width: 14,
                height: 14,
                marginLeft: -7,
                marginTop: -7,
                borderRadius: '50%',
                border: '3px solid var(--lemon)',
                pointerEvents: 'none'
              }}
            />
          )}
        </FieldCanvas>
        <div style={{ marginTop: 8 }}>
          <ClassLegend extra={showTest ? <span className="vz-legend-item">속 빈 점 = 시험 데이터</span> : undefined} />
        </div>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <div>
          <div className="vz-panel-title" style={{ marginBottom: 6 }}>
            숲 속 나무들 하나하나 (각자 다른 표본으로 자람)
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {forest.slice(0, MINI).map((f, i) => (
              <div key={`${seed}-${randomFeat}-${i}`} style={{ width: miniW, opacity: i < n ? 1 : 0.35, transition: 'opacity 0.3s' }}>
                <FieldCanvas
                  height={Math.round(miniW * 0.8)}
                  points={[]}
                  predict={(x, y) => predictTree(f.tree, x, y)}
                  res={36}
                  ariaLabel={`나무 ${i + 1}의 경계`}
                />
                <div style={{ fontSize: 11.5, fontWeight: 750, textAlign: 'center', marginTop: 2, color: i < n ? 'var(--text)' : 'var(--text-3)' }}>
                  나무 {i + 1} · {Math.round(treeAcc[i] * 100)}%
                </div>
              </div>
            ))}
          </div>
        </div>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12, alignItems: 'start' }}>
          <div className="vz-stats">
            <Stat label="투표하는 나무" value={`${n}그루`} tone="brand" />
            <Stat label="혼자 자란 나무" value={`${Math.round(SINGLE_ACC * 100)}%`} sub="시험 정확도" />
            <Stat label="숲의 시험 정확도" value={`${Math.round(acc * 100)}%`} tone={acc > SINGLE_ACC ? 'good' : undefined} />
          </div>
          <div>
            <div className="vz-panel-title">나무 수에 따른 시험 정확도</div>
            <Plot
              xDomain={[1, MAX_TREES]}
              yDomain={[0.5, 1]}
              height={140}
              margin={{ top: 8, right: 12, bottom: 34, left: 42 }}
              xTicks={6}
              yTicks={3}
              xLabel="나무 수"
              yFormat={(v) => `${Math.round(v * 100)}%`}
              ariaLabel="나무 수에 따른 숲의 시험 정확도"
            >
              {({ x, y }) => (
                <g>
                  <line x1={x(1)} x2={x(MAX_TREES)} y1={y(SINGLE_ACC)} y2={y(SINGLE_ACC)} stroke={c.series[3]} strokeWidth={2} strokeDasharray="5 4" />
                  <path d={linePath(curve.map((a, i) => [x(i + 1), y(a)]))} fill="none" stroke={c.series[1]} strokeWidth={3} />
                  <line x1={x(n)} x2={x(n)} y1={y(0.5)} y2={y(1)} strokeWidth={2} strokeDasharray="4 3" style={{ stroke: 'var(--lemon)' }} />
                  <circle cx={x(n)} cy={y(acc)} r={5} fill={c.series[1]} />
                </g>
              )}
            </Plot>
            <Legend
              items={[
                { color: c.series[1], label: '숲 (다수결)' },
                { color: c.series[3], label: '혼자 자란 깊은 나무', dashed: true }
              ]}
            />
          </div>
        </div>
      </div>

      <Controls>
        <Slider label="나무 수" value={n} min={1} max={MAX_TREES} step={1} onChange={(v) => setN(Math.round(v))} format={(v) => `${v}그루`} />
        <Toggle label="갈림길마다 특징 무작위로 고르기" checked={randomFeat} onChange={setRandomFeat} />
        <Toggle label="시험 데이터 보기" checked={showTest} onChange={setShowTest} />
        <Button size="sm" variant="ghost" onClick={() => setSeed((s) => s + 1)}>
          🎲 다른 숲 키우기
        </Button>
      </Controls>
    </div>
  );
}
