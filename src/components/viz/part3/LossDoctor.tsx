import { useEffect, useMemo, useState } from 'react';
import type { VizProps } from '../Viz';
import { Plot, linePath } from '../core/plot';
import { Button, Controls, Legend, Note, Segmented, Slider, Stat } from '../core/ui';
import { useRafLoop, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { rng } from '../core/math';

type Sym = 'A' | 'B' | 'C' | 'D' | 'E' | 'H';
const EPOCHS = 100;
const LN10 = Math.log(10);

const SYMPTOMS: { id: Sym; short: string; title: string; look: string }[] = [
  { id: 'A', short: 'A 폭발', title: '증상 A: 손실이 폭발한다 (NaN)', look: '내려가다가 널뛰더니 하늘로 솟구쳐 NaN이 돼요.' },
  { id: 'B', short: 'B 정체', title: '증상 B: 찍기 수준에서 정체', look: '손실이 ln 10 ≈ 2.30 근처에서 꿈쩍도 안 해요.' },
  { id: 'C', short: 'C 과적합', title: '증상 C: 훈련은 좋은데 검증이 나쁘다', look: '훈련 손실은 바닥인데, 검증 손실은 어느 순간부터 다시 올라가요.' },
  { id: 'D', short: 'D 역주행', title: '증상 D: 잘 되다가 무너진다', look: '한참 잘 내려가다가 후반에 갑자기 다시 올라가요.' },
  { id: 'E', short: 'E 재현 불가', title: '증상 E: 돌릴 때마다 결과가 다르다', look: '같은 설정인데 실행할 때마다 곡선이 제각각이에요.' }
];

interface Rx {
  label: string;
  effect: 'fix' | 'info' | 'warn' | 'stop';
  say: string;
}
const RX: Record<Exclude<Sym, 'H'>, Rx[]> = {
  A: [
    { label: '① 학습률을 1/10로', effect: 'fix', say: '손실이 매끄럽게 내려가요! 발산의 첫 용의자는 언제나 학습률이에요.' },
    { label: '② 손실 속 수치 지뢰 확인 (log 0, 0÷0)', effect: 'info', say: '이번 환자는 수식 문제는 아니었어요. 그래도 꼭 확인할 항목이에요.' },
    { label: '③ 기울기 클리핑 + 노름 기록', effect: 'warn', say: '폭발 보험으로 좋아요. 하지만 근본 원인(학습률)부터 고쳐야 해요.' },
    { label: '④ 데이터에 극단값·NaN 확인', effect: 'info', say: '데이터는 깨끗했어요. 다른 용의자를 찾아봐요.' }
  ],
  B: [
    { label: '① 결선 버그 수색', effect: 'fix', say: '범인 발견! 손실에 로짓 대신 확률을 넣고 있었어요(소프트맥스 이중 적용). 고치니 쑥 내려가요.' },
    { label: '② 학습률 탐색 (3e-4부터 10배 간격)', effect: 'info', say: '학습률도 의심할 만하지만, 정체의 1순위 용의자는 결선 버그예요.' },
    { label: '③ 입력 표준화 확인', effect: 'info', say: '입력 크기는 잘 맞춰져 있었어요.' },
    { label: '④ 모델 구조부터 바꾸기', effect: 'warn', say: '아직 일러요! 모델은 데이터·설정·결선을 모두 무죄로 만든 다음에 의심해요.' }
  ],
  C: [
    { label: '① 훈련·검증 전처리가 같은지 확인', effect: 'info', say: '전처리는 같았어요. 진짜 과적합이네요 — 다음 처방으로!' },
    { label: '② 데이터 증강 + 정규화 강화', effect: 'fix', say: '검증 손실이 훈련 손실을 바짝 따라와요. 외우는 대신 배우기 시작했어요!' },
    { label: '③ 조기 종료', effect: 'stop', say: '검증 손실이 가장 낮았던 지점에서 멈추면 그때의 모델을 쓸 수 있어요.' },
    { label: '④ 모델 줄이기', effect: 'info', say: '이것도 방법이지만, 증강·정규화부터 해 보는 게 순서예요.' }
  ],
  D: [
    { label: '① 학습률 감쇠 스케줄 추가', effect: 'fix', say: '후반에 보폭을 줄이니 골짜기 바닥에 얌전히 머물러요!' },
    { label: '② 무너진 지점의 배치 열어 보기', effect: 'info', say: '이번엔 데이터는 멀쩡했어요. 학습률 쪽이 범인이에요.' }
  ],
  E: [
    { label: '① 시드·버전·설정을 기록하고 고정', effect: 'fix', say: '이제 같은 설정이면 같은 결과! 중요한 비교는 시드 여러 개의 평균으로 해요.' },
    { label: '② 그냥 한 번 더 돌려 보기', effect: 'warn', say: '또 달라요… 운에 맡기면 원인을 영영 못 찾아요.' }
  ]
};

/** Procedural loss curves for a symptom (optionally after the key treatment). */
function makeCurves(sym: Sym, treated: boolean, seed: number) {
  const r = rng(seed);
  const nz = (s: number) => r.normal() * s;
  const healthy = (tau: number, floor: number, gap: number) => {
    const train: number[] = [];
    const val: number[] = [];
    for (let t = 0; t < EPOCHS; t++) {
      const base = floor + (LN10 - floor) * Math.exp(-t / tau);
      train.push(Math.max(0.02, base + nz(0.02)));
      val.push(Math.max(0.05, base + gap * (1 - Math.exp(-t / 25)) + nz(0.03)));
    }
    return { train, val };
  };
  if (sym === 'H') return { ...healthy(16, 0.15, 0.1), runs: [] as number[][] };
  if (treated) {
    if (sym === 'A') return { ...healthy(26, 0.2, 0.1), runs: [] };
    if (sym === 'B') return { ...healthy(14, 0.18, 0.12), runs: [] };
    if (sym === 'C') {
      const train: number[] = [];
      const val: number[] = [];
      for (let t = 0; t < EPOCHS; t++) {
        const tr = 0.22 + (LN10 - 0.22) * Math.exp(-t / 14);
        train.push(tr + nz(0.02));
        val.push(tr + 0.12 + nz(0.03));
      }
      return { train, val, runs: [] };
    }
    if (sym === 'D') return { ...healthy(16, 0.16, 0.1), runs: [] };
    if (sym === 'E') {
      const runs = [0, 1, 2].map(() => {
        const out: number[] = [];
        for (let t = 0; t < EPOCHS; t++) out.push(0.18 + (LN10 - 0.18) * Math.exp(-t / 17) + nz(0.015));
        return out;
      });
      return { train: runs[0], val: [] as number[], runs };
    }
  }
  if (sym === 'A') {
    const train: number[] = [];
    const val: number[] = [];
    for (let t = 0; t < EPOCHS; t++) {
      if (t < 14) {
        const b = 1.4 + (LN10 - 1.4) * Math.exp(-t / 5);
        train.push(b + nz(0.05));
        val.push(b + 0.05 + nz(0.05));
      } else if (t < 30) {
        const k = t - 14;
        const b = 1.5 + 0.08 * k * k * (0.6 + 0.4 * Math.abs(Math.sin(k * 1.7)));
        train.push(b + nz(0.1));
        val.push(b * 1.05 + nz(0.1));
      } else {
        train.push(NaN);
        val.push(NaN);
      }
    }
    return { train, val, runs: [] };
  }
  if (sym === 'B') {
    const train: number[] = [];
    const val: number[] = [];
    for (let t = 0; t < EPOCHS; t++) {
      train.push(LN10 - 0.01 * (1 - Math.exp(-t / 10)) + nz(0.012));
      val.push(LN10 + nz(0.015));
    }
    return { train, val, runs: [] };
  }
  if (sym === 'C') {
    const train: number[] = [];
    const val: number[] = [];
    for (let t = 0; t < EPOCHS; t++) {
      train.push(Math.max(0.02, 0.03 + (LN10 - 0.03) * Math.exp(-t / 11) + nz(0.015)));
      const v = t < 22 ? 0.85 + (LN10 - 0.85) * Math.exp(-t / 8) : 0.88 + 0.0095 * (t - 22) ** 1.15;
      val.push(v + nz(0.03));
    }
    return { train, val, runs: [] };
  }
  if (sym === 'D') {
    const h = healthy(15, 0.15, 0.1);
    for (let t = 62; t < EPOCHS; t++) {
      const k = t - 62;
      h.train[t] += 0.03 * k + 0.25 * Math.abs(Math.sin(k * 0.9)) * (k / 38);
      h.val[t] += 0.035 * k + 0.3 * Math.abs(Math.sin(k * 0.7)) * (k / 38);
    }
    return { ...h, runs: [] };
  }
  // E: three seeds, three very different stories
  const shapes = [
    [12, 0.15],
    [30, 0.55],
    [9, 0.35]
  ];
  const runs = shapes.map(([tau, floor], i) => {
    const out: number[] = [];
    for (let t = 0; t < EPOCHS; t++) {
      const bump = i === 2 && t > 40 && t < 55 ? 0.4 * Math.sin(((t - 40) / 15) * Math.PI) : 0;
      out.push(floor + (LN10 - floor) * Math.exp(-t / tau) + bump + nz(0.03));
    }
    return out;
  });
  return { train: runs[0], val: [] as number[], runs };
}

type Tab = 'atlas' | 'quiz' | 'first';

export default function LossDoctor({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [tab, setTab] = useState<Tab>('atlas');
  const [sym, setSym] = useState<Exclude<Sym, 'H'>>('A');
  const [treated, setTreated] = useState(false);
  const [stopAt, setStopAt] = useState<number | null>(null);
  const [said, setSaid] = useState<Rx | null>(null);
  const [seed, setSeed] = useState(1);
  const [p, setP] = useState(1);
  // quiz
  const [qi, setQi] = useState(0);
  const [qAnswer, setQAnswer] = useState<Sym | null>(null);
  const [score, setScore] = useState({ right: 0, total: 0 });
  // first-loss check
  const [K, setK] = useState(10);
  const [first, setFirst] = useState(6.2);

  const quizOrder: Sym[] = ['C', 'A', 'H', 'B', 'D', 'C', 'B', 'A', 'D', 'H'];
  const qSym = quizOrder[qi % quizOrder.length];
  const curves = useMemo(() => (tab === 'quiz' ? makeCurves(qSym, false, 100 + qi) : makeCurves(sym, treated, seed)), [tab, qSym, qi, sym, treated, seed]);

  useEffect(() => setP(reduce ? 1 : 0), [sym, treated, seed, qi, tab, reduce]);
  useRafLoop((dt) => setP((v) => Math.min(1, v + dt / 2.2)), p < 1);

  const shown = Math.max(2, Math.ceil(p * EPOCHS));
  const minVal = curves.val.length ? curves.val.reduce((best, v, i) => (Number.isFinite(v) && v < curves.val[best] ? i : best), 0) : 0;

  const chart = (
    <Plot
      xDomain={[0, EPOCHS - 1]}
      yDomain={[0, 3]}
      aspect={expanded ? 2.4 : 1.9}
      maxHeight={expanded ? 460 : 320}
      xLabel="에포크 (훈련 진행)"
      yLabel="손실"
      xTicks={5}
      yTicks={3}
      ariaLabel="훈련 손실과 검증 손실 곡선"
    >
      {({ x, y, clip, inner }) => {
        const toPts = (arr: number[]) => {
          const segs: [number, number][][] = [[]];
          arr.slice(0, shown).forEach((v, i) => {
            if (!Number.isFinite(v)) {
              if (segs[segs.length - 1].length) segs.push([]);
              return;
            }
            segs[segs.length - 1].push([x(i), y(Math.min(3.4, v))]);
          });
          return segs.filter((s) => s.length > 1);
        };
        const nanAt = curves.train.findIndex((v) => !Number.isFinite(v));
        return (
          <g>
            <g clipPath={clip}>
              <line x1={inner.left} x2={inner.left + inner.width} y1={y(LN10)} y2={y(LN10)} stroke={c.axis} strokeDasharray="6 5" />
              {curves.runs.length > 0
                ? curves.runs.map((run, i) => toPts(run).map((s, k) => <path key={`${i}-${k}`} d={linePath(s)} fill="none" stroke={c.series[i === 0 ? 0 : i === 1 ? 2 : 3]} strokeWidth={2.6} />))
                : (
                  <>
                    {toPts(curves.val).map((s, k) => (
                      <path key={`v${k}`} d={linePath(s)} fill="none" stroke={c.series[2]} strokeWidth={3} />
                    ))}
                    {toPts(curves.train).map((s, k) => (
                      <path key={`t${k}`} d={linePath(s)} fill="none" stroke={c.series[0]} strokeWidth={3} />
                    ))}
                  </>
                )}
              {stopAt !== null && tab === 'atlas' && p >= 1 && (
                <g>
                  <line x1={x(stopAt)} x2={x(stopAt)} y1={inner.top} y2={inner.top + inner.height} stroke={c.pos} strokeWidth={2} strokeDasharray="4 4" />
                  <circle cx={x(stopAt)} cy={y(curves.val[stopAt])} r={6} fill={c.pos} />
                </g>
              )}
            </g>
            <text x={inner.left + inner.width - 4} y={y(LN10) - 6} textAnchor="end" style={{ fontSize: 11.5, fontWeight: 800, fill: 'var(--text-2)' }}>
              찍기 수준 ln10 ≈ 2.30
            </text>
            {nanAt > 0 && shown > nanAt && (
              <text x={x(nanAt)} y={inner.top + 16} textAnchor="middle" style={{ fontSize: 14, fontWeight: 900, fill: c.neg }}>
                💥 NaN!
              </text>
            )}
            {stopAt !== null && tab === 'atlas' && p >= 1 && (
              <text x={x(stopAt) + 6} y={inner.top + 16} style={{ fontSize: 11.5, fontWeight: 850, fill: c.pos }}>
                여기서 멈춤 (검증 최저)
              </text>
            )}
          </g>
        );
      }}
    </Plot>
  );

  const legend = curves.runs.length
    ? [
        { color: c.series[0], label: '실행 1' },
        { color: c.series[2], label: '실행 2' },
        { color: c.series[3], label: '실행 3' }
      ]
    : [
        { color: c.series[0], label: '훈련 손실' },
        { color: c.series[2], label: '검증 손실' }
      ];

  const symInfo = SYMPTOMS.find((s) => s.id === sym)!;
  const lnK = Math.log(K);
  const firstOk = Math.abs(first - lnK) < Math.max(0.25, lnK * 0.12);

  return (
    <div>
      {tab !== 'first' && (
        <div className="vz-pad" style={{ paddingBottom: 0 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <strong style={{ fontSize: 15 }}>{tab === 'quiz' ? `🩺 진단 퀴즈 ${(qi % quizOrder.length) + 1}번 환자` : symInfo.title}</strong>
            <Legend items={legend} />
          </div>
          {chart}
        </div>
      )}

      {tab === 'atlas' && (
        <div className="vz-pad" style={{ display: 'grid', gap: 10 }}>
          <p style={{ margin: 0, fontSize: 14, color: 'var(--text-2)' }}>
            <b>진찰 소견:</b> {treated ? '처방이 들었어요! 곡선을 보세요.' : symInfo.look}
          </p>
          <div className="vz-panel-title">처방전 — 위에서부터 순서대로 의심해요</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {RX[sym].map((rx) => (
              <Button
                key={rx.label}
                size="sm"
                active={said?.label === rx.label}
                onClick={() => {
                  setSaid(rx);
                  if (rx.effect === 'fix') setTreated(true);
                  if (rx.effect === 'stop') setStopAt(minVal);
                  if (rx.effect === 'warn' && sym === 'E') setSeed((s) => s + 7);
                }}
              >
                {rx.label}
              </Button>
            ))}
          </div>
          {said ? (
            <Note mood={said.effect === 'fix' ? 'cheer' : said.effect === 'warn' ? 'think' : 'smile'} tone={said.effect === 'fix' ? 'good' : said.effect === 'warn' ? 'warn' : undefined}>
              {said.say}
            </Note>
          ) : (
            <Note mood="think">환자(학습 곡선)를 보고 처방을 하나씩 눌러 보세요. 진짜 원인을 찾으면 곡선이 건강해져요!</Note>
          )}
        </div>
      )}

      {tab === 'quiz' && (
        <div className="vz-pad" style={{ display: 'grid', gap: 10 }}>
          <div className="vz-panel-title">이 학습 곡선의 병명은?</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {[...SYMPTOMS.filter((s) => s.id !== 'E').map((s) => ({ id: s.id, label: s.short })), { id: 'H' as Sym, label: '건강해요 💚' }].map((o) => {
              const picked = qAnswer === o.id;
              const right = qAnswer !== null && o.id === qSym;
              return (
                <Button
                  key={o.id}
                  size="sm"
                  variant={right ? 'primary' : 'default'}
                  active={picked}
                  disabled={qAnswer !== null}
                  onClick={() => {
                    setQAnswer(o.id);
                    setScore((s) => ({ right: s.right + (o.id === qSym ? 1 : 0), total: s.total + 1 }));
                  }}
                >
                  {o.label}
                </Button>
              );
            })}
          </div>
          {qAnswer !== null && (
            <Note mood={qAnswer === qSym ? 'cheer' : 'think'} tone={qAnswer === qSym ? 'good' : 'warn'}>
              {qAnswer === qSym ? '정답! ' : `아쉬워요. 정답은 "${qSym === 'H' ? '건강' : SYMPTOMS.find((s) => s.id === qSym)!.short}". `}
              {qSym === 'H' ? '훈련·검증 손실이 함께 내려가고 간격도 작아요. 건강한 학습이에요.' : SYMPTOMS.find((s) => s.id === qSym)!.look}
            </Note>
          )}
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <Stat label="맞힌 환자" value={`${score.right} / ${score.total}`} tone="brand" />
            <Button variant="primary" onClick={() => { setQi((i) => i + 1); setQAnswer(null); }} disabled={qAnswer === null}>
              다음 환자 →
            </Button>
          </div>
        </div>
      )}

      {tab === 'first' && (
        <div className="vz-pad" style={{ display: 'grid', gap: 14 }}>
          <p style={{ margin: 0, fontSize: 14.5, color: 'var(--text-2)' }}>
            학습을 시작하기 전, 아무렇게나 찍는 모델의 손실은 <b>ln K</b>(K = 클래스 수)여야 해요. 첫 손실이 이 값과 크게 다르면 시작부터 무언가 틀린 거예요.
          </p>
          <svg viewBox="0 0 640 110" className="vz-svg" role="img" aria-label="첫 손실과 ln K 비교" style={{ touchAction: 'pan-y' }}>
            {(() => {
              const sx = (v: number) => 30 + (Math.min(10, Math.max(0, v)) / 10) * 580;
              return (
                <g>
                  <rect x={30} y={48} width={580} height={14} rx={7} fill="var(--surface-3)" />
                  <rect x={sx(lnK * 0.88) } y={48} width={Math.max(6, sx(lnK * 1.12) - sx(lnK * 0.88))} height={14} rx={7} fill={withAlpha(c.pos, 0.45)} />
                  {[0, 2, 4, 6, 8, 10].map((v) => (
                    <text key={v} x={sx(v)} y={84} textAnchor="middle" className="vz-tick">
                      {v}
                    </text>
                  ))}
                  <line x1={sx(lnK)} x2={sx(lnK)} y1={36} y2={70} stroke={c.pos} strokeWidth={3} />
                  <text x={sx(lnK)} y={28} textAnchor="middle" style={{ fontSize: 12.5, fontWeight: 850, fill: c.pos }}>
                    기대값 ln {K} = {lnK.toFixed(2)}
                  </text>
                  <circle cx={sx(first)} cy={55} r={10} fill={firstOk ? c.pos : c.neg} stroke={c.surface} strokeWidth={3} />
                  <text x={sx(first)} y={104} textAnchor="middle" style={{ fontSize: 12.5, fontWeight: 850, fill: firstOk ? c.pos : c.neg }}>
                    내 첫 손실 {first.toFixed(1)}
                  </text>
                </g>
              );
            })()}
          </svg>
          <Note mood={firstOk ? 'cheer' : 'wow'} tone={firstOk ? 'good' : 'warn'}>
            {firstOk
              ? '정상 범위예요! 이제 한 배치 암기 테스트로 넘어가요.'
              : `기대값 ${lnK.toFixed(2)}와 많이 달라요. 손실 결선(로짓 대신 확률을 넣었나?), 출력 크기, 레이블 범위를 먼저 의심해 보세요.`}
          </Note>
        </div>
      )}

      <Controls>
        <Segmented
          label="모드"
          options={[
            { value: 'atlas', label: '🩺 증상 도감' },
            { value: 'quiz', label: '❓ 진단 퀴즈' },
            { value: 'first', label: '🧮 첫 손실 검산' }
          ]}
          value={tab}
          onChange={(v) => setTab(v as Tab)}
        />
        {tab === 'atlas' && (
          <Segmented
            label="증상"
            options={SYMPTOMS.map((s) => ({ value: s.id, label: s.short }))}
            value={sym}
            onChange={(v) => {
              setSym(v as Exclude<Sym, 'H'>);
              setTreated(false);
              setStopAt(null);
              setSaid(null);
            }}
          />
        )}
        {tab === 'first' && (
          <>
            <Segmented
              label="클래스 수 K"
              options={[
                { value: 2, label: '2' },
                { value: 10, label: '10' },
                { value: 100, label: '100' },
                { value: 1000, label: '1000' }
              ]}
              value={K}
              onChange={(v) => setK(Number(v))}
            />
            <Slider label="학습 전 첫 손실" value={first} min={0} max={10} step={0.1} onChange={setFirst} format={(v) => v.toFixed(1)} accent="var(--peach)" />
          </>
        )}
      </Controls>
    </div>
  );
}
