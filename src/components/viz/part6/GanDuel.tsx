import { useMemo, useRef, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Controls, Legend, Note, PlayBar, Segmented, Stat } from '../core/ui';
import { Plot, linePath } from '../core/plot';
import { useElementSize, useInView, useInterval } from '../core/hooks';
import { useWide } from '../part5/_layout';
import { useVizColors, withAlpha } from '../core/colors';
import { linspace } from '../core/math';
import type { AlgoMood } from '@/components/mascot/Algo';
import { SCENARIOS, TinyGan, pReal } from './_gan';

type Scenario = keyof typeof SCENARIOS;
const XS = linspace(-6.5, 8.5, 260);
const YMAX = 0.75;

/** A real, tiny GAN: watch the forger (generator) and the judge (discriminator) compete. */
export default function GanDuel({ expanded }: VizProps) {
  const c = useVizColors();
  const [scenario, setScenario] = useState<Scenario>('good');
  const gan = useRef<TinyGan | null>(null);
  if (!gan.current) gan.current = new TinyGan(SCENARIOS.good.cfg);
  const [, setTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [rootRef, size] = useElementSize<HTMLDivElement>();
  const inView = useInView(rootRef);
  const wide = useWide(size.width, 520);

  const reset = (s: Scenario = scenario) => {
    setPlaying(false);
    gan.current = new TinyGan(SCENARIOS[s].cfg);
    setTick((t) => t + 1);
  };

  const g = gan.current;
  const done = g.steps >= 700;

  useInterval(
    () => {
      const n = Math.max(1, Math.round(speed));
      for (let i = 0; i < n; i++) g.step();
      if (g.steps >= 700) setPlaying(false);
      setTick((t) => t + 1);
    },
    playing && inView ? (speed < 1 ? 160 : 70) : null
  );

  const curves = useMemo(() => {
    const real = XS.map((x) => [x, pReal(x)] as [number, number]);
    const fake = XS.map((x) => [x, g.pGen(x)] as [number, number]);
    const disc = XS.map((x) => [x, g.D(x)] as [number, number]);
    return { real, fake, disc };
  }, [g, g.steps, scenario]); // eslint-disable-line react-hooks/exhaustive-deps

  const overlap = g.overlap();
  const acc = g.accuracy();
  const modes = g.modesCovered();
  const stranded = g.gen.some((comp) => comp.m > 6.2);
  const word = c.theme === 'dark' ? { real: '분홍', fake: '하늘색', d: '연두' } : { real: '초록', fake: '보라', d: '주황' };

  let note: { text: ReactNode; mood: AlgoMood; tone?: 'good' | 'bad' | 'warn' };
  if (g.steps === 0)
    note =
      scenario === 'good'
        ? { text: <>{word.real} 산은 <strong>진짜 데이터</strong>(봉우리 두 개), {word.fake} 산은 <strong>생성기가 만든 가짜</strong>예요. {word.d} 점선은 판별기가 "여긴 진짜 같아!"라고 믿는 정도. ▶ 시작을 눌러 대결을 지켜보세요!</>, mood: 'smile' }
        : scenario === 'strong'
          ? { text: <>이번엔 판별기가 아주 <strong>예리</strong>하고, 생성기는 진짜와 <strong>멀리 떨어진 곳</strong>에서 출발해요. 어떻게 될까요?</>, mood: 'think' }
          : { text: <>이번 생성기는 <strong>한 갈래</strong>로만 그릴 수 있어요. 봉우리가 둘인 진짜를 어떻게 흉내 낼까요?</>, mood: 'think' };
  else if (scenario === 'good' && overlap > 0.93)
    note = { text: <>위조가 진품과 거의 <strong>구별 불가</strong>! 판별기 정확도가 {Math.round(acc * 100)}%(찍기 수준)까지 떨어졌어요. 이게 GAN이 노리는 균형점이에요. 🎉</>, mood: 'cheer', tone: 'good' };
  else if (scenario === 'strong' && stranded && g.steps > 60)
    note = { text: <>판별기가 너무 세서 처음부터 가짜를 <strong>단번에 간파</strong>했어요. "다 틀렸어!"만 들은 한 갈래는 어디로 가야 할지 <strong>신호를 잃고</strong> 데이터 밖 엉뚱한 곳(오른쪽 끝)에 멈춰 버렸어요.</>, mood: 'wow', tone: 'bad' };
  else if (scenario === 'collapse' && modes === 1 && g.steps > 60)
    note = { text: <><strong>모드 붕괴!</strong> 생성기가 판별기를 속이기 쉬운 <strong>한 종류</strong>(한쪽 봉우리)만 계속 만들어요. 다른 봉우리 같은 가짜는 영영 안 나와요 — 숫자 생성기가 "1"만 그리는 그 실패예요.</>, mood: 'wow', tone: 'warn' };
  else note = { text: <>판별기는 진짜와 가짜를 가르도록, 생성기는 판별기를 <strong>속이도록</strong> 번갈아 배워요. {word.d} 점선이 높은 쪽으로 {word.fake} 산이 옮겨 가는 게 보이죠?</>, mood: 'happy' };

  const dColor = c.series[2];
  return (
    <div ref={rootRef}>
      <div className="vz-pad" style={{ paddingBottom: 4 }}>
        <Plot
          xDomain={[-6.5, 8.5]}
          yDomain={[0, YMAX]}
          aspect={expanded ? 2.4 : wide ? 1.8 : 1.25}
          maxHeight={expanded ? 560 : 400}
          xLabel="데이터 값 (예: 그림의 특징 하나)"
          yLabel="얼마나 자주 나오나"
          yFormat={(v) => (v === 0 ? '0' : '')}
          ariaLabel="진짜 데이터 분포, 생성기 분포, 판별기 확신 곡선"
        >
          {({ x, y, clip, inner }) => {
            const area = (pts: [number, number][]) =>
              `${linePath(pts.map(([px, py]) => [x(px), y(Math.min(YMAX, py))] as [number, number]))}L${x(pts[pts.length - 1][0])},${y(0)}L${x(pts[0][0])},${y(0)}Z`;
            return (
              <g clipPath={clip}>
                <path d={area(curves.real)} fill={withAlpha(hex(c.series[1], '#2fbf98'), 0.28)} stroke={c.series[1]} strokeWidth={2.5} />
                <path d={area(curves.fake)} fill={withAlpha(hex(c.series[0], '#7c6cf6'), 0.28)} stroke={c.series[0]} strokeWidth={2.5} />
                <path d={linePath(curves.real.map(([px, py]) => [x(px), y(Math.min(YMAX, py))] as [number, number]))} fill="none" stroke={c.series[1]} strokeWidth={2} strokeDasharray="1 4" strokeLinecap="round" />
                <path
                  d={linePath(curves.disc.map(([px, d]) => [x(px), y(d * YMAX)] as [number, number]))}
                  fill="none"
                  stroke={dColor}
                  strokeWidth={3}
                  strokeDasharray="7 5"
                />
                <line x1={inner.left} x2={inner.left + inner.width} y1={y(0.5 * YMAX)} y2={y(0.5 * YMAX)} stroke={dColor} strokeOpacity={0.35} strokeDasharray="2 5" />
                <text x={inner.left + inner.width - 6} y={y(0.5 * YMAX) - 6} textAnchor="end" style={{ fontSize: 11, fontWeight: 750, fill: dColor, paintOrder: 'stroke', stroke: c.theme === 'dark' ? '#0b1030' : '#fffdf9', strokeWidth: 3.5, strokeLinejoin: 'round' }}>
                  판별기 50% = 모르겠다
                </text>
                {g.lastReal.map((v, i) => (
                  <circle key={`r${i}`} cx={x(v)} cy={y(0.06)} r={3.6} fill={c.series[1]} opacity={0.8} />
                ))}
                {g.lastFake.map((v, i) => (
                  <circle key={`f${i}`} cx={x(Math.min(8.4, v))} cy={y(0.025)} r={3.6} fill={c.series[0]} opacity={0.8} />
                ))}
              </g>
            );
          }}
        </Plot>
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, marginTop: 6 }}>
          <Legend
            items={[
              { color: c.series[1], label: '진짜 데이터 (점 = 진짜 샘플)' },
              { color: c.series[0], label: '생성기의 가짜 (점 = 가짜 샘플)' },
              { color: dColor, label: '판별기의 "진짜다!" 확신', dashed: true }
            ]}
          />
        </div>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12, paddingTop: 6 }}>
        <div className="vz-stats">
          <Stat label="훈련 단계" value={g.steps} />
          <Stat label="판별기 정확도" value={`${Math.round(acc * 100)}%`} sub="50%면 구별 못 함" tone={acc < 0.6 && g.steps > 0 ? 'good' : undefined} />
          <Stat label="진짜와 닮은 정도" value={`${Math.round(overlap * 100)}%`} tone={overlap > 0.9 ? 'good' : overlap > 0.6 ? 'warn' : 'bad'} />
          <Stat label="흉내 낸 봉우리" value={`${modes} / 2`} tone={modes === 2 ? 'good' : 'warn'} />
        </div>
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
      </div>

      <Controls>
        <Segmented
          label="실험 고르기"
          options={(Object.keys(SCENARIOS) as Scenario[]).map((k) => ({ value: k, label: SCENARIOS[k].label }))}
          value={scenario}
          onChange={(v) => {
            setScenario(v as Scenario);
            reset(v as Scenario);
          }}
        />
        <PlayBar
          playing={playing}
          onToggle={() => {
            if (done) reset();
            setPlaying((p) => !p);
          }}
          onStep={() => {
            g.step();
            setTick((t) => t + 1);
          }}
          onReset={() => reset()}
          speed={speed}
          onSpeed={setSpeed}
        />
      </Controls>
    </div>
  );
}

const hex = (col: string, fb: string) => (col.startsWith('#') ? col : fb);
