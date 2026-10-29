import { useEffect, useState, type ReactNode } from 'react';
import type { VizProps } from '../Viz';
import { Button, Controls, Legend, Note, Segmented, Slider, Stat, Toggle } from '../core/ui';
import { Plot, linePath } from '../core/plot';
import { useInterval, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors, withAlpha } from '../core/colors';
import { pct, rng, sigmoid } from '../core/math';

/*
 * Model-monitoring simulator. A music app predicts "will this user like the song?" from age.
 * The model is a per-age-bin majority vote learned from labelled data; age bins it never saw
 * copy their nearest seen neighbour (that guessing is where data drift bites).
 * Every simulated week draws 400 users from the current world. PSI compares this week's age
 * histogram with the model's training data; labels arrive `delay` weeks late.
 */
const A0 = 10;
const BW = 2;
const NB = 30;
const WEEKS = 52;
const PER_WEEK = 400;
const TRAIN_N = 600;
const PSI_ALARM = 0.2;
const ACC_ALARM = 0.8;
const OLD_MAX = 0.3;
const STORY = { data: 8, concept: 28 };

const binOf = (x: number) => Math.min(NB - 1, Math.max(0, Math.floor((x - A0) / BW)));
const center = (b: number) => A0 + (b + 0.5) * BW;

/** True P(like | age). The trend `m` (0 → 1) morphs "teens to 30s like it" into "20s to mid-40s like it". */
const before = (x: number) => 0.04 + 0.9 * sigmoid((x - 14) / 0.7) * (1 - sigmoid((x - 40) / 1));
const after = (x: number) => 0.04 + 0.9 * sigmoid((x - 20) / 0.5) * (1 - sigmoid((x - 48) / 0.6));
const pLike = (x: number, m: number) => (1 - m) * before(x) + m * after(x);

interface Counts {
  n: number[];
  k: number[];
}
interface Model {
  labels: number[];
  seen: boolean[];
  ref: number[];
  hist: number[];
  trainedAt: number;
  version: number;
}
interface WeekRec extends Counts {
  psi: number;
  acc: number;
  alarm: boolean;
}
type LogKind = 'event' | 'alarm' | 'ok' | 'same' | 'fail';
interface LogItem {
  week: number;
  text: string;
  kind: LogKind;
}
interface Sim {
  week: number;
  train: Counts;
  recs: WeekRec[];
  models: Model[];
  cur: number;
  old: number;
  trend: number;
  dataAt: number | null;
  conceptAt: number | null;
  log: LogItem[];
  lastRetrain: number;
  alarmPsi: boolean;
  alarmAcc: boolean;
}

function sample(seed: number, old: number, trend: number, N: number): Counts {
  const r = rng(seed);
  const n = new Array<number>(NB).fill(0);
  const k = new Array<number>(NB).fill(0);
  for (let i = 0; i < N; i++) {
    const x = r.next() < old ? 50 + 6 * r.normal() : 22 + 4.5 * r.normal();
    const b = binOf(x);
    n[b]++;
    if (r.next() < pLike(x, trend)) k[b]++;
  }
  return { n, k };
}

const total = (n: number[]) => n.reduce((a, b) => a + b, 0);

/** 10 coarse bins with add-½ smoothing, the usual way PSI avoids log(0). */
function hist10(n: number[]) {
  const h = new Array<number>(10).fill(0);
  n.forEach((v, i) => (h[Math.floor(i / 3)] += v));
  const s = total(h);
  return h.map((v) => (v + 0.5) / (s + 5));
}

const psiOf = (p: number[], q: number[]) => p.reduce((acc, pi, i) => acc + (pi - q[i]) * Math.log(pi / q[i]), 0);

function trainModel(c: Counts, trainedAt: number, version: number): Model {
  const seen = c.n.map((v) => v >= 3);
  const raw = c.n.map((v, i) => (seen[i] ? (2 * c.k[i] >= v ? 1 : 0) : -1));
  const labels = raw.map((l, i) => {
    if (l >= 0) return l;
    for (let d = 1; d < NB; d++) {
      if (i - d >= 0 && raw[i - d] >= 0) return raw[i - d];
      if (i + d < NB && raw[i + d] >= 0) return raw[i + d];
    }
    return 1;
  });
  const s = total(c.n) || 1;
  return { labels, seen, ref: hist10(c.n), hist: c.n.map((v) => v / s), trainedAt, version };
}

function accuracy(m: Model, c: Counts) {
  let right = 0;
  let all = 0;
  for (let i = 0; i < NB; i++) {
    all += c.n[i];
    right += m.labels[i] ? c.k[i] : c.n[i] - c.k[i];
  }
  return all ? right / all : 1;
}

function addCounts(cs: Counts[]): Counts {
  const n = new Array<number>(NB).fill(0);
  const k = new Array<number>(NB).fill(0);
  for (const c of cs) {
    for (let i = 0; i < NB; i++) {
      n[i] += c.n[i];
      k[i] += c.k[i];
    }
  }
  return { n, k };
}

function initSim(): Sim {
  const train = sample(999, 0, 0, TRAIN_N);
  return { week: 0, train, recs: [], models: [trainModel(train, 0, 1)], cur: 0, old: 0, trend: 0, dataAt: null, conceptAt: null, log: [], lastRetrain: -99, alarmPsi: false, alarmAcc: false };
}

/** Latest accuracy the team can actually see: predictions made `delay` weeks ago. */
const knownAcc = (s: Sim, delay: number) => (s.week - delay >= 1 ? s.recs[s.week - delay - 1].acc : null);

function retrain(s: Sim, delay: number, auto: boolean): Sim {
  const L = s.week - delay;
  const who = auto ? '🤖 ' : '';
  if (L < 2) return { ...s, lastRetrain: s.week, log: [...s.log, { week: s.week, text: `${who}재학습 보류: 정답이 도착한 주가 아직 부족해요`, kind: 'same' }] };
  // train on the 4 labelled weeks before the newest one; that newest week is the test (chapter 8's rule)
  const cand = trainModel(addCounts(s.recs.slice(Math.max(0, L - 5), L - 1)), s.week, s.models.length + 1);
  const test = s.recs[L - 1];
  const accOld = accuracy(s.models[s.cur], test);
  const accNew = accuracy(cand, test);
  if (accNew < accOld - 0.01) {
    return { ...s, lastRetrain: s.week, log: [...s.log, { week: s.week, text: `${who}평가 게이트 탈락 ❌ 새 모델 ${pct(accNew)} < 지금 모델 ${pct(accOld)} → 배포 안 함`, kind: 'fail' }] };
  }
  const same = cand.labels.every((l, i) => l === s.models[s.cur].labels[i]);
  const text = same
    ? `${who}재학습 v${cand.version}: 답이 그대로예요 — ${L}주차까지 도착한 정답엔 아직 변화가 안 담겼어요`
    : `${who}재학습 → 배포 ✅ v${cand.version}: 시험 주(${L}주차) 정확도 ${pct(accOld)} → ${pct(accNew)}`;
  return { ...s, models: [...s.models, cand], cur: s.models.length, lastRetrain: s.week, log: [...s.log, { week: s.week, text, kind: same ? 'same' : 'ok' }] };
}

function step(s: Sim, story: boolean, delay: number, auto: boolean): Sim {
  const week = s.week + 1;
  const log = [...s.log];
  let { dataAt, conceptAt } = s;
  if (story && week === STORY.data && dataAt === null) {
    dataAt = week;
    log.push({ week, text: '👨‍👩‍👧 부모님 세대가 앱에 가입하기 시작했어요', kind: 'event' });
  }
  if (story && week === STORY.concept && conceptAt === null) {
    conceptAt = week;
    log.push({ week, text: '📻 유행이 바뀌기 시작했어요', kind: 'event' });
  }
  const old = dataAt !== null && week >= dataAt ? Math.min(OLD_MAX, 0.06 * (week - dataAt + 1)) : s.old;
  const trend = conceptAt !== null && week >= conceptAt ? Math.min(1, (week - conceptAt + 1) / 3) : s.trend;
  const c = sample(5000 + week * 13, old, trend, PER_WEEK);
  const m = s.models[s.cur];
  const psi = psiOf(hist10(c.n), m.ref);
  const rec: WeekRec = { ...c, psi, acc: accuracy(m, c), alarm: false };
  let next: Sim = { ...s, week, old, trend, dataAt, conceptAt, log, recs: [...s.recs, rec] };
  const known = knownAcc(next, delay);
  const alarmPsi = psi > PSI_ALARM;
  const alarmAcc = known !== null && known < ACC_ALARM;
  rec.alarm = alarmPsi || alarmAcc;
  if (alarmPsi && !s.alarmPsi) log.push({ week, text: `📈 입력 경보! PSI ${psi.toFixed(2)}`, kind: 'alarm' });
  if (alarmAcc && !s.alarmAcc) log.push({ week, text: `📉 성능 경보! 확인된 정확도 ${pct(known ?? 0)}`, kind: 'alarm' });
  next = { ...next, alarmPsi, alarmAcc };
  if (auto && (alarmPsi || alarmAcc) && week - s.lastRetrain >= 3) next = retrain(next, delay, true);
  return next;
}

export default function DriftMonitor({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [sim, setSim] = useState<Sim>(initSim);
  const [playing, setPlaying] = useState(false);
  const [story, setStory] = useState(true);
  const [delay, setDelay] = useState(3);
  const [auto, setAuto] = useState(false);
  const [oracle, setOracle] = useState(true);

  const done = sim.week >= WEEKS;
  useInterval(() => setSim((s) => (s.week >= WEEKS ? s : step(s, story, delay, auto))), playing && !done ? (reduce ? 1200 : 650) : null);
  useEffect(() => {
    if (done) setPlaying(false);
  }, [done]);

  const reset = (nextStory = story) => {
    setPlaying(false);
    setStory(nextStory);
    setSim(initSim());
  };
  const startEvent = (kind: 'data' | 'concept') =>
    setSim((s) => {
      const week = s.week + 1;
      if (kind === 'data' && s.dataAt === null) return { ...s, dataAt: week, log: [...s.log, { week, text: '👨‍👩‍👧 부모님 세대가 앱에 가입하기 시작해요', kind: 'event' }] };
      if (kind === 'concept' && s.conceptAt === null) return { ...s, conceptAt: week, log: [...s.log, { week, text: '📻 유행이 바뀌기 시작해요', kind: 'event' }] };
      return s;
    });

  const model = sim.models[sim.cur];
  const last = sim.recs[sim.week - 1];
  const shown: Counts = last ?? sim.train;
  const shownTotal = total(shown.n) || 1;
  const share = shown.n.map((v) => v / shownTotal);
  const truthLabel = (b: number) => (pLike(center(b), sim.trend) > 0.5 ? 1 : 0);
  const wrongBin = (b: number) => model.labels[b] !== truthLabel(b);
  const known = knownAcc(sim, delay);
  const trueAcc = last?.acc ?? accuracy(model, sim.train);
  const psiNow = last?.psi ?? 0;
  const yMax = Math.max(0.06, ...share, ...model.hist) * 1.15;
  const histH = expanded ? 280 : 220;
  const accH = expanded ? 180 : 140;
  const psiH = expanded ? 130 : 104;
  const retrains = sim.log.filter((l) => l.kind === 'ok' || l.kind === 'same' || l.kind === 'fail');
  const fresh = sim.log.filter((l) => l.week === sim.week);
  const freshRetrain = [...fresh].reverse().find((l) => l.kind === 'ok' || l.kind === 'same' || l.kind === 'fail');

  let note: { t: ReactNode; mood: 'smile' | 'happy' | 'think' | 'wow' | 'cheer' | 'wink'; tone?: 'good' | 'warn' | 'bad' };
  if (done) {
    const below = sim.recs.filter((r) => r.acc < ACC_ALARM).length;
    const caught = (at: number | null) => {
      if (at === null) return null;
      const w = sim.recs.findIndex((r, i) => i + 1 >= at && r.alarm);
      return w < 0 ? -1 : w + 1 - at;
    };
    const dLag = caught(sim.dataAt);
    const cLag = caught(sim.conceptAt);
    const lag = (v: number | null, name: string) => (v === null ? null : v < 0 ? <>{name}는 끝내 경보가 안 울렸어요. </> : <>{name}는 <strong>{v}주 만에</strong> 경보가 울렸어요. </>);
    note = {
      t: (
        <>
          1년 끝! 🎉 진짜 정확도가 80% 아래였던 주는 <strong>{below}주</strong>예요. {lag(dLag, '데이터 드리프트')}
          {lag(cLag, '컨셉 드리프트')}
          {auto
            ? '🤖 자동 파이프라인이 경보 → 재학습 → 평가 → 배포를 알아서 했어요. 그래도 정답이 늦게 오는 만큼은 늦을 수밖에 없어요. 정답 지연을 줄이면 어떨까요?'
            : below > 8
              ? '🤖 자동 재학습을 켜고 다시 돌려 보세요. 나쁜 주가 얼마나 줄어드나요?'
              : '모니터링과 재학습이 모델을 젊게 지켰어요!'}
        </>
      ),
      mood: below > 8 && !auto ? 'think' : 'cheer',
      tone: below > 8 && !auto ? 'warn' : 'good'
    };
  } else if (freshRetrain) {
    note =
      freshRetrain.kind === 'ok'
        ? { t: <>새 모델 배포! 최근 데이터를 배워서 다시 정확해졌어요. 기준 분포도 새 학습 데이터로 바뀌어서 점선이 이동했죠?</>, mood: 'cheer', tone: 'good' }
        : freshRetrain.kind === 'fail'
          ? { t: <>새 모델이 시험에서 지금 모델보다 못해서 <strong>평가 게이트</strong>가 막았어요. 나쁜 모델이 몰래 배포되는 사고를 막는 장치예요.</>, mood: 'think', tone: 'warn' }
          : { t: <>다시 학습했지만 달라진 게 없어요. 정답이 <strong>{delay}주 늦게</strong> 오니까, 최근에 생긴 변화는 아직 학습 데이터에 없거든요. 정답이 쌓일 때까지 기다렸다가 다시 해 보세요.</>, mood: 'think', tone: 'warn' };
  } else if (sim.alarmPsi && sim.alarmAcc) {
    note = { t: <>경보가 두 개나 울려요! 손님도 바뀌었고 성능도 떨어졌어요. <strong>🔁 재학습</strong>으로 최근 데이터를 배우게 하세요.</>, mood: 'wow', tone: 'bad' };
  } else if (sim.alarmPsi) {
    note = {
      t: (
        <>
          📈 입력 경보! 이번 주 손님 나이가 학습 때와 크게 달라졌어요(<strong>데이터 드리프트</strong>). 정답 없이도 바로 알아챘죠. 정확도는 정답이 오는 {delay}주 뒤에야 확인돼요.
          {oracle && trueAcc < ACC_ALARM && <> 🔮 사실 진짜 정확도는 벌써 {pct(trueAcc)}예요 — 모델이 처음 보는 나이대를 엉터리로 짐작하고 있어요.</>}
        </>
      ),
      mood: 'wow',
      tone: 'warn'
    };
  } else if (sim.alarmAcc) {
    note = { t: <>📉 성능 경보! 그런데 입력 분포(PSI)는 멀쩡해요. 손님 나이는 그대로인데 <strong>좋아하는 규칙</strong>이 바뀐 거예요 — <strong>컨셉 드리프트</strong>. 정답이 도착하고 나서야 알 수 있었어요.</>, mood: 'think', tone: 'bad' };
  } else if (oracle && trueAcc < ACC_ALARM) {
    note = { t: <>🔮 쉿! 진짜 정확도는 {pct(trueAcc)}로 떨어졌는데 경보가 안 울렸어요. 손님 나이는 그대로라 PSI로는 못 잡고, 정답은 {delay}주 뒤에 오거든요.</>, mood: 'wink', tone: 'warn' };
  } else if (sim.week === 0) {
    note = {
      t: (
        <>
          음악 앱의 AI가 손님 <strong>나이</strong>만 보고 "이 노래를 좋아할까?"를 맞혀요. 위쪽 띠가 모델의 답이에요. ▶ 재생을 누르면 1주씩 시간이 흘러요.
          {story ? ' 세상이 바뀌면 모델은 어떻게 될까요?' : ' 재생한 뒤 아래 버튼으로 세상을 바꿔 보세요!'}
        </>
      ),
      mood: 'smile'
    };
  } else {
    note = { t: <>모델이 잘 일하고 있어요. 막대(이번 주 손님)와 점선(학습 때 손님)이 겹치면 데이터가 그대로라는 뜻이에요. 그래도 세상은 언제든 바뀌니 계속 지켜봐요 👀</>, mood: 'happy', tone: 'good' };
  }

  const logColor: Record<LogKind, string> = { event: c.series[1], alarm: c.neg, ok: c.pos, same: c.axis, fail: c.series[4] };
  const weekMarkers = (x: (v: number) => number, top: number, bottom: number, withIcons: boolean) => (
    <g>
      {retrains.map((r, i) => (
        <g key={`r${i}`}>
          <line x1={x(r.week)} x2={x(r.week)} y1={top} y2={bottom} stroke={r.kind === 'ok' ? c.pos : r.kind === 'fail' ? c.series[4] : c.axis} strokeWidth={r.kind === 'ok' ? 2 : 1.2} strokeDasharray={r.kind === 'ok' ? undefined : '3 3'} />
          {withIcons && (
            <text x={x(r.week)} y={top + 11} textAnchor="middle" fontSize={11}>
              🔁
            </text>
          )}
        </g>
      ))}
      {[
        { at: sim.dataAt, icon: '👨‍👩‍👧' },
        { at: sim.conceptAt, icon: '📻' }
      ].map(
        (e) =>
          e.at !== null &&
          e.at <= sim.week && (
            <g key={e.icon}>
              <line x1={x(e.at)} x2={x(e.at)} y1={top} y2={bottom} stroke={c.series[1]} strokeWidth={1.5} strokeDasharray="1 3" />
              {withIcons && (
                <text x={x(e.at)} y={bottom - 4} textAnchor="middle" fontSize={12}>
                  {e.icon}
                </text>
              )}
            </g>
          )
      )}
    </g>
  );

  const accPts = (getter: (r: WeekRec, w: number) => number | null) =>
    sim.recs.flatMap((r, i) => {
      const v = getter(r, i + 1);
      return v === null ? [] : [[i + 1, v] as [number, number]];
    });
  const measuredPts = accPts((r, w) => (w <= sim.week - delay ? r.acc : null));
  const truePts = accPts((r) => r.acc);
  const psiPts = sim.recs.map((r, i) => [i + 1, r.psi] as [number, number]);

  return (
    <div>
      <div className="vz-pad" style={{ display: 'grid', gap: 14, gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', paddingBottom: 0 }}>
        <div>
          <div className="vz-panel-title">{sim.week === 0 ? '학습 데이터의 손님 나이' : `${sim.week}주차 손님 나이 vs 학습 때`}</div>
          <Plot xDomain={[A0, A0 + NB * BW]} yDomain={[0, yMax]} height={histH} margin={{ top: 30, right: 12, bottom: 36, left: 44 }} xTicks={6} yTicks={3} xLabel="손님 나이" yFormat={(v) => `${Math.round(v * 100)}%`} ariaLabel="이번 주 손님 나이 분포와 학습 때 분포, 모델의 답">
            {({ x, y, inner }) => {
              const bw = x(A0 + BW) - x(A0);
              const pY = (p: number) => inner.top + inner.height * (1 - p);
              const truth: [number, number][] = [];
              for (let a = A0; a <= A0 + NB * BW; a += 0.25) truth.push([x(a), pY(pLike(a, sim.trend))]);
              const steps: [number, number][] = [];
              model.hist.forEach((h, b) => {
                steps.push([x(A0 + b * BW), y(h)]);
                steps.push([x(A0 + (b + 1) * BW), y(h)]);
              });
              return (
                <g>
                  <text x={inner.left - 6} y={19} textAnchor="end" fontSize={11} fontWeight={700} fill="var(--viz-text)">
                    모델 답
                  </text>
                  {model.labels.map((l, b) => (
                    <g key={`m${b}`}>
                      <rect x={x(A0 + b * BW) + 0.5} y={8} width={Math.max(1, bw - 1)} height={15} rx={2} fill={l ? withAlpha(c.pos, model.seen[b] ? 0.85 : 0.35) : withAlpha(c.text, model.seen[b] ? 0.14 : 0.07)} />
                      {!model.seen[b] && bw > 7 && (
                        <text x={x(center(b))} y={19.5} textAnchor="middle" fontSize={10} fontWeight={800} fill="var(--viz-text)" opacity={0.8}>
                          ?
                        </text>
                      )}
                    </g>
                  ))}
                  {share.map((s, b) => (
                    <rect key={`h${b}`} x={x(A0 + b * BW) + 1} y={y(s)} width={Math.max(1, bw - 2)} height={Math.max(0, y(0) - y(s))} rx={2} fill={oracle && wrongBin(b) ? withAlpha(c.neg, 0.85) : withAlpha(c.brand, 0.7)} />
                  ))}
                  <path d={linePath(steps)} fill="none" stroke="var(--viz-text)" strokeWidth={1.6} strokeDasharray="4 3" opacity={0.75} />
                  {oracle && (
                    <g>
                      <line x1={inner.left} x2={inner.left + inner.width} y1={pY(0.5)} y2={pY(0.5)} stroke={c.series[2]} strokeWidth={1} strokeDasharray="2 4" opacity={0.6} />
                      <path d={linePath(truth)} fill="none" stroke={c.series[2]} strokeWidth={2.4} />
                    </g>
                  )}
                </g>
              );
            }}
          </Plot>
          <Legend
            items={[
              { color: c.brand, label: '이번 주 손님' },
              { color: c.text, label: '학습 때 손님', dashed: true },
              { color: c.pos, label: '모델: "좋아할 거야" (? = 본 적 없어 짐작)' },
              ...(oracle
                ? [
                    { color: c.neg, label: '모델이 틀리는 나이대' },
                    { color: c.series[2], label: '🔮 진짜로 좋아할 확률' }
                  ]
                : [])
            ]}
          />
        </div>
        <div>
          <div className="vz-panel-title">모니터링 대시보드</div>
          <Plot xDomain={[0, WEEKS]} yDomain={[0.4, 1]} height={accH} margin={{ top: 16, right: 12, bottom: 14, left: 44 }} xTicks={6} yTicks={3} xFormat={() => ''} yFormat={(v) => `${Math.round(v * 100)}%`} ariaLabel="주차별 정확도: 확인된 정확도와 진짜 정확도">
            {({ x, y, inner, clip }) => {
              const w0 = Math.max(0, sim.week - delay);
              return (
                <g>
                  <text x={inner.left + 4} y={inner.top - 4} fontSize={11} fontWeight={700} fill="var(--viz-text)">
                    정확도
                  </text>
                  {sim.week > 0 && delay > 0 && (
                    <g>
                      <rect x={x(w0)} y={inner.top} width={Math.max(0, x(sim.week) - x(w0))} height={inner.height} fill={withAlpha(c.series[4], 0.16)} />
                      {x(sim.week) - x(w0) > 14 && (
                        <text x={(x(w0) + x(sim.week)) / 2} y={inner.top + inner.height - 6} textAnchor="middle" fontSize={10} fill="var(--viz-text)">
                          {x(sim.week) - x(w0) > 60 ? '⏳ 정답 대기' : '⏳'}
                        </text>
                      )}
                    </g>
                  )}
                  <line x1={inner.left} x2={inner.left + inner.width} y1={y(ACC_ALARM)} y2={y(ACC_ALARM)} stroke={c.neg} strokeWidth={1.2} strokeDasharray="5 4" />
                  <text x={inner.left + inner.width - 2} y={y(ACC_ALARM) - 4} textAnchor="end" fontSize={10} fill={c.neg}>
                    경보선 80%
                  </text>
                  {weekMarkers(x, inner.top, inner.top + inner.height, true)}
                  <g clipPath={clip}>
                    {oracle && <path d={linePath(truePts.map(([w, v]) => [x(w), y(Math.max(0.4, v))]))} fill="none" stroke={c.series[2]} strokeWidth={2} strokeDasharray="5 4" />}
                    <path d={linePath(measuredPts.map(([w, v]) => [x(w), y(Math.max(0.4, v))]))} fill="none" stroke={c.series[3]} strokeWidth={3} strokeLinejoin="round" />
                  </g>
                </g>
              );
            }}
          </Plot>
          <Plot xDomain={[0, WEEKS]} yDomain={[0, 0.6]} height={psiH} margin={{ top: 16, right: 12, bottom: 40, left: 44 }} xTicks={6} yTicks={3} xLabel="주차" yFormat={(v) => v.toFixed(1)} ariaLabel="주차별 입력 분포 차이 PSI">
            {({ x, y, inner, clip }) => (
              <g>
                <text x={inner.left + 4} y={inner.top - 4} fontSize={11} fontWeight={700} fill="var(--viz-text)">
                  입력 분포 차이 (PSI)
                </text>
                <line x1={inner.left} x2={inner.left + inner.width} y1={y(PSI_ALARM)} y2={y(PSI_ALARM)} stroke={c.neg} strokeWidth={1.2} strokeDasharray="5 4" />
                <text x={inner.left + inner.width - 2} y={y(PSI_ALARM) - 4} textAnchor="end" fontSize={10} fill={c.neg}>
                  경보선 0.2
                </text>
                {weekMarkers(x, inner.top, inner.top + inner.height, false)}
                <g clipPath={clip}>
                  <path d={linePath(psiPts.map(([w, v]) => [x(w), y(Math.min(0.6, v))]))} fill="none" stroke={c.series[0]} strokeWidth={2.5} strokeLinejoin="round" />
                </g>
                {psiPts.map(([w, v]) =>
                  v > 0.6 ? (
                    <text key={w} x={x(w)} y={inner.top + 9} textAnchor="middle" fontSize={9} fill={c.series[0]}>
                      ▲
                    </text>
                  ) : null
                )}
              </g>
            )}
          </Plot>
          <Legend
            items={[
              { color: c.series[3], label: `확인된 정확도 (정답 ${delay}주 뒤 도착)` },
              ...(oracle ? [{ color: c.series[2], label: '🔮 진짜 정확도', dashed: true }] : []),
              { color: c.series[0], label: 'PSI' }
            ]}
          />
        </div>
      </div>
      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <Note mood={note.mood} tone={note.tone}>
          {note.t}
        </Note>
        <div className="vz-stats">
          <Stat label="시간" value={`${sim.week}주차`} sub={`/ ${WEEKS}주 (1년)`} />
          <Stat label="입력 분포 차이 PSI" value={psiNow.toFixed(2)} tone={psiNow > PSI_ALARM ? 'bad' : 'good'} sub="0.2를 넘으면 경보" />
          <Stat label="확인된 정확도" value={known === null ? '정답 대기 중' : pct(known)} tone={known === null ? undefined : known < ACC_ALARM ? 'bad' : 'good'} sub={known === null ? `정답이 ${delay}주 뒤에 와요` : `${sim.week - delay}주차 예측 기준`} />
          {oracle && <Stat label="🔮 진짜 정확도" value={pct(trueAcc)} tone={trueAcc < ACC_ALARM ? 'bad' : 'good'} sub="현실에선 아무도 몰라요" />}
          <Stat label="배포된 모델" value={`v${model.version}`} sub={`${model.trainedAt}주차에 학습`} />
        </div>
        {story ? (
          <div style={{ fontSize: 13, color: 'var(--text-2)' }}>
            📅 <strong>이야기 모드</strong>: {STORY.data}주차에 👨‍👩‍👧 부모님 세대가 가입하고(데이터 드리프트), {STORY.concept}주차에 📻 유행이 바뀌어요(컨셉 드리프트).
          </div>
        ) : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
            <span style={{ fontSize: 13, fontWeight: 700 }}>🌍 세상을 바꾸기:</span>
            <Button onClick={() => startEvent('data')} disabled={sim.dataAt !== null || done}>
              👨‍👩‍👧 부모님 세대 가입 (데이터 드리프트)
            </Button>
            <Button onClick={() => startEvent('concept')} disabled={sim.conceptAt !== null || done}>
              📻 유행이 바뀜 (컨셉 드리프트)
            </Button>
          </div>
        )}
        {sim.log.length > 0 && (
          <div style={{ display: 'grid', gap: 4, fontSize: 13 }} aria-live="polite">
            {sim.log
              .slice(-4)
              .reverse()
              .map((l, i) => (
                <div key={`${l.week}-${i}-${l.text}`} style={{ display: 'flex', gap: 8, alignItems: 'baseline', opacity: i === 0 ? 1 : 0.75 }}>
                  <span style={{ minWidth: 44, fontWeight: 800, color: logColor[l.kind] }}>{l.week}주</span>
                  <span>{l.text}</span>
                </div>
              ))}
          </div>
        )}
      </div>
      <Controls>
        <Button variant="primary" onClick={() => setPlaying((p) => !p)} disabled={done}>
          {playing ? '❚❚ 멈춤' : sim.week === 0 ? '▶ 1년 돌려 보기' : '▶ 계속'}
        </Button>
        <Button onClick={() => setSim((s) => (s.week >= WEEKS ? s : step(s, story, delay, auto)))} disabled={done}>
          +1주
        </Button>
        <Button onClick={() => setSim((s) => retrain(s, delay, false))} disabled={sim.week === 0 || done}>
          🔁 재학습
        </Button>
        <Toggle label="🤖 자동 재학습" checked={auto} onChange={setAuto} />
        <Toggle label="🔮 진짜 정보 보기" checked={oracle} onChange={setOracle} />
        <Slider label="정답이 오는 데 걸리는 시간" value={delay} min={0} max={8} step={1} onChange={setDelay} format={(v) => `${v}주`} accent="var(--peach)" />
        <Segmented
          label="모드"
          options={[
            { value: 'story', label: '📅 이야기' },
            { value: 'free', label: '🧪 실험' }
          ]}
          value={story ? 'story' : 'free'}
          onChange={(v) => reset(v === 'story')}
        />
        <Button variant="ghost" onClick={() => reset()}>
          ↺ 처음부터
        </Button>
      </Controls>
    </div>
  );
}
