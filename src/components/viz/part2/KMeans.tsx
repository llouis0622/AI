import { useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';
import type { VizProps } from '../Viz';
import { MiniChart, Plot, linePath } from '../core/plot';
import { Button, Controls, Note, PlayBar, Segmented, Slider, Stat } from '../core/ui';
import { useElementSize, useInterval, useRafLoop, usePrefersReducedMotion } from '../core/hooks';
import { useVizColors } from '../core/colors';
import { range, rng } from '../core/math';
import { makeDataset } from './_data';

type DS = 'blobs' | 'book' | 'moons';
type Phase = 'init' | 'assigned' | 'updated' | 'done';
type V2 = [number, number];

const BOOK = [1, 2, 3, 10, 11, 12];
const bookX = (v: number) => 0.06 + (v / 13) * 0.88;
/** unit-square squared distance → squared distance in the book's number units */
const BOOK_SCALE = (13 / 0.88) ** 2;

function makePoints(ds: DS): V2[] {
  if (ds === 'book') return BOOK.map((v) => [bookX(v), 0.5]);
  if (ds === 'moons') return makeDataset('moons', 90, 3).map((p) => [p.x, p.y]);
  const r = rng(21);
  const centers: V2[] = [
    [0.24, 0.7],
    [0.74, 0.74],
    [0.52, 0.26]
  ];
  return range(96).map((i) => {
    const c = centers[i % 3];
    return [Math.min(0.97, Math.max(0.03, c[0] + r.normal() * 0.075)), Math.min(0.97, Math.max(0.03, c[1] + r.normal() * 0.075))];
  });
}

const d2 = (a: V2, b: V2) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
const nearest = (p: V2, cs: V2[]) => {
  let best = 0;
  for (let i = 1; i < cs.length; i++) if (d2(p, cs[i]) < d2(p, cs[best])) best = i;
  return best;
};
const sse = (pts: V2[], cs: V2[], asg: number[] | null) => pts.reduce((s, p, i) => s + d2(p, cs[asg ? asg[i] : nearest(p, cs)]), 0);
function means(pts: V2[], asg: number[], cs: V2[]): V2[] {
  return cs.map((c, k) => {
    const mine = pts.filter((_, i) => asg[i] === k);
    if (!mine.length) return c;
    return [mine.reduce((s, p) => s + p[0], 0) / mine.length, mine.reduce((s, p) => s + p[1], 0) / mine.length];
  });
}

function randomInit(pts: V2[], k: number, seed: number): V2[] {
  const r = rng(seed * 101 + 7);
  const idx: number[] = [];
  while (idx.length < k) {
    const i = r.int(pts.length);
    if (!idx.includes(i)) idx.push(i);
  }
  return idx.map((i) => [pts[i][0] + 0.01, pts[i][1] + 0.01]);
}

/** Converged SSE for k = 1..6 (k-means++ style seeding, best of a few runs) for the elbow chart. */
function elbow(pts: V2[]) {
  return range(6).map((i) => {
    const k = i + 1;
    let best = Infinity;
    for (let run = 0; run < 4; run++) {
      const r = rng(1000 + run * 31 + k);
      const cs: V2[] = [pts[r.int(pts.length)]];
      while (cs.length < k) {
        const ds = pts.map((p) => Math.min(...cs.map((c) => d2(p, c))));
        const tot = ds.reduce((s, v) => s + v, 0);
        let u = r.next() * tot;
        let j = 0;
        while (j < ds.length - 1 && u > ds[j]) u -= ds[j++];
        cs.push(pts[j]);
      }
      let cur = cs;
      for (let it = 0; it < 50; it++) {
        const asg = pts.map((p) => nearest(p, cur));
        cur = means(pts, asg, cur);
      }
      best = Math.min(best, sse(pts, cur, null));
    }
    return best;
  });
}

/** k-means, one visible step at a time: ① assign each point, ② move each centre to its group's mean. */
export default function KMeans({ expanded }: VizProps) {
  const c = useVizColors();
  const reduce = usePrefersReducedMotion();
  const [ds, setDs] = useState<DS>('blobs');
  const [k, setK] = useState(3);
  const [seed, setSeed] = useState(2);
  const pts = useMemo(() => makePoints(ds), [ds]);
  const initFor = (d: DS, kk: number, s: number): V2[] => (d === 'book' ? [[bookX(2), 0.5], [bookX(3), 0.5]] : randomInit(makePoints(d), kk, s));
  const [cents, setCents] = useState<V2[]>(() => initFor('blobs', 3, 2));
  const [trails, setTrails] = useState<V2[][]>(() => initFor('blobs', 3, 2).map((p) => [p]));
  const [asg, setAsg] = useState<number[] | null>(null);
  const [phase, setPhase] = useState<Phase>('init');
  const [iter, setIter] = useState(0);
  const [hist, setHist] = useState<number[]>([]);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [placing, setPlacing] = useState<V2[] | null>(null);
  const anim = useRef<{ from: V2[]; t: number } | null>(null);
  const [animT, setAnimT] = useState(1);
  const [box, { width }] = useElementSize<HTMLDivElement>();
  const elbowSse = useMemo(() => (ds === 'book' ? null : elbow(pts)), [ds, pts]);

  const kk = ds === 'book' ? 2 : k;
  const colors = range(6).map((i) => c.series[i]);

  const restart = (d: DS = ds, nk: number = kk, s: number = seed, start?: V2[]) => {
    const cs = start ?? initFor(d, nk, s);
    setCents(cs);
    setTrails(cs.map((p) => [p]));
    setAsg(null);
    setPhase('init');
    setIter(0);
    setHist([]);
    setPlaying(false);
    anim.current = null;
    setAnimT(1);
  };

  const step = () => {
    if (placing) return;
    if (phase === 'init' || phase === 'updated') {
      const next = pts.map((p) => nearest(p, cents));
      if (phase === 'updated' && asg && next.every((v, i) => v === asg[i])) {
        setPhase('done');
        setPlaying(false);
        return;
      }
      setAsg(next);
      setPhase('assigned');
      setHist((h) => [...h, sse(pts, cents, next)]);
    } else if (phase === 'assigned' && asg) {
      const next = means(pts, asg, cents);
      anim.current = { from: cents, t: 0 };
      setAnimT(reduce ? 1 : 0);
      setCents(next);
      setTrails((ts) => ts.map((t, i) => [...t, next[i]]));
      setPhase('updated');
      setIter((n) => n + 1);
      setHist((h) => [...h, sse(pts, next, asg)]);
    }
  };

  useRafLoop((dt) => setAnimT((t) => Math.min(1, t + dt * 2.2)), animT < 1);
  useInterval(
    () => {
      if (phase === 'done' || iter > 40) setPlaying(false);
      else if (animT >= 1) step();
    },
    playing ? 950 / speed : null
  );

  // drawn positions (animated)
  const shown: V2[] = cents.map((p, i) => {
    const a = anim.current;
    if (!a || animT >= 1 || !a.from[i]) return p;
    const e = 1 - (1 - animT) ** 3;
    return [a.from[i][0] + (p[0] - a.from[i][0]) * e, a.from[i][1] + (p[1] - a.from[i][1]) * e];
  });

  const W = Math.max(280, width);
  const H = ds === 'book' ? 190 : Math.round(expanded ? Math.min(560, Math.max(320, W * 0.55)) : Math.min(420, Math.max(270, W * 0.62)));
  const X = (v: number) => v * W;
  const Y = (v: number) => (1 - v) * H;
  const curSse = sse(pts, cents, asg);

  const onClick = (e: ReactMouseEvent<SVGSVGElement>) => {
    if (!placing) return;
    const r = e.currentTarget.getBoundingClientRect();
    const p: V2 = [(e.clientX - r.left) / r.width, ds === 'book' ? 0.5 : 1 - (e.clientY - r.top) / r.height];
    const next = [...placing, p];
    if (next.length >= kk) {
      setPlacing(null);
      restart(ds, kk, seed, next);
    } else setPlacing(next);
  };

  let note: { text: string; mood: 'smile' | 'happy' | 'think' | 'wow' | 'wink' | 'cheer'; tone?: 'good' | 'warn' | 'bad' };
  if (placing) note = { text: `지도 위를 ${kk - placing.length}번 더 눌러 중심(✖)을 찍어 주세요. 일부러 한쪽 구석에 몰아 찍어 보는 것도 재밌어요!`, mood: 'wink' };
  else if (phase === 'init')
    note = { text: `중심(✖) ${kk}개를 아무 데나 찍었어요. "한 걸음"을 누르면 ① 배정 — 모든 점이 가장 가까운 중심의 색으로 물들어요.`, mood: 'smile' };
  else if (phase === 'assigned') note = { text: '① 배정 끝! 각 점이 가장 가까운 중심 편이 됐어요. 다음은 ② 갱신 — 중심이 자기 무리의 한가운데(평균)로 이사해요.', mood: 'happy' };
  else if (phase === 'updated') note = { text: `② 갱신 끝! 중심이 무리의 평균 위치로 옮겼고, 거리 제곱합이 ${curSse.toFixed(3)}(으)로 줄었어요. 다시 배정해 볼까요?`, mood: 'happy' };
  else if (ds === 'book')
    note = { text: `책의 예와 똑같죠? {1,2,3}과 {10,11,12}로 나뉘고 중심은 2와 11에 안착했어요. ${iter}번 갱신만에 끝! 🎉`, mood: 'cheer', tone: 'good' };
  else if (ds === 'moons')
    note = { text: '멈추긴 했는데… 초승달 두 개를 직선으로 반씩 잘라 버렸어요. k-평균은 "둥근 무리"를 가정하기 때문이에요. 이런 모양은 밀도 기반 방법(DBSCAN)이 맡아요.', mood: 'think', tone: 'warn' };
  else note = { text: `더 이상 소속이 안 바뀌어요 — 끝! 🎉 ${iter}번 반복만에 안착했어요. 중심을 다르게 찍으면 결과가 달라질 수도 있어요.`, mood: 'cheer', tone: 'good' };

  const phaseLabel = placing ? '중심 찍는 중' : phase === 'init' ? '시작 전' : phase === 'assigned' ? '① 배정' : phase === 'updated' ? '② 갱신' : '끝!';
  const elbowK = elbowSse
    ? (() => {
        let bestK = 2;
        let bestBend = -Infinity;
        for (let i = 1; i < 5; i++) {
          const bend = elbowSse[i - 1] - 2 * elbowSse[i] + elbowSse[i + 1];
          if (bend > bestBend) {
            bestBend = bend;
            bestK = i + 1;
          }
        }
        return bestK;
      })()
    : 0;

  return (
    <div ref={box}>
      <div className="vz-pad" style={{ paddingBottom: 0 }}>
        <div style={{ position: 'relative', borderRadius: 14, overflow: 'hidden', border: '1px solid var(--border)', background: 'var(--surface-2)' }}>
          {width > 0 && (
            <svg
              className="vz-svg"
              width={W}
              height={H}
              viewBox={`0 0 ${W} ${H}`}
              role="img"
              aria-label="k-평균: 점들이 가장 가까운 중심의 색으로 물들고 중심이 무리의 평균으로 이동하는 모습"
              onClick={onClick}
              style={{ display: 'block', cursor: placing ? 'crosshair' : 'default' }}
            >
              {ds === 'book' && (
                <g>
                  <line x1={X(0.03)} x2={X(0.97)} y1={Y(0.5)} y2={Y(0.5)} stroke={c.axis} strokeWidth={2} />
                  {range(14).map((v) => (
                    <g key={v}>
                      <line x1={X(bookX(v))} x2={X(bookX(v))} y1={Y(0.5) - 4} y2={Y(0.5) + 4} stroke={c.axis} />
                      <text x={X(bookX(v))} y={Y(0.5) + 46} textAnchor="middle" className="vz-tick" style={{ fill: c.text }}>
                        {v}
                      </text>
                    </g>
                  ))}
                </g>
              )}
              {/* lines from points to their centre */}
              {asg &&
                phase !== 'done' &&
                pts.map((p, i) => (
                  <line key={`l${i}`} x1={X(p[0])} y1={Y(p[1])} x2={X(shown[asg[i]][0])} y2={Y(shown[asg[i]][1])} stroke={colors[asg[i]]} strokeWidth={1} opacity={phase === 'assigned' ? 0.45 : 0.18} />
                ))}
              {/* trails */}
              {trails.map((t, i) =>
                t.length > 1 ? (
                  <path key={`t${i}`} d={linePath(t.map((p) => [X(p[0]), Y(p[1])]))} fill="none" stroke={colors[i]} strokeWidth={2} strokeDasharray="4 4" opacity={0.8} />
                ) : null
              )}
              {pts.map((p, i) => (
                <circle
                  key={`p${i}`}
                  cx={X(p[0])}
                  cy={Y(p[1])}
                  r={ds === 'book' ? 9 : 5.5}
                  fill={asg ? colors[asg[i]] : c.theme === 'dark' ? '#8d97c4' : '#a49db8'}
                  stroke={c.surface}
                  strokeWidth={1.5}
                  style={{ transition: 'fill 0.35s' }}
                />
              ))}
              {/* centres */}
              {shown.map((p, i) => (
                <g key={`c${i}`} transform={`translate(${X(p[0])} ${Y(p[1])})`}>
                  <circle r={15} fill={colors[i]} opacity={0.22} />
                  <path d="M-8,-8 L8,8 M8,-8 L-8,8" stroke={c.surface} strokeWidth={8} strokeLinecap="round" />
                  <path d="M-8,-8 L8,8 M8,-8 L-8,8" stroke={colors[i]} strokeWidth={4.5} strokeLinecap="round" />
                  {ds === 'book' && (
                    <text y={-20} textAnchor="middle" style={{ fontSize: 12.5, fontWeight: 850, fill: colors[i] }}>
                      중심 {((p[0] - 0.06) * (13 / 0.88)).toFixed(1)}
                    </text>
                  )}
                </g>
              ))}
              {placing?.map((p, i) => (
                <g key={`pl${i}`} transform={`translate(${X(p[0])} ${Y(p[1])})`}>
                  <path d="M-8,-8 L8,8 M8,-8 L-8,8" stroke={colors[i]} strokeWidth={4.5} strokeLinecap="round" />
                </g>
              ))}
            </svg>
          )}
          <span
            style={{
              position: 'absolute',
              left: 8,
              top: 8,
              padding: '3px 10px',
              borderRadius: 999,
              fontSize: 12.5,
              fontWeight: 850,
              background: 'color-mix(in srgb, var(--surface-solid) 85%, transparent)',
              border: '1px solid var(--border)',
              pointerEvents: 'none'
            }}
          >
            {phaseLabel}
          </span>
        </div>
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 12 }}>
        <PlayBar
          playing={playing}
          onToggle={() => {
            if (phase === 'done') restart();
            setPlaying((p) => !p);
          }}
          onStep={phase === 'done' || placing ? undefined : step}
          onReset={() => restart()}
          speed={speed}
          onSpeed={setSpeed}
        />
        <Note mood={note.mood} tone={note.tone}>
          {note.text}
        </Note>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, alignItems: 'start' }}>
          <div style={{ display: 'grid', gap: 10 }}>
            <div className="vz-stats">
              <Stat label="반복" value={iter} />
              <Stat label="단계" value={phaseLabel} tone="brand" />
              <Stat label="거리 제곱합" value={(ds === 'book' ? curSse * BOOK_SCALE : curSse).toFixed(ds === 'book' ? 1 : 3)} tone={phase === 'done' ? 'good' : undefined} sub="작을수록 잘 뭉침" />
            </div>
            <div>
              <div className="vz-panel-title">걸음마다 거리 제곱합</div>
              <MiniChart series={[ds === 'book' ? hist.map((v) => v * BOOK_SCALE) : hist]} colors={[c.series[0]]} height={100} xLabel="걸음" />
            </div>
          </div>
          {elbowSse && (
            <div>
              <div className="vz-panel-title">k를 바꾸면? (끝까지 돌린 거리 제곱합)</div>
              <Plot
                xDomain={[1, 6]}
                yDomain={[0, Math.max(...elbowSse) * 1.08]}
                height={170}
                margin={{ top: 10, right: 12, bottom: 36, left: 46 }}
                xTicks={5}
                yTicks={3}
                xLabel="무리 수 k"
                yFormat={(v) => v.toFixed(1)}
                ariaLabel="무리 수 k에 따른 최종 거리 제곱합 (팔꿈치 그래프)"
              >
                {({ x, y }) => (
                  <g>
                    <path d={linePath(elbowSse.map((v, i) => [x(i + 1), y(v)]))} fill="none" stroke={c.series[0]} strokeWidth={3} />
                    {elbowSse.map((v, i) => (
                      <circle key={i} cx={x(i + 1)} cy={y(v)} r={i + 1 === kk ? 8 : 4.5} fill={c.series[0]} stroke={c.surface} strokeWidth={2} style={i + 1 === kk ? { fill: 'var(--lemon)' } : undefined} />
                    ))}
                    {ds === 'blobs' && (
                      <text x={x(elbowK) + 10} y={y(elbowSse[elbowK - 1]) - 10} style={{ fontSize: 12, fontWeight: 850, fill: c.ink }}>
                        💪 팔꿈치 (k={elbowK})
                      </text>
                    )}
                  </g>
                )}
              </Plot>
            </div>
          )}
        </div>
      </div>

      <Controls>
        <Segmented
          label="데이터"
          options={[
            { value: 'blobs', label: '세 무리' },
            { value: 'book', label: '책 속 1차원 예' },
            { value: 'moons', label: '초승달 (함정)' }
          ]}
          value={ds}
          onChange={(v) => {
            setDs(v);
            setPlacing(null);
            const nk = v === 'book' ? 2 : v === 'moons' ? 2 : 3;
            setK(nk);
            restart(v, nk, seed);
          }}
        />
        <Slider
          label="무리 수 k"
          value={kk}
          min={2}
          max={6}
          step={1}
          disabled={ds === 'book'}
          onChange={(v) => {
            const nk = Math.round(v);
            setK(nk);
            setPlacing(null);
            restart(ds, nk, seed);
          }}
          format={(v) => `${v}개`}
        />
        <Button
          size="sm"
          onClick={() => {
            const s = seed + 1;
            setSeed(s);
            setPlacing(null);
            restart(ds, kk, s);
          }}
          disabled={ds === 'book'}
        >
          🎲 중심 다시 찍기
        </Button>
        <Button size="sm" variant={placing ? 'primary' : 'ghost'} onClick={() => (placing ? setPlacing(null) : (restart(), setPlacing([])))} disabled={ds === 'book'}>
          📍 {placing ? '찍기 취소' : '중심 직접 찍기'}
        </Button>
      </Controls>
    </div>
  );
}
