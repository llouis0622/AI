import type { ReactNode } from 'react';
import { CodeStory, type StoryStep } from '../core/story';
import type { VizColors } from '../core/colors';
import { Arrow, Chip, Curve, Defs, Txt, VB, mix, seg, soft } from './_labsA';

/* ---------- the lab's BPE, run for real on a toy corpus ---------- */

const CORPUS = 'low lower newest widest low newest lower widest newest low';
const enc = new TextEncoder();
const dec = new TextDecoder();

type Pair = [number, number];

function mergeIds(ids: number[], pair: Pair, newId: number) {
  const out: number[] = [];
  for (let i = 0; i < ids.length; ) {
    if (i < ids.length - 1 && ids[i] === pair[0] && ids[i + 1] === pair[1]) {
      out.push(newId);
      i += 2;
    } else {
      out.push(ids[i]);
      i += 1;
    }
  }
  return out;
}

function countPairs(ids: number[]) {
  const counts = new Map<string, number>();
  for (let i = 0; i < ids.length - 1; i++) {
    const k = `${ids[i]},${ids[i + 1]}`;
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  return counts;
}

function train(text: string, nMerges: number) {
  let ids = [...enc.encode(text)];
  const vocab = new Map<number, number[]>();
  for (let i = 0; i < 256; i++) vocab.set(i, [i]);
  const merges: { pair: Pair; id: number; count: number; lenAfter: number }[] = [];
  const history = [ids];
  for (let k = 0; k < nMerges; k++) {
    let best: string | null = null;
    let bc = 0;
    for (const [key, v] of countPairs(ids)) {
      if (v > bc) {
        bc = v;
        best = key;
      }
    }
    if (!best) break;
    const pair = best.split(',').map(Number) as Pair;
    const id = 256 + k;
    ids = mergeIds(ids, pair, id);
    vocab.set(id, [...vocab.get(pair[0])!, ...vocab.get(pair[1])!]);
    merges.push({ pair, id, count: bc, lenAfter: ids.length });
    history.push(ids);
  }
  return { merges, vocab, history };
}

const T = train(CORPUS, 8);
const BYTES0 = T.history[0];
const show = (id: number) => dec.decode(new Uint8Array(T.vocab.get(id)!)).replace(/ /g, '␣');

/** encode() exactly like the lab: repeatedly apply the earliest-learned applicable merge. */
function encodeTrace(text: string) {
  let ids = [...enc.encode(text)];
  const rank = new Map(T.merges.map((m) => [`${m.pair[0]},${m.pair[1]}`, m.id]));
  const trace = [ids];
  while (ids.length >= 2) {
    let best: Pair | null = null;
    let bestRank = Infinity;
    for (let i = 0; i < ids.length - 1; i++) {
      const r = rank.get(`${ids[i]},${ids[i + 1]}`);
      if (r !== undefined && r < bestRank) {
        bestRank = r;
        best = [ids[i], ids[i + 1]];
      }
    }
    if (!best) break;
    ids = mergeIds(ids, best, bestRank);
    trace.push(ids);
  }
  return trace;
}
const LOWEST = encodeTrace('lowest');
const TOP_PAIRS = [...countPairs(BYTES0).entries()]
  .sort((a, b) => b[1] - a[1])
  .slice(0, 6)
  .map(([k, v]) => ({ pair: k.split(',').map(Number) as Pair, count: v }));
const HAK = [...enc.encode('학')];

/** Draw a row of tokens as chips, wrapping by width. */
function ChipRow({
  ids,
  x,
  y,
  maxW,
  c,
  t = 1,
  highlight,
  fs = 13
}: {
  ids: number[];
  x: number;
  y: number;
  maxW: number;
  c: VizColors;
  t?: number;
  highlight?: (id: number, i: number) => boolean;
  fs?: number;
}) {
  const out: ReactNode[] = [];
  let cx = x;
  let cy = y;
  ids.forEach((id, i) => {
    const label = show(id);
    const w = Math.max(20, label.length * (fs * 0.62) + 10);
    if (cx + w > x + maxW) {
      cx = x;
      cy += 32;
    }
    const merged = id >= 256;
    out.push(
      <Chip
        key={i}
        x={cx + w / 2}
        y={cy}
        text={label}
        w={w}
        h={26}
        fs={fs}
        mono
        color={merged ? c.series[2] : c.series[0]}
        active={highlight?.(id, i)}
        t={Math.min(1, t * 1.5 - (i / ids.length) * 0.5)}
        c={c}
      />
    );
    cx += w + 4;
  });
  return <g>{out}</g>;
}

const steps: StoryStep[] = [
  {
    title: '바이트에서 시작: 어떤 글자든 0~255',
    text: (
      <>
        텍스트를 <code>utf-8</code>로 인코딩하면 어떤 언어든 <strong>0~255 사이 숫자(바이트)</strong>가 돼요. 영어 글자는 1바이트, 한글 한 글자는 <strong>3바이트</strong>예요. 그래서 처음 어휘는 바이트 256개 — 처음 보는 글자가 와도 절대 막히지 않아요.
      </>
    ),
    code: `# 바이트 수준에서 시작: 유니코드 무엇이 와도 0~255로 표현된다.
ids = list(text.encode("utf-8"))
self.vocab = {i: bytes([i]) for i in range(256)}`,
    highlight: [2, 3],
    render: (t, c) => {
      const low = [...enc.encode('low')];
      return (
        <g>
          <Defs />
          <Txt x={110} y={44} size={14} weight={850}>
            "low"
          </Txt>
          {low.map((b, i) => (
            <g key={i}>
              <Chip x={60 + i * 50} y={86} text={String.fromCharCode(b)} w={40} h={32} fs={16} mono color={c.series[0]} t={seg(t, 0.05 + i * 0.08, 0.25 + i * 0.08)} c={c} />
              <Txt x={60 + i * 50} y={126} size={13} weight={800} color={c.series[0]} mono opacity={seg(t, 0.25 + i * 0.08, 0.4 + i * 0.08)}>
                {b}
              </Txt>
            </g>
          ))}
          <Txt x={110} y={156} size={12} weight={700} color="var(--text-2)" opacity={seg(t, 0.4, 0.55)}>
            영어 1글자 = 1바이트
          </Txt>
          <Txt x={350} y={44} size={14} weight={850}>
            "학"
          </Txt>
          <g opacity={seg(t, 0.4, 0.6)}>
            <Chip x={350} y={86} text="학" w={44} h={34} fs={17} color={c.series[3]} c={c} />
          </g>
          {HAK.map((b, i) => (
            <g key={i}>
              <Arrow x1={350} y1={106} x2={300 + i * 50} y2={150} color={c.series[3]} width={1.5} t={seg(t, 0.55, 0.7)} head={false} />
              <Chip x={300 + i * 50} y={168} text={String(b)} w={42} h={28} fs={13} mono color={c.series[3]} t={seg(t, 0.65 + i * 0.06, 0.8 + i * 0.06)} c={c} />
            </g>
          ))}
          <Txt x={350} y={210} size={12} weight={700} color="var(--text-2)" opacity={seg(t, 0.8, 0.95)}>
            한글 1글자 = 3바이트
          </Txt>
          <g opacity={seg(t, 0.85, 1)}>
            <rect x={60} y={236} width={360} height={40} rx={14} fill={soft(c.series[1], 0.12)} stroke={soft(c.series[1], 0.6)} />
            <Txt x={240} y={261} size={13} weight={800}>
              처음 어휘 = 바이트 256개 (모르는 글자 없음!)
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '이웃한 쌍을 세어 본다',
    text: (
      <>
        말뭉치 전체를 훑으며 <strong>바로 옆에 붙은 두 토큰의 쌍</strong>이 몇 번 나오는지 세요. 이 장난감 말뭉치에서는 여러 쌍이 5번씩 나와 <strong>공동 1등</strong>이에요. 동점이면 <code>most_common</code>은 먼저 나온 쌍, 즉 <code>l·o</code>를 골라요.
      </>
    ),
    code: `pairs = Counter(zip(ids, ids[1:]))         # 인접 쌍 빈도
if not pairs:
    break
best = pairs.most_common(1)[0][0]          # 최빈 쌍`,
    highlight: [1, 4],
    render: (t, c) => {
      const max = TOP_PAIRS[0].count;
      return (
        <g>
          <Defs />
          <Txt x={16} y={26} size={12.5} weight={800} anchor="start" color="var(--text-2)">
            말뭉치 (␣ = 공백, {BYTES0.length}바이트)
          </Txt>
          <ChipRow ids={BYTES0.slice(0, 23)} x={16} y={50} maxW={450} c={c} t={seg(t, 0, 0.35)} fs={12.5} highlight={(_, i) => i < 2} />
          <Txt x={458} y={86} size={14} weight={900} anchor="end" color="var(--text-2)" opacity={seg(t, 0.2, 0.35)}>
            …
          </Txt>
          <Txt x={16} y={124} size={12.5} weight={800} anchor="start" color="var(--text-2)" opacity={seg(t, 0.35, 0.5)}>
            쌍 빈도 순위
          </Txt>
          {TOP_PAIRS.map((p, i) => {
            const w = (p.count / max) * 230 * seg(t, 0.4 + i * 0.06, 0.6 + i * 0.06);
            const label = `${show(p.pair[0])} · ${show(p.pair[1])}`;
            return (
              <g key={i}>
                <Txt x={110} y={150 + i * 24} size={13} weight={800} anchor="end" mono>
                  {label}
                </Txt>
                <rect x={120} y={137 + i * 24} width={w} height={17} rx={5} fill={i === 0 ? c.series[2] : soft(c.series[2], 0.45)} />
                <Txt x={126 + w} y={150 + i * 24} size={12} weight={850} anchor="start">
                  {p.count}
                </Txt>
              </g>
            );
          })}
          <Txt x={420} y={160} size={13} weight={850} color={c.series[2]} opacity={seg(t, 0.8, 1)}>
            1등: {show(TOP_PAIRS[0].pair[0])}·{show(TOP_PAIRS[0].pair[1])}
          </Txt>
        </g>
      );
    }
  },
  {
    title: '가장 흔한 쌍을 새 토큰으로 합친다',
    text: (
      <>
        1등 쌍 <code>l·o</code>를 새 토큰 <strong>256번 "lo"</strong>로 바꿔요. 말뭉치에서 그 쌍이 나오는 곳마다 두 칸이 한 칸으로 합쳐지니, 전체 길이가 <strong>{BYTES0.length} → {T.history[1].length}</strong>로 줄어요.
      </>
    ),
    code: `@staticmethod
def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out`,
    highlight: [5, 6],
    render: (t, c) => {
      const before = BYTES0.slice(0, 18);
      const after = mergeIds(before, T.merges[0].pair, T.merges[0].id);
      const p = seg(t, 0.2, 0.7);
      return (
        <g>
          <Defs />
          <Txt x={16} y={34} size={12.5} weight={800} anchor="start" color="var(--text-2)">
            합치기 전
          </Txt>
          <ChipRow ids={before} x={16} y={60} maxW={450} c={c} fs={13} highlight={(id, i) => (id === T.merges[0].pair[0] && before[i + 1] === T.merges[0].pair[1]) || (id === T.merges[0].pair[1] && before[i - 1] === T.merges[0].pair[0])} />
          <Arrow x1={240} y1={110} x2={240} y2={150} color={c.series[2]} t={seg(t, 0.15, 0.35)} flow />
          <Txt x={254} y={135} size={12.5} weight={800} anchor="start" color={c.series[2]} opacity={seg(t, 0.25, 0.4)}>
            {`l·o → ${T.merges[0].id}번 "lo"`}
          </Txt>
          <Txt x={16} y={176} size={12.5} weight={800} anchor="start" color="var(--text-2)" opacity={p}>
            합친 뒤
          </Txt>
          <g opacity={p}>
            <ChipRow ids={after} x={16} y={202} maxW={450} c={c} t={p} fs={13} highlight={(id) => id === T.merges[0].id} />
          </g>
          <Txt x={240} y={280} size={13} weight={850} opacity={seg(t, 0.75, 1)}>
            전체 길이 {BYTES0.length} → {T.history[1].length}
          </Txt>
        </g>
      );
    }
  },
  {
    title: '반복하면 어휘가 자란다',
    text: (
      <>
        같은 일을 원하는 어휘 크기가 될 때까지 반복해요. <code>lo → low</code>, <code>es → est</code>처럼 조각이 점점 단어다워져요. 그런데 <strong>"est␣"</strong>처럼 공백까지 붙은 조각도 생겨요 — 실전 토크나이저가 먼저 단어 단위로 나누는(사전분할) 이유예요.
      </>
    ),
    code: `for new_id in range(256, vocab_size):
    pairs = Counter(zip(ids, ids[1:]))         # 인접 쌍 빈도
    if not pairs:
        break
    best = pairs.most_common(1)[0][0]          # 최빈 쌍
    ids = self._merge(ids, best, new_id)
    self.merges[best] = new_id
    self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]`,
    highlight: [6, 7, 8],
    render: (t, c) => {
      const n = T.merges.length;
      return (
        <g>
          <Defs />
          <Txt x={16} y={26} size={12.5} weight={800} anchor="start" color="var(--text-2)">
            배운 병합 규칙 (순서가 중요!)
          </Txt>
          {T.merges.map((m, i) => {
            const on = seg(t, 0.05 + i * 0.08, 0.15 + i * 0.08);
            const row = i % 4;
            const col = Math.floor(i / 4);
            const x = 16 + col * 150;
            const y = 44 + row * 40;
            return (
              <g key={m.id} opacity={on} transform={`translate(${(1 - on) * -10} 0)`}>
                <rect x={x} y={y} width={142} height={32} rx={9} fill={soft(c.series[2], 0.1)} stroke={soft(c.series[2], 0.5)} />
                <Txt x={x + 10} y={y + 21} size={12} weight={850} anchor="start" mono color="var(--text-2)">
                  {m.id}
                </Txt>
                <Txt x={x + 134} y={y + 21} size={13} weight={850} anchor="end" mono>
                  {show(m.id)}
                </Txt>
              </g>
            );
          })}
          <Curve
            x={322}
            y={56}
            w={140}
            h={110}
            fn={(k) => T.history[Math.min(n, Math.max(0, Math.round(k)))].length}
            xDomain={[0, n]}
            yDomain={[0, BYTES0.length]}
            t={seg(t, 0.2, 0.9)}
            color={c.series[0]}
            n={n}
            xLabel="병합 횟수"
            yLabel="말뭉치 길이"
            marker={n}
            markerLabel={String(T.history[n].length)}
            markerAnchor="end"
          />
          <g opacity={seg(t, 0.75, 1)}>
            <rect x={16} y={216} width={448} height={56} rx={14} fill={soft(c.series[4], 0.1)} stroke={soft(c.series[4], 0.6)} />
            <Txt x={240} y={240} size={13} weight={800}>
              {BYTES0.length}바이트 → {T.history[n].length}토큰. 자주 쓰는 조각일수록 하나로 뭉쳐요
            </Txt>
            <Txt x={240} y={262} size={12} weight={700} color="var(--text-2)">
              "est␣"처럼 공백이 붙은 조각 → 실전에선 사전분할로 막아요
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '인코딩: 배운 순서대로 적용한다',
    text: (
      <>
        처음 보는 단어 <strong>"lowest"</strong>도 인코딩할 수 있어요. 적용 가능한 병합 중 <strong>가장 먼저 배운 것(번호가 작은 것)</strong>부터 하나씩 적용하면 <code>l o w e s t</code> → <code>low</code> + <code>est</code>, 토큰 2개가 돼요. 순서를 무시하면 학습 때와 다르게 잘려요.
      </>
    ),
    code: `def encode(self, text: str) -> list[int]:
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pairs = set(zip(ids, ids[1:]))
        # 적용 가능한 merge 중 가장 먼저 학습된 것(id가 작은 것)부터
        candidates = [p for p in pairs if p in self.merges]
        if not candidates:
            break
        best = min(candidates, key=lambda p: self.merges[p])
        ids = self._merge(ids, best, self.merges[best])
    return ids`,
    highlight: [6, 9, 10],
    render: (t, c) => (
      <g>
        <Defs />
        <Txt x={16} y={28} size={12.5} weight={800} anchor="start" color="var(--text-2)">
          "lowest" (말뭉치에 없던 단어)
        </Txt>
        {LOWEST.map((ids, k) => {
          const on = seg(t, 0.05 + k * 0.17, 0.2 + k * 0.17);
          return (
            <g key={k} opacity={on}>
              <Txt x={28} y={64 + k * 44} size={12} weight={850} anchor="start" color="var(--text-2)">
                {k === 0 ? '시작' : `${k}단계`}
              </Txt>
              <ChipRow ids={ids} x={84} y={60 + k * 44} maxW={330} c={c} fs={14} highlight={(id) => k > 0 && id === Math.max(...ids.filter((v) => v >= 256), -1)} />
            </g>
          );
        })}
        <g opacity={seg(t, 0.85, 1)}>
          <Txt x={420} y={64 + (LOWEST.length - 1) * 44} size={13} weight={900} color={c.series[2]}>
            토큰 {LOWEST[LOWEST.length - 1].length}개!
          </Txt>
        </g>
      </g>
    )
  },
  {
    title: '디코딩: 이어 붙이면 원문 그대로',
    text: (
      <>
        디코딩은 각 토큰의 바이트를 <strong>그냥 이어 붙이면</strong> 끝이에요. 바이트 수준이라 <code>decode(encode(s)) == s</code>가 항상 성립해요(무손실). 토큰 경계가 한글 3바이트 중간에 걸려도 <code>errors="replace"</code>로 안전하게 처리해요.
      </>
    ),
    code: `def decode(self, ids: list[int]) -> str:
    data = b"".join(self.vocab[i] for i in ids)
    # 잘못된 토큰 경계로 바이트가 깨질 수 있으니 replace로 안전하게
    return data.decode("utf-8", errors="replace")`,
    highlight: [2, 4],
    render: (t, c) => {
      const toks = LOWEST[LOWEST.length - 1];
      const bytes = toks.flatMap((id) => T.vocab.get(id)!);
      const p = seg(t, 0.25, 0.6);
      return (
        <g>
          <Defs />
          <Txt x={240} y={34} size={12.5} weight={800} color="var(--text-2)">
            토큰
          </Txt>
          {toks.map((id, i) => (
            <Chip key={i} x={200 + i * 80} y={64} text={show(id)} w={70} h={32} fs={15} mono color={c.series[2]} t={seg(t, 0, 0.2)} c={c} />
          ))}
          <Arrow x1={240} y1={86} x2={240} y2={116} color={c.series[0]} t={seg(t, 0.15, 0.3)} />
          <Txt x={240} y={140} size={12.5} weight={800} color="var(--text-2)" opacity={p}>
            바이트 이어 붙이기
          </Txt>
          {bytes.map((b, i) => (
            <g key={i} opacity={p}>
              <Chip x={mix(240, 90 + i * 60, p)} y={168} text={String(b)} w={50} h={28} fs={13} mono color={c.series[0]} c={c} />
            </g>
          ))}
          <Arrow x1={240} y1={188} x2={240} y2={218} color={c.series[0]} t={seg(t, 0.6, 0.75)} />
          <g opacity={seg(t, 0.75, 0.95)}>
            <rect x={150} y={226} width={180} height={44} rx={14} fill={soft(c.pos, 0.14)} stroke={c.pos} />
            <Txt x={240} y={254} size={18} weight={900} mono>
              "lowest" ✓
            </Txt>
          </g>
        </g>
      );
    }
  }
];

/** Code lab 07 walkthrough: byte-level BPE training, encoding and decoding on a toy corpus. */
export default function Lab07BpeTrain() {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={2} />;
}
