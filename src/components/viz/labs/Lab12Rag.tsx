import type { VizProps } from '../Viz';
import { CodeStory, type StoryStep } from '../core/story';
import type { VizColors } from '../core/colors';
import { VB, FS, seg, alpha, rich, dd, T, Box, Arrow, Pill, Card, Pulse } from './_LabKitB';

/* A toy "meaning map": 9 chunks from 3 documents, drawn on the unit circle. */
const CX = 300;
const CY = 168;
const R = 96;
const TOPICS = [
  { name: '배포 절차', file: 'deploy.md', angles: [22, 35, 48] },
  { name: '휴가 규정', file: 'leave.md', angles: [75, 88, 100] },
  { name: '보안 정책', file: 'security.md', angles: [118, 128, 138] }
];
const QUERY_ANGLE = 38;
const DOTS = TOPICS.flatMap((tp, ti) =>
  tp.angles.map((a, k) => ({
    topic: ti,
    idx: k,
    angle: a,
    file: tp.file,
    // unnormalised length before "normalize_embeddings=True"
    len: [0.55, 1.35, 0.8, 1.2, 0.65, 1.05, 0.7, 1.4, 0.9][ti * 3 + k],
    score: Math.cos(((a - QUERY_ANGLE) * Math.PI) / 180)
  }))
);
const RANKED = [...DOTS].sort((a, b) => b.score - a.score);
const TOP = RANKED.slice(0, 5);

const rad = (deg: number) => (deg * Math.PI) / 180;
const onCircle = (deg: number, r = R): [number, number] => [CX + r * Math.cos(rad(deg)), CY - r * Math.sin(rad(deg))];
const topicColor = (c: VizColors, ti: number) => [c.series[0], c.series[2], c.series[3]][ti];

function MeaningMap({ c, t, normalize, showQuery }: { c: VizColors; t: number; normalize: number; showQuery: number }) {
  return (
    <g>
      <circle cx={CX} cy={CY} r={R} fill="none" stroke="var(--border-strong)" strokeWidth={1.5} strokeDasharray="4 5" />
      <line x1={CX - R - 10} y1={CY} x2={CX + R + 10} y2={CY} stroke="var(--border)" />
      <line x1={CX} y1={CY - R - 10} x2={CX} y2={CY + R + 10} stroke="var(--border)" />
      <circle cx={CX} cy={CY} r={3} fill="var(--text-3)" />
      {DOTS.map((d, i) => {
        const len = d.len + (1 - d.len) * normalize;
        const [x, y] = onCircle(d.angle, R * len);
        const appear = seg(t, 0.05 + i * 0.04, 0.25);
        return <circle key={i} cx={x} cy={y} r={7 * appear} fill={topicColor(c, d.topic)} stroke="var(--surface-solid)" strokeWidth={2} />;
      })}
      {showQuery > 0 && (
        <g>
          <Arrow x1={CX} y1={CY} x2={onCircle(QUERY_ANGLE, R * 1.0)[0]} y2={onCircle(QUERY_ANGLE, R * 1.0)[1]} color={c.neg} t={showQuery} width={3} />
        </g>
      )}
    </g>
  );
}

const steps: StoryStep[] = [
  {
    title: '청킹 — 문단 경계를 존중하며 자르기',
    text: rich('긴 문서를 그대로 넣을 수는 없어요. 문단을 차곡차곡 담다가 **800자를 넘으면 잘라** 새 조각을 시작해요. 이때 앞 조각의 끝 **150자를 겹쳐 붙여서**, 경계에 걸친 문장도 앞뒤 문맥을 잃지 않게 해요.'),
    code: dd(String.raw`def chunk_text(text: str, source: str) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) > CHUNK_SIZE and buf:
            chunks.append(buf)
            buf = buf[-CHUNK_OVERLAP:] + "\n" + p     # 오버랩: 경계에 걸친 문맥 보존
        else:
            buf = (buf + "\n\n" + p) if buf else p`),
    highlight: [5, 6, 7],
    render: (t, c) => {
      const paras = [
        { y: 62, n: 2 },
        { y: 96, n: 3 },
        { y: 142, n: 2 },
        { y: 176, n: 3 },
        { y: 222, n: 2 },
        { y: 256, n: 1 }
      ];
      const chunks = [
        { y0: 54, y1: 132, color: c.series[0] },
        { y0: 116, y1: 210, color: c.series[2] },
        { y0: 196, y1: 268, color: c.series[3] }
      ];
      return (
        <g>
          <rect x={14} y={24} width={152} height={262} rx={12} fill="var(--surface-solid)" stroke="var(--border-strong)" strokeWidth={1.5} />
          <T x={26} y={45} size={FS.s} weight={800}>
            📄 deploy.md
          </T>
          {paras.map((p, i) =>
            Array.from({ length: p.n }, (_, k) => (
              <rect key={`${i}-${k}`} x={28} y={p.y + k * 10} width={k === p.n - 1 ? 82 : 122} height={6} rx={3} fill="var(--border-strong)" opacity={0.9} />
            ))
          )}
          {chunks.map((ch, i) => {
            const a = seg(t, 0.12 + i * 0.22, 0.3);
            return (
              <g key={i} opacity={a}>
                <rect x={20} y={ch.y0} width={140} height={ch.y1 - ch.y0} rx={8} fill={alpha(ch.color, 0.16)} stroke={ch.color} strokeWidth={2} />
                <Arrow x1={164} y1={(ch.y0 + ch.y1) / 2} x2={214} y2={52 + i * 86} color={ch.color} t={a} />
              </g>
            );
          })}
          {[0, 1].map((i) => {
            const a = seg(t, 0.4 + i * 0.22, 0.3);
            const y0 = chunks[i + 1].y0;
            const y1 = chunks[i].y1;
            return (
              <g key={`ov${i}`} opacity={a}>
                <rect x={20} y={y0} width={140} height={y1 - y0} fill="url(#lab12-hatch)" />
              </g>
            );
          })}
          <defs>
            <pattern id="lab12-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="3" height="7" fill="var(--text-3)" opacity="0.35" />
            </pattern>
          </defs>
          {chunks.map((ch, i) => {
            const a = seg(t, 0.2 + i * 0.22, 0.3);
            return (
              <g key={`card${i}`}>
                <Box x={218} y={26 + i * 86} w={188} h={58} color={ch.color} t={a} label={`청크 ${i}`} sub={`deploy.md · idx ${i}`} />
              </g>
            );
          })}
          <T x={218} y={292} size={FS.m} weight={700} color="var(--text-2)" opacity={seg(t, 0.75, 0.25)}>
            ▨ 빗금 = 겹쳐 붙인 150자
          </T>
        </g>
      );
    }
  },
  {
    title: '임베딩 — 조각을 "의미 좌표"로',
    text: rich('임베딩 모델이 각 조각을 숫자 벡터(여기선 2차원으로 단순화)로 바꿔요. 비슷한 내용끼리 가까이 모이죠. `normalize_embeddings=True`로 **길이를 모두 1**로 맞추면 점들이 원 위에 올라가고, 이제 **내적이 곧 코사인 유사도**가 돼요.'),
    code: dd(String.raw`        vecs = self.embedder.encode([c["text"] for c in self.chunks],
                                    normalize_embeddings=True,   # 정규화 → 내적 = 코사인
                                    batch_size=64, show_progress_bar=True)
        self.index = faiss.IndexFlatIP(vecs.shape[1])            # 소규모는 전수(Flat)로 충분
        self.index.add(np.asarray(vecs, dtype=np.float32))`),
    highlight: [2, 4, 5],
    render: (t, c) => (
      <g>
        {TOPICS.map((tp, i) => (
          <g key={i}>
            <Box x={6} y={40 + i * 76} w={94} h={50} color={topicColor(c, i)} t={seg(t, i * 0.08, 0.3)} label={tp.name} sub={tp.file} labelSize={FS.s} />
            <Arrow x1={102} y1={65 + i * 76} x2={116} y2={150} color={topicColor(c, i)} t={seg(t, 0.15 + i * 0.05, 0.25)} />
          </g>
        ))}
        <Box x={118} y={118} w={74} h={64} color={c.series[1]} t={seg(t, 0.2, 0.3)} label="임베딩" sub="bge-m3" />
        <Arrow x1={194} y1={150} x2={214} y2={150} color={c.series[1]} t={seg(t, 0.35, 0.2)} />
        <MeaningMap c={c} t={seg(t, 0.3, 0.7)} normalize={seg(t, 0.65, 0.35)} showQuery={0} />
        <T x={CX} y={26} anchor="middle" size={FS.s} weight={800} color="var(--text-2)">
          {seg(t, 0.65, 0.35) < 1 ? '길이가 제각각인 벡터…' : '정규화: 모두 반지름 1인 원 위로!'}
        </T>
        <g opacity={seg(t, 0.85, 0.15)}>
          <T x={412} y={78} anchor="end" size={FS.m} weight={800} color={topicColor(c, 0)}>
            {TOPICS[0].name}
          </T>
          <T x={onCircle(88, R + 24)[0]} y={onCircle(88, R + 24)[1]} anchor="middle" size={FS.m} weight={800} color={topicColor(c, 1)}>
            {TOPICS[1].name}
          </T>
          <T x={onCircle(130, R + 38)[0]} y={onCircle(130, R + 38)[1]} anchor="middle" size={FS.m} weight={800} color={topicColor(c, 2)}>
            {TOPICS[2].name}
          </T>
        </g>
      </g>
    )
  },
  {
    title: '검색 — 질문과 가장 가까운 조각 k개',
    text: rich('질문도 같은 모델로 벡터(빨간 화살표)를 만들고, 모든 조각과 내적을 계산해 **점수가 높은 5개**를 뽑아요. Flat 인덱스는 하나하나 전부 비교해요. 눈여겨볼 점: 4·5위처럼 **별로 관련 없는 조각도 k개를 채우려고 딸려 와요**.'),
    code: dd(String.raw`    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        qv = self.embedder.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(qv, dtype=np.float32), k)
        return [{**self.chunks[i], "score": float(s)}
                for s, i in zip(scores[0], ids[0]) if i != -1]`),
    highlight: [2, 3],
    render: (t, c) => {
      const q = seg(t, 0.05, 0.3);
      return (
        <g>
          <Pill x={92} y={34} label="질문: 배포 절차는?" color={c.neg} t={q} />
          <MeaningMap c={c} t={1} normalize={1} showQuery={q} />
          {TOP.map((d, i) => {
            const a = seg(t, 0.35 + i * 0.11, 0.2);
            const [x, y] = onCircle(d.angle);
            return (
              <g key={i} opacity={a}>
                <line x1={CX} y1={CY} x2={x} y2={y} stroke={topicColor(c, d.topic)} strokeWidth={1.5} strokeDasharray="3 3" />
                <circle cx={x} cy={y} r={11} fill="none" stroke={topicColor(c, d.topic)} strokeWidth={2.5} />
                <rect x={14} y={62 + i * 42} width={168} height={34} rx={10} fill={alpha(topicColor(c, d.topic), i < 3 ? 0.18 : 0.08)} stroke={topicColor(c, d.topic)} strokeWidth={1.5} strokeDasharray={i < 3 ? undefined : '4 4'} />
                <T x={24} y={84 + i * 42} size={FS.s} weight={850}>
                  {i + 1}위
                </T>
                <T x={62} y={84 + i * 42} size={FS.m} weight={650} mono>
                  {d.file.replace('.md', '')}#{d.idx}
                </T>
                <T x={174} y={84 + i * 42} anchor="end" size={FS.m} weight={800} mono color={i < 3 ? 'var(--text)' : 'var(--text-3)'}>
                  {d.score.toFixed(3)}
                </T>
              </g>
            );
          })}
        </g>
      );
    }
  },
  {
    title: '프롬프트 조립 — 근거를 번호 붙여 넣기',
    text: rich('뽑힌 조각마다 **[1], [2]… 번호와 출처·유사도**를 붙여 하나의 컨텍스트로 이어 붙여요. 질문은 맨 마지막에. 번호가 있어야 모델이 답 끝에 "어디서 봤는지" 표시할 수 있어요.'),
    code: dd(String.raw`def answer(store: VectorStore, question: str) -> str:
    hits = store.search(question)
    context = "\n\n".join(
        f"[{i+1}] (출처: {h['source']}, 유사도 {h['score']:.3f})\n{h['text']}"
        for i, h in enumerate(hits))`),
    highlight: [3, 4, 5],
    render: (t, c) => (
      <g>
        {TOP.map((d, i) => {
          const fly = seg(t, 0.1 + i * 0.12, 0.3);
          const x0 = 16;
          const y0 = 40 + i * 46;
          const x1 = 214;
          const y1 = 88 + i * 30;
          const x = x0 + (x1 - x0) * fly;
          const y = y0 + (y1 - y0) * fly;
          return (
            <g key={i}>
              <rect x={x} y={y} width={176} height={26} rx={8} fill={alpha(topicColor(c, d.topic), 0.2)} stroke={topicColor(c, d.topic)} strokeWidth={1.5} />
              <T x={x + 10} y={y + 18} size={FS.m} weight={800} mono>
                [{i + 1}] {d.file} · {d.score.toFixed(3)}
              </T>
            </g>
          );
        })}
        <rect x={204} y={22} width={200} height={268} rx={14} fill="none" stroke={c.series[1]} strokeWidth={2} strokeDasharray="6 5" />
        <T x={214} y={44} size={FS.s} weight={850} color={c.series[1]}>
          🧾 완성된 프롬프트
        </T>
        <T x={214} y={70} size={FS.m} weight={750} color="var(--text-2)">
          문서 발췌:
        </T>
        <g opacity={seg(t, 0.75, 0.2)}>
          <rect x={214} y={244} width={180} height={34} rx={10} fill={alpha(c.neg, 0.16)} stroke={c.neg} strokeWidth={1.5} />
          <T x={224} y={266} size={FS.s} weight={850}>
            질문: 배포 절차는?
          </T>
        </g>
      </g>
    )
  },
  {
    title: '생성 — 발췌만 근거로, 출처까지',
    text: rich('시스템 프롬프트로 규칙을 줘요: **발췌에 없으면 지어내지 말 것, 쓴 발췌 번호를 [1][3]처럼 적을 것**. 그래서 답이 틀렸는지 출처로 확인할 수 있고, 문서에 없는 질문엔 "찾을 수 없다"고 솔직하게 답해요.'),
    code: dd(String.raw`SYSTEM = """당신은 제공된 문서 발췌만을 근거로 답하는 어시스턴트입니다.
- 발췌에 없는 내용은 지어내지 말고 "문서에서 찾을 수 없다"고 답합니다.
- 답변 끝에 사용한 발췌 번호를 [1][3] 형식으로 표기합니다."""`),
    highlight: [2, 3],
    render: (t, c) => {
      const a1 = seg(t, 0.05, 0.25);
      const a2 = seg(t, 0.3, 0.25);
      const a3 = seg(t, 0.55, 0.3);
      return (
        <g>
          <Box x={12} y={110} w={96} h={70} color={c.series[1]} t={a1} label="프롬프트" sub="발췌 + 질문" />
          <Arrow x1={110} y1={145} x2={146} y2={145} color={c.series[1]} t={a2} />
          <Box x={148} y={105} w={90} h={80} color={c.series[0]} t={a2} label="🤖 LLM" sub="claude" />
          <Arrow x1={240} y1={145} x2={262} y2={145} color={c.series[0]} t={a3} />
          {!(t < 0.55) && <Pulse path="M110 145 L146 145 M240 145 L262 145" color={c.series[0]} dur={1.4} />}
          <Card
            x={264}
            y={56}
            w={146}
            title="✅ 답변"
            lines={['1) 테스트 통과 확인', '2) 스테이징 배포', '3) 승인 후 운영 반영', '[1][2]']}
            color={c.pos}
            t={a3}
          />
          <g opacity={seg(t, 0.8, 0.2)}>
            <Pill x={300} y={226} label="[1] deploy#1" color={c.series[0]} mono size={FS.m} />
            <Pill x={300} y={260} label="[2] deploy#2" color={c.series[0]} mono size={FS.m} />
            <T x={14} y={250} size={FS.m} weight={750} color="var(--text-2)">
              문서에 없는 질문이라면?
            </T>
            <T x={14} y={272} size={FS.s} weight={850} color={c.neg}>
              → "문서에서 찾을 수 없다"
            </T>
          </g>
        </g>
      );
    }
  },
  {
    title: '검색 평가 — Recall@k와 MRR',
    text: rich('RAG가 틀리는 이유는 대부분 **검색 실패**예요. (질문, 정답 조각) 쌍을 모아 정답 조각이 몇 위에 나왔는지 봐요. **Recall@5** = 5위 안에 들어온 비율, **MRR** = 1/순위의 평균(1위일수록 높음). 정답이 아예 없으면 어떤 LLM도 못 맞혀요.'),
    code: dd(String.raw`    hit, rr = 0, 0.0
    for ex in eval_set:
        results = store.search(ex["question"], k)
        for rank, r in enumerate(results, start=1):
            if r["source"] == ex["relevant_source"] and r["idx"] == ex["relevant_idx"]:
                hit += 1
                rr += 1.0 / rank
                break`),
    highlight: [6, 7],
    render: (t, c) => {
      const rows = [
        { q: '배포 절차는?', rank: 1 },
        { q: '휴가 신청 방법?', rank: 3 },
        { q: '비밀번호 규칙?', rank: 0 },
        { q: '롤백 방법?', rank: 2 },
        { q: '연차 이월?', rank: 1 }
      ];
      const hits = rows.filter((r) => r.rank > 0).length;
      const mrr = rows.reduce((s, r) => s + (r.rank ? 1 / r.rank : 0), 0) / rows.length;
      return (
        <g>
          <T x={16} y={30} size={FS.m} weight={800} color="var(--text-3)">
            평가 질문
          </T>
          <T x={196} y={30} size={FS.m} weight={800} color="var(--text-3)" anchor="middle">
            정답 순위
          </T>
          <T x={272} y={30} size={FS.m} weight={800} color="var(--text-3)" anchor="middle">
            1/순위
          </T>
          {rows.map((r, i) => {
            const a = seg(t, 0.05 + i * 0.1, 0.2);
            const y = 44 + i * 38;
            const ok = r.rank > 0;
            return (
              <g key={i} opacity={a}>
                <rect x={10} y={y} width={300} height={30} rx={9} fill={ok ? alpha(c.pos, 0.12) : alpha(c.neg, 0.12)} stroke={ok ? c.pos : c.neg} strokeWidth={1.3} />
                <T x={20} y={y + 21} size={FS.s} weight={700}>
                  {r.q}
                </T>
                <T x={196} y={y + 21} size={FS.s} weight={850} anchor="middle" color={ok ? 'var(--text)' : c.neg}>
                  {ok ? `${r.rank}위` : '못 찾음 ✗'}
                </T>
                <T x={272} y={y + 21} size={FS.s} weight={800} anchor="middle" mono>
                  {ok ? (1 / r.rank).toFixed(2) : '0'}
                </T>
              </g>
            );
          })}
          <g opacity={seg(t, 0.62, 0.25)}>
            <Box x={322} y={44} w={88} h={84} color={c.series[0]} label={`${(hits / rows.length).toFixed(2)}`} sub="Recall@5" labelSize={22} />
            <Box x={322} y={140} w={88} h={84} color={c.series[2]} label={mrr.toFixed(2)} sub="MRR" labelSize={22} />
          </g>
          <T x={10} y={252 + 18} size={FS.m} weight={750} color="var(--text-2)" opacity={seg(t, 0.85, 0.15)}>
            못 찾은 20%는 프롬프트가 아니라 청킹·임베딩을 고쳐야 해요.
          </T>
        </g>
      );
    }
  }
];

export default function Lab12Rag(_: VizProps) {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={2} />;
}
