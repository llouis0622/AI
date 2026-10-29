import type { VizProps } from '../Viz';
import { CodeStory, type StoryStep } from '../core/story';
import type { VizColors } from '../core/colors';
import { VB, FS, seg, alpha, rich, dd, T, Box, Arrow, Pill, Cells } from './_LabKitB';
import { softmax } from '../core/math';

/* A toy shared embedding space: each concept has an image point (square) and a text point (circle). */
const CONCEPTS = [
  { icon: '🐶', name: '골든리트리버', text: 'dog', img: [262, 82], txt: [282, 96] },
  { icon: '🐱', name: '고양이', text: 'cat', img: [234, 132], txt: [252, 146] },
  { icon: '🚲', name: '자전거', text: 'bike', img: [376, 98], txt: [392, 118] },
  { icon: '🍜', name: '라면 한 그릇', text: 'ramen', img: [330, 214], txt: [352, 228] },
  { icon: '🌃', name: '야경 사진', text: 'night', img: [240, 230], txt: [222, 250] }
];
const SIMS = [0.31, 0.24, 0.19, 0.17, 0.16];
const PROBS = softmax(SIMS.map((s) => s * 100));

function MapFrame({ c }: { c: VizColors }) {
  return (
    <g>
      <rect x={200} y={44} width={212} height={232} rx={14} fill={alpha(c.series[0], 0.05)} stroke="var(--border-strong)" strokeDasharray="5 5" />
      <T x={306} y={36} anchor="middle" size={FS.m} weight={800} color="var(--text-2)">
        같은 임베딩 공간
      </T>
    </g>
  );
}

function ConceptDots({ c, t, upto = CONCEPTS.length, start = 0 }: { c: VizColors; t: number; upto?: number; start?: number }) {
  return (
    <g>
      {CONCEPTS.slice(0, upto).map((k, i) => {
        const a = i < start ? 1 : seg(t, 0.3 + (i - start) * 0.1, 0.2);
        return (
          <g key={i} opacity={a}>
            <line x1={k.img[0]} y1={k.img[1]} x2={k.txt[0]} y2={k.txt[1]} stroke={c.series[i % 6]} strokeWidth={1.5} strokeDasharray="3 3" />
            <rect x={k.img[0] - 7} y={k.img[1] - 7} width={14} height={14} rx={3} fill={c.series[i % 6]} />
            <circle cx={k.txt[0]} cy={k.txt[1]} r={7} fill="var(--surface-solid)" stroke={c.series[i % 6]} strokeWidth={3} />
            <T x={k.img[0] - 12} y={k.img[1] + 5} anchor="end" size={13}>
              {k.icon}
            </T>
          </g>
        );
      })}
    </g>
  );
}

const steps: StoryStep[] = [
  {
    title: '두 인코더, 하나의 공간',
    text: rich('CLIP은 **이미지 인코더**와 **텍스트 인코더**를 4억 쌍의 (사진, 설명)으로 함께 학습했어요. 그 결과 강아지 사진의 벡터(■)와 "a photo of a dog"의 벡터(●)가 **같은 공간에서 가까이** 놓여요.'),
    code: dd(String.raw`model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()`),
    highlight: [1, 2, 3],
    render: (t, c) => (
      <g>
        <MapFrame c={c} />
        <g opacity={seg(t, 0, 0.2)}>
          <rect x={14} y={56} width={52} height={52} rx={10} fill="var(--surface-solid)" stroke="var(--border-strong)" />
          <T x={40} y={92} anchor="middle" size={30}>
            🐶
          </T>
        </g>
        <Arrow x1={68} y1={82} x2={88} y2={82} color={c.series[0]} t={seg(t, 0.1, 0.2)} />
        <Box x={90} y={58} w={92} h={50} color={c.series[0]} t={seg(t, 0.12, 0.25)} label="이미지 인코더" sub="ViT-B-32" labelSize={FS.m} />
        <Arrow x1={184} y1={82} x2={252} y2={82} color={c.series[0]} t={seg(t, 0.25, 0.2)} />
        <Pill x={60} y={196} label="a photo of a dog" color={c.series[1]} t={seg(t, 0.05, 0.2)} size={11.5} mono />
        <Arrow x1={60} y1={212} x2={88} y2={236} color={c.series[1]} t={seg(t, 0.15, 0.2)} />
        <Box x={90} y={212} w={92} h={50} color={c.series[1]} t={seg(t, 0.18, 0.25)} label="텍스트 인코더" sub="Transformer" labelSize={FS.m} />
        <Arrow x1={184} y1={230} x2={276} y2={102} color={c.series[1]} t={seg(t, 0.3, 0.2)} />
        <ConceptDots c={c} t={t} />
      </g>
    )
  },
  {
    title: '제로샷 분류 ① — 클래스를 문장으로',
    text: rich('분류 레이블을 따로 학습하지 않아요. 내가 정한 클래스 이름을 **"a photo of {c}"** 문장으로 바꿔 텍스트 벡터를 만들 뿐이에요. 그러면 학습 데이터에 없던 조합(라면 한 그릇!)도 클래스가 될 수 있어요.'),
    code: dd(String.raw`# 내가 정의한 임의의 클래스 — 학습 데이터에 이 조합이 있었을 필요가 없다
classes = ["골든리트리버", "고양이", "자전거", "라면 한 그릇", "야경 사진"]
# 프롬프트 템플릿: 단어만 넣는 것보다 문장이 텍스트 인코더 분포에 가깝다
prompts = [f"a photo of {c}" for c in classes]`),
    highlight: [2, 4],
    render: (t, c) => (
      <g>
        <MapFrame c={c} />
        {CONCEPTS.map((k, i) => (
          <g key={i} opacity={seg(t, 0.05 + i * 0.1, 0.2)}>
            <rect x={14} y={52 + i * 44} width={170} height={32} rx={9} fill={alpha(c.series[i % 6], 0.14)} stroke={c.series[i % 6]} strokeWidth={1.5} />
            <T x={24} y={73 + i * 44} size={FS.m} weight={800}>
              a photo of {k.name}
            </T>
            <circle cx={k.txt[0]} cy={k.txt[1]} r={7} fill="var(--surface-solid)" stroke={c.series[i % 6]} strokeWidth={3} />
          </g>
        ))}
        <g opacity={seg(t, 0.7, 0.2)}>
          <rect x={CONCEPTS[0].img[0] - 8} y={CONCEPTS[0].img[1] - 8} width={16} height={16} rx={3} fill={c.neg} />
          <T x={CONCEPTS[0].img[0] + 12} y={CONCEPTS[0].img[1] - 14} size={FS.m} weight={850} color={c.neg}>
            test.jpg 🐕
          </T>
        </g>
      </g>
    )
  },
  {
    title: '제로샷 분류 ② — 코사인 × 100 → softmax',
    text: rich('벡터를 정규화한 뒤 내적하면 **코사인 유사도**예요. 값 차이는 0.07 정도로 작지만 **100을 곱해 softmax** 하면 확신이 뚜렷해져요(100은 학습된 온도의 근사). 골든리트리버가 압도적 1위!'),
    code: dd(String.raw`    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)`),
    highlight: [3],
    render: (t, c) => (
      <g>
        <T x={14} y={34} size={FS.s} weight={850}>
          코사인 유사도
        </T>
        <T x={236} y={34} size={FS.s} weight={850}>
          확률 (×100 → softmax)
        </T>
        {CONCEPTS.map((k, i) => {
          const a = seg(t, 0.05 + i * 0.08, 0.2);
          const b = seg(t, 0.5 + i * 0.06, 0.2);
          const y = 52 + i * 44;
          return (
            <g key={i}>
              <g opacity={a}>
                <T x={14} y={y + 15} size={12} weight={750}>
                  {k.icon} {k.name}
                </T>
                <rect x={14} y={y + 22} width={170} height={10} rx={5} fill="var(--surface-3)" />
                <rect x={14} y={y + 22} width={170 * (SIMS[i] / 0.35)} height={10} rx={5} fill={c.series[i % 6]} />
                <T x={196} y={y + 31} size={11.5} weight={800} mono>
                  {SIMS[i].toFixed(2)}
                </T>
              </g>
              <g opacity={b}>
                <rect x={236} y={y + 22} width={126} height={10} rx={5} fill="var(--surface-3)" />
                <rect x={236} y={y + 22} width={126 * PROBS[i]} height={10} rx={5} fill={i === 0 ? c.pos : c.series[i % 6]} />
                <T x={410} y={y + 31} anchor="end" size={11.5} weight={800} mono>
                  {PROBS[i] < 0.001 ? '<0.001' : PROBS[i].toFixed(3)}
                </T>
              </g>
            </g>
          );
        })}
        <Arrow x1={200} y1={290} x2={232} y2={290} color={c.series[3]} t={seg(t, 0.45, 0.15)} label="×100" labelDy={-8} />
      </g>
    )
  },
  {
    title: '텍스트로 이미지 찾기',
    text: rich('사진 폴더를 **한 번만** 임베딩해 두면, 아무 문장으로나 검색돼요. 질문 문장의 벡터와 모든 사진 벡터의 유사도를 구해 **상위 k장**을 돌려줘요. RAG의 이미지판이에요.'),
    code: dd(String.raw`@torch.no_grad()
def search(query, paths, feats, k=5):
    t = model.encode_text(tokenizer([query]).to(device))
    t /= t.norm(dim=-1, keepdim=True)
    sims = (feats @ t.T).squeeze(1)
    for score, idx in zip(*sims.topk(k)):
        print(f"{score.item():.3f}  {paths[idx]}")`),
    highlight: [3, 5, 6],
    render: (t, c) => {
      const photos = [
        { e: '🌅', s: 0.33 },
        { e: '🏖️', s: 0.27 },
        { e: '🌃', s: 0.22 },
        { e: '🐶', s: 0.12 },
        { e: '🍜', s: 0.1 },
        { e: '🚲', s: 0.11 },
        { e: '🌄', s: 0.25 },
        { e: '🐱', s: 0.09 }
      ];
      const ranked = photos.map((p, i) => ({ ...p, i })).sort((a, b) => b.s - a.s);
      const rankOf = (i: number) => ranked.findIndex((r) => r.i === i);
      return (
        <g>
          <Pill x={210} y={34} label='🔎 "노을 지는 바다"' color={c.neg} t={seg(t, 0, 0.2)} size={FS.s} />
          {photos.map((p, i) => {
            const col = i % 4;
            const row = Math.floor(i / 4);
            const x = 30 + col * 96;
            const y = 64 + row * 104;
            const rank = rankOf(i);
            const hl = rank < 3 && seg(t, 0.35 + rank * 0.12, 0.2) > 0;
            return (
              <g key={i} opacity={seg(t, 0.05 + i * 0.04, 0.2)}>
                <rect x={x} y={y} width={76} height={76} rx={12} fill="var(--surface-solid)" stroke={hl ? c.pos : 'var(--border-strong)'} strokeWidth={hl ? 3.5 : 1.5} />
                <T x={x + 38} y={y + 50} anchor="middle" size={34}>
                  {p.e}
                </T>
                <T x={x + 38} y={y + 94} anchor="middle" size={11.5} weight={800} mono color={hl ? 'var(--text)' : 'var(--text-3)'}>
                  {p.s.toFixed(2)}
                </T>
                {hl && <Pill x={x + 70} y={y + 6} label={`${rank + 1}위`} color={c.pos} solid size={11.5} />}
              </g>
            );
          })}
        </g>
      );
    }
  },
  {
    title: '대조 손실 — 대각선이 정답',
    text: rich('배치 안의 B장 사진과 B개 설명으로 **B×B 유사도 표**를 만들어요. i번째 사진의 짝은 i번째 설명, 즉 **대각선**이 정답이에요. 행 방향(사진→설명)과 열 방향(설명→사진) 교차 엔트로피로 **대각선은 높이고 나머지는 낮춰요**.'),
    code: dd(String.raw`def clip_loss(img_feat, txt_feat, temperature=0.07):
    """배치 (B, D) 두 개. 대각선이 정답 쌍이다."""
    img = F.normalize(img_feat, dim=-1)
    txt = F.normalize(txt_feat, dim=-1)
    logits = img @ txt.T / temperature          # (B, B) 유사도 행렬
    labels = torch.arange(len(img))             # i번째 이미지의 짝은 i번째 텍스트
    # 두 방향의 교차엔트로피: 이미지→텍스트 검색과 텍스트→이미지 검색을 동시에
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2`),
    highlight: [5, 6, 8, 9],
    render: (t, c) => {
      const train = seg(t, 0.3, 0.6);
      const B = CONCEPTS.length;
      const cell = 40;
      const x0 = 120;
      const y0 = 60;
      return (
        <g>
          {CONCEPTS.map((k, i) => (
            <g key={i}>
              <T x={x0 - 12} y={y0 + i * (cell + 4) + 27} anchor="end" size={20}>
                {k.icon}
              </T>
              <T x={x0 + i * (cell + 4) + cell / 2} y={y0 - 10} anchor="middle" size={11.5} weight={800} mono>
                {k.text}
              </T>
            </g>
          ))}
          <Cells
            x={x0}
            y={y0}
            rows={B}
            cols={B}
            cell={cell}
            gap={4}
            values={(r, k) => (r === k ? 0.35 + 0.6 * train : 0.4 - 0.3 * train + 0.08 * Math.sin(r * 3 + k))}
            color={c.series[0]}
            highlight={new Set(CONCEPTS.map((_, i) => `${i},${i}`))}
            highlightColor={c.pos}
            t={seg(t, 0.05, 0.25)}
          />
          <T x={x0 + (B * (cell + 4)) / 2} y={y0 + B * (cell + 4) + 20} anchor="middle" size={FS.m} weight={800} color={c.pos} opacity={seg(t, 0.6, 0.2)}>
            학습할수록 대각선 ↑, 나머지 ↓
          </T>
          <T x={24} y={y0 - 10} size={11.5} weight={750} color="var(--text-3)">
            사진 ↓ 설명 →
          </T>
        </g>
      );
    }
  },
  {
    title: '배치가 클수록 배울 게 많아요',
    text: rich('한 사진 입장에서는 **정답 1개 대 오답 B-1개**를 고르는 문제예요. 배치가 크면 헷갈리는 오답이 많아져 학습 신호가 좋아져요. CLIP이 **수만 장짜리 배치**로 학습된 이유예요.'),
    code: dd(String.raw`    logits = img @ txt.T / temperature          # (B, B) 유사도 행렬
    labels = torch.arange(len(img))             # i번째 이미지의 짝은 i번째 텍스트`),
    highlight: [1, 2],
    render: (t, c) => {
      const grid = (B: number, x: number, cell: number, a: number) => (
        <Cells
          x={x}
          y={72}
          rows={B}
          cols={B}
          cell={cell}
          gap={2}
          values={(r, k) => (r === k ? 0.9 : 0.2)}
          color={c.series[0]}
          highlight={new Set(Array.from({ length: B }, (_, i) => `${i},${i}`))}
          highlightColor={c.pos}
          t={a}
        />
      );
      return (
        <g>
          <T x={70} y={50} anchor="middle" size={FS.s} weight={850}>
            B = 4
          </T>
          {grid(4, 22, 22, seg(t, 0.05, 0.25))}
          <T x={70} y={200} anchor="middle" size={FS.m} weight={800} color="var(--text-2)">
            정답 1 : 오답 3
          </T>
          <T x={290} y={50} anchor="middle" size={FS.s} weight={850}>
            B = 12
          </T>
          {grid(12, 186, 15, seg(t, 0.35, 0.35))}
          <T x={290} y={290} anchor="middle" size={FS.m} weight={800} color={c.pos} opacity={seg(t, 0.7, 0.2)}>
            정답 1 : 오답 11 → 더 많이 배워요
          </T>
        </g>
      );
    }
  },
  {
    title: '다음 단계 — 이미지와 대화하는 VLM',
    text: rich('CLIP은 "정렬"까지예요. 이미지에 대해 **대화**하는 VLM은 보통 **CLIP 계열 비전 인코더(동결) → 프로젝션 층 → LLM**으로 이미지 특징을 LLM의 토큰 공간에 넣어요. LLaVA가 대표적이고, 코드랩 11의 **LoRA**가 그대로 쓰여요.'),
    render: (t, c) => (
      <g>
        <g opacity={seg(t, 0, 0.2)}>
          <rect x={14} y={110} width={60} height={60} rx={12} fill="var(--surface-solid)" stroke="var(--border-strong)" />
          <T x={44} y={152} anchor="middle" size={32}>
            🐶
          </T>
        </g>
        <Arrow x1={76} y1={140} x2={92} y2={140} color={c.series[0]} t={seg(t, 0.1, 0.15)} />
        <Box x={94} y={106} w={92} h={68} color={c.series[0]} t={seg(t, 0.12, 0.2)} label="🧊 CLIP" sub="비전 (동결)" />
        <Arrow x1={188} y1={140} x2={204} y2={140} color={c.series[2]} t={seg(t, 0.3, 0.15)} />
        <Box x={206} y={112} w={70} h={56} color={c.series[2]} t={seg(t, 0.32, 0.2)} label="프로젝션" sub="학습" labelSize={FS.m} />
        <Arrow x1={278} y1={140} x2={294} y2={140} color={c.series[1]} t={seg(t, 0.45, 0.15)} />
        <Box x={296} y={100} w={110} h={80} color={c.series[1]} t={seg(t, 0.47, 0.2)} label="🤖 LLM" sub="LoRA로 조정" />
        <g opacity={seg(t, 0.65, 0.25)}>
          <rect x={60} y={214} width={300} height={44} rx={14} fill={alpha(c.pos, 0.14)} stroke={c.pos} strokeWidth={1.8} />
          <T x={210} y={242} anchor="middle" size={FS.s} weight={800}>
            "강아지가 초록 공을 물고 있어요."
          </T>
        </g>
      </g>
    )
  }
];

export default function Lab18Clip(_: VizProps) {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={2} />;
}
