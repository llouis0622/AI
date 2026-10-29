import { CodeStory, type StoryStep } from '../core/story';
import { Arrow, Box, Chip, Curve, Defs, Emoji, Txt, VB, mix, seg, soft } from './_labsA';

/* Illustrative WordPiece split (the real klue/bert-base pieces may differ). */
const TOKENS = ['[CLS]', '연기', '##가', '미쳤', '##다', '최고', '##의', '영화', '[SEP]'];
const TOK_W = TOKENS.map((tk) => (tk.startsWith('[') ? 50 : 42));
const TOK_X = TOK_W.map((w, i) => 11 + TOK_W.slice(0, i).reduce((a, b) => a + b + 8, 0) + w / 2);
const BATCH = [9, 6, 12];
const ACC = [0.5, 0.83, 0.86, 0.875, 0.885, 0.889, 0.893, 0.897, 0.899, 0.901, 0.899, 0.902];

function lrSchedule(s: number) {
  const peak = 2e-5;
  if (s < 0.1) return (peak * s) / 0.1;
  return peak * Math.max(0, (1 - s) / 0.9);
}

const steps: StoryStep[] = [
  {
    title: '데이터: 영화 리뷰 20만 개와 긍정/부정',
    text: (
      <>
        NSMC는 네이버 영화 리뷰에 <strong>긍정(1) / 부정(0)</strong> 라벨이 붙은 데이터예요(학습 15만, 테스트 5만). <code>datasets</code>로 한 줄에 내려받고, 내용이 비어 있는 리뷰는 <code>filter</code>로 걸러내요.
      </>
    ),
    code: `ds = load_dataset("e9t/nsmc", trust_remote_code=True)     # train 15만 / test 5만
ds = ds.filter(lambda x: x["document"] is not None and len(x["document"]) > 0)`,
    highlight: [1, 2],
    render: (t, c) => {
      const cards = [
        { text: '연기가 미쳤다 최고의 영화', label: '긍정 1', emoji: '👍', col: c.pos },
        { text: '시간이 아깝다 정말', label: '부정 0', emoji: '👎', col: c.neg },
        { text: '스토리는 약한데 배우가 다 살렸다', label: '긍정 1', emoji: '👍', col: c.pos },
        { text: '(빈 리뷰)', label: '제거', emoji: '🗑️', col: '#9aa0b4' }
      ];
      return (
        <g>
          <Defs />
          {cards.map((cd, i) => {
            const p = seg(t, 0.05 + i * 0.15, 0.3 + i * 0.15);
            const removed = i === 3 && seg(t, 0.75, 0.95) > 0;
            return (
              <g key={i} opacity={p * (removed ? 1 - seg(t, 0.75, 0.95) * 0.7 : 1)} transform={`translate(${(1 - p) * -20} 0)`}>
                <rect x={30} y={34 + i * 56} width={300} height={44} rx={12} fill="var(--surface-solid)" stroke="var(--border-strong)" />
                <Txt x={44} y={61 + i * 56} size={13.5} weight={700} anchor="start">
                  {cd.text}
                </Txt>
                <rect x={346} y={40 + i * 56} width={110} height={32} rx={16} fill={soft(cd.col, 0.16)} stroke={cd.col} />
                <Txt x={401} y={61 + i * 56} size={13} weight={850}>
                  {cd.emoji} {cd.label}
                </Txt>
                {removed && <line x1={36} x2={324} y1={56 + i * 56} y2={56 + i * 56} stroke={c.neg} strokeWidth={2} />}
              </g>
            );
          })}
          <Txt x={240} y={278} size={12} weight={750} color="var(--text-2)" opacity={seg(t, 0.85, 1)}>
            × 200,000 리뷰
          </Txt>
        </g>
      );
    }
  },
  {
    title: '토크나이즈 + 동적 패딩',
    text: (
      <>
        모델과 짝인 토크나이저가 문장을 <strong>서브워드 조각</strong>으로 나누고 앞뒤에 [CLS], [SEP]을 붙여요. 미리 128칸으로 다 채우지 않고, 배치를 만들 때 <strong>그 배치에서 가장 긴 문장 길이까지만</strong> 채워요(동적 패딩) — 계산 낭비가 확 줄어요.
      </>
    ),
    code: `tok = AutoTokenizer.from_pretrained(MODEL)

def tokenize(batch):
    # 여기서는 패딩하지 않는다 — 배치 시점에 collator가 배치 내 최장 길이로 패딩(동적 패딩)
    return tok(batch["document"], truncation=True, max_length=128)

data_collator=DataCollatorWithPadding(tok),   # 동적 패딩`,
    highlight: [5, 7],
    render: (t, c) => {
      const dyn = seg(t, 0.55, 0.85);
      const maxLen = Math.max(...BATCH);
      const cellW = 12;
      return (
        <g>
          <Defs />
          {TOKENS.map((tk, i) => (
            <Chip
              key={i}
              x={TOK_X[i]}
              y={50}
              text={tk}
              w={TOK_W[i]}
              h={28}
              fs={tk.startsWith('[') ? 10.5 : 12.5}
              color={tk.startsWith('[') ? c.series[3] : c.series[0]}
              t={seg(t, i * 0.04, 0.15 + i * 0.04)}
              c={c}
            />
          ))}
          <Txt x={240} y={88} size={11.5} weight={700} color="var(--text-2)" opacity={seg(t, 0.3, 0.45)}>
            "##"은 앞 조각에 붙는다는 표시 (분할은 예시)
          </Txt>
          <Txt x={30} y={122} size={12.5} weight={850} anchor="start">
            배치 3문장
          </Txt>
          <Txt x={450} y={122} size={12.5} weight={850} anchor="end" color={dyn > 0.5 ? c.pos : c.neg}>
            {dyn > 0.5 ? `동적: 최장 ${maxLen}칸까지만` : '고정: 128칸까지 패딩'}
          </Txt>
          {BATCH.map((len, r) => {
            const padTo = mix(32, maxLen, dyn);
            return (
              <g key={r}>
                {Array.from({ length: Math.round(padTo) }, (_, k) => (
                  <rect key={k} x={30 + k * cellW} y={136 + r * 30} width={cellW - 2} height={22} rx={3} fill={k < len ? soft(c.series[0], 0.65) : 'var(--surface-3)'} stroke={k < len ? 'none' : 'var(--border)'} opacity={30 + k * cellW > 450 ? 0 : 1} />
                ))}
              </g>
            );
          })}
          <g opacity={seg(t, 0.85, 1)}>
            <Txt x={240} y={248} size={12.5} weight={800}>
              회색(패딩) 칸 = 계산 낭비 → 동적 패딩으로 대부분 사라져요
            </Txt>
            <Txt x={240} y={270} size={11.5} weight={700} color="var(--text-2)">
              비슷한 길이끼리 묶으면(group_by_length) 더 빨라져요
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '모델: 사전학습된 BERT + 새 분류 헤드',
    text: (
      <>
        <code>AutoModelForSequenceClassification</code>은 사전학습된 BERT 본체 위에 <strong>[CLS] 벡터를 받는 작은 분류 층</strong>을 얹어요. 이 층은 새로 만들어져서 "새로 초기화된다"는 경고가 뜨는 게 정상이에요. 출력은 부정/긍정 점수 2개예요.
      </>
    ),
    code: `model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=2, id2label={0: "부정", 1: "긍정"})`,
    highlight: [1, 2],
    render: (t, c) => {
      const layers = 6;
      return (
        <g>
          <Defs />
          {Array.from({ length: layers }, (_, i) => (
            <g key={i} opacity={seg(t, i * 0.06, 0.2 + i * 0.06)}>
              <rect x={40} y={226 - i * 30} width={190} height={24} rx={7} fill={soft(c.series[1], 0.22)} stroke={c.series[1]} strokeWidth={1.5} />
              <Txt x={135} y={243 - i * 30} size={11.5} weight={750}>
                {i === layers - 1 ? 'Transformer 층 12 (…)' : `Transformer 층 ${i + 1}`}
              </Txt>
            </g>
          ))}
          <Txt x={135} y={52} size={12.5} weight={850} color={c.series[1]} opacity={seg(t, 0.3, 0.45)}>
            BERT 본체 (사전학습 ✓)
          </Txt>
          <g opacity={seg(t, 0.4, 0.55)}>
            <rect x={250} y={60} width={40} height={40} rx={8} fill={c.series[3]} />
            <Txt x={270} y={84} size={10} weight={900} color="#fff">
              CLS
            </Txt>
            <Txt x={270} y={118} size={11} weight={750} color="var(--text-2)">
              768차원
            </Txt>
          </g>
          <Arrow x1={232} y1={80} x2={246} y2={80} color={c.series[3]} t={seg(t, 0.4, 0.5)} />
          <Arrow x1={294} y1={80} x2={320} y2={80} color={c.series[3]} t={seg(t, 0.5, 0.6)} />
          <Box x={324} y={56} w={70} h={48} label="Linear" sub="768→2" subFs={10.5} fs={13} color={c.series[2]} glow={t > 0.7 && t < 0.999} t={seg(t, 0.55, 0.7)} c={c} />
          <Txt x={359} y={124} size={11.5} weight={850} color={c.series[2]} opacity={seg(t, 0.65, 0.8)}>
            ✨ 새로 초기화
          </Txt>
          <Arrow x1={396} y1={80} x2={414} y2={80} color={c.series[2]} t={seg(t, 0.7, 0.8)} />
          <g opacity={seg(t, 0.78, 0.95)}>
            <rect x={418} y={54} width={52} height={22} rx={6} fill={soft(c.neg, 0.18)} />
            <Txt x={444} y={70} size={11.5} weight={850}>
              부정
            </Txt>
            <rect x={418} y={84} width={52} height={22} rx={6} fill={soft(c.pos, 0.3)} stroke={c.pos} />
            <Txt x={444} y={100} size={11.5} weight={850}>
              긍정
            </Txt>
          </g>
          <g opacity={seg(t, 0.85, 1)}>
            <rect x={262} y={160} width={200} height={70} rx={14} fill={soft(c.series[4], 0.1)} stroke={soft(c.series[4], 0.6)} />
            <Emoji x={292} y={203} size={22}>
              ⚠️
            </Emoji>
            <Txt x={380} y={190} size={11.5} weight={800}>
              "Some weights … newly
            </Txt>
            <Txt x={380} y={208} size={11.5} weight={800}>
              initialized" → 정상이에요
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '작은 학습률(2e-5) + 웜업',
    text: (
      <>
        사전학습 지식을 지키면서 살짝만 적응시키려면 학습률을 <strong>1e-5 ~ 5e-5</strong>로 작게 잡아요. 앞 10%는 0에서 천천히 올리는 웜업이에요. 1e-3처럼 크게 잡으면 배운 걸 다 잊어버리는 <strong>파국적 망각</strong>으로 정확도가 찍기 수준(50%)까지 무너질 수 있어요.
      </>
    ),
    code: `num_train_epochs=2,
per_device_train_batch_size=64,
per_device_eval_batch_size=256,
learning_rate=2e-5,                # BERT 파인튜닝의 표준 범위: 1e-5 ~ 5e-5
warmup_ratio=0.1,
weight_decay=0.01,
bf16=True,                         # Ampere+ GPU. 아니면 fp16=True 또는 둘 다 False`,
    highlight: [4, 5],
    render: (t, c) => (
      <g>
        <Defs />
        <Curve x={50} y={40} w={220} h={160} fn={lrSchedule} xDomain={[0, 1]} yDomain={[0, 2.2e-5]} t={seg(t, 0.05, 0.6)} color={c.series[0]} xLabel="학습 진행" yLabel="학습률" n={100} marker={0.1} markerLabel="2e-5" />
        <rect x={50} y={40} width={22} height={160} fill={soft(c.series[4], 0.14)} opacity={seg(t, 0.2, 0.4)} />
        <Txt x={61} y={216} size={11} weight={850} color={c.series[4]} opacity={seg(t, 0.2, 0.4)}>
          웜업
        </Txt>
        <g opacity={seg(t, 0.55, 0.85)}>
          <rect x={300} y={44} width={168} height={160} rx={16} fill={soft(c.neg, 0.08)} stroke={soft(c.neg, 0.5)} />
          <Txt x={384} y={70} size={12.5} weight={850} color={c.neg}>
            lr = 1e-3 이라면?
          </Txt>
          <Curve x={320} y={86} w={128} h={86} fn={(s) => (s < 0.15 ? 0.6 + s * 1.5 : Math.max(0.5, 0.82 - (s - 0.15) * 1.6))} xDomain={[0, 1]} yDomain={[0.4, 1]} t={seg(t, 0.6, 0.95)} color={c.neg} n={60} />
          <Txt x={384} y={192} size={11.5} weight={800} color="var(--text-2)">
            정확도 → 50% (망각)
          </Txt>
        </g>
        <Txt x={240} y={262} size={12.5} weight={800} opacity={seg(t, 0.85, 1)}>
          백본은 작게 — 03 전이학습의 원칙이 텍스트에서도 그대로!
        </Txt>
      </g>
    )
  },
  {
    title: 'Trainer: 평가하고, 최고 체크포인트를 기억한다',
    text: (
      <>
        <code>Trainer</code>가 학습 루프, 혼합 정밀도, 평가, 저장을 대신 돌려 줘요. 500 스텝마다 테스트 정확도를 재고, <code>load_best_model_at_end</code> 덕분에 끝나면 <strong>가장 좋았던 체크포인트</strong>로 자동 복원돼요. 2 에포크면 약 90%!
      </>
    ),
    code: `trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=DataCollatorWithPadding(tok),   # 동적 패딩
    compute_metrics=compute_metrics,
)

trainer.train()`,
    highlight: [1, 10],
    render: (t, c) => {
      const best = ACC.indexOf(Math.max(...ACC));
      return (
        <g>
          <Defs />
          <Curve x={52} y={40} w={300} h={170} fn={(k) => ACC[Math.min(ACC.length - 1, Math.max(0, Math.round(k)))]} xDomain={[0, ACC.length - 1]} yDomain={[0.45, 0.95]} t={seg(t, 0.05, 0.8)} color={c.series[3]} xLabel="평가 시점 (500 스텝마다)" yLabel="정확도" n={ACC.length - 1} />
          {t > 0.8 && (
            <g>
              <circle cx={52 + (best / (ACC.length - 1)) * 300} cy={40 + 170 - ((ACC[best] - 0.45) / 0.5) * 170} r={8} fill={c.pos} stroke="var(--surface-solid)" strokeWidth={2.5} />
              <Emoji x={52 + (best / (ACC.length - 1)) * 300} y={40 + 170 - ((ACC[best] - 0.45) / 0.5) * 170 - 16} size={16}>
                💾
              </Emoji>
            </g>
          )}
          <g opacity={seg(t, 0.8, 1)}>
            <rect x={366} y={70} width={104} height={110} rx={14} fill={soft(c.pos, 0.12)} stroke={c.pos} />
            <Txt x={418} y={96} size={12} weight={850}>
              최고 기록
            </Txt>
            <Txt x={418} y={130} size={24} weight={900} color={c.pos}>
              {Math.round(ACC[best] * 1000) / 10}%
            </Txt>
            <Txt x={418} y={160} size={11} weight={750} color="var(--text-2)">
              끝나면 자동 복원
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '추론: pipeline 한 줄이면 끝',
    text: (
      <>
        저장한 모델 폴더를 <code>pipeline</code>에 넣으면 문장 리스트를 바로 분류해요. 토크나이저도 함께 저장했기 때문에 가능한 일이에요 — <strong>모델 + 토크나이저</strong>가 완결된 한 세트예요. 서비스에서는 배치를 직접 제어하는 방식도 써요.
      </>
    ),
    code: `clf = pipeline("text-classification", model="./bert-nsmc-final", device=0)
print(clf(["연기가 미쳤다 최고의 영화", "시간이 아깝다 정말"]))
# [{'label': '긍정', 'score': 0.99}, {'label': '부정', 'score': 0.99}]`,
    highlight: [1, 2],
    render: (t, c) => {
      const rows = [
        { text: '연기가 미쳤다 최고의 영화', label: '긍정', score: 0.99, col: c.pos, emoji: '😍' },
        { text: '시간이 아깝다 정말', label: '부정', score: 0.99, col: c.neg, emoji: '😩' }
      ];
      return (
        <g>
          <Defs />
          <Box x={190} y={110} w={100} h={56} label="pipeline" sub="BERT 분류기" subFs={10.5} fs={14} color={c.series[1]} t={seg(t, 0.05, 0.25)} c={c} />
          {rows.map((r, i) => (
            <g key={i}>
              <g opacity={seg(t, 0.1 + i * 0.1, 0.3 + i * 0.1)}>
                <rect x={10} y={70 + i * 80} width={164} height={50} rx={12} fill="var(--surface-solid)" stroke="var(--border-strong)" />
                <Txt x={92} y={100 + i * 80} size={12} weight={750}>
                  {r.text}
                </Txt>
              </g>
              <Arrow x1={176} y1={95 + i * 80} x2={188} y2={128 + i * 20} color="var(--viz-axis)" width={2} t={seg(t, 0.3, 0.4)} />
              <Arrow x1={292} y1={128 + i * 20} x2={306} y2={95 + i * 80} color={r.col} width={2.5} t={seg(t, 0.5 + i * 0.1, 0.6 + i * 0.1)} />
              <g opacity={seg(t, 0.6 + i * 0.12, 0.8 + i * 0.12)}>
                <rect x={310} y={70 + i * 80} width={160} height={50} rx={14} fill={soft(r.col, 0.14)} stroke={r.col} />
                <Txt x={390} y={92 + i * 80} size={14} weight={900}>
                  {r.emoji} {r.label}
                </Txt>
                <Txt x={390} y={110 + i * 80} size={11.5} weight={750} color="var(--text-2)" mono>
                  score {r.score}
                </Txt>
              </g>
            </g>
          ))}
          <Txt x={240} y={262} size={12} weight={750} color="var(--text-2)" opacity={seg(t, 0.9, 1)}>
            저장할 때 토크나이저도 함께 → 어디서든 그대로 불러 쓸 수 있어요
          </Txt>
        </g>
      );
    }
  }
];

/** Code lab 10 walkthrough: Hugging Face fine-tuning workflow for a Korean sentiment classifier. */
export default function Lab10BertFinetune() {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={1.9} />;
}
