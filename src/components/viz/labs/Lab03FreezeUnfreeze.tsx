import { CodeStory, type StoryStep } from '../core/story';
import type { VizColors } from '../core/colors';
import { Arrow, Bars, Box, Defs, Emoji, Txt, VB, seg, soft } from './_labsA';

const BLOCKS = [
  { id: 'stem', label: 'stem', x: 14, w: 56, feat: '선·색' },
  { id: 'layer1', label: 'layer1', x: 78, w: 60, feat: '모서리' },
  { id: 'layer2', label: 'layer2', x: 146, w: 64, feat: '질감' },
  { id: 'layer3', label: 'layer3', x: 218, w: 68, feat: '부분' },
  { id: 'layer4', label: 'layer4', x: 294, w: 74, feat: '모양' }
];
const HEAD = { x: 384, w: 82 };
const Y = 96;
const H = 62;
const TOTAL = 23.7;

/**
 * The ResNet-50 strip shared by every step.
 * ice[i] ∈ [0,1]: how frozen block i is; gradTo: index of the leftmost block that receives gradients.
 */
function Net({
  c,
  t,
  ice,
  gradTo,
  headNew = 1,
  lr,
  trainable
}: {
  c: VizColors;
  t: number;
  ice: number[];
  gradTo: number;
  headNew?: number;
  lr?: { body: string; head: string };
  trainable: number;
}) {
  const gradT = seg(t, 0.45, 0.95);
  const gx0 = HEAD.x + HEAD.w - 4;
  const target = gradTo >= BLOCKS.length ? HEAD.x + 6 : BLOCKS[gradTo].x + 6;
  return (
    <g>
      <Txt x={200} y={52} size={13} weight={800} color="var(--text-2)">
        ResNet-50 몸통 · ImageNet 사전학습 ✓
      </Txt>
      <line x1={14} x2={368} y1={62} y2={62} stroke="var(--border-strong)" strokeWidth={1.5} />
      {BLOCKS.map((b, i) => (
        <g key={b.id}>
          <Box x={b.x} y={Y} w={b.w} h={H} label={b.label} fs={12.5} color={c.series[1]} c={c} />
          <Txt x={b.x + b.w / 2} y={Y + H + 18} size={11.5} weight={700} color="var(--text-2)">
            {b.feat}
          </Txt>
          {ice[i] > 0.01 && (
            <g opacity={ice[i]}>
              <rect x={b.x - 3} y={Y - 3} width={b.w + 6} height={H + 6} rx={13} fill={c.theme === 'dark' ? 'rgba(125, 220, 255, 0.22)' : 'rgba(140, 200, 255, 0.35)'} stroke="#7cc8ff" strokeWidth={1.5} />
              <Emoji x={b.x + b.w - 8} y={Y + 12} size={15}>
                ❄️
              </Emoji>
            </g>
          )}
        </g>
      ))}
      <g opacity={1 - headNew}>
        <Box x={HEAD.x} y={Y - 30 * headNew} w={HEAD.w} h={H} label="fc 1000" sub="옛 헤드" subFs={11} fs={13} color={c.neg} c={c} />
      </g>
      <g opacity={headNew}>
        <Box x={HEAD.x} y={Y + 30 * (1 - headNew)} w={HEAD.w} h={H} label="fc 102 ✨" sub="새 헤드" subFs={11} fs={13} color={c.series[3]} c={c} />
      </g>
      <Txt x={HEAD.x + HEAD.w / 2} y={Y + H + 18} size={11.5} weight={700} color="var(--text-2)">
        꽃 102종
      </Txt>
      {gradTo <= BLOCKS.length && (
        <g>
          <Arrow x1={gx0} y1={Y + H + 32} x2={target} y2={Y + H + 32} color={c.series[2]} width={3} t={gradT} flow label="← 기울기" labelDy={20} />
          {gradTo > 0 && gradT > 0.95 && (
            <Emoji x={target - 14} y={Y + H + 38} size={16} className="labA-bob">
              🛑
            </Emoji>
          )}
        </g>
      )}
      {lr && (
        <g opacity={seg(t, 0.55, 0.8)}>
          <Txt x={190} y={Y - 12} size={12.5} weight={850} color={c.series[1]}>
            lr 1e-4 (살살)
          </Txt>
          <Txt x={HEAD.x + HEAD.w / 2} y={Y - 12} size={12.5} weight={850} color={c.series[3]}>
            lr 1e-3 (크게)
          </Txt>
        </g>
      )}
      <g opacity={seg(t, 0.3, 0.55)}>
        <Txt x={14} y={252} size={12.5} weight={800} anchor="start">
          학습하는 파라미터
        </Txt>
        <rect x={130} y={240} width={250} height={16} rx={8} fill="var(--surface-3)" />
        <rect x={130} y={240} width={Math.max(4, (trainable / TOTAL) * 250 * seg(t, 0.35, 0.8))} height={16} rx={8} fill={c.series[2]} />
        <Txt x={388} y={253} size={13} weight={850} anchor="start">
          {trainable.toFixed(1)}M
        </Txt>
        <Txt x={255} y={278} size={11.5} weight={700} color="var(--text-2)">
          전체 {TOTAL}M 중
        </Txt>
      </g>
    </g>
  );
}

const steps: StoryStep[] = [
  {
    title: '사전학습 ResNet-50을 빌리고, 헤드만 새로',
    text: (
      <>
        ImageNet 1000종으로 이미 학습된 ResNet-50을 가져와요. 몸통은 <strong>선 → 질감 → 부분 → 모양</strong> 순으로 쓸모 있는 특징을 이미 알고 있어요. 마지막 분류 층(fc)만 <strong>꽃 102종용 새 층</strong>으로 바꾸는데, 이 층은 무작위로 시작해요.
      </>
    ),
    code: `model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# 분류 헤드 교체: 1000 클래스 → 102 클래스. 새 층은 무작위 초기화된다.
model.fc = nn.Linear(model.fc.in_features, num_classes)`,
    highlight: [1, 3],
    render: (t, c) => (
      <g>
        <Defs />
        <Net c={c} t={t} ice={[0, 0, 0, 0, 0]} gradTo={99} headNew={seg(t, 0.3, 0.75)} trainable={TOTAL} />
      </g>
    )
  },
  {
    title: '선형 프로브: 몸통은 얼리고 헤드만 학습',
    text: (
      <>
        <code>requires_grad = False</code>로 몸통 전체를 <strong>얼려요(동결)</strong>. 기울기는 새 헤드까지만 흐르고 멈춰요. 학습할 파라미터가 <strong>0.2M</strong>뿐이라 데이터가 아주 적어도(클래스당 10장!) 과적합 걱정이 적어요.
      </>
    ),
    code: `if mode == "linear":
    # 선형 프로브: 백본 전체 동결, 헤드만 학습.
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc")`,
    highlight: [4],
    render: (t, c) => {
      const ice = BLOCKS.map((_, i) => seg(t, 0.05 + i * 0.06, 0.25 + i * 0.06));
      return (
        <g>
          <Defs />
          <Net c={c} t={t} ice={ice} gradTo={BLOCKS.length} trainable={0.21} />
        </g>
      );
    }
  },
  {
    title: '부분 해동: layer4만 녹인다',
    text: (
      <>
        마지막 스테이지 <strong>layer4</strong>는 "고양이 귀 모양" 같은 과제 특화 특징을 담아요. 여기만 녹여서(해동) 꽃에 맞게 고쳐 쓰면, 저수준 특징(선·질감)은 그대로 지키면서 더 잘 맞출 수 있어요. 학습 파라미터는 <strong>15.2M</strong>.
      </>
    ),
    code: `elif mode == "partial":
    # 부분 해동: 마지막 스테이지(layer4)와 헤드만 학습.
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(("layer4", "fc"))`,
    highlight: [4],
    render: (t, c) => {
      const melt = seg(t, 0.1, 0.45);
      return (
        <g>
          <Defs />
          <Net c={c} t={t} ice={[1, 1, 1, 1, 1 - melt]} gradTo={4} trainable={15.2} />
        </g>
      );
    }
  },
  {
    title: '전체 파인튜닝: 몸통은 살살, 헤드는 크게',
    text: (
      <>
        데이터가 충분하면 전부 녹여요. 단, 사전학습 층은 이미 좋은 자리에 있으니 <strong>작은 학습률(1e-4)</strong>로, 무작위로 시작한 헤드는 <strong>큰 학습률(1e-3)</strong>로 움직여요. 그래야 헤드의 큰 기울기가 몸통을 망가뜨리지 않아요.
      </>
    ),
    code: `head_params = [p for n, p in model.named_parameters() if n.startswith("fc")]
body_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
return torch.optim.AdamW([
    {"params": body_params, "lr": 1e-4},
    {"params": head_params, "lr": 1e-3},
], weight_decay=1e-4)`,
    highlight: [4, 5],
    render: (t, c) => {
      const melt = seg(t, 0.05, 0.35);
      return (
        <g>
          <Defs />
          <Net c={c} t={t} ice={BLOCKS.map(() => 1 - melt)} gradTo={0} lr={{ body: '1e-4', head: '1e-3' }} trainable={TOTAL} />
        </g>
      );
    }
  },
  {
    title: '결과: 클래스당 10장으로도 95%',
    text: (
      <>
        10 에포크 뒤 대략 <strong>선형 프로브 88% · 부분 해동 93% · 전체 95%</strong>. 함정 하나: 동결해도 <code>model.train()</code>은 BatchNorm의 평균 통계를 계속 바꿔요. 그래서 선형 프로브에선 몸통을 <code>eval()</code>로 두고 헤드만 <code>train()</code>으로 켜요.
      </>
    ),
    code: `model.train()
# 주의: 동결층이 있어도 model.train()은 그 층의 BatchNorm 통계를 갱신한다.
# 선형 프로브에서 백본 통계까지 완전히 고정하려면 백본을 eval로 둔다.
if args.mode == "linear":
    model.eval()
    model.fc.train()`,
    highlight: [4, 5, 6],
    render: (t, c) => (
      <g>
        <Defs />
        <Bars
          x={40}
          y={36}
          w={250}
          h={170}
          values={[0.88, 0.93, 0.95].map((v) => v - 0.8)}
          labels={['선형 프로브', '부분 해동', '전체']}
          color={c.series[1]}
          hiColor={c.series[3]}
          highlight={2}
          t={seg(t, 0.05, 0.6)}
          max={0.17}
          fs={12.5}
          valueText={(v) => `${Math.round((v + 0.8) * 100)}%`}
        />
        <Txt x={165} y={262} size={11.5} weight={700} color="var(--text-2)">
          (세로축은 80%부터)
        </Txt>
        <g opacity={seg(t, 0.55, 0.85)}>
          <rect x={314} y={50} width={154} height={150} rx={16} fill={soft(c.series[4], 0.12)} stroke={soft(c.series[4], 0.7)} />
          <Emoji x={391} y={86} size={24}>
            ⚠️
          </Emoji>
          <Txt x={391} y={112} size={13} weight={850}>
            BatchNorm 함정
          </Txt>
          <Txt x={391} y={136} size={11.5} weight={700} color="var(--text-2)">
            얼려도 train()이면
          </Txt>
          <Txt x={391} y={154} size={11.5} weight={700} color="var(--text-2)">
            평균 통계는 바뀌어요
          </Txt>
          <Txt x={391} y={180} size={12} weight={850} color={c.series[4]} mono>
            → 몸통은 eval()
          </Txt>
        </g>
      </g>
    )
  }
];

/** Code lab 03 walkthrough: linear probe → partial unfreeze → full fine-tune. */
export default function Lab03FreezeUnfreeze() {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={1.8} />;
}
