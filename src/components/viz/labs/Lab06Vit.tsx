import { CodeStory, type StoryStep } from '../core/story';
import type { VizColors } from '../core/colors';
import { rng, softmax } from '../core/math';
import { Arrow, Bars, Box, Defs, Emoji, Txt, VB, mix, seg, soft } from './_labsA';

/* A 16×16 pixel-art bird on a branch (each drawn pixel stands for 2×2 real pixels of the 32×32 image). */
const ART = [
  '................',
  '................',
  '......rrrr......',
  '.....rrrrrr.....',
  '....rrrwkrrry...',
  '....rrrrrrrryy..',
  '...rrrrrrrrr....',
  '..rrrrrrrrrr....',
  '..rrrrrrrrr.....',
  '...rrrrrrr......',
  '....rrrrr.......',
  '.....y..y.......',
  'bbbbbbbbbbbbbbbb',
  '..........gg....',
  '.........gggg...',
  '................'
];

function pixColor(ch: string, c: VizColors) {
  switch (ch) {
    case 'r':
      return c.theme === 'dark' ? '#ff4d6d' : '#f2626b';
    case 'y':
      return c.theme === 'dark' ? '#ffd23f' : '#f5b925';
    case 'w':
      return '#ffffff';
    case 'k':
      return '#2a2540';
    case 'b':
      return c.theme === 'dark' ? '#b07a4a' : '#9a6b43';
    case 'g':
      return c.theme === 'dark' ? '#2ef2c0' : '#5fb336';
    default:
      return c.theme === 'dark' ? '#1b3a8f' : '#cfe7ff';
  }
}

/** One 4×4 patch (= 2×2 drawn pixels) of the art, drawn at (x, y) with size s. */
function Patch({ pr, pc, x, y, s, c }: { pr: number; pc: number; x: number; y: number; s: number; c: VizColors }) {
  const h = s / 2;
  return (
    <g>
      {[0, 1].map((dr) =>
        [0, 1].map((dc) => <rect key={`${dr}${dc}`} x={x + dc * h} y={y + dr * h} width={h + 0.3} height={h + 0.3} fill={pixColor(ART[pr * 2 + dr][pc * 2 + dc], c)} />)
      )}
    </g>
  );
}

const R = rng(11);
const VEC = Array.from({ length: 9 }, () => Array.from({ length: 6 }, () => R.next()));
const CLASSES = ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭'];
const PROBS = softmax([0.2, -0.6, 3.1, 0.9, 0.4, 0.5, 0.7, -0.2, -0.4, -0.9]);

const steps: StoryStep[] = [
  {
    title: '사진을 4×4 조각(패치) 64개로 자른다',
    text: (
      <>
        ViT는 사진을 <strong>겹치지 않는 4×4 조각</strong>으로 잘라요. 32×32 사진이면 8×8 = <strong>64개 패치</strong>. 신기하게도 <code>kernel = stride = 4</code>인 Conv2d 하나가 "자르기 + 선형 변환"을 한 번에 해 줘요.
      </>
    ),
    code: `class PatchEmbed(nn.Module):
    """이미지 → 패치 시퀀스. stride=patch인 Conv2d 하나가 '자르기+선형투영'을 동시에 한다."""
    def __init__(self, img_size=32, patch=4, d_model=192):
        super().__init__()
        self.n_patches = (img_size // patch) ** 2          # 8×8 = 64개
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)`,
    highlight: [5, 6],
    render: (t, c) => {
      const gap = mix(0, 5, seg(t, 0.35, 0.8));
      const s = 18;
      const x0 = 30;
      const y0 = 50;
      const patches = [];
      for (let pr = 0; pr < 8; pr++)
        for (let pc = 0; pc < 8; pc++)
          patches.push(<Patch key={`${pr}-${pc}`} pr={pr} pc={pc} x={x0 + pc * (s + gap)} y={y0 + pr * (s + gap)} s={s} c={c} />);
      const gridOn = seg(t, 0.1, 0.35);
      return (
        <g>
          <Defs />
          <Txt x={110} y={36} size={13} weight={800} color="var(--text-2)">
            32×32 사진
          </Txt>
          {patches}
          {gap < 0.5 &&
            Array.from({ length: 9 }, (_, k) => (
              <g key={k} opacity={gridOn}>
                <line x1={x0 + k * s} x2={x0 + k * s} y1={y0} y2={y0 + 8 * s} stroke="var(--surface-solid)" strokeWidth={1.5} />
                <line x1={x0} x2={x0 + 8 * s} y1={y0 + k * s} y2={y0 + k * s} stroke="var(--surface-solid)" strokeWidth={1.5} />
              </g>
            ))}
          <g opacity={seg(t, 0.55, 0.85)}>
            <rect x={262} y={64} width={200} height={120} rx={16} fill={soft(c.series[0], 0.1)} stroke={soft(c.series[0], 0.6)} />
            <Txt x={362} y={94} size={15} weight={850}>
              8 × 8 = 64개 패치
            </Txt>
            <Txt x={362} y={120} size={12.5} weight={700} color="var(--text-2)">
              패치 하나 = 4×4 픽셀 × 3색
            </Txt>
            <Txt x={362} y={142} size={12.5} weight={700} color="var(--text-2)">
              = 48개 숫자
            </Txt>
            <Txt x={362} y={168} size={12.5} weight={850} color={c.series[0]}>
              "단어"처럼 다룰 거예요
            </Txt>
          </g>
          <Txt x={110} y={272} size={11.5} weight={650} color="var(--text-2)" opacity={seg(t, 0.6, 0.9)}>
            (그림은 단순하게 16×16으로 그렸어요)
          </Txt>
        </g>
      );
    }
  },
  {
    title: '패치를 한 줄로: (B, 64, 192) 시퀀스',
    text: (
      <>
        각 패치(48개 숫자)는 선형 변환을 거쳐 <strong>192차원 벡터</strong> 하나가 돼요. <code>flatten</code>과 <code>transpose</code>로 64개를 한 줄로 세우면, 이미지 분류가 <strong>"64단어짜리 문장 분류"</strong>로 바뀌어요.
      </>
    ),
    code: `def forward(self, x):                                  # (B, 3, 32, 32)
    x = self.proj(x)                                   # (B, D, 8, 8)
    return x.flatten(2).transpose(1, 2)                # (B, 64, D) — 패치 시퀀스`,
    highlight: [2, 3],
    render: (t, c) => {
      const s = 14;
      const x0 = 30;
      const y0 = 24;
      const fly = seg(t, 0.15, 0.6);
      const vec = seg(t, 0.55, 0.85);
      const tokens = [];
      for (let k = 0; k < 9; k++) {
        const pr = Math.floor(k / 8);
        const pc = k % 8;
        const sx = x0 + pc * s;
        const sy = y0 + pr * s;
        const tx = 36 + k * 46;
        const ty = 160;
        const x = mix(sx, tx, fly);
        const y = mix(sy, ty, fly);
        const size = mix(s, 30, fly);
        tokens.push(
          <g key={k}>
            <Patch pr={pr} pc={pc} x={x} y={y} s={size} c={c} />
            <rect x={x} y={y} width={size} height={size} fill="none" stroke={c.series[0]} strokeWidth={1.5} rx={3} />
            {VEC[k].map((v, j) => (
              <rect key={j} x={tx + 4} y={200 + j * 9} width={22} height={7} rx={2} fill={soft(c.series[0], 0.15 + 0.8 * v)} opacity={vec} />
            ))}
          </g>
        );
      }
      return (
        <g>
          <Defs />
          <g opacity={1 - fly * 0.6}>
            {Array.from({ length: 8 }, (_, pr) =>
              Array.from({ length: 8 }, (_, pc) => (pr === 0 || (pr === 1 && pc === 0) ? null : <Patch key={`${pr}-${pc}`} pr={pr} pc={pc} x={x0 + pc * s} y={y0 + pr * s} s={s} c={c} />))
            )}
            <rect x={x0 - 2} y={y0 - 2} width={8 * s + 4} height={8 * s + 4} rx={4} fill="none" stroke="var(--border-strong)" />
          </g>
          {tokens}
          <Txt x={452} y={180} size={16} weight={900} opacity={fly}>
            …
          </Txt>
          <Txt x={240} y={278} size={13} weight={800} opacity={vec}>
            패치마다 192차원 벡터 → 64개가 한 줄로 = (B, 64, 192)
          </Txt>
          <Arrow x1={170} y1={80} x2={270} y2={80} color={c.series[0]} t={seg(t, 0.05, 0.25)} label="flatten + transpose" flow />
          <Txt x={350} y={84} size={12} weight={750} color="var(--text-2)" anchor="start" opacity={seg(t, 0.2, 0.4)}>
            왼쪽 위부터 차례로
          </Txt>
        </g>
      );
    }
  },
  {
    title: '[CLS] 토큰을 붙이고 위치를 더한다',
    text: (
      <>
        맨 앞에 학습 가능한 특별 토큰 <strong>[CLS]</strong>를 붙여요 — 나중에 "사진 전체 요약"을 담는 자리예요. 어텐션은 순서를 모르니, 각 자리에 <strong>위치 임베딩</strong>을 더해 "몇 번째 패치인지" 알려 줘요. 그래서 65개 토큰이 돼요.
      </>
    ),
    code: `x = self.patch(x)                                  # (B, 64, D)
cls = self.cls.expand(x.shape[0], -1, -1)
x = torch.cat([cls, x], dim=1) + self.pos          # (B, 65, D)`,
    highlight: [2, 3],
    render: (t, c) => {
      const slide = seg(t, 0.1, 0.4);
      const pos = seg(t, 0.45, 0.85);
      return (
        <g>
          <Defs />
          {Array.from({ length: 8 }, (_, k) => {
            const x = mix(30 + k * 46, 30 + (k + 1) * 46, slide);
            return (
              <g key={k}>
                <Patch pr={0} pc={k} x={x} y={92} s={30} c={c} />
                <rect x={x} y={92} width={30} height={30} fill="none" stroke={c.series[0]} strokeWidth={1.5} rx={3} />
                {VEC[k].map((v, j) => (
                  <rect key={j} x={x + 4} y={132 + j * 9} width={22} height={7} rx={2} fill={soft(c.series[0], 0.15 + 0.8 * v)} />
                ))}
                <g opacity={pos}>
                  <rect x={x + 4} y={192} width={22} height={7} rx={2} fill={soft(c.series[4], 0.3 + 0.6 * (0.5 + 0.5 * Math.sin(k * 0.9)))} />
                  <Txt x={x + 15} y={216} size={11} weight={750} color="var(--text-2)">
                    {k + 1}
                  </Txt>
                </g>
              </g>
            );
          })}
          <g transform={`translate(${mix(-60, 0, slide)} 0)`} opacity={slide}>
            <rect x={30} y={92} width={30} height={30} rx={6} fill={c.series[3]} />
            <Txt x={45} y={112} size={14} weight={900} color="#fff">
              ★
            </Txt>
            {VEC[8].map((v, j) => (
              <rect key={j} x={34} y={132 + j * 9} width={22} height={7} rx={2} fill={soft(c.series[3], 0.2 + 0.7 * v)} />
            ))}
            <Txt x={45} y={80} size={12.5} weight={850} color={c.series[3]}>
              [CLS]
            </Txt>
            <g opacity={pos}>
              <rect x={34} y={192} width={22} height={7} rx={2} fill={soft(c.series[4], 0.5)} />
              <Txt x={45} y={216} size={11} weight={750} color="var(--text-2)">
                0
              </Txt>
            </g>
          </g>
          <Txt x={452} y={112} size={16} weight={900}>
            …
          </Txt>
          <g opacity={pos}>
            <Txt x={14} y={198} size={14} weight={900} anchor="start" color={c.series[4]}>
              +
            </Txt>
            <Txt x={240} y={248} size={13} weight={800}>
              + 위치 임베딩 (몇 번째 자리인지)
            </Txt>
            <Txt x={240} y={272} size={12.5} weight={700} color="var(--text-2)">
              [CLS] 1개 + 패치 64개 = 65개 토큰 (B, 65, 192)
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: 'Transformer 블록 ×6: 패치끼리 서로 바라본다',
    text: (
      <>
        블록마다 <strong>어텐션</strong>으로 모든 패치가 다른 모든 패치를 참고해요 — 멀리 떨어진 날개와 부리도 한 번에 연결돼요. Pre-LN 구조라 <strong>LayerNorm을 먼저</strong> 통과시키고, 결과를 원래 값에 <strong>더하는(잔차)</strong> 길이 기울기의 고속도로가 돼요.
      </>
    ),
    code: `def forward(self, x):
    x = x + self.attn(self.ln1(x))                     # 잔차: 그래디언트의 고속도로
    x = x + self.mlp(self.ln2(x))
    return x`,
    highlight: [2, 3],
    render: (t, c) => {
      const arc = seg(t, 0.1, 0.6);
      const xs = Array.from({ length: 7 }, (_, k) => 30 + k * 40);
      const weights = [0, 0.25, 0.9, 0.7, 0.15, 0.55, 0.3];
      return (
        <g>
          <Defs />
          {xs.map((x, k) => (
            <g key={k}>
              {k === 0 ? (
                <rect x={x} y={214} width={28} height={28} rx={6} fill={c.series[3]} />
              ) : (
                <g>
                  <Patch pr={3 + (k % 2)} pc={1 + k} x={x} y={214} s={28} c={c} />
                  <rect x={x} y={214} width={28} height={28} fill="none" stroke={c.series[0]} strokeWidth={1.5} rx={3} />
                </g>
              )}
            </g>
          ))}
          <Txt x={44} y={262} size={11.5} weight={850} color={c.series[3]}>
            CLS
          </Txt>
          {xs.slice(1).map((x, k) => {
            const w = weights[k + 1];
            const mid = (xs[0] + 14 + x + 14) / 2;
            const hgt = 40 + (x - xs[0]) * 0.55;
            return (
              <path
                key={k}
                d={`M${xs[0] + 14} 212 Q${mid} ${212 - hgt} ${x + 14} 212`}
                fill="none"
                stroke={c.series[2]}
                strokeWidth={1 + w * 5}
                opacity={(0.25 + 0.75 * w) * arc}
                strokeLinecap="round"
                pathLength={1}
                strokeDasharray={`${arc} 1`}
              />
            );
          })}
          <Txt x={150} y={60} size={12.5} weight={800} color={c.series[2]} opacity={arc}>
            굵을수록 많이 참고 (어텐션 가중치)
          </Txt>
          {/* Pre-LN block */}
          <g opacity={seg(t, 0.4, 0.75)}>
            <rect x={322} y={30} width={146} height={226} rx={16} fill={soft(c.series[1], 0.06)} stroke={soft(c.series[1], 0.5)} />
            <Txt x={395} y={50} size={12.5} weight={850}>
              블록 하나 (×6)
            </Txt>
            <path d="M380 252 V234 M380 208 V198 M380 170 V164 M380 148 V138 M380 112 V102 M380 74 V70" stroke="var(--viz-axis)" strokeWidth={2} fill="none" />
            <Box x={350} y={208} w={60} h={26} label="LN" fs={12} color={c.series[5]} c={c} />
            <Box x={350} y={170} w={60} h={28} label="MHA" fs={12} color={c.series[2]} c={c} />
            <Box x={350} y={112} w={60} h={26} label="LN" fs={12} color={c.series[5]} c={c} />
            <Box x={350} y={74} w={60} h={28} label="MLP" fs={12} color={c.series[1]} c={c} />
            <circle cx={380} cy={156} r={8} fill="var(--surface-solid)" stroke={c.series[3]} strokeWidth={2} />
            <Txt x={380} y={161} size={13} weight={900} color={c.series[3]}>
              +
            </Txt>
            <circle cx={380} cy={62} r={8} fill="var(--surface-solid)" stroke={c.series[3]} strokeWidth={2} />
            <Txt x={380} y={67} size={13} weight={900} color={c.series[3]}>
              +
            </Txt>
            <path d="M380 248 L 440 248 L 440 156 L 388 156" fill="none" stroke={c.series[3]} strokeWidth={2} strokeDasharray="4 3" />
            <path d="M380 150 L 440 150 L 440 62 L 388 62" fill="none" stroke={c.series[3]} strokeWidth={2} strokeDasharray="4 3" />
            <Txt x={452} y={110} size={11} weight={850} color={c.series[3]} anchor="start">
              잔차
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '[CLS] 토큰 하나로 분류한다',
    text: (
      <>
        6개 블록을 지나는 동안 [CLS] 토큰은 모든 패치를 참고해 <strong>사진 전체의 요약</strong>이 돼요. 마지막엔 <code>x[:, 0]</code>, 즉 [CLS] 자리만 꺼내 LayerNorm → 선형 층에 넣어 10개 클래스 점수를 만들어요.
      </>
    ),
    code: `x = self.blocks(x)
return self.head(self.ln(x[:, 0]))                 # CLS 토큰만 분류에 사용`,
    highlight: [2],
    render: (t, c) => (
      <g>
        <Defs />
        <rect x={24} y={60} width={40} height={40} rx={8} fill={c.series[3]} />
        <Txt x={44} y={86} size={16} weight={900} color="#fff">
          ★
        </Txt>
        <Txt x={44} y={124} size={12} weight={850} color={c.series[3]}>
          [CLS]
        </Txt>
        <Txt x={44} y={142} size={11} weight={700} color="var(--text-2)">
          x[:, 0]
        </Txt>
        <Arrow x1={70} y1={80} x2={110} y2={80} color={c.series[3]} t={seg(t, 0.05, 0.2)} flow />
        <Box x={114} y={60} w={52} h={40} label="LN" fs={13} color={c.series[5]} t={seg(t, 0.1, 0.25)} c={c} />
        <Arrow x1={170} y1={80} x2={204} y2={80} color={c.series[3]} t={seg(t, 0.2, 0.32)} flow />
        <Box x={208} y={60} w={60} h={40} label="head" sub="192→10" subFs={10.5} fs={13} color={c.series[3]} t={seg(t, 0.25, 0.4)} c={c} />
        <Arrow x1={238} y1={106} x2={238} y2={138} color={c.series[3]} t={seg(t, 0.35, 0.45)} flow />
        <Txt x={250} y={126} size={11.5} weight={750} color="var(--text-2)" anchor="start" opacity={seg(t, 0.4, 0.5)}>
          점수 10개 → softmax
        </Txt>
        <Bars
          x={40}
          y={150}
          w={420}
          h={90}
          values={PROBS}
          labels={CLASSES}
          color={c.series[0]}
          hiColor={c.series[0]}
          highlight={2}
          t={seg(t, 0.4, 0.9)}
          max={1}
          fs={11}
          valueText={(v, i) => (i === 2 ? `${Math.round(v * 100)}%` : '')}
        />
        <g opacity={seg(t, 0.8, 1)}>
          <Emoji x={360} y={92} size={30}>
            🐦
          </Emoji>
          <Txt x={410} y={88} size={14} weight={850} anchor="start">
            "새!"
          </Txt>
        </g>
      </g>
    )
  },
  {
    title: '작은 데이터에서는 왜 CNN이 이길까?',
    text: (
      <>
        30 에포크 뒤 ViT는 약 <strong>85%</strong>, 같은 조건의 ResNet은 약 <strong>93%</strong>. ViT에는 "가까운 픽셀끼리 관련 있다"는 귀띔(지역성 편향)이 없어서 그것까지 데이터로 배워야 하거든요. 대신 아주 큰 데이터로 사전학습하면 역전돼요. 그리고 ViT는 <strong>AdamW + 웜업</strong>이 사실상 필수예요.
      </>
    ),
    code: `opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                                            total_steps=epochs * len(train_dl), pct_start=0.1)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)`,
    highlight: [1, 2, 3],
    render: (t, c) => (
      <g>
        <Defs />
        <Txt x={140} y={32} size={13} weight={800} color="var(--text-2)">
          CIFAR-10, 처음부터 학습
        </Txt>
        <Bars
          x={50}
          y={50}
          w={180}
          h={170}
          values={[0.93 - 0.7, 0.85 - 0.7]}
          labels={['ResNet', 'ViT']}
          color={c.series[1]}
          hiColor={c.series[1]}
          highlight={0}
          t={seg(t, 0.05, 0.5)}
          max={0.25}
          fs={13}
          valueText={(v) => `${Math.round((v + 0.7) * 100)}%`}
        />
        <Txt x={140} y={268} size={11.5} weight={700} color="var(--text-2)">
          (세로축은 70%부터)
        </Txt>
        <g opacity={seg(t, 0.5, 0.85)}>
          <rect x={268} y={50} width={200} height={170} rx={16} fill={soft(c.series[0], 0.08)} stroke={soft(c.series[0], 0.5)} />
          <Txt x={368} y={78} size={13} weight={850}>
            거대 데이터로 사전학습하면
          </Txt>
          <Emoji x={320} y={136} size={34}>
            🧱
          </Emoji>
          <Txt x={320} y={166} size={11.5} weight={750} color="var(--text-2)">
            CNN: 편향 내장
          </Txt>
          <Emoji x={416} y={136} size={34} className="labA-bob">
            🚀
          </Emoji>
          <Txt x={416} y={166} size={11.5} weight={750} color="var(--text-2)">
            ViT: 역전!
          </Txt>
          <Txt x={368} y={204} size={12} weight={800} color={c.series[0]}>
            귀납 편향 ↔ 데이터 양의 거래
          </Txt>
        </g>
      </g>
    )
  }
];

/** Code lab 06 walkthrough: patchify → tokens + CLS + position → blocks → CLS head. */
export default function Lab06Vit() {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={1.9} />;
}
