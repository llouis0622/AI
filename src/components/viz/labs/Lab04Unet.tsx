import { CodeStory, type StoryStep } from '../core/story';
import type { VizColors } from '../core/colors';
import { Arrow, Chip, Defs, Txt, VB, mix, seg, soft } from './_labsA';

/* ---------- a drawn pet photo and its 3-class mask ---------- */

function CatShape({ fill, stroke, sw = 0 }: { fill: string; stroke?: string; sw?: number }) {
  return (
    <g fill={fill} stroke={stroke} strokeWidth={sw} strokeLinejoin="round">
      <path d="M-34 -18 L-30 -52 L-8 -32 Z" />
      <path d="M34 -18 L30 -52 L8 -32 Z" />
      <ellipse cx={0} cy={0} rx={40} ry={34} />
      <ellipse cx={0} cy={44} rx={30} ry={22} />
    </g>
  );
}

function PetPhoto({ x, y, c }: { x: number; y: number; c: VizColors }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      <rect x={-70} y={-80} width={140} height={160} rx={12} fill={soft(c.series[3], 0.22)} />
      <rect x={-70} y={30} width={140} height={50} fill={soft(c.series[1], 0.3)} />
      <CatShape fill={c.series[4]} />
      <circle cx={-14} cy={-4} r={5} fill="var(--text)" />
      <circle cx={14} cy={-4} r={5} fill="var(--text)" />
      <path d="M-6 10 Q0 15 6 10" fill="none" stroke="var(--text)" strokeWidth={2} strokeLinecap="round" />
    </g>
  );
}

function PetMask({ x, y, c, blur = 0, id }: { x: number; y: number; c: VizColors; blur?: number; id: string }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      {blur > 0 && (
        <defs>
          <filter id={id} x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation={blur} />
          </filter>
        </defs>
      )}
      <rect x={-70} y={-80} width={140} height={160} rx={12} fill="var(--surface-3)" />
      <g filter={blur > 0 ? `url(#${id})` : undefined}>
        <CatShape fill={c.series[0]} stroke={c.series[2]} sw={6} />
      </g>
    </g>
  );
}

/* ---------- the U ---------- */

const ENC = [
  { x: 40, cy: 66, h: 72, w: 10, label: '128²×32', id: 's1' },
  { x: 82, cy: 130, h: 46, w: 16, label: '64²×64', id: 's2' },
  { x: 124, cy: 186, h: 30, w: 24, label: '32²×128', id: 's3' }
];
const BOT = { x: 206, cy: 236, h: 20, w: 36, label: '16²×256' };
const DEC = [
  { x: 316, cy: 186, h: 30, w: 24, label: 'd3', id: 'd3' },
  { x: 366, cy: 130, h: 46, w: 16, label: 'd2', id: 'd2' },
  { x: 410, cy: 66, h: 72, w: 10, label: 'd1', id: 'd1' }
];

function Slab({ x, cy, w, h, col, t, label, labelLeft = false }: { x: number; cy: number; w: number; h: number; col: string; t: number; label?: string; labelLeft?: boolean }) {
  return (
    <g opacity={Math.min(1, t * 2)}>
      <rect x={x} y={cy - (h / 2) * Math.min(1, t * 1.3)} width={w} height={h * Math.min(1, t * 1.3)} rx={3} fill={soft(col, 0.35)} stroke={col} strokeWidth={2} />
      {label && t > 0.6 && (
        <Txt x={labelLeft ? x - 6 : x + w + 6} y={cy + 4} size={11.5} weight={750} anchor={labelLeft ? 'end' : 'start'} color="var(--text-2)" mono>
          {label}
        </Txt>
      )}
    </g>
  );
}

function UNetFigure({ c, enc, dec, skip }: { c: VizColors; enc: number; dec: number; skip: number }) {
  return (
    <g>
      {ENC.map((b, i) => (
        <Slab key={b.id} {...b} col={c.series[0]} t={seg(enc, i * 0.25, i * 0.25 + 0.3)} label={b.label} labelLeft={false} />
      ))}
      {ENC.slice(0, 2).map((b, i) => (
        <Arrow key={`p${i}`} x1={b.x + b.w / 2} y1={b.cy + b.h / 2 + 4} x2={ENC[i + 1].x + 2} y2={ENC[i + 1].cy - ENC[i + 1].h / 2 - 4} color={c.series[0]} width={2} t={seg(enc, i * 0.25 + 0.2, i * 0.25 + 0.35)} />
      ))}
      <Arrow x1={ENC[2].x + ENC[2].w / 2} y1={ENC[2].cy + ENC[2].h / 2 + 4} x2={BOT.x + 2} y2={BOT.cy - 4} color={c.series[0]} width={2} t={seg(enc, 0.7, 0.85)} />
      <Slab {...BOT} col={c.series[4]} t={seg(enc, 0.8, 1)} />
      {enc > 0.3 && (
        <Txt x={78} y={268} size={12} weight={800} color={c.series[0]} opacity={seg(enc, 0.3, 0.6)}>
          MaxPool ↓ 해상도 절반
        </Txt>
      )}
      {/* decoder */}
      <Arrow x1={BOT.x + BOT.w - 2} y1={BOT.cy - 4} x2={DEC[0].x + 4} y2={DEC[0].cy + DEC[0].h / 2 + 4} color={c.series[3]} width={2} t={seg(dec, 0, 0.2)} />
      {DEC.map((b, i) => (
        <Slab key={b.id} {...b} col={c.series[3]} t={seg(dec, 0.1 + i * 0.28, 0.4 + i * 0.28)} label={b.label} />
      ))}
      {DEC.slice(0, 2).map((b, i) => (
        <Arrow key={`u${i}`} x1={b.x + b.w / 2} y1={b.cy - b.h / 2 - 4} x2={DEC[i + 1].x + 2} y2={DEC[i + 1].cy + DEC[i + 1].h / 2 + 4} color={c.series[3]} width={2} t={seg(dec, 0.3 + i * 0.28, 0.45 + i * 0.28)} />
      ))}
      {dec > 0.2 && (
        <Txt x={400} y={268} size={12} weight={800} color={c.series[3]} opacity={seg(dec, 0.2, 0.5)}>
          업샘플 ↑ 해상도 두 배
        </Txt>
      )}
      {/* skips */}
      {ENC.map((b, i) => {
        const d = DEC[2 - i];
        const p = seg(skip, (2 - i) * 0.3, (2 - i) * 0.3 + 0.35);
        return (
          <g key={`k${i}`}>
            <Arrow x1={b.x + b.w + 72} y1={b.cy} x2={d.x - 6} y2={d.cy} color={c.series[2]} width={3} t={p} flow />
            {p > 0.9 && (
              <Txt x={(b.x + b.w + 72 + d.x) / 2} y={b.cy - 8} size={11.5} weight={850} color={c.series[2]}>
                skip · concat
              </Txt>
            )}
          </g>
        );
      })}
    </g>
  );
}

function circleOverlap(r1: number, r2: number, d: number) {
  if (d >= r1 + r2) return 0;
  if (d <= Math.abs(r1 - r2)) return Math.PI * Math.min(r1, r2) ** 2;
  const a = r1 * r1 * Math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1));
  const b = r2 * r2 * Math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2));
  const k = 0.5 * Math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2));
  return a + b - k;
}

const PIX = [
  '..........',
  '...####...',
  '..######..',
  '.########.',
  '.########.',
  '..######..',
  '...####...',
  '..........'
];

const steps: StoryStep[] = [
  {
    title: '목표: 픽셀 하나하나에 정답 달기',
    text: (
      <>
        분류는 사진 한 장에 답 하나지만, 세그멘테이션은 <strong>픽셀마다</strong> "반려동물 / 배경 / 경계" 중 하나를 맞혀요. 정답 마스크를 줄일 땐 반드시 <strong>최근접 보간</strong>! 쌍선형 보간은 레이블 1과 2를 섞어 1.5 같은 엉뚱한 값을 만들어요.
      </>
    ),
    code: `img = v2.functional.resize(img, [IMG_SIZE, IMG_SIZE])
# 마스크는 최근접 보간! 쌍선형 보간은 레이블 값을 섞어버린다.
mask = v2.functional.resize(mask, [IMG_SIZE, IMG_SIZE],
                            interpolation=v2.InterpolationMode.NEAREST)`,
    highlight: [3, 4],
    render: (t, c) => {
      const reveal = seg(t, 0.25, 0.75);
      return (
        <g>
          <Defs />
          <defs>
            <clipPath id="labA-l04-clip">
              <rect x={262} y={20} width={140 * reveal} height={170} />
            </clipPath>
          </defs>
          <Txt x={110} y={22} size={13} weight={800} color="var(--text-2)">
            입력 사진 (128×128)
          </Txt>
          <PetPhoto x={110} y={112} c={c} />
          <Arrow x1={190} y1={112} x2={252} y2={112} color={c.series[0]} t={seg(t, 0.1, 0.3)} label="U-Net" flow />
          <Txt x={332} y={22} size={13} weight={800} color="var(--text-2)">
            정답 마스크
          </Txt>
          <rect x={262} y={32} width={140} height={160} rx={12} fill="none" stroke="var(--border-strong)" strokeDasharray="4 4" />
          <g clipPath="url(#labA-l04-clip)">
            <PetMask x={332} y={112} c={c} id="labA-l04-m1" />
          </g>
          <g opacity={seg(t, 0.6, 0.85)}>
            <Chip x={290} y={214} text="전경" w={52} h={24} fs={12} color={c.series[0]} active c={c} />
            <Chip x={350} y={214} text="경계" w={52} h={24} fs={12} color={c.series[2]} active c={c} />
            <Chip x={410} y={214} text="배경" w={52} h={24} fs={12} color="#9aa0b4" c={c} />
          </g>
          <g opacity={seg(t, 0.75, 1)}>
            <rect x={40} y={242} width={400} height={44} rx={14} fill={soft(c.series[4], 0.12)} stroke={soft(c.series[4], 0.6)} />
            <Txt x={240} y={262} size={12.5} weight={800}>
              레이블 1, 2 사이를 쌍선형 보간 → 1.5 ✗ · 최근접 → 1 또는 2 ✓
            </Txt>
            <Txt x={240} y={279} size={11.5} weight={650} color="var(--text-2)">
              그리고 저장된 값 1~3을 0~2로 바꿔(-1) 클래스 번호로 써요
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '인코더: 해상도는 줄이고 채널은 늘린다',
    text: (
      <>
        <code>enc → pool</code>을 반복할 때마다 해상도는 절반(128 → 64 → 32 → 16), 채널은 두 배(32 → 64 → 128 → 256)가 돼요. 사진이 점점 작고 두꺼운 "요약"으로 압축되며 <strong>"무엇이 있는가"</strong>를 잘 알게 되지만, <strong>"정확히 어디인가"</strong>는 흐려져요.
      </>
    ),
    code: `s1 = self.enc1(x)                             # (B, 32, 128, 128)
s2 = self.enc2(self.pool(s1))                 # (B, 64, 64, 64)
s3 = self.enc3(self.pool(s2))                 # (B, 128, 32, 32)
b = self.bottleneck(self.pool(s3))            # (B, 256, 16, 16)`,
    highlight: [1, 2, 3, 4],
    render: (t, c) => (
      <g>
        <Defs />
        <Txt x={120} y={22} size={13} weight={850} color={c.series[0]}>
          인코더 (내려가기)
        </Txt>
        <UNetFigure c={c} enc={seg(t, 0, 0.95)} dec={0} skip={0} />
        <g opacity={seg(t, 0.8, 1)}>
          <Txt x={360} y={70} size={13} weight={800}>
            작아지고(해상도↓)
          </Txt>
          <Txt x={360} y={92} size={13} weight={800}>
            두꺼워지고(채널↑)
          </Txt>
          <Txt x={360} y={120} size={12} weight={700} color="var(--text-2)">
            "무엇" ↑ · "어디" ↓
          </Txt>
        </g>
      </g>
    )
  },
  {
    title: '디코더 + 스킵 연결: "어디"를 되살린다',
    text: (
      <>
        디코더는 <code>ConvTranspose2d</code>로 해상도를 다시 두 배씩 키워요. 핵심은 <code>torch.cat</code> — 같은 해상도의 인코더 특징을 <strong>그대로 이어붙여(스킵 연결)</strong> 잃어버린 위치 정보를 되살려요. 이 세 줄을 지우면 경계가 뭉개진답니다.
      </>
    ),
    code: `d3 = self.dec3(torch.cat([self.up3(b), s3], dim=1))
d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
return self.head(d1)                          # (B, 3, 128, 128) — 픽셀별 로짓`,
    highlight: [1, 2, 3],
    render: (t, c) => (
      <g>
        <Defs />
        <Txt x={240} y={22} size={13} weight={850}>
          U자 모양: 내려갔다가 다시 올라오기
        </Txt>
        <UNetFigure c={c} enc={1} dec={seg(t, 0, 0.6)} skip={seg(t, 0.35, 1)} />
      </g>
    )
  },
  {
    title: '1×1 conv 헤드: 픽셀마다 점수 3개',
    text: (
      <>
        마지막 <strong>1×1 합성곱</strong>은 픽셀마다 따로 작동하는 작은 분류기예요. 각 픽셀의 32채널 특징을 받아 <strong>전경·배경·경계 점수 3개</strong>를 내고, 가장 큰 점수가 그 픽셀의 답이 돼요. 출력 크기는 입력과 같은 128×128이에요.
      </>
    ),
    code: `self.head = nn.Conv2d(chs[0], n_classes, 1)   # 1×1 conv: 픽셀별 분류기

pred = model(x.to(device)).argmax(1).cpu()`,
    highlight: [1, 3],
    render: (t, c) => {
      const fill = seg(t, 0.45, 0.95);
      const cell = 18;
      const gx = 250;
      const gy = 48;
      const cells = [];
      for (let r = 0; r < PIX.length; r++)
        for (let k = 0; k < PIX[0].length; k++) {
          const inside = PIX[r][k] === '#';
          const edge =
            inside &&
            [
              [r - 1, k],
              [r + 1, k],
              [r, k - 1],
              [r, k + 1]
            ].some(([rr, kk]) => PIX[rr]?.[kk] !== '#');
          const order = (r * PIX[0].length + k) / (PIX.length * PIX[0].length);
          const on = fill > order;
          const col = edge ? c.series[2] : inside ? c.series[0] : '#9aa0b4';
          cells.push(
            <rect key={`${r}-${k}`} x={gx + k * (cell + 2)} y={gy + r * (cell + 2)} width={cell} height={cell} rx={4} fill={on ? (inside ? col : 'var(--surface-3)') : 'var(--surface-2)'} stroke="var(--border)" />
          );
        }
      const scores = [2.1, 0.3, -0.5];
      const names = ['전경', '배경', '경계'];
      const cols = [c.series[0], '#9aa0b4', c.series[2]];
      return (
        <g>
          <Defs />
          <Txt x={110} y={30} size={13} weight={800} color="var(--text-2)">
            픽셀 하나의 점수
          </Txt>
          <rect x={40} y={46} width={140} height={110} rx={14} fill={soft(c.series[3], 0.08)} stroke={soft(c.series[3], 0.5)} />
          {scores.map((s, i) => {
            const w = Math.max(0, s) * 40 * seg(t, 0.05, 0.35);
            return (
              <g key={names[i]}>
                <Txt x={52} y={78 + i * 30} size={12.5} weight={800} anchor="start">
                  {names[i]}
                </Txt>
                <rect x={92} y={66 + i * 30} width={Math.max(3, w)} height={16} rx={5} fill={cols[i]} opacity={i === 0 ? 1 : 0.5} />
                <Txt x={98 + Math.max(3, w)} y={79 + i * 30} size={11.5} weight={750} anchor="start" color="var(--text-2)">
                  {s}
                </Txt>
              </g>
            );
          })}
          <Txt x={110} y={180} size={12.5} weight={850} opacity={seg(t, 0.3, 0.45)}>
            argmax → "전경"!
          </Txt>
          <Arrow x1={186} y1={128} x2={240} y2={128} color={c.series[3]} t={seg(t, 0.35, 0.5)} label="모든 픽셀" flow />
          {cells}
          <Txt x={350} y={226} size={12} weight={700} color="var(--text-2)">
            128×128 픽셀 전부 → 마스크 완성
          </Txt>
        </g>
      );
    }
  },
  {
    title: '손실: CE + Dice로 작은 전경도 놓치지 않기',
    text: (
      <>
        픽셀별 교차엔트로피(CE)만 쓰면 넓은 배경이 손실을 지배해서, 작은 전경을 다 틀려도 점수가 좋아 보일 수 있어요. <strong>Dice = 2·겹친 넓이 / (두 넓이의 합)</strong>은 클래스별 겹침 비율이라 크기에 흔들리지 않아요. 그래서 둘을 더해 써요.
      </>
    ),
    code: `def dice_loss(logits, target, eps=1e-6):
    """Dice = 2|A∩B| / (|A|+|B|). 클래스 불균형(작은 전경)에 강하다."""
    probs = logits.softmax(dim=1)
    target_1h = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * target_1h).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_1h.sum(dim=(2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()

loss = F.cross_entropy(logits, y) + dice_loss(logits, y)`,
    highlight: [5, 6, 7, 9],
    render: (t, c) => {
      const r1 = 52;
      const r2 = 46;
      const d = mix(110, 12, seg(t, 0.15, 0.85));
      const inter = circleOverlap(r1, r2, d);
      const dice = (2 * inter) / (Math.PI * r1 * r1 + Math.PI * r2 * r2);
      const cx = 170;
      const cy = 140;
      return (
        <g>
          <Defs />
          <defs>
            <clipPath id="labA-l04-dice">
              <circle cx={cx - d / 2} cy={cy} r={r1} />
            </clipPath>
          </defs>
          <circle cx={cx - d / 2} cy={cy} r={r1} fill={soft(c.series[0], 0.18)} stroke={c.series[0]} strokeWidth={2.5} />
          <circle cx={cx + d / 2} cy={cy} r={r2} fill={soft(c.series[3], 0.18)} stroke={c.series[3]} strokeWidth={2.5} strokeDasharray="6 4" />
          <circle cx={cx + d / 2} cy={cy} r={r2} fill={c.series[2]} opacity={0.55} clipPath="url(#labA-l04-dice)" />
          <Txt x={cx - d / 2 - 20} y={cy - r1 - 10} size={12.5} weight={850} color={c.series[0]}>
            정답 A
          </Txt>
          <Txt x={cx + d / 2 + 22} y={cy - r2 - 10} size={12.5} weight={850} color={c.series[3]}>
            예측 B
          </Txt>
          <g>
            <rect x={310} y={70} width={156} height={120} rx={16} fill={soft(c.series[2], 0.1)} stroke={soft(c.series[2], 0.6)} />
            <Txt x={388} y={100} size={13} weight={800}>
              Dice 점수
            </Txt>
            <Txt x={388} y={140} size={30} weight={900} color={c.series[2]}>
              {dice.toFixed(2)}
            </Txt>
            <Txt x={388} y={170} size={12} weight={700} color="var(--text-2)">
              손실 = 1 - Dice = {(1 - dice).toFixed(2)}
            </Txt>
          </g>
          <Txt x={240} y={262} size={12.5} weight={700} color="var(--text-2)" opacity={seg(t, 0.7, 1)}>
            주황색 = 겹친 부분 A∩B · 많이 겹칠수록 Dice → 1
          </Txt>
        </g>
      );
    }
  }
];

/** Code lab 04 walkthrough: U-Net encoder/decoder, skip connections, 1×1 head, CE + Dice. */
export default function Lab04Unet() {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={1.9} />;
}
