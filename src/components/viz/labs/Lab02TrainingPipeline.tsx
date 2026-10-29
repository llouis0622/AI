import { CodeStory, type StoryStep } from '../core/story';
import type { VizColors } from '../core/colors';
import { Arrow, Box, Chip, Curve, Defs, Emoji, Tensor, Txt, VB, mix, seg, soft } from './_labsA';

/** A tiny cartoon "car" photo (CIFAR-10 style thumbnail). */
function CarPhoto({ x, y, s = 1, flip = 0, shift = 0, c }: { x: number; y: number; s?: number; flip?: number; shift?: number; c: VizColors }) {
  const sx = 1 - 2 * flip;
  return (
    <g transform={`translate(${x} ${y})`}>
      <rect width={64 * s} height={64 * s} rx={8} fill={soft(c.series[3], 0.18)} stroke={c.series[3]} strokeWidth={1.5} />
      <g transform={`translate(${32 * s + shift} ${36 * s}) scale(${sx * s} ${s})`}>
        <rect x={-22} y={-6} width={44} height={14} rx={5} fill={c.series[2]} />
        <path d="M-12 -6 L-6 -16 L10 -16 L16 -6 Z" fill={c.series[2]} opacity={0.85} />
        <rect x={-4} y={-14} width={10} height={7} rx={2} fill="var(--surface-solid)" opacity={0.9} />
        <circle cx={-12} cy={9} r={5} fill="var(--text)" />
        <circle cx={12} cy={9} r={5} fill="var(--text)" />
        <circle cx={19} cy={-1} r={2} fill={c.series[4]} />
      </g>
    </g>
  );
}

const ACC = [0.58, 0.67, 0.71, 0.74, 0.73, 0.78, 0.8, 0.79, 0.83, 0.84, 0.83, 0.86, 0.87, 0.88, 0.875, 0.89, 0.905, 0.91, 0.918, 0.921];

/** OneCycleLR (cosine) with pct_start 0.1, max_lr 0.1, div_factor 25, final_div 1e4. */
function oneCycle(s: number) {
  const max = 0.1;
  const init = max / 25;
  const min = init / 1e4;
  if (s <= 0.1) return init + (max - init) * (1 - Math.cos((Math.PI * s) / 0.1)) / 2;
  return min + (max - min) * (1 + Math.cos((Math.PI * (s - 0.1)) / 0.9)) / 2;
}

const steps: StoryStep[] = [
  {
    title: '데이터: 증강은 훈련에만',
    text: (
      <>
        훈련 이미지는 매번 조금씩 <strong>잘리고(RandomCrop) 뒤집혀서(Flip)</strong> 들어가요. 같은 사진도 매 에포크 다르게 보이니 외우기 어렵죠. 하지만 검증 이미지는 <strong>항상 그대로</strong> 둬야 에포크끼리 성적을 공정하게 비교할 수 있어요.
      </>
    ),
    code: `train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])`,
    highlight: [2, 3],
    render: (t, c) => {
      const flip = seg(t, 0.25, 0.45);
      const shift = mix(0, 6, seg(t, 0.1, 0.25)) * (1 - seg(t, 0.55, 0.7));
      return (
        <g>
          <Defs />
          <Txt x={18} y={40} size={14} weight={850} anchor="start" color={c.series[0]}>
            🏋️ 훈련
          </Txt>
          <CarPhoto x={22} y={52} flip={flip > 0.5 ? 1 : 0} shift={shift} s={1} c={c} />
          <rect x={18} y={48} width={72} height={72} rx={10} fill="none" stroke={c.series[0]} strokeDasharray="4 4" opacity={seg(t, 0.05, 0.2)} />
          <Txt x={56} y={140} size={12} weight={700} color="var(--text-2)">
            {flip > 0.5 ? '좌우 뒤집기!' : '살짝 잘라내기'}
          </Txt>
          <Arrow x1={98} y1={86} x2={168} y2={86} color={c.series[0]} t={seg(t, 0.3, 0.5)} flow />
          <Box x={174} y={64} w={104} h={44} label="ToTensor" sub="+ Normalize" subFs={11.5} fs={14} color={c.series[1]} t={seg(t, 0.35, 0.55)} c={c} />
          <Arrow x1={282} y1={86} x2={330} y2={86} color={c.series[0]} t={seg(t, 0.45, 0.6)} flow />
          <Tensor x={340} y={58} w={108} h={56} label="배치" shape="(256, 3, 32, 32)" color={c.series[0]} t={seg(t, 0.5, 0.7)} c={c} />
          <Txt x={394} y={146} size={11.5} weight={700} color="var(--text-2)" opacity={seg(t, 0.6, 0.8)}>
            shuffle=True · 섞어서
          </Txt>

          <line x1={18} x2={462} y1={170} y2={170} stroke="var(--border)" strokeWidth={1.5} strokeDasharray="4 6" />

          <Txt x={18} y={200} size={14} weight={850} anchor="start" color={c.series[3]}>
            🧪 검증
          </Txt>
          <CarPhoto x={22} y={210} c={c} />
          <Txt x={56} y={292} size={12} weight={700} color="var(--text-2)">
            항상 그대로
          </Txt>
          <Arrow x1={98} y1={242} x2={168} y2={242} color={c.series[3]} t={seg(t, 0.55, 0.7)} flow />
          <Box x={174} y={220} w={104} h={44} label="ToTensor" sub="+ Normalize" subFs={11.5} fs={14} color={c.series[1]} t={seg(t, 0.6, 0.75)} c={c} />
          <Arrow x1={282} y1={242} x2={330} y2={242} color={c.series[3]} t={seg(t, 0.7, 0.85)} flow />
          <Tensor x={340} y={214} w={108} h={56} label="배치" shape="(512, 3, 32, 32)" color={c.series[3]} t={seg(t, 0.75, 0.95)} c={c} />
        </g>
      );
    }
  },
  {
    title: '모델: 작은 사진에 맞춘 ResNet-18',
    text: (
      <>
        원래 ResNet-18의 첫 층(7×7, stride 2)과 maxpool은 224×224 사진용이라 32×32 사진을 <strong>시작하자마자 8×8로 뭉개요</strong>. 그래서 첫 층을 3×3, stride 1로 바꾸고 maxpool은 빼서(Identity) 해상도를 지켜요.
      </>
    ),
    code: `m = models.resnet18(num_classes=10)
m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
m.maxpool = nn.Identity()
return m`,
    highlight: [2, 3],
    render: (t, c) => {
      const swap = seg(t, 0.25, 0.6);
      const stages = [
        { label: 'layer1', res: 32 },
        { label: 'layer2', res: 16 },
        { label: 'layer3', res: 8 },
        { label: 'layer4', res: 4 }
      ];
      const oldRes = [8, 8, 4, 2, 1];
      return (
        <g>
          <Defs />
          <g opacity={1 - swap} transform={`translate(0 ${-swap * 24})`}>
            <Box x={14} y={46} w={78} h={46} label="7×7 conv" sub="stride 2" subFs={11} fs={13} color={c.neg} c={c} />
            <Box x={100} y={46} w={64} h={46} label="maxpool" fs={12.5} color={c.neg} c={c} />
          </g>
          <g opacity={swap} transform={`translate(0 ${(1 - swap) * 18})`}>
            <Box x={14} y={46} w={78} h={46} label="3×3 conv" sub="stride 1" subFs={11} fs={13} color={c.pos} glow={swap >= 1} c={c} />
            <Box x={100} y={46} w={64} h={46} label="Identity" fs={12.5} color="var(--viz-axis)" dashed c={c} />
          </g>
          {stages.map((s, i) => {
            const res = swap > 0.5 ? s.res : oldRes[i + 1];
            const h = 18 + res * 2.4;
            const x = 186 + i * 62;
            return (
              <g key={s.label} opacity={seg(t, 0.05 + i * 0.05, 0.25 + i * 0.05)}>
                <rect x={x} y={70 - h / 2 + 20} width={48} height={h} rx={8} fill={soft(c.series[1], 0.2)} stroke={c.series[1]} strokeWidth={2} />
                <Txt x={x + 24} y={70 + h / 2 + 38} size={12} weight={750} color="var(--text-2)">
                  {s.label}
                </Txt>
                <Txt x={x + 24} y={94} size={13} weight={850}>
                  {res}²
                </Txt>
              </g>
            );
          })}
          <Box x={436} y={74} w={34} h={34} label="10" fs={13} color={c.series[3]} t={seg(t, 0.3, 0.45)} c={c} />
          <Txt x={453} y={128} size={11.5} weight={700} color="var(--text-2)">
            fc
          </Txt>
          <g opacity={seg(t, 0.65, 0.9)}>
            <rect x={40} y={204} width={400} height={66} rx={16} fill={soft(swap > 0.5 ? c.pos : c.neg, 0.12)} stroke={soft(swap > 0.5 ? c.pos : c.neg, 0.6)} />
            <Txt x={240} y={231} size={14.5} weight={800}>
              32×32 사진이 layer1에서도 32×32 그대로!
            </Txt>
            <Txt x={240} y={254} size={12.5} weight={650} color="var(--text-2)">
              (원래 구조였다면 layer1부터 8×8 — 디테일이 사라져요)
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '한 스텝의 순서: scale → backward → unscale → clip → step',
    text: (
      <>
        AMP(혼합 정밀도)는 순전파를 fp16으로 돌려 메모리를 절반으로 줄여요. 대신 작은 기울기가 0으로 사라질 수 있어 <strong>손실을 키워서(scale) 역전파</strong>한 뒤, 클리핑 전에 <strong>원래 크기로 되돌려야(unscale)</strong> 해요. 순서가 바뀌면 클리핑 기준(5.0)이 엉망이 돼요.
      </>
    ),
    code: `optimizer.zero_grad(set_to_none=True)
with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
    loss = criterion(model(x), y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                       # 클리핑 전 언스케일 필수
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
scaler.step(optimizer)
scaler.update()
scheduler.step()                                 # OneCycle은 스텝 단위`,
    highlight: [4, 5, 6, 7],
    render: (t, c) => {
      const items = [
        { label: 'zero_grad', col: c.series[5] },
        { label: '순전파 (fp16)', col: c.series[0] },
        { label: '손실', col: c.series[2] },
        { label: 'scale·backward', col: c.series[2] },
        { label: 'unscale_', col: c.series[4] },
        { label: 'clip (5.0)', col: c.series[4] },
        { label: 'step·update', col: c.series[1] },
        { label: 'scheduler', col: c.series[3] }
      ];
      const pos = (i: number) => {
        const row = Math.floor(i / 4);
        const col = row === 0 ? i % 4 : 3 - (i % 4);
        return { x: 18 + col * 114, y: 50 + row * 108 };
      };
      const active = Math.min(items.length - 1, Math.floor(t * items.length * 1.02));
      return (
        <g>
          <Defs />
          {items.slice(0, -1).map((_, i) => {
            const a = pos(i);
            const b = pos(i + 1);
            const sameRow = a.y === b.y;
            return sameRow ? (
              <Arrow key={i} x1={a.x + (b.x > a.x ? 100 : 0)} y1={a.y + 22} x2={b.x + (b.x > a.x ? 0 : 100)} y2={b.y + 22} color="var(--viz-axis)" width={2} t={seg(t, i * 0.11, i * 0.11 + 0.12)} />
            ) : (
              <Arrow key={i} x1={a.x + 50} y1={a.y + 46} x2={b.x + 50} y2={b.y - 2} color="var(--viz-axis)" width={2} t={seg(t, i * 0.11, i * 0.11 + 0.12)} bend={-30} />
            );
          })}
          {items.map((it, i) => {
            const p = pos(i);
            const on = i === active;
            return (
              <g key={it.label}>
                <Box x={p.x} y={p.y} w={100} h={44} label={it.label} fs={it.label.length > 11 ? 11.5 : 12.5} color={it.col} t={seg(t, i * 0.11, i * 0.11 + 0.1)} glow={on && t < 0.999} c={c} />
                <Txt x={p.x + 8} y={p.y - 6} size={11} weight={800} anchor="start" color="var(--text-2)" opacity={seg(t, i * 0.11, i * 0.11 + 0.1)}>
                  {i + 1}
                </Txt>
              </g>
            );
          })}
          <g opacity={seg(t, 0.25, 0.45)}>
            <rect x={124} y={30} width={230} height={74} rx={14} fill="none" stroke={c.series[0]} strokeDasharray="5 5" opacity={0.6} />
            <Txt x={300} y={24} size={11.5} weight={800} color={c.series[0]}>
              autocast (fp16) 구간
            </Txt>
          </g>
          <g opacity={seg(t, 0.55, 0.75)}>
            <Txt x={240} y={238} size={13.5} weight={800}>
              4 키워서 역전파 → 5 되돌리고 → 6 자르고 → 7 한 걸음
            </Txt>
            <Txt x={240} y={262} size={12.5} weight={650} color="var(--text-2)">
              OneCycle 스케줄러는 에포크가 아니라 스텝마다 한 번씩!
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: 'OneCycle 학습률: 천천히 올렸다가 길게 내린다',
    text: (
      <>
        학습률을 처음부터 0.1로 주면 무작위 가중치가 크게 흔들려요. OneCycle은 <strong>앞 10% 동안 0.004 → 0.1로 올리고(웜업)</strong>, 나머지 동안 코사인 곡선을 따라 거의 0까지 천천히 내려요. 그래서 총 스텝 수(에포크 × 배치 수)를 미리 알려 줘야 해요.
      </>
    ),
    code: `optimizer = torch.optim.SGD(model.parameters(), lr=CFG["lr"],
                            momentum=0.9, weight_decay=CFG["weight_decay"],
                            nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=CFG["lr"],
    total_steps=CFG["epochs"] * len(train_dl),
    pct_start=0.1,               # 앞 10%는 웜업
)`,
    highlight: [4, 5, 6, 7],
    render: (t, c) => {
      const x0 = 60;
      const y0 = 40;
      const w = 380;
      const h = 190;
      const sx = (v: number) => x0 + v * w;
      const sy = (v: number) => y0 + h - (v / 0.105) * h;
      return (
        <g>
          <Defs />
          <rect x={x0} y={y0} width={w * 0.1} height={h} fill={soft(c.series[4], 0.14)} opacity={seg(t, 0.2, 0.4)} />
          <Txt x={x0 + w * 0.05} y={y0 + 16} size={11.5} weight={800} color={c.series[4]} opacity={seg(t, 0.2, 0.4)}>
            웜업
          </Txt>
          <Curve x={x0} y={y0} w={w} h={h} fn={oneCycle} xDomain={[0, 1]} yDomain={[0, 0.105]} t={seg(t, 0.05, 0.9)} color={c.series[0]} xLabel="학습 진행 (전체 스텝)" yLabel="학습률" n={140} />
          <Txt x={x0 - 8} y={sy(0.1) + 4} size={11.5} weight={700} anchor="end" color="var(--text-2)">
            0.1
          </Txt>
          <Txt x={x0 - 8} y={sy(0) + 4} size={11.5} weight={700} anchor="end" color="var(--text-2)">
            0
          </Txt>
          {t > 0.3 && (
            <g>
              <circle cx={sx(0.1)} cy={sy(0.1)} r={6} fill={c.series[0]} stroke="var(--surface-solid)" strokeWidth={2} />
              <Txt x={sx(0.1) + 10} y={sy(0.1) - 8} size={12.5} weight={800} anchor="start">
                최고점 0.1 (10% 지점)
              </Txt>
            </g>
          )}
          {t > 0.9 && (
            <Txt x={sx(0.96)} y={sy(0.004) - 12} size={12.5} weight={800} anchor="end">
              끝에서는 거의 0 → 미세 조정
            </Txt>
          )}
          <Txt x={sx(0.1) + 8} y={sy(0) - 8} size={11.5} weight={700} anchor="start" color="var(--text-2)" opacity={seg(t, 0.1, 0.3)}>
            ← 0.004에서 출발
          </Txt>
        </g>
      );
    }
  },
  {
    title: '검증: eval()과 no_grad()는 다른 스위치',
    text: (
      <>
        <code>model.eval()</code>은 BatchNorm·Dropout을 <strong>추론 모드</strong>로 바꾸는 스위치, <code>torch.no_grad()</code>는 역전파용 <strong>계산 그래프를 저장하지 않는</strong> 스위치예요. 검증에는 둘 다 켜야 정확하고 가벼워요. 20 에포크면 테스트 정확도가 약 92%까지 올라가요.
      </>
    ),
    code: `def evaluate(model, loader, device):
    model.eval()                      # BatchNorm/Dropout을 추론 모드로
    correct = total = 0
    with torch.no_grad():             # 그래프 생성 중단 — eval()과는 별개 기능
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total`,
    highlight: [2, 4],
    render: (t, c) => {
      const s1 = seg(t, 0.05, 0.3);
      const s2 = seg(t, 0.25, 0.5);
      const sw = (x: number, y: number, on: number, col: string) => (
        <g transform={`translate(${x} ${y})`}>
          <rect width={52} height={28} rx={14} fill={on > 0.5 ? col : 'var(--surface-3)'} stroke={col} strokeWidth={1.5} />
          <circle cx={mix(14, 38, on)} cy={14} r={10} fill="var(--surface-solid)" />
        </g>
      );
      return (
        <g>
          <Defs />
          <g opacity={Math.min(1, t * 4)}>
            <rect x={14} y={24} width={214} height={92} rx={16} fill={soft(c.series[0], 0.1)} stroke={soft(c.series[0], 0.6)} />
            {sw(28, 40, s1, c.series[0])}
            <Txt x={92} y={60} size={14.5} weight={850} anchor="start" mono>
              model.eval()
            </Txt>
            <Txt x={28} y={92} size={12.5} weight={700} anchor="start" color="var(--text-2)">
              BatchNorm: 학습 때 모은 평균 사용
            </Txt>
            <Txt x={28} y={108} size={12.5} weight={700} anchor="start" color="var(--text-2)">
              Dropout: 끔 (모든 뉴런 사용)
            </Txt>
          </g>
          <g opacity={seg(t, 0.2, 0.35)}>
            <rect x={14} y={128} width={214} height={92} rx={16} fill={soft(c.series[1], 0.1)} stroke={soft(c.series[1], 0.6)} />
            {sw(28, 144, s2, c.series[1])}
            <Txt x={92} y={164} size={14.5} weight={850} anchor="start" mono>
              no_grad()
            </Txt>
            <Txt x={28} y={196} size={12.5} weight={700} anchor="start" color="var(--text-2)">
              계산 그래프 저장 안 함
            </Txt>
            <Txt x={28} y={212} size={12.5} weight={700} anchor="start" color="var(--text-2)">
              → 메모리↓ 속도↑
            </Txt>
          </g>
          <Curve x={268} y={40} w={190} h={170} fn={(e) => ACC[Math.min(19, Math.max(0, Math.round(e)))]} xDomain={[0, 19]} yDomain={[0.5, 1]} t={seg(t, 0.45, 0.95)} color={c.series[3]} xLabel="에포크" yLabel="테스트 정확도" n={19} marker={19} markerLabel="92%" markerAnchor="end" />
          <Txt x={240} y={270} size={12.5} weight={700} color="var(--text-2)" opacity={seg(t, 0.85, 1)}>
            두 스위치는 따로따로! 하나만 켜면 결과가 틀리거나 메모리를 낭비해요
          </Txt>
        </g>
      );
    }
  },
  {
    title: '체크포인트: 최고 기록일 때만 저장',
    text: (
      <>
        마지막 에포크가 최고 성능이라는 보장은 없어요. 그래서 <strong>지금까지의 최고 기록을 넘을 때만</strong> 저장해요 — 사실상의 조기 종료죠. 가중치만이 아니라 <strong>정확도·에포크·설정(CFG)</strong>까지 함께 저장해야 3주 뒤에도 무슨 모델인지 알 수 있어요.
      </>
    ),
    code: `if acc > best_acc:                                   # 최고 성능만 저장
    best_acc = acc
    torch.save({"model": model.state_dict(),
                "acc": acc, "epoch": epoch, "cfg": CFG}, CFG["ckpt_path"])`,
    highlight: [1, 3, 4],
    render: (t, c) => {
      const x0 = 30;
      const base = 196;
      const bw = 16;
      const shown = Math.round(seg(t, 0.05, 0.8) * ACC.length);
      let best = 0;
      return (
        <g>
          <Defs />
          <line x1={x0 - 4} x2={x0 + ACC.length * 21} y1={base} y2={base} stroke="var(--viz-axis)" strokeWidth={1.5} />
          {ACC.map((a, i) => {
            if (i >= shown) return null;
            const isBest = a > best;
            if (isBest) best = a;
            const h = (a - 0.5) * 300;
            const x = x0 + i * 21;
            return (
              <g key={i}>
                <rect x={x} y={base - h} width={bw} height={h} rx={4} fill={isBest ? c.series[1] : soft(c.series[1], 0.35)} />
                {isBest && (
                  <Emoji x={x + bw / 2} y={base - h - 8} size={14}>
                    💾
                  </Emoji>
                )}
              </g>
            );
          })}
          <Txt x={x0} y={base + 20} size={12} weight={700} anchor="start" color="var(--text-2)">
            에포크 1 → 20 (진한 막대 = 새 최고 기록 → 저장)
          </Txt>
          <g opacity={seg(t, 0.8, 1)}>
            <Txt x={240} y={246} size={13} weight={800}>
              ckpt_best.pt 안에 든 것
            </Txt>
            {['model', 'acc 0.921', 'epoch 19', 'cfg'].map((s, i) => (
              <Chip key={s} x={102 + i * 92} y={274} text={s} w={84} h={28} fs={12.5} color={c.series[3]} c={c} mono />
            ))}
          </g>
        </g>
      );
    }
  }
];

/** Code lab 02 walkthrough: a reusable PyTorch training pipeline. */
export default function Lab02TrainingPipeline() {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={1.9} />;
}
