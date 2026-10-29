import { CodeStory, type StoryStep } from '../core/story';
import type { VizColors } from '../core/colors';
import { Arrow, Box, Defs, Emoji, Txt, VB, mix, seg, soft } from './_labsA';

type B4 = [number, number, number, number];

function iou(a: B4, b: B4) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  const area = (r: B4) => (r[2] - r[0]) * (r[3] - r[1]);
  return inter / (area(a) + area(b) - inter + 1e-7);
}

/* Candidate detections around two pedestrians (+ one stray low-score box). */
const CANDS: { id: string; box: B4; score: number }[] = [
  { id: 'A', box: [64, 94, 118, 248], score: 0.98 },
  { id: 'E', box: [184, 94, 238, 248], score: 0.95 },
  { id: 'B', box: [58, 100, 122, 240], score: 0.91 },
  { id: 'F', box: [178, 104, 244, 244], score: 0.88 },
  { id: 'C', box: [70, 88, 126, 236], score: 0.86 },
  { id: 'G', box: [190, 86, 250, 236], score: 0.7 },
  { id: 'D', box: [52, 120, 110, 258], score: 0.62 },
  { id: 'H', box: [252, 42, 292, 88], score: 0.31 }
];

/** Run the lab's NMS and record each round (who was kept, who got removed with which IoU). */
function nmsRounds(thresh = 0.5) {
  let order = [...CANDS].sort((a, b) => b.score - a.score);
  const rounds: { keep: string; removed: { id: string; iou: number }[] }[] = [];
  while (order.length) {
    const best = order[0];
    const rest = order.slice(1);
    const removed = rest.filter((o) => iou(best.box, o.box) > thresh).map((o) => ({ id: o.id, iou: iou(best.box, o.box) }));
    rounds.push({ keep: best.id, removed });
    order = rest.filter((o) => iou(best.box, o.box) <= thresh);
  }
  return rounds;
}
const ROUNDS = nmsRounds();

function Person({ x, c }: { x: number; c: VizColors }) {
  return (
    <g>
      <circle cx={x} cy={114} r={14} fill={c.series[4]} />
      <rect x={x - 15} y={130} width={30} height={64} rx={12} fill={c.series[0]} opacity={0.85} />
      <path d={`M${x - 8} 192 L${x - 12} 244 M${x + 8} 192 L${x + 12} 244`} stroke={c.series[0]} strokeWidth={8} strokeLinecap="round" />
    </g>
  );
}

function Street({ c, t = 1 }: { c: VizColors; t?: number }) {
  return (
    <g opacity={Math.min(1, t * 3)}>
      <rect x={14} y={30} width={290} height={240} rx={14} fill={soft(c.series[3], 0.12)} stroke="var(--border)" />
      <rect x={14} y={232} width={290} height={38} fill={soft(c.series[1], 0.18)} />
      <Person x={91} c={c} />
      <Person x={211} c={c} />
      <Emoji x={272} y={74} size={22}>
        🌳
      </Emoji>
    </g>
  );
}

function BoxRect({ b, color, label, opacity = 1, width = 2.5, dashed = false }: { b: B4; color: string; label?: string; opacity?: number; width?: number; dashed?: boolean }) {
  return (
    <g opacity={opacity}>
      <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} rx={4} fill="none" stroke={color} strokeWidth={width} strokeDasharray={dashed ? '5 4' : undefined} />
      {label && (
        <g>
          <rect x={b[0]} y={b[1] - 17} width={label.length * 7.4 + 10} height={16} rx={4} fill={color} />
          <text x={b[0] + 5} y={b[1] - 5} style={{ fontSize: 11.5, fontWeight: 800, fill: '#fff', fontFamily: 'var(--font-sans)' }}>
            {label}
          </text>
        </g>
      )}
    </g>
  );
}

const steps: StoryStep[] = [
  {
    title: 'IoU: 겹친 넓이 ÷ 합친 넓이',
    text: (
      <>
        두 박스가 얼마나 겹치는지는 <strong>IoU = 교집합 / 합집합</strong>으로 재요. 교집합 박스의 왼쪽 위는 두 박스 왼쪽 위의 <strong>max</strong>, 오른쪽 아래는 <strong>min</strong>이에요. 안 겹치면 폭이 음수가 되니 <code>clamp(min=0)</code>으로 0을 만들어요.
      </>
    ),
    code: `lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])                # (N,M,2)
rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])                # (N,M,2)
wh = (rb - lt).clamp(min=0)                     # 겹치지 않으면 0
inter = wh[..., 0] * wh[..., 1]                 # (N,M)
return inter / (area1[:, None] + area2[None, :] - inter + 1e-7)`,
    highlight: [1, 2, 3, 5],
    render: (t, c) => {
      const gt: B4 = [60, 70, 196, 206];
      const p = seg(t, 0.1, 0.85);
      const ox = mix(250, 78, p);
      const oy = mix(150, 82, p);
      const pr: B4 = [ox, oy, ox + 130, oy + 130];
      const v = iou(gt, pr);
      const lt = [Math.max(gt[0], pr[0]), Math.max(gt[1], pr[1])];
      const rb = [Math.min(gt[2], pr[2]), Math.min(gt[3], pr[3])];
      const has = rb[0] > lt[0] && rb[1] > lt[1];
      return (
        <g>
          <Defs />
          <rect x={gt[0]} y={gt[1]} width={gt[2] - gt[0]} height={gt[3] - gt[1]} fill={soft(c.series[1], 0.14)} stroke={c.series[1]} strokeWidth={3} rx={4} />
          <rect x={pr[0]} y={pr[1]} width={130} height={130} fill={soft(c.series[3], 0.12)} stroke={c.series[3]} strokeWidth={3} strokeDasharray="7 4" rx={4} />
          {has && <rect x={lt[0]} y={lt[1]} width={rb[0] - lt[0]} height={rb[1] - lt[1]} fill={c.series[2]} opacity={0.5} />}
          {has && (
            <g>
              <circle cx={lt[0]} cy={lt[1]} r={5} fill={c.series[2]} />
              <circle cx={rb[0]} cy={rb[1]} r={5} fill={c.series[2]} />
              <Txt x={lt[0] + 8} y={lt[1] + 18} size={11.5} weight={850} anchor="start">
                lt = max
              </Txt>
              <Txt x={rb[0] - 8} y={rb[1] - 10} size={11.5} weight={850} anchor="end">
                rb = min
              </Txt>
            </g>
          )}
          <Txt x={gt[0] + 4} y={gt[1] - 8} size={12.5} weight={850} anchor="start" color={c.series[1]}>
            정답 박스
          </Txt>
          <Txt x={pr[0] + 126} y={pr[1] + 148} size={12.5} weight={850} anchor="end" color={c.series[3]}>
            예측 박스
          </Txt>
          <g>
            <rect x={356} y={40} width={110} height={92} rx={16} fill={soft(c.series[2], 0.12)} stroke={soft(c.series[2], 0.6)} />
            <Txt x={411} y={66} size={13} weight={800}>
              IoU
            </Txt>
            <Txt x={411} y={108} size={30} weight={900} color={c.series[2]}>
              {v.toFixed(2)}
            </Txt>
          </g>
          <Txt x={411} y={160} size={12} weight={700} color="var(--text-2)">
            {v > 0.5 ? '0.5 넘음 → 같은 물체!' : v > 0 ? '조금 겹침' : '안 겹침 = 0'}
          </Txt>
        </g>
      );
    }
  },
  {
    title: '모델은 한 물체에 박스를 잔뜩 낸다',
    text: (
      <>
        탐지 모델은 한 사람 주변에 <strong>비슷한 박스를 여러 개</strong> 쏟아내요. 각각 "사람일 확률" 점수가 붙어 있죠. NMS는 먼저 이 박스들을 <strong>점수 높은 순서로 줄 세우는 것</strong>부터 시작해요.
      </>
    ),
    code: `def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5):
    """Non-Maximum Suppression: 점수 높은 박스부터 채택, 많이 겹치는 후보는 제거."""
    order = scores.argsort(descending=True)
    keep = []`,
    highlight: [3],
    render: (t, c) => (
      <g>
        <Defs />
        <Street c={c} t={t} />
        {CANDS.map((d, i) => (
          <BoxRect key={d.id} b={d.box} color={c.series[2]} opacity={seg(t, 0.1 + i * 0.07, 0.2 + i * 0.07)} width={2} />
        ))}
        <Txt x={392} y={40} size={13} weight={850}>
          점수 순서
        </Txt>
        {CANDS.map((d, i) => (
          <g key={`r${d.id}`} opacity={seg(t, 0.35 + i * 0.06, 0.45 + i * 0.06)}>
            <rect x={330} y={50 + i * 26} width={124} height={22} rx={7} fill={soft(c.series[2], 0.12 + d.score * 0.2)} />
            <Txt x={342} y={66 + i * 26} size={12.5} weight={800} anchor="start" mono>
              {`${i + 1}. 박스 ${d.id}`}
            </Txt>
            <Txt x={446} y={66 + i * 26} size={12.5} weight={850} anchor="end" mono>
              {d.score.toFixed(2)}
            </Txt>
          </g>
        ))}
      </g>
    )
  },
  {
    title: 'NMS: 최고 점수는 남기고, 많이 겹치는 후보는 지운다',
    text: (
      <>
        1등 박스를 채택하고, 그 박스와 <strong>IoU가 0.5를 넘는 후보는 "같은 사람"</strong>으로 보고 지워요. 남은 것 중 다시 1등을 채택… 후보가 없어질 때까지 반복해요. 겹치지 않는 박스는 점수가 낮아도 살아남으니, 나중에 점수 기준(0.8)으로 한 번 더 걸러요.
      </>
    ),
    code: `while order.numel() > 0:
    best = order[0]
    keep.append(best.item())
    if order.numel() == 1:
        break
    ious = box_iou(boxes[best].unsqueeze(0), boxes[order[1:]]).squeeze(0)
    order = order[1:][ious <= iou_thresh]       # 임계값 이하만 생존`,
    highlight: [2, 3, 6, 7],
    render: (t, c) => {
      const nR = ROUNDS.length;
      const span = 0.85 / nR;
      const state = new Map<string, { kept?: number; removed?: number; iou?: number }>();
      ROUNDS.forEach((r, k) => {
        const p = seg(t, 0.05 + k * span, 0.05 + (k + 1) * span);
        state.set(r.keep, { kept: p });
        r.removed.forEach((o) => state.set(o.id, { removed: p, iou: o.iou }));
      });
      const keptIds = ROUNDS.filter((_, k) => seg(t, 0.05 + k * span, 0.05 + (k + 1) * span) > 0.3).map((r) => r.keep);
      return (
        <g>
          <Defs />
          <Street c={c} />
          {CANDS.map((d) => {
            const s = state.get(d.id) ?? {};
            const kept = (s.kept ?? 0) > 0.3;
            const gone = s.removed ?? 0;
            const color = kept ? (d.score < 0.8 ? '#9aa0b4' : c.pos) : c.series[2];
            return (
              <g key={d.id}>
                <BoxRect b={d.box} color={color} opacity={1 - gone * 0.85} width={kept ? 3.5 : 2} dashed={gone > 0.4} label={kept ? `${d.id} ${d.score.toFixed(2)}` : undefined} />
                {gone > 0.3 && gone < 1 && s.iou !== undefined && (
                  <Txt x={(d.box[0] + d.box[2]) / 2} y={d.box[3] + 16} size={11.5} weight={850} color={c.neg} opacity={1 - gone}>
                    IoU {s.iou.toFixed(2)} ✗
                  </Txt>
                )}
              </g>
            );
          })}
          <g>
            <Txt x={390} y={50} size={13} weight={850}>
              keep =
            </Txt>
            {keptIds.map((id, i) => (
              <g key={id}>
                <rect x={340 + (i % 3) * 38} y={64} width={32} height={30} rx={8} fill={CANDS.find((d) => d.id === id)!.score < 0.8 ? '#9aa0b4' : c.pos} />
                <Txt x={356 + (i % 3) * 38} y={84} size={14} weight={900} color="#fff">
                  {id}
                </Txt>
              </g>
            ))}
          </g>
          <g opacity={seg(t, 0.85, 1)}>
            <Txt x={392} y={140} size={12} weight={750} color="var(--text-2)">
              H는 아무와도 안 겹쳐서
            </Txt>
            <Txt x={392} y={158} size={12} weight={750} color="var(--text-2)">
              살아남았지만 점수 0.31
            </Txt>
            <Txt x={392} y={182} size={12.5} weight={850} color={c.series[2]}>
              → scores &gt; 0.8로 탈락
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: 'Faster R-CNN: 헤드만 갈아끼운다 (91 → 2 클래스)',
    text: (
      <>
        COCO로 사전학습된 Faster R-CNN은 백본(ResNet-50+FPN)이 특징을 뽑고, <strong>RPN이 "물체가 있을 법한 영역"</strong>을 제안하고, ROI 헤드가 클래스와 박스를 다듬어요. 우리는 마지막 <code>box_predictor</code>만 <strong>배경+사람 2클래스</strong>로 바꿔 파인튜닝해요.
      </>
    ),
    code: `model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
# 헤드 교체: COCO 91 클래스 → 우리 2 클래스(배경 + 사람)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)`,
    highlight: [3, 4],
    render: (t, c) => {
      const swap = seg(t, 0.55, 0.85);
      const blocks = [
        { x: 10, w: 70, label: '이미지', sub: '' },
        { x: 96, w: 92, label: '백본', sub: 'ResNet50+FPN' },
        { x: 204, w: 76, label: 'RPN', sub: '후보 영역' },
        { x: 296, w: 70, label: 'ROI 헤드', sub: '' }
      ];
      return (
        <g>
          <Defs />
          {blocks.map((b, i) => (
            <g key={b.label}>
              <Box x={b.x} y={96} w={b.w} h={58} label={b.label} sub={b.sub || undefined} subFs={10.5} fs={13.5} color={i === 0 ? c.series[3] : c.series[1]} t={seg(t, i * 0.1, i * 0.1 + 0.2)} c={c} />
              {i < blocks.length - 1 && <Arrow x1={b.x + b.w + 2} y1={125} x2={blocks[i + 1].x - 4} y2={125} color="var(--viz-axis)" width={2} t={seg(t, i * 0.1 + 0.1, i * 0.1 + 0.25)} />}
            </g>
          ))}
          <Arrow x1={368} y1={125} x2={384} y2={125} color="var(--viz-axis)" width={2} t={seg(t, 0.4, 0.5)} />
          <g opacity={1 - swap} transform={`translate(0 ${-swap * 30})`}>
            <Box x={388} y={96} w={84} h={58} label="91 클래스" sub="COCO" subFs={11} fs={13} color={c.neg} c={c} />
          </g>
          <g opacity={swap} transform={`translate(0 ${(1 - swap) * 30})`}>
            <Box x={388} y={96} w={84} h={58} label="2 클래스" sub="배경·사람" subFs={11} fs={13} color={c.pos} glow={swap >= 1} c={c} />
          </g>
          <Txt x={430} y={84} size={11.5} weight={800} color="var(--text-2)">
            box_predictor
          </Txt>
          <g opacity={seg(t, 0.3, 0.55)}>
            <rect x={204} y={176} width={76} height={50} rx={10} fill={soft(c.series[2], 0.08)} stroke={soft(c.series[2], 0.5)} strokeDasharray="4 4" />
            {[0, 1, 2, 3, 4].map((k) => (
              <rect key={k} x={212 + k * 9} y={184 + (k % 2) * 8} width={26} height={22} rx={3} fill="none" stroke={c.series[2]} strokeWidth={1.2} opacity={0.8} />
            ))}
            <Txt x={242} y={244} size={11.5} weight={700} color="var(--text-2)">
              "여기 뭔가 있다" 후보들
            </Txt>
          </g>
          <Txt x={240} y={284} size={12.5} weight={800} opacity={seg(t, 0.85, 1)}>
            몸통의 사전학습 지식은 그대로, 마지막 분류 칸만 우리 문제에 맞게!
          </Txt>
        </g>
      );
    }
  },
  {
    title: '학습: 손실이 딕셔너리로 나온다',
    text: (
      <>
        학습 모드의 탐지 모델은 예측 대신 <strong>손실 4가지가 든 딕셔너리</strong>를 돌려줘요 — RPN의 "물체냐/아니냐"와 박스 위치, ROI 헤드의 클래스와 박스 위치. 이걸 <strong>모두 더해</strong> 한 번에 역전파해요.
      </>
    ),
    code: `loss_dict = model(imgs, targets)
loss = sum(loss_dict.values())
opt.zero_grad(set_to_none=True)
loss.backward()
opt.step()`,
    highlight: [1, 2],
    render: (t, c) => {
      const parts = [
        { k: 'loss_classifier', v: 0.21, col: c.series[0] },
        { k: 'loss_box_reg', v: 0.18, col: c.series[1] },
        { k: 'loss_objectness', v: 0.05, col: c.series[3] },
        { k: 'loss_rpn_box_reg', v: 0.03, col: c.series[4] }
      ];
      const go = seg(t, 0.45, 0.85);
      let acc = 0;
      const total = parts.reduce((a, p) => a + p.v, 0);
      const scale = 520;
      const sx0 = 120;
      const sScale = 0.55;
      return (
        <g>
          <Defs />
          {parts.map((p, i) => {
            const x0 = 196;
            const y0 = 40 + i * 38;
            const w = p.v * scale * seg(t, i * 0.08, i * 0.08 + 0.25);
            const tx = mix(x0, sx0 + acc * scale * sScale, go);
            const ty = mix(y0, 232, go);
            const ww = mix(w, p.v * scale * sScale, go);
            acc += p.v;
            return (
              <g key={p.k}>
                <Txt x={186} y={y0 + 15} size={12.5} weight={750} anchor="end" mono>
                  {p.k}
                </Txt>
                <rect x={x0} y={y0} width={w} height={22} rx={5} fill={p.col} />
                <Txt x={x0 + w + 8} y={y0 + 15} size={12} weight={800} anchor="start">
                  {p.v.toFixed(2)}
                </Txt>
                {go > 0 && <rect x={tx} y={ty} width={ww} height={22} rx={5} fill={p.col} opacity={0.4 + 0.6 * go} />}
              </g>
            );
          })}
          <g opacity={seg(t, 0.8, 1)}>
            <Txt x={sx0} y={222} size={13} weight={850} anchor="start">
              loss = sum(...) = {total.toFixed(2)}
            </Txt>
            <Arrow x1={sx0 + total * scale * sScale + 10} y1={243} x2={410} y2={243} color={c.series[2]} t={seg(t, 0.85, 1)} label="backward()" labelDy={-10} flow />
          </g>
        </g>
      );
    }
  },
  {
    title: '추론: 박스·라벨·점수, 그리고 YOLO',
    text: (
      <>
        <code>eval()</code> 모드에서는 이미지만 넣으면 <strong>boxes · labels · scores</strong>가 나와요. 점수 0.8 이상만 남기면 두 사람이 깔끔하게 잡혀요. 빠른 실무용으로는 ultralytics YOLO가 사실상 표준 — 추론이 <code>model("street.jpg")</code> 한 줄이에요.
      </>
    ),
    code: `model.eval()
img, _ = ds[0]
with torch.no_grad():
    pred = model([img.to(device)])[0]
keep = pred["scores"] > 0.8`,
    highlight: [4, 5],
    render: (t, c) => {
      const kept = CANDS.filter((d) => ['A', 'E'].includes(d.id));
      const stray = CANDS.find((d) => d.id === 'H')!;
      return (
        <g>
          <Defs />
          <Street c={c} />
          {kept.map((d, i) => (
            <BoxRect key={d.id} b={d.box} color={c.pos} width={3.5} label={`사람 ${d.score.toFixed(2)}`} opacity={seg(t, 0.1 + i * 0.15, 0.3 + i * 0.15)} />
          ))}
          <BoxRect b={stray.box} color="#9aa0b4" dashed opacity={1 - seg(t, 0.4, 0.7) * 0.6} />
          {t > 0.4 && (
            <g opacity={seg(t, 0.4, 0.6)} stroke={c.neg} strokeWidth={3} strokeLinecap="round">
              <line x1={stray.box[0] + 4} y1={stray.box[1] + 4} x2={stray.box[2] - 4} y2={stray.box[3] - 4} />
              <line x1={stray.box[2] - 4} y1={stray.box[1] + 4} x2={stray.box[0] + 4} y2={stray.box[3] - 4} />
            </g>
          )}
          <g opacity={seg(t, 0.45, 0.65)}>
            <rect x={318} y={226} width={150} height={40} rx={12} fill={soft(c.neg, 0.1)} stroke={soft(c.neg, 0.5)} />
            <Txt x={393} y={251} size={12.5} weight={850} color={c.neg}>
              나무 박스 H: 0.31 &lt; 0.8 탈락
            </Txt>
          </g>
          <g opacity={seg(t, 0.6, 0.9)}>
            <rect x={318} y={60} width={150} height={150} rx={16} fill={soft(c.series[3], 0.1)} stroke={soft(c.series[3], 0.6)} />
            <Txt x={393} y={88} size={13.5} weight={850}>
              YOLO 한 줄 ⚡
            </Txt>
            <Txt x={393} y={118} size={11.5} weight={750} mono>
              model = YOLO(...)
            </Txt>
            <Txt x={393} y={140} size={11.5} weight={750} mono>
              model("street.jpg")
            </Txt>
            <Txt x={393} y={172} size={11.5} weight={700} color="var(--text-2)">
              학습·평가·내보내기도
            </Txt>
            <Txt x={393} y={190} size={11.5} weight={700} color="var(--text-2)">
              모두 한 줄씩
            </Txt>
          </g>
        </g>
      );
    }
  }
];

/** Code lab 05 walkthrough: IoU, NMS, Faster R-CNN head swap, loss dict, inference. */
export default function Lab05Detection() {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={2.2} />;
}
