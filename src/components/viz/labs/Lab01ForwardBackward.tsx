import { CodeStory, type StoryStep } from '../core/story';
import { rng, softmax } from '../core/math';
import { Arrow, Bars, Box, Cells, Chip, Curve, Defs, Tensor, Txt, VB, mix, seg, soft } from './_labsA';

/* A 12×12 "7" (the real MNIST image is 28×28; drawn smaller so the cells are visible). */
const SEVEN = [
  '............',
  '.##########.',
  '.##########.',
  '........###.',
  '.......###..',
  '......###...',
  '.....###....',
  '.....##.....',
  '....###.....',
  '....##......',
  '...###......',
  '............'
];
const INK = SEVEN.map((row) => [...row].map((ch) => (ch === '#' ? 1 : 0)));

const R = rng(7);
const Z1 = Array.from({ length: 5 }, () => Array.from({ length: 5 }, () => R.normal() * 0.8));
const LOGITS = [-1.2, 0.3, 0.8, 1.1, -0.5, 0.2, -0.8, 4.2, 0.9, 1.5];
const PROBS = softmax(LOGITS);

const steps: StoryStep[] = [
  {
    title: '그림 한 장을 숫자 784개로 편다',
    text: (
      <>
        MNIST 한 장은 28×28 = <strong>784칸</strong>짜리 밝기 그림이에요. <code>reshape(-1, 784)</code>로 한 줄로 펴면 숫자 784개짜리 벡터가 되고, 128장을 묶으면 행렬 <strong>X (128, 784)</strong>가 돼요. 정답 숫자는 <code>one_hot</code>으로 10칸 중 한 칸만 1인 벡터가 돼요.
      </>
    ),
    code: `x_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_tr = tr.targets.numpy()

def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out`,
    highlight: [1, 6],
    render: (t, c) => {
      const cell = 10;
      const gx = 26;
      const gy = 34;
      const sx = 40;
      const sy = 218;
      const sw = 400 / 144;
      const ghost = [];
      const cells = [];
      for (let r = 0; r < 12; r++)
        for (let k = 0; k < 12; k++) {
          const v = INK[r][k];
          ghost.push(
            <rect key={`g${r}-${k}`} x={gx + k * (cell + 1)} y={gy + r * (cell + 1)} width={cell} height={cell} rx={2} fill={v ? soft(c.series[0], 0.35) : soft(c.series[0], 0.07)} />
          );
          const p = seg(t, 0.12 + r * 0.04, 0.36 + r * 0.04);
          const idx = r * 12 + k;
          cells.push(
            <rect
              key={idx}
              x={mix(gx + k * (cell + 1), sx + idx * sw, p)}
              y={mix(gy + r * (cell + 1), sy, p)}
              width={mix(cell, Math.max(1.2, sw - 0.6), p)}
              height={mix(cell, 24, p)}
              rx={p > 0.5 ? 0.6 : 2}
              opacity={Math.min(1, t * 6)}
              fill={v ? c.series[0] : soft(c.series[0], 0.1)}
            />
          );
        }
      return (
        <g>
          <Defs />
          <Txt x={92} y={22} size={13} weight={750} color="var(--text-2)">
            손글씨 한 장 (28×28)
          </Txt>
          {ghost}
          {cells}
          <Txt x={40} y={sy - 10} size={13} weight={750} color="var(--text-2)" anchor="start" opacity={seg(t, 0.55, 0.8)}>
            한 줄로 편 784개 숫자 (0 = 검정, 1 = 흰색)
          </Txt>
          <Arrow x1={168} y1={96} x2={300} y2={92} color={c.series[0]} t={seg(t, 0.55, 0.75)} label="reshape(-1, 784)" flow />
          <Tensor x={316} y={58} w={120} h={66} label="X" shape="(128, 784)" color={c.series[0]} t={seg(t, 0.62, 0.85)} c={c} />
          <Txt x={376} y={162} size={12.5} weight={650} color="var(--text-2)" opacity={seg(t, 0.7, 0.9)}>
            128장을 한 번에 (배치)
          </Txt>
          <Txt x={26} y={278} size={13} weight={750} color="var(--text-2)" anchor="start" opacity={seg(t, 0.8, 1)}>
            정답 7 → one_hot
          </Txt>
          {Array.from({ length: 10 }, (_, d) => (
            <Chip key={d} x={182 + d * 28} y={273} text={d === 7 ? '1' : '0'} w={24} h={24} fs={13} color={c.series[3]} active={d === 7} t={seg(t, 0.82 + d * 0.012, 1)} c={c} mono />
          ))}
        </g>
      );
    }
  },
  {
    title: '1층: X·W₁ + b₁, 그리고 ReLU',
    text: (
      <>
        행렬 곱 한 번으로 128장 모두의 1층 점수 <strong>z₁ (128, 256)</strong>을 한꺼번에 계산해요. 그다음 ReLU가 <strong>음수는 0으로, 양수는 그대로</strong> 통과시켜요. 그래서 <code>W₁</code>을 He 초기화(분산 2/fan_in)로 시작하죠 — 절반이 꺼져도 신호 크기가 유지되게요.
      </>
    ),
    code: `# He 초기화: ReLU가 절반을 죽이므로 분산을 2/fan_in으로.
self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, d_hidden)).astype(np.float32)
self.b1 = np.zeros(d_hidden, dtype=np.float32)

self.z1 = X @ self.W1 + self.b1
self.h = np.maximum(0, self.z1)          # ReLU`,
    highlight: [5, 6],
    render: (t, c) => {
      const relu = seg(t, 0.55, 0.9);
      return (
        <g>
          <Defs />
          <Tensor x={14} y={92} w={66} h={58} label="X" shape="(128, 784)" color={c.series[0]} t={seg(t, 0, 0.25)} c={c} />
          <Txt x={96} y={128} size={22} weight={800} opacity={seg(t, 0.15, 0.3)}>
            ×
          </Txt>
          <Box x={110} y={78} w={56} h={86} label="W₁" sub="784×256" subFs={11} color={c.series[1]} t={seg(t, 0.15, 0.4)} c={c} />
          <Txt x={184} y={128} size={22} weight={800} opacity={seg(t, 0.3, 0.45)}>
            =
          </Txt>
          <g opacity={seg(t, 0.3, 0.5)}>
            <Txt x={232} y={84} size={15} weight={850}>
              z₁
            </Txt>
            <Cells x={202} y={94} rows={5} cols={5} cell={10} gap={2} color={c.series[1]} t={seg(t, 0.3, 0.55)} value={(r, k) => Math.min(1, Math.abs(Z1[r][k]))} />
            {Z1.map((row, r) =>
              row.map((v, k) =>
                v < 0 ? <rect key={`n${r}${k}`} x={202 + k * 12} y={94 + r * 12} width={10} height={10} rx={2.5} fill={c.neg} opacity={0.75 * seg(t, 0.4, 0.55)} /> : null
              )
            )}
            <Txt x={232} y={176} size={12} weight={650} color="var(--text-2)" mono>
              (128, 256)
            </Txt>
          </g>
          <Arrow x1={270} y1={124} x2={360} y2={124} color={c.series[2]} t={seg(t, 0.45, 0.65)} flow />
          <g opacity={seg(t, 0.45, 0.62)} transform="translate(292 70)">
            <rect x={-6} y={-6} width={50} height={40} rx={8} fill="var(--surface-solid)" stroke={c.series[2]} strokeWidth={1.5} />
            <path d="M0 26 L19 26 L38 4" fill="none" stroke={c.series[2]} strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
            <Txt x={19} y={50} size={12.5} weight={800}>
              ReLU
            </Txt>
          </g>
          <g opacity={seg(t, 0.55, 0.75)}>
            <Txt x={404} y={84} size={15} weight={850}>
              h
            </Txt>
            {Z1.map((row, r) =>
              row.map((v, k) => {
                const off = v < 0;
                return (
                  <rect
                    key={`h${r}${k}`}
                    x={374 + k * 12}
                    y={94 + r * 12}
                    width={10}
                    height={10}
                    rx={2.5}
                    fill={off ? (relu > 0.5 ? 'var(--surface-3)' : c.neg) : soft(c.series[1], 0.15 + 0.8 * Math.min(1, v))}
                    stroke={off && relu > 0.5 ? 'var(--border-strong)' : 'none'}
                  />
                );
              })
            )}
            <Txt x={404} y={176} size={12} weight={650} color="var(--text-2)" mono>
              (128, 256)
            </Txt>
          </g>
          <g opacity={seg(t, 0.8, 1)}>
            <rect x={60} y={214} width={360} height={52} rx={14} fill={soft(c.series[2], 0.12)} stroke={soft(c.series[2], 0.6)} />
            <Txt x={240} y={237} size={14} weight={750}>
              빨간 칸(음수) → 회색 칸(0): 신호가 꺼져요
            </Txt>
            <Txt x={240} y={256} size={12.5} weight={650} color="var(--text-2)">
              살아남은 양수만 다음 층으로 가요
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '2층과 소프트맥스: 점수를 확률로',
    text: (
      <>
        <code>h @ W₂</code>로 숫자 0~9 각각의 점수(로짓) 10개가 나와요. 소프트맥스는 점수를 <strong>합이 1인 확률</strong>로 바꿔요. 계산 전에 최댓값을 빼는 건 <code>exp</code>가 너무 커져 넘치지 않게 하는 안전장치예요 — 결과는 똑같아요.
      </>
    ),
    code: `self.z2 = self.h @ self.W2 + self.b2
# 수치 안정 소프트맥스: 최댓값을 빼도 결과는 동일하다.
z = self.z2 - self.z2.max(axis=1, keepdims=True)
e = np.exp(z)
self.probs = e / e.sum(axis=1, keepdims=True)
return self.probs`,
    highlight: [1, 3, 4, 5],
    render: (t, c) => {
      const base = 136;
      const lg = seg(t, 0.1, 0.5);
      const pr = seg(t, 0.55, 0.95);
      return (
        <g>
          <Defs />
          <Txt x={110} y={30} size={14} weight={800}>
            점수(로짓) z₂
          </Txt>
          <line x1={14} x2={206} y1={base} y2={base} stroke="var(--viz-axis)" strokeWidth={1.5} />
          {LOGITS.map((v, d) => {
            const h = v * 22 * lg;
            return (
              <g key={d}>
                <rect x={16 + d * 19} y={h >= 0 ? base - h : base} width={15} height={Math.abs(h)} rx={3} fill={d === 7 ? c.series[0] : v >= 0 ? soft(c.series[0], 0.55) : soft(c.neg, 0.7)} />
                <Txt x={23.5 + d * 19} y={base + 46} size={11.5} weight={d === 7 ? 850 : 650} color={d === 7 ? 'var(--text)' : 'var(--text-2)'}>
                  {d}
                </Txt>
              </g>
            );
          })}
          <Arrow x1={214} y1={130} x2={262} y2={130} color={c.series[2]} t={seg(t, 0.45, 0.6)} label="softmax" labelDy={-10} flow />
          <Txt x={364} y={30} size={14} weight={800}>
            확률 (합 = 1)
          </Txt>
          <Bars
            x={270}
            y={52}
            w={190}
            h={150}
            values={PROBS}
            labels={PROBS.map((_, d) => d)}
            color={c.series[3]}
            hiColor={c.series[3]}
            highlight={7}
            t={pr}
            max={1}
            fs={11.5}
            valueText={(v, d) => (d === 7 ? `${Math.round(v * 100)}%` : '')}
          />
          <g opacity={seg(t, 0.85, 1)}>
            <rect x={130} y={238} width={220} height={40} rx={20} fill={soft(c.series[3], 0.14)} stroke={c.series[3]} />
            <Txt x={240} y={264} size={14.5} weight={800}>
              "7일 확률 {Math.round(PROBS[7] * 100)}%" 라고 답해요
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '손실: 정답 확률이 낮을수록 큰 벌점',
    text: (
      <>
        교차엔트로피 손실은 <strong>-log(정답 확률)</strong>이에요. 처음엔 아무것도 몰라서 정답 확률이 1/10쯤이고 손실은 <strong>-log(0.1) ≈ 2.30</strong>이에요 — 첫 배치 손실이 이 값 근처가 아니면 버그를 의심하세요. 정답을 80%로 맞히면 손실은 0.22로 뚝 떨어져요.
      </>
    ),
    code: `def loss(self, Y):
    # 교차엔트로피. 1e-12는 log(0) 방지.
    n = Y.shape[0]
    return -np.sum(Y * np.log(self.probs + 1e-12)) / n`,
    highlight: [4],
    render: (t, c) => {
      const p = mix(0.1, 0.8, seg(t, 0.45, 1));
      const x0 = 64;
      const y0 = 30;
      const w = 300;
      const h = 200;
      const sx = (v: number) => x0 + v * w;
      const sy = (v: number) => y0 + h - (Math.min(4, v) / 4) * h;
      return (
        <g>
          <Defs />
          <Curve x={x0} y={y0} w={w} h={h} fn={(v) => -Math.log(Math.max(v, 0.02))} xDomain={[0, 1]} yDomain={[0, 4]} t={seg(t, 0, 0.45)} color={c.series[2]} xLabel="정답의 확률 p" yLabel="손실 = -log p" />
          <Txt x={sx(0)} y={y0 + h + 16} size={11.5} weight={650} color="var(--text-2)">
            0
          </Txt>
          <Txt x={sx(1)} y={y0 + h + 16} size={11.5} weight={650} color="var(--text-2)">
            1
          </Txt>
          <line x1={x0} x2={x0 + w} y1={sy(2.303)} y2={sy(2.303)} stroke={c.series[4]} strokeDasharray="4 5" strokeWidth={1.5} opacity={seg(t, 0.3, 0.5)} />
          <Txt x={x0 + w + 6} y={sy(2.303) + 4} size={12} weight={750} anchor="start" color="var(--text-2)" opacity={seg(t, 0.3, 0.5)}>
            2.30
          </Txt>
          {t > 0.4 && (
            <g>
              <circle cx={sx(0.1)} cy={sy(2.303)} r={5} fill={c.series[4]} />
              <Txt x={sx(0.1) + 8} y={sy(2.303) - 10} size={12} weight={750} anchor="start" color="var(--text-2)">
                처음 (p ≈ 0.1)
              </Txt>
              <circle cx={sx(p)} cy={sy(-Math.log(p))} r={8} fill={c.series[2]} stroke="var(--surface-solid)" strokeWidth={2.5} />
              <Txt x={Math.min(sx(p) + 4, 400)} y={sy(-Math.log(p)) - 14} size={13} weight={850} anchor="middle">
                p={p.toFixed(2)} → {(-Math.log(p)).toFixed(2)}
              </Txt>
            </g>
          )}
          <g opacity={seg(t, 0.85, 1)}>
            <Txt x={420} y={120} size={26}>
              🎯
            </Txt>
            <Txt x={420} y={146} size={12.5} weight={750} color="var(--text-2)">
              확신 있게
            </Txt>
            <Txt x={420} y={163} size={12.5} weight={750} color="var(--text-2)">
              맞히면 0에 가까워요
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '역전파: (probs - Y) / n 에서 거꾸로',
    text: (
      <>
        소프트맥스+교차엔트로피를 합쳐 미분하면 놀랍게도 <strong>(probs - Y) / n</strong> 한 줄이 돼요. 여기서 출발해 순전파의 반대 순서로, 각 층의 국소 미분을 곱하며 거슬러 올라가요. ReLU는 꺼졌던 칸(z₁ ≤ 0)으로는 기울기를 <strong>통과시키지 않아요</strong>.
      </>
    ),
    code: `dz2 = (self.probs - Y) / n                        # (n, 10)
dW2 = self.h.T @ dz2                              # (256, 10)
db2 = dz2.sum(axis=0)
dh = dz2 @ self.W2.T                              # (n, 256)
dz1 = dh * (self.z1 > 0)                          # ReLU의 미분: 통과 or 0
dW1 = self.X.T @ dz1                              # (784, 256)`,
    highlight: [1, 2, 4, 5, 6],
    render: (t, c) => {
      const nodes = [
        { x: 18, label: 'X', col: c.series[0] },
        { x: 84, label: 'W₁', col: c.series[1] },
        { x: 150, label: 'ReLU', col: c.series[2] },
        { x: 216, label: 'h', col: c.series[1] },
        { x: 282, label: 'W₂', col: c.series[1] },
        { x: 348, label: 'probs', col: c.series[3] },
        { x: 414, label: '손실', col: c.series[2] }
      ];
      const w = 50;
      const y = 44;
      const grads = [
        { from: 6, to: 5, text: 'dz2 = (probs - Y) / n' },
        { from: 5, to: 4, text: 'dW2 = hᵀ @ dz2' },
        { from: 4, to: 3, text: 'dh = dz2 @ W2ᵀ' },
        { from: 3, to: 2, text: 'dz1 = dh * (z1 > 0)' },
        { from: 2, to: 1, text: 'dW1 = Xᵀ @ dz1' }
      ];
      return (
        <g>
          <Defs />
          {nodes.slice(0, -1).map((n, i) => (
            <Arrow key={`f${i}`} x1={n.x + w} y1={y + 12} x2={nodes[i + 1].x} y2={y + 12} color="var(--viz-axis)" width={1.5} t={1} />
          ))}
          {nodes.map((n, i) => {
            const hit = grads.findIndex((g) => g.to === i);
            const arrived = hit >= 0 && seg(t, 0.08 + hit * 0.17, 0.2 + hit * 0.17) >= 1 && (i === 1 || i === 4);
            return <Box key={n.label} x={n.x} y={y} w={w} h={36} label={n.label} fs={n.label.length > 3 ? 12 : 15} color={n.col} glow={arrived} c={c} />;
          })}
          <Txt x={240} y={30} size={12} weight={700} color="var(--text-2)">
            순전파 →
          </Txt>
          {grads.map((g, k) => {
            const a = nodes[g.from];
            const b = nodes[g.to];
            const p = seg(t, 0.08 + k * 0.17, 0.2 + k * 0.17);
            return <Arrow key={`b${k}`} x1={a.x + w / 2} y1={y + 40} x2={b.x + w / 2} y2={y + 40} bend={-22} color={c.series[2]} width={3} t={p} flow />;
          })}
          <Txt x={240} y={128} size={12} weight={700} color={c.series[2]}>
            ← 역전파 (기울기)
          </Txt>
          {grads.map((g, k) => (
            <g key={`l${k}`} opacity={seg(t, 0.12 + k * 0.17, 0.24 + k * 0.17)}>
              <rect x={70} y={146 + k * 28} width={340} height={23} rx={7} fill={soft(c.series[2], k === 0 ? 0.22 : 0.1)} />
              <Txt x={84} y={162 + k * 28} size={13} weight={750} anchor="start" mono>
                {`${k + 1}. ${g.text}`}
              </Txt>
            </g>
          ))}
        </g>
      );
    }
  },
  {
    title: '한 걸음 내려가고, 반복한다',
    text: (
      <>
        구한 기울기의 <strong>반대 방향으로 lr(=0.1)만큼</strong> 모든 가중치를 옮기면 한 스텝 끝! 배치 128장마다 순전파 → 손실 → 역전파 → 스텝을 반복하면 10 에포크 뒤 테스트 정확도가 <strong>약 97.5%</strong>까지 올라가요. <code>loss.backward()</code>가 하던 일을 전부 직접 짠 거예요.
      </>
    ),
    code: `def step(self, grads, lr):
    for name, g in grads.items():
        setattr(self, name, getattr(self, name) - lr * g)

model.forward(x_tr[b])
total_loss += model.loss(Y_tr[b]) * len(b)
grads = model.backward(Y_tr[b])
model.step(grads, lr)`,
    highlight: [3, 5, 6, 7, 8],
    render: (t, c) => {
      const ws = [0.6, -0.3, 0.9, 0.2, -0.7];
      const gs = [0.5, -0.6, 0.4, -0.3, -0.5];
      const mv = seg(t, 0.15, 0.55);
      const base = 150;
      const k = 70;
      return (
        <g>
          <Defs />
          <Txt x={115} y={30} size={14} weight={800}>
            가중치 w ← w - lr·g
          </Txt>
          <line x1={20} x2={210} y1={base} y2={base} stroke="var(--viz-axis)" strokeWidth={1.5} />
          {ws.map((w0, i) => {
            const w1 = w0 - gs[i] * 0.8;
            const v = mix(w0, w1, mv);
            const x = 32 + i * 36;
            const top0 = base - w0 * k;
            const top1 = base - w1 * k;
            return (
              <g key={i}>
                <rect
                  x={x}
                  y={Math.min(top0, base)}
                  width={24}
                  height={Math.abs(w0 * k)}
                  rx={4}
                  fill="none"
                  stroke="var(--viz-axis)"
                  strokeDasharray="3 3"
                  opacity={seg(t, 0.15, 0.3)}
                />
                <rect x={x} y={v >= 0 ? base - v * k : base} width={24} height={Math.abs(v * k)} rx={4} fill={soft(c.series[1], 0.78)} />
                <Arrow x1={x + 12} y1={top0} x2={x + 12} y2={top1} color={c.series[2]} width={2.5} t={seg(t, 0.2, 0.55)} />
              </g>
            );
          })}
          <Txt x={115} y={258} size={12.5} weight={700} color="var(--text-2)">
            점선 = 이전 값, 화살표 = 기울기 반대 방향
          </Txt>
          <Curve
            x={258}
            y={44}
            w={200}
            h={150}
            fn={(e) => 0.07 + 2.23 * Math.exp(-e * 1.25) + 0.03 * Math.exp(-e * 0.1)}
            xDomain={[0, 10]}
            yDomain={[0, 2.5]}
            t={seg(t, 0.35, 0.95)}
            color={c.series[2]}
            xLabel="에포크"
            yLabel="손실"
            marker={10}
          />
          <g opacity={seg(t, 0.85, 1)}>
            <rect x={286} y={228} width={174} height={40} rx={20} fill={soft(c.pos, 0.15)} stroke={c.pos} />
            <Txt x={373} y={254} size={14} weight={850}>
              테스트 정확도 97.5% 🎉
            </Txt>
          </g>
        </g>
      );
    }
  }
];

/** Code lab 01 walkthrough: NumPy MLP forward pass, loss, backward pass, update. */
export default function Lab01ForwardBackward() {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={1.9} />;
}
