import type { VizProps } from '../Viz';
import { CodeStory, type StoryStep } from '../core/story';
import { VB, FS, seg, alpha, rich, dd, T, Box, Arrow, Pill, Cells } from './_LabKitB';
import { digit, blur, noise, diffuse, clamp01, ALPHA_BAR, BETA, type Img } from './_digitsB';

const X0 = digit('3');
const EPS = noise(5);
const EPS_HAT = EPS.map((r, i) => r.map((v, j) => v * 0.9 + 0.25 * Math.sin(i * 1.7 + j * 2.3)));
const ab = (t: number) => ALPHA_BAR[Math.min(999, Math.max(0, t))];
const noiseImg = (e: Img) => e.map((r) => r.map((v) => clamp01(0.5 + v * 0.3)));

function Pic({ img, x, y, cell = 6, color, t = 1, label, sub }: { img: Img; x: number; y: number; cell?: number; color: string; t?: number; label?: string; sub?: string }) {
  const n = img.length;
  const size = n * (cell + 1) - 1;
  return (
    <g opacity={Math.min(1, t * 1.6)}>
      <rect x={x - 3} y={y - 3} width={size + 6} height={size + 6} rx={7} fill="var(--surface-solid)" stroke="var(--border-strong)" />
      <Cells x={x} y={y} rows={n} cols={n} cell={cell} gap={1} values={(r, k) => img[r][k]} color={color} />
      {label && (
        <T x={x + size / 2} y={y + size + 18} anchor="middle" size={FS.m} weight={800} mono>
          {label}
        </T>
      )}
      {sub && (
        <T x={x + size / 2} y={y + size + 34} anchor="middle" size={11.5} weight={700} color="var(--text-2)">
          {sub}
        </T>
      )}
    </g>
  );
}

/* sinusoidal time embedding exactly like the lab's time_embed, with dim = 32 */
function timeEmbed(t: number, dim = 32) {
  const half = dim / 2;
  const out: number[] = [];
  for (let k = 0; k < half; k++) out.push(Math.sin(t * Math.exp((-Math.log(10000) * k) / half)));
  for (let k = 0; k < half; k++) out.push(Math.cos(t * Math.exp((-Math.log(10000) * k) / half)));
  return out;
}

const steps: StoryStep[] = [
  {
    title: '노이즈 스케줄 — 미리 전부 계산해 두기',
    text: rich('1000단계 동안 매번 조금씩(**β**) 노이즈를 넣어요. β를 곱해 쌓은 **ᾱ(알파 바)** 는 "원래 그림이 얼마나 남았나"예요. 500단계면 벌써 8% 남짓, 1000단계면 거의 0 — 순수한 노이즈가 돼요.'),
    code: dd(String.raw`class Schedule:
    def __init__(self):
        self.beta = torch.linspace(1e-4, 0.02, T, device=device)     # 선형 스케줄
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)            # ᾱ_t`),
    highlight: [3, 5],
    render: (t, c) => {
      const P = { x: 44, y: 40, w: 330, h: 190 };
      const n = Math.max(2, Math.round(1000 * seg(t, 0.05, 0.6)));
      const pA: string[] = [];
      const pB: string[] = [];
      for (let i = 0; i < n; i += 10) {
        const px = P.x + (P.w * i) / 999;
        pA.push(`${px.toFixed(1)},${(P.y + P.h - P.h * ALPHA_BAR[i]).toFixed(1)}`);
        pB.push(`${px.toFixed(1)},${(P.y + P.h - (P.h * BETA[i]) / 0.02).toFixed(1)}`);
      }
      const marks = [250, 500, 999];
      return (
        <g>
          <line x1={P.x} y1={P.y + P.h} x2={P.x + P.w} y2={P.y + P.h} stroke="var(--viz-axis)" strokeWidth={1.5} />
          <line x1={P.x} y1={P.y} x2={P.x} y2={P.y + P.h} stroke="var(--viz-axis)" strokeWidth={1.5} />
          <line x1={P.x + P.w} y1={P.y} x2={P.x + P.w} y2={P.y + P.h} stroke="var(--viz-axis)" strokeWidth={1} strokeDasharray="3 3" />
          {[0, 250, 500, 750, 1000].map((v) => (
            <T key={v} x={P.x + (P.w * Math.min(v, 999)) / 999} y={P.y + P.h + 17} anchor="middle" size={11.5} weight={650} color="var(--text-2)" mono>
              {v}
            </T>
          ))}
          <T x={P.x + P.w / 2} y={P.y + P.h + 36} anchor="middle" size={FS.m} weight={750} color="var(--text-2)">
            단계 t
          </T>
          <T x={P.x - 6} y={P.y + 5} anchor="end" size={11.5} weight={700} color={c.series[0]} mono>
            1
          </T>
          <T x={P.x - 6} y={P.y + P.h} anchor="end" size={11.5} weight={700} color={c.series[0]} mono>
            0
          </T>
          <T x={P.x + P.w + 4} y={P.y + 5} size={11.5} weight={700} color={c.series[2]} mono>
            0.02
          </T>
          <polyline points={pA.join(' ')} fill="none" stroke={c.series[0]} strokeWidth={3.5} strokeLinejoin="round" />
          <polyline points={pB.join(' ')} fill="none" stroke={c.series[2]} strokeWidth={2.5} strokeDasharray="6 4" />
          <g opacity={seg(t, 0.55, 0.2)}>
            <Pill x={150} y={28} label="ᾱ: 남은 신호의 비율" color={c.series[0]} size={FS.m} />
            <Pill x={318} y={28} label="β: 매 단계 노이즈" color={c.series[2]} size={FS.m} />
          </g>
          {marks.map((m, i) => {
            const px = P.x + (P.w * m) / 999;
            const py = P.y + P.h - P.h * ALPHA_BAR[m];
            const a = seg(t, 0.62 + i * 0.1, 0.15);
            return (
              <g key={m} opacity={a}>
                <circle cx={px} cy={py} r={5.5} fill={c.series[0]} stroke="var(--surface-solid)" strokeWidth={2} />
                <T x={px + (i === 2 ? -6 : 8)} y={py - 10 - (i === 2 ? 36 : 0)} anchor={i === 2 ? 'end' : 'start'} size={FS.m} weight={800}>
                  {`t=${m === 999 ? 1000 : m}: ${(ALPHA_BAR[m] * 100).toFixed(m === 999 ? 3 : 0)}%`}
                </T>
              </g>
            );
          })}
        </g>
      );
    }
  },
  {
    title: '순방향 — 아무 단계로나 한 번에 점프',
    text: rich('단계를 하나하나 밟을 필요가 없어요. 닫힌 형태 공식 **x_t = √ᾱ·x₀ + √(1-ᾱ)·ε** 하나로 원하는 t의 노이즈 그림을 바로 만들어요. 신호(√ᾱ)는 줄고 노이즈(√(1-ᾱ))는 늘어요.'),
    code: dd(String.raw`    def add_noise(self, x0, t, eps):
        """순방향 닫힌 형태: x_t = √ᾱ·x0 + √(1-ᾱ)·ε"""
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps`),
    highlight: [3, 4],
    render: (t, c) => {
      const ts = [0, 100, 300, 600, 999];
      return (
        <g>
          {ts.map((tt, i) => {
            const a = ab(tt);
            const x = 12 + i * 81;
            const sig = Math.sqrt(a);
            const noi = Math.sqrt(1 - a);
            const show = seg(t, 0.05 + i * 0.13, 0.2);
            return (
              <g key={tt} opacity={show}>
                <Pic img={diffuse(X0, EPS, a)} x={x + 4} y={60} cell={6} color={c.series[0]} label={`t=${tt === 999 ? 1000 : tt}`} />
                <rect x={x + 4} y={170} width={69} height={12} rx={6} fill="var(--surface-3)" />
                <rect x={x + 4} y={170} width={69 * sig} height={12} rx={6} fill={c.series[0]} />
                <rect x={x + 4} y={196} width={69} height={12} rx={6} fill="var(--surface-3)" />
                <rect x={x + 4} y={196} width={69 * noi} height={12} rx={6} fill={c.series[2]} />
                <rect x={x + 16} y={218} width={8} height={8} rx={2} fill={c.series[0]} />
                <T x={x + 28} y={226} size={11.5} weight={750} color="var(--text-2)" mono>
                  {sig.toFixed(2)}
                </T>
                <rect x={x + 16} y={234} width={8} height={8} rx={2} fill={c.series[2]} />
                <T x={x + 28} y={242} size={11.5} weight={750} color="var(--text-2)" mono>
                  {noi.toFixed(2)}
                </T>
              </g>
            );
          })}
          <T x={12} y={34} size={FS.s} weight={850}>
            같은 그림 x₀ + 같은 노이즈 ε, t만 바꿔서
          </T>
          <g opacity={seg(t, 0.75, 0.2)}>
            <rect x={20} y={262} width={14} height={10} rx={3} fill={c.series[0]} />
            <T x={40} y={272} size={FS.m} weight={750} color="var(--text-2)">
              신호 √ᾱ
            </T>
            <rect x={130} y={262} width={14} height={10} rx={3} fill={c.series[2]} />
            <T x={150} y={272} size={FS.m} weight={750} color="var(--text-2)">
              노이즈 √(1-ᾱ)
            </T>
          </g>
        </g>
      );
    }
  },
  {
    title: '학습 — "더해진 노이즈를 맞혀라"',
    text: rich('매번 **무작위 t**를 골라 그림에 노이즈 ε를 섞고(x_t), 네트워크에게 **섞인 노이즈를 맞히게** 해요. 손실은 그냥 **MSE** 하나. 두 네트워크의 줄다리기가 없으니 GAN보다 훨씬 안정적이에요.'),
    code: dd(String.raw`            t = torch.randint(0, T, (x0.shape[0],), device=device)   # 시점 무작위
            eps = torch.randn_like(x0)
            xt = sched.add_noise(x0, t, eps)
            loss = F.mse_loss(model(xt, t), eps)                     # 노이즈 맞추기 — 이게 전부`),
    highlight: [1, 3, 4],
    render: (t, c) => {
      const a = ab(420);
      return (
        <g>
          <Pic img={X0} x={18} y={40} cell={5} color={c.series[0]} t={seg(t, 0, 0.2)} label="x₀" />
          <Pic img={noiseImg(EPS)} x={18} y={170} cell={5} color={c.series[2]} t={seg(t, 0.08, 0.2)} label="ε" />
          <Arrow x1={82} y1={70} x2={112} y2={120} color={c.series[0]} t={seg(t, 0.18, 0.2)} />
          <Arrow x1={82} y1={200} x2={112} y2={150} color={c.series[2]} t={seg(t, 0.22, 0.2)} />
          <Pic img={diffuse(X0, EPS, a)} x={118} y={106} cell={5} color={c.series[0]} t={seg(t, 0.3, 0.2)} label="x_t" />
          <Pill x={150} y={86} label="t = 420 🎲" color={c.series[3]} t={seg(t, 0.3, 0.2)} size={11.5} />
          <Arrow x1={180} y1={136} x2={212} y2={136} color={c.series[1]} t={seg(t, 0.4, 0.2)} />
          <Box x={214} y={106} w={74} h={60} color={c.series[1]} t={seg(t, 0.42, 0.25)} label="U-Net" sub="ε_θ(x_t, t)" />
          <Arrow x1={290} y1={136} x2={318} y2={136} color={c.series[1]} t={seg(t, 0.55, 0.2)} />
          <Pic img={noiseImg(EPS_HAT)} x={322} y={106} cell={5} color={c.series[2]} t={seg(t, 0.6, 0.2)} label="ε̂ (예측)" />
          <path d="M82 236 C 200 290, 300 280, 352 182" fill="none" stroke={c.series[2]} strokeWidth={2} strokeDasharray="5 5" opacity={seg(t, 0.7, 0.2)} />
          <g opacity={seg(t, 0.78, 0.2)}>
            <Pill x={250} y={252} label="MSE(ε, ε̂) ↓" color={c.neg} size={FS.m} />
          </g>
        </g>
      );
    }
  },
  {
    title: '시간 임베딩 — "지금 몇 단계야?" 이름표',
    text: rich('하나의 네트워크가 거의 깨끗한 x₁₀도, 거의 노이즈인 x₉₉₀도 처리해야 해요. 그래서 t를 **사인·코사인 무늬**로 바꿔 각 블록에 넣어 줘요. 이 이름표가 없으면 얼마나 센 노이즈를 빼야 할지 몰라요.'),
    code: dd(String.raw`def time_embed(t, dim=128):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)`),
    highlight: [3, 4, 5],
    render: (t, c) => {
      const rows = [10, 250, 500, 990];
      return (
        <g>
          <T x={14} y={32} size={FS.s} weight={850}>
            t마다 다른 무늬 (32칸으로 줄여서)
          </T>
          {rows.map((tt, i) => {
            const emb = timeEmbed(tt);
            const a = seg(t, 0.08 + i * 0.15, 0.25);
            return (
              <g key={tt} opacity={a}>
                <T x={60} y={70 + i * 52} anchor="end" size={FS.m} weight={850} mono>
                  t={tt}
                </T>
                <Cells x={70} y={54 + i * 52} rows={1} cols={32} cell={9} gap={1.2} values={(_, k) => (emb[k] + 1) / 2} color={i % 2 === 0 ? c.series[0] : c.series[1]} />
              </g>
            );
          })}
          <T x={70} y={270} size={FS.m} weight={750} color="var(--text-2)" opacity={seg(t, 0.75, 0.2)}>
            ← sin 16칸 | cos 16칸 →  (왼쪽은 빠르게, 오른쪽은 천천히 변해요)
          </T>
        </g>
      );
    }
  },
  {
    title: '노이즈 예측기 — 작은 U-Net',
    text: rich('28×28 → 14×14 → 7×7로 줄였다가 다시 키우는 **U자 모양**이에요. 줄일 때의 결과를 키울 때 **옆으로 이어 붙이는 스킵 연결**이 세세한 위치 정보를 지켜 줘요. 모든 블록에 시간 이름표(t)가 들어가요.'),
    code: dd(String.raw`        temb = self.t_mlp(time_embed(t))
        h1 = self.inc(x, temb)                                   # (B, 64, 28, 28)
        h2 = self.down1(self.pool(h1), temb)                     # (B, 128, 14, 14)
        h3 = self.down2(self.pool(h2), temb)                     # (B, 256, 7, 7)
        m = self.mid(h3, temb)
        u2 = self.up2(torch.cat([m, h3], 1), temb)
        u2 = F.interpolate(u2, scale_factor=2)                   # (B, 128, 14, 14)
        u1 = self.up1(torch.cat([u2, h2], 1), temb)
        u1 = F.interpolate(u1, scale_factor=2)                   # (B, 64, 28, 28)
        return self.out(torch.cat([u1, h1], 1))                  # 예측 노이즈`),
    highlight: [6, 8, 10],
    render: (t, c) => {
      const blocks = [
        { x: 18, y: 60, h: 120, label: '28²', ch: '64' },
        { x: 92, y: 120, h: 80, label: '14²', ch: '128' },
        { x: 166, y: 180, h: 50, label: '7²', ch: '256' },
        { x: 240, y: 120, h: 80, label: '14²', ch: '128' },
        { x: 314, y: 60, h: 120, label: '28²', ch: '64' }
      ];
      const order = [0, 1, 2, 3, 4];
      return (
        <g>
          {order.map((i) => {
            const b = blocks[i];
            const a = seg(t, 0.05 + i * 0.1, 0.2);
            const col = i < 2 ? c.series[0] : i === 2 ? c.series[3] : c.series[1];
            return (
              <g key={i} opacity={a}>
                <rect x={b.x} y={b.y} width={56} height={b.h} rx={10} fill={alpha(col, 0.18)} stroke={col} strokeWidth={2} />
                <T x={b.x + 28} y={b.y + b.h / 2} anchor="middle" size={FS.s} weight={850} mono>
                  {b.label}
                </T>
                <T x={b.x + 28} y={b.y + b.h / 2 + 18} anchor="middle" size={11.5} weight={700} color="var(--text-2)" mono>
                  ×{b.ch}
                </T>
                <T x={b.x + 28} y={b.y - 8} anchor="middle" size={11.5} weight={800} color={c.series[2]}>
                  ⏱t
                </T>
              </g>
            );
          })}
          <Arrow x1={76} y1={150} x2={90} y2={160} color={c.series[0]} t={seg(t, 0.15, 0.15)} />
          <Arrow x1={150} y1={190} x2={164} y2={200} color={c.series[0]} t={seg(t, 0.25, 0.15)} />
          <Arrow x1={224} y1={200} x2={238} y2={190} color={c.series[1]} t={seg(t, 0.35, 0.15)} />
          <Arrow x1={298} y1={160} x2={312} y2={150} color={c.series[1]} t={seg(t, 0.45, 0.15)} />
          {[
            { from: blocks[0], to: blocks[4], y: 42 },
            { from: blocks[1], to: blocks[3], y: 104 }
          ].map((s, i) => (
            <g key={i} opacity={seg(t, 0.6 + i * 0.12, 0.2)}>
              <path
                d={`M${s.from.x + 28} ${s.from.y} C ${s.from.x + 28} ${s.y - 8}, ${s.to.x + 28} ${s.y - 8}, ${s.to.x + 28} ${s.to.y}`}
                fill="none"
                stroke={c.series[4]}
                strokeWidth={2.5}
                strokeDasharray="6 4"
              />
            </g>
          ))}
          <g opacity={seg(t, 0.8, 0.2)}>
            <Pill x={210} y={262} label="⋯⋯ 스킵 연결: torch.cat" color={c.series[4]} size={FS.m} />
          </g>
        </g>
      );
    }
  },
  {
    title: '샘플링 — 노이즈에서 숫자로 1000걸음',
    text: rich('순수 노이즈 x_T에서 출발해, 매 단계 **예측한 노이즈를 조금씩 빼며** x₀까지 되돌아가요. 생성 한 번에 **순전파 1000번** — 확산 모델이 느린 이유예요. (그림은 정답을 아는 "완벽한 예측기"로 흉내 낸 거예요.)'),
    code: dd(String.raw`    x = torch.randn(n, 1, 28, 28, device=device)                 # x_T: 순수 노이즈
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, tt)
        a, ab, b = sched.alpha[t], sched.alpha_bar[t], sched.beta[t]
        # 평균: 예측 노이즈를 빼서 한 단계 덜 노이즈한 상태로
        x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()`),
    highlight: [1, 2, 4, 7],
    render: (t, c) => {
      const ts = [999, 700, 450, 250, 100, 0];
      const cur = seg(t, 0.05, 0.85);
      const counter = Math.round(999 - 999 * cur);
      return (
        <g>
          {ts.map((tt, i) => {
            const a = seg(t, 0.05 + i * 0.14, 0.18);
            const x = 10 + i * 68;
            return (
              <g key={tt}>
                <Pic img={diffuse(X0, EPS, ab(tt))} x={x + 3} y={86} cell={5} color={c.series[1]} t={a} label={`t=${tt === 999 ? 1000 : tt}`} />
                {i < ts.length - 1 && <Arrow x1={x + 62} y1={112} x2={x + 70} y2={112} color={c.series[1]} t={seg(t, 0.12 + i * 0.14, 0.1)} />}
              </g>
            );
          })}
          <T x={210} y={48} anchor="middle" size={FS.h} weight={900} mono>
            t = {counter}
          </T>
          <rect x={60} y={190} width={300} height={14} rx={7} fill="var(--surface-3)" />
          <rect x={60} y={190} width={300 * cur} height={14} rx={7} fill={c.series[1]} />
          <T x={210} y={230} anchor="middle" size={FS.s} weight={800} color="var(--text-2)">
            U-Net 순전파 {Math.round(1000 * cur)} / 1000번
          </T>
        </g>
      );
    }
  },
  {
    title: '노이즈 재주입 — 다양성의 비밀',
    text: rich('역방향의 각 단계는 정답을 하나로 정하는 게 아니라 **분포에서 뽑는 일**이에요. 그래서 마지막 단계만 빼고 **노이즈를 조금 다시 더해요**. 이 줄을 지우면 샘플이 흐릿해지고 모두 비슷해져요 — 직접 지워서 비교해 보세요.'),
    code: dd(String.raw`        x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
        if t > 0:                                                # 마지막 단계 빼고 노이즈 재주입
            x = x + b.sqrt() * torch.randn_like(x)
    return (x.clamp(-1, 1) + 1) / 2`),
    highlight: [2, 3],
    render: (t, c) => {
      const crisp = ['3', '8', '2'].map((d) => digit(d));
      const avg = blur(
        crisp[0].map((row, i) => row.map((v, j) => (v + crisp[1][i][j] + crisp[2][i][j]) / 3)),
        1
      );
      return (
        <g>
          <T x={14} y={44} size={FS.s} weight={850} color={c.pos}>
            ✅ 재주입 있음 — 또렷하고 다양해요
          </T>
          {crisp.map((img, i) => (
            <Pic key={i} img={img} x={34 + i * 126} y={58} cell={5} color={c.series[1]} t={seg(t, 0.1 + i * 0.1, 0.2)} />
          ))}
          <T x={14} y={170} size={FS.s} weight={850} color={c.neg} opacity={seg(t, 0.45, 0.2)}>
            ❌ 재주입 없음 — 흐릿하고 다 비슷해요
          </T>
          {[0, 1, 2].map((i) => (
            <Pic key={i} img={avg} x={34 + i * 126} y={184} cell={5} color={c.series[1]} t={seg(t, 0.55 + i * 0.1, 0.2)} />
          ))}
        </g>
      );
    }
  }
];

export default function Lab15Ddpm(_: VizProps) {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={2} />;
}
