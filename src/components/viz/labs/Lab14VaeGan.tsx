import type { VizProps } from '../Viz';
import { CodeStory, type StoryStep } from '../core/story';
import { VB, FS, seg, alpha, rich, dd, T, Box, Arrow, Pill, Cells, Pulse } from './_LabKitB';
import { digit, blur, noise, mix, clamp01, type Img } from './_digitsB';

const THREE_IMG = digit('3');
const THREE_BLUR = blur(THREE_IMG, 1);
const SAMPLES = ['3', '8', '1', '7'].map((d) => blur(digit(d), 1));
const NOISES = [11, 12, 13, 14].map((s) => noise(s).map((r) => r.map((v) => clamp01(0.5 + v * 0.35))));

function Pic({ img, x, y, cell = 9, color, t = 1, label }: { img: Img; x: number; y: number; cell?: number; color: string; t?: number; label?: string }) {
  const n = img.length;
  const size = n * (cell + 1) - 1;
  return (
    <g>
      <rect x={x - 4} y={y - 4} width={size + 8} height={size + 8} rx={8} fill="var(--surface-solid)" stroke="var(--border-strong)" />
      <Cells x={x} y={y} rows={n} cols={n} cell={cell} gap={1} values={(r, k) => img[r][k]} color={color} t={t} />
      {label && (
        <T x={x + size / 2} y={y + size + 20} anchor="middle" size={FS.m} weight={750} color="var(--text-2)">
          {label}
        </T>
      )}
    </g>
  );
}

/** Gaussian bell drawn in a box: mean mu, std sd, over x in [-3.5, 3.5]. */
function Bell({ x, y, w, h, mu, sd, color, fill = 0.18, t = 1 }: { x: number; y: number; w: number; h: number; mu: number; sd: number; color: string; fill?: number; t?: number }) {
  const pts: string[] = [];
  const peak = 1 / sd;
  for (let i = 0; i <= 60; i++) {
    const u = -3.5 + (7 * i) / 60;
    const pdf = Math.exp(-0.5 * ((u - mu) / sd) ** 2) / sd;
    const px = x + (w * i) / 60;
    const py = y + h - (h * pdf * t) / Math.max(peak, 1);
    pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
  }
  return (
    <g>
      <polygon points={`${x},${y + h} ${pts.join(' ')} ${x + w},${y + h}`} fill={alpha(color, fill)} stroke="none" />
      <polyline points={pts.join(' ')} fill="none" stroke={color} strokeWidth={2.5} />
      <line x1={x} y1={y + h} x2={x + w} y2={y + h} stroke="var(--border-strong)" />
    </g>
  );
}
const bx = (x: number, w: number, u: number) => x + (w * (u + 3.5)) / 7;

function Funnel({ x, y, w, h0, h1, color, t = 1, label }: { x: number; y: number; w: number; h0: number; h1: number; color: string; t?: number; label?: string }) {
  const cy = y;
  return (
    <g opacity={Math.min(1, t * 1.6)}>
      <path d={`M${x} ${cy - h0 / 2} L${x + w} ${cy - h1 / 2} L${x + w} ${cy + h1 / 2} L${x} ${cy + h0 / 2} Z`} fill={alpha(color, 0.18)} stroke={color} strokeWidth={2} strokeLinejoin="round" />
      {label && (
        <T x={x + w / 2} y={cy + 5} anchor="middle" size={FS.s} weight={850}>
          {label}
        </T>
      )}
    </g>
  );
}

const MU = 0.9;
const SD = 0.45;
const EPS = -0.8;
const Z = MU + SD * EPS;

const steps: StoryStep[] = [
  {
    title: 'VAE 인코더 — 숫자 하나 대신 "분포"를 출력',
    text: rich('이미지(784칸)를 400 → 200으로 줄인 뒤, 마지막에 **두 갈래**로 나눠요. 하나는 평균 **μ**, 하나는 퍼진 정도 **logσ²**. "이 이미지는 잠재 공간의 이 근처쯤이야"라고 **구름 모양으로** 알려 주는 거예요.'),
    code: dd(String.raw`        self.enc = nn.Sequential(
            nn.Linear(784, 400), nn.ReLU(),
            nn.Linear(400, 200), nn.ReLU())
        self.fc_mu = nn.Linear(200, Z)        # 인코더는 분포(μ, logσ²)를 출력한다
        self.fc_logvar = nn.Linear(200, Z)`),
    highlight: [4, 5],
    render: (t, c) => (
      <g>
        <Pic img={THREE_IMG} x={18} y={96} color={c.series[0]} t={seg(t, 0, 0.3)} label="x (28×28)" />
        <Arrow x1={122} y1={146} x2={140} y2={146} color={c.series[0]} t={seg(t, 0.2, 0.2)} />
        <Funnel x={144} y={146} w={70} h0={130} h1={70} color={c.series[0]} t={seg(t, 0.25, 0.3)} label="400" />
        <Funnel x={218} y={146} w={44} h0={70} h1={46} color={c.series[0]} t={seg(t, 0.4, 0.3)} label="200" />
        <Arrow x1={264} y1={140} x2={300} y2={98} color={c.series[0]} t={seg(t, 0.55, 0.2)} />
        <Arrow x1={264} y1={152} x2={300} y2={196} color={c.series[1]} t={seg(t, 0.6, 0.2)} />
        <g opacity={seg(t, 0.65, 0.25)}>
          <Cells x={306} y={60} rows={1} cols={8} cell={11} values={(_, k) => [0.8, 0.3, 0.6, 0.9, 0.2, 0.5, 0.7, 0.4][k]} color={c.series[0]} />
          <T x={306} y={50} size={FS.s} weight={850} color={c.series[0]}>
            μ (평균) · Z=16
          </T>
          <Cells x={306} y={186} rows={1} cols={8} cell={11} values={(_, k) => [0.3, 0.5, 0.2, 0.4, 0.6, 0.3, 0.2, 0.5][k]} color={c.series[1]} />
          <T x={306} y={176} size={FS.s} weight={850} color={c.series[1]}>
            logσ² (퍼짐) · Z=16
          </T>
          <T x={306} y={236} size={FS.m} weight={700} color="var(--text-2)">
            → "대략 이 근처"라는
          </T>
          <T x={306} y={254} size={FS.m} weight={700} color="var(--text-2)">
            구름 모양 분포
          </T>
        </g>
      </g>
    )
  },
  {
    title: '재파라미터화 트릭 — z = μ + σ·ε',
    text: rich('분포에서 바로 "뽑기"를 하면 그 순간 미분이 끊겨요. 그래서 무작위는 바깥의 **ε ~ N(0, 1)** 에 맡기고, **z = μ + σ·ε** 라는 평범한 계산으로 바꿔요. 이제 기울기가 μ와 σ까지 흘러갈 수 있어요.'),
    code: dd(String.raw`    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        # 재파라미터화 트릭: z = μ + σ·ε.
        # '샘플링'을 미분 가능한 연산 + 외부 노이즈로 바꾼다 — 이게 없으면 역전파 불가.
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.dec(z), mu, logvar`),
    highlight: [6, 7],
    render: (t, c) => {
      const L = { x: 14, y: 150, w: 180, h: 90 };
      const Rb = { x: 226, y: 90, w: 180, h: 150 };
      const drop = seg(t, 0.1, 0.25);
      const scale = seg(t, 0.4, 0.25);
      const ex = bx(L.x, L.w, EPS);
      const zx = bx(Rb.x, Rb.w, Z);
      return (
        <g>
          <T x={L.x} y={L.y - 18} size={FS.s} weight={850} color={c.series[3]}>
            🎲 ε ~ N(0, 1) (바깥의 무작위)
          </T>
          <Bell {...L} mu={0} sd={1} color={c.series[3]} />
          <circle cx={ex} cy={L.y - 4 + (L.h + 4) * drop} r={6} fill={c.series[3]} opacity={drop > 0 ? 1 : 0} />
          <T x={ex} y={L.y + L.h + 18} anchor="middle" size={FS.m} weight={800} mono>
            ε = {EPS}
          </T>
          <T x={Rb.x} y={Rb.y - 18} size={FS.s} weight={850} color={c.series[0]}>
            q(z|x) = N(μ, σ²)
          </T>
          <Bell {...Rb} mu={MU} sd={SD} color={c.series[0]} />
          <line x1={bx(Rb.x, Rb.w, MU)} y1={Rb.y + 8} x2={bx(Rb.x, Rb.w, MU)} y2={Rb.y + Rb.h} stroke={c.series[0]} strokeDasharray="4 4" />
          <T x={bx(Rb.x, Rb.w, MU) + 4} y={Rb.y + 20} size={FS.m} weight={800} color={c.series[0]}>
            μ
          </T>
          <Arrow x1={ex + 6} y1={L.y + L.h - 6} x2={zx - 4} y2={Rb.y + Rb.h - 6} color={c.series[2]} t={scale} label="× σ  + μ" labelDy={-10} />
          <g opacity={seg(t, 0.62, 0.2)}>
            <circle cx={zx} cy={Rb.y + Rb.h - 4} r={8} fill={c.series[2]} stroke="var(--surface-solid)" strokeWidth={2} />
            <T x={zx} y={Rb.y + Rb.h + 20} anchor="middle" size={FS.m} weight={850} mono>
              z = {Z.toFixed(2)}
            </T>
          </g>
          <g opacity={seg(t, 0.78, 0.2)}>
            <Pill x={318} y={36} label="기울기 → μ, σ 로 흐름 ✓" color={c.pos} size={FS.m} />
          </g>
        </g>
      );
    }
  },
  {
    title: '손실 = 재구성 + KL',
    text: rich('두 가지를 함께 줄여요. **재구성 항**: 복원한 그림이 원래 그림과 얼마나 같은가(픽셀별 BCE). **KL 항**: 인코더의 구름 N(μ, σ²)이 기준 구름 N(0, 1)에서 너무 멀어지지 않게. 덕분에 잠재 공간이 빈틈없이 정돈돼요.'),
    code: dd(String.raw`def vae_loss(recon_logits, x, mu, logvar):
    # 재구성 항: 픽셀별 베르누이 로그가능도
    recon = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum")
    # KL 항: q(z|x)=N(μ,σ²) 와 p(z)=N(0,1) 사이 — 가우시안이라 닫힌 형태
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon + kl) / x.shape[0]`),
    highlight: [3, 5, 6],
    render: (t, c) => {
      const K = { x: 226, y: 86, w: 180, h: 120 };
      const pull = seg(t, 0.35, 0.5);
      const mu = MU * (1 - 0.6 * pull);
      const sd = SD + (1 - SD) * 0.55 * pull;
      return (
        <g>
          <Pic img={THREE_IMG} x={16} y={58} cell={7} color={c.series[0]} t={seg(t, 0, 0.3)} label="원본 x" />
          <Pic img={THREE_BLUR} x={118} y={58} cell={7} color={c.series[1]} t={seg(t, 0.1, 0.3)} label="복원" />
          <T x={107} y={98} anchor="middle" size={FS.h} weight={900} color="var(--text-2)">
            →
          </T>
          <T x={16} y={40} size={FS.s} weight={850} color={c.series[1]}>
            재구성 항 (BCE)
          </T>
          <T x={K.x} y={40} size={FS.s} weight={850} color={c.series[0]}>
            KL 항
          </T>
          <Bell {...K} mu={0} sd={1} color={c.series[3]} fill={0.1} />
          <Bell {...K} mu={mu} sd={sd} color={c.series[0]} />
          <T x={K.x + K.w} y={K.y + K.h + 17} anchor="end" size={FS.m} weight={750} color="var(--text-2)">
            N(0,1) 쪽으로 당겨져요
          </T>
          {[
            { label: '재구성', v: 0.62 - 0.12 * pull, color: c.series[1] },
            { label: 'KL', v: 0.34 - 0.14 * pull, color: c.series[0] }
          ].map((b, i) => (
            <g key={i} opacity={seg(t, 0.2 + i * 0.1, 0.25)}>
              <T x={16} y={250 + i * 30} size={FS.m} weight={800}>
                {b.label}
              </T>
              <rect x={70} y={238 + i * 30} width={300} height={16} rx={8} fill="var(--surface-3)" />
              <rect x={70} y={238 + i * 30} width={300 * b.v} height={16} rx={8} fill={b.color} />
            </g>
          ))}
          <T x={380} y={264} size={FS.h} weight={900}>
            +
          </T>
        </g>
      );
    }
  },
  {
    title: '생성 — 인코더 없이 z만 뽑아서',
    text: rich('학습이 끝나면 **인코더는 필요 없어요**. 기준 분포 N(0, 1)에서 z를 뽑아 디코더에 넣으면 세상에 없던 숫자가 나와요. VAE 그림이 조금 **흐릿한** 건 픽셀 평균 쪽으로 수렴하는 성질 때문이에요.'),
    code: dd(String.raw`    # 생성: 사전분포 N(0,1)에서 z를 뽑아 디코딩 — 인코더는 이제 필요 없다
    with torch.no_grad():
        z = torch.randn(64, Z, device=device)
        samples = torch.sigmoid(model.dec(z)).view(-1, 1, 28, 28)
        save_image(samples, "vae_samples.png", nrow=8)`),
    highlight: [3, 4],
    render: (t, c) => (
      <g>
        <T x={16} y={36} size={FS.s} weight={850} color={c.series[3]}>
          🎲 z ~ N(0, 1)
        </T>
        <Bell x={16} y={52} w={120} h={70} mu={0} sd={1} color={c.series[3]} />
        {[0, 1, 2, 3].map((i) => (
          <circle key={i} cx={bx(16, 120, [-1.2, 0.3, 1.1, -0.4][i])} cy={128} r={5} fill={c.series[3]} opacity={seg(t, 0.05 + i * 0.08, 0.15)} />
        ))}
        <Arrow x1={140} y1={100} x2={170} y2={100} color={c.series[1]} t={seg(t, 0.3, 0.2)} />
        <Funnel x={174} y={100} w={64} h0={36} h1={96} color={c.series[1]} t={seg(t, 0.35, 0.3)} label="dec" />
        {SAMPLES.map((img, i) => (
          <g key={i}>
            <Pic img={img} x={46 + i * 92} y={174} cell={6} color={c.series[1]} t={seg(t, 0.5 + i * 0.1, 0.25)} />
          </g>
        ))}
        <Arrow x1={240} y1={110} x2={300} y2={160} color={c.series[1]} t={seg(t, 0.45, 0.2)} />
        <g opacity={seg(t, 0.2, 0.3)}>
          <rect x={286} y={44} width={120} height={56} rx={12} fill="none" stroke="var(--border-strong)" strokeDasharray="5 5" />
          <T x={346} y={68} anchor="middle" size={FS.s} weight={800} color="var(--text-3)">
            인코더
          </T>
          <T x={346} y={88} anchor="middle" size={FS.m} weight={700} color="var(--text-3)">
            (쉬는 중 💤)
          </T>
        </g>
      </g>
    )
  },
  {
    title: 'GAN — 위조꾼 G와 감별사 D',
    text: rich('GAN은 손실 함수 대신 **두 네트워크의 대결**로 배워요. 생성자 G는 노이즈 (64×1×1)를 전치 합성곱으로 7×7 → 14×14 → **28×28** 그림으로 키우고, 판별자 D는 진짜인지 가짜인지 점수(로짓)를 매겨요.'),
    code: dd(String.raw`        self.net = nn.Sequential(
            nn.ConvTranspose2d(Z, 256, 7, 1, 0, bias=False),   # → (256, 7, 7)
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # → (128, 14, 14)
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),               # → (1, 28, 28)
            nn.Tanh())                                          # 출력 [-1, 1]`),
    highlight: [2, 4, 6],
    render: (t, c) => (
      <g>
        <T x={14} y={34} size={FS.s} weight={850} color={c.series[2]}>
          🎨 생성자 G
        </T>
        <g opacity={seg(t, 0, 0.2)}>
          <Cells x={22} y={84} rows={4} cols={1} cell={12} values={(r) => NOISES[0][r][0]} color={c.series[2]} />
          <T x={28} y={150} anchor="middle" size={FS.m} weight={800}>
            z
          </T>
        </g>
        {[
          { s: 7, x: 64, px: 36 },
          { s: 14, x: 118, px: 52 },
          { s: 28, x: 186, px: 72 }
        ].map((b, i) => {
          const a = seg(t, 0.15 + i * 0.15, 0.25);
          const px = b.px;
          return (
            <g key={i} opacity={a}>
              <rect x={b.x} y={110 - px / 2} width={px} height={px} rx={6} fill={alpha(c.series[2], 0.12 + i * 0.1)} stroke={c.series[2]} strokeWidth={2} />
              <T x={b.x + px / 2} y={110 + px / 2 + 18} anchor="middle" size={FS.m} weight={800} mono>
                {b.s}×{b.s}
              </T>
            </g>
          );
        })}
        <Arrow x1={38} y1={110} x2={62} y2={110} color={c.series[2]} t={seg(t, 0.1, 0.2)} />
        <Arrow x1={102} y1={110} x2={116} y2={110} color={c.series[2]} t={seg(t, 0.28, 0.2)} />
        <Arrow x1={172} y1={110} x2={184} y2={110} color={c.series[2]} t={seg(t, 0.43, 0.2)} />
        <Box x={292} y={60} w={112} h={100} color={c.series[0]} t={seg(t, 0.55, 0.25)} label="🕵️ D" sub="진짜/가짜?" />
        <Arrow x1={260} y1={110} x2={290} y2={110} color={c.series[2]} t={seg(t, 0.6, 0.2)} label="가짜" />
        <g opacity={seg(t, 0.65, 0.25)}>
          <Pic img={THREE_IMG} x={306} y={196} cell={5} color={c.series[3]} label="진짜 x" />
          <Arrow x1={348} y1={192} x2={348} y2={164} color={c.series[3]} />
        </g>
        <g opacity={seg(t, 0.8, 0.2)}>
          <Pill x={130} y={228} label="출력은 Tanh → [-1, 1]" color={c.series[2]} size={FS.m} />
          <Pill x={130} y={262} label="D 출력 = 진짜일 로짓" color={c.series[0]} size={FS.m} />
        </g>
      </g>
    )
  },
  {
    title: 'D 스텝 — 진짜는 1, 가짜는 0',
    text: rich('감별사부터 훈련해요. 진짜 그림엔 1, 가짜엔 0을 맞히게. 이때 가짜에 **`detach()`** 를 붙여서 D의 손실이 **G까지 거슬러 올라가지 않게 끊어요** ✂️. 빼먹으면 학습이 꼬이는 GAN의 급소예요.'),
    code: dd(String.raw`            # --- D 스텝: 진짜→1, 가짜→0 ---
            z = torch.randn(B, Z, 1, 1, device=device)
            fake = G(z).detach()                 # detach: D 스텝에서 G로 그래디언트 차단
            loss_d = bce(D(real), ones) + bce(D(fake), zeros)
            opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()`),
    highlight: [3, 4, 5],
    render: (t, c) => (
      <g>
        <Box x={14} y={150} w={80} h={60} color={c.series[2]} label="🎨 G" t={seg(t, 0, 0.2)} />
        <Pic img={NOISES[1]} x={128} y={156} cell={4} color={c.series[2]} t={seg(t, 0.05, 0.2)} />
        <Arrow x1={96} y1={180} x2={124} y2={180} color={c.series[2]} t={seg(t, 0.1, 0.2)} />
        <Pic img={THREE_IMG} x={128} y={50} cell={4} color={c.series[3]} t={seg(t, 0.05, 0.2)} />
        <Box x={250} y={96} w={96} h={74} color={c.series[0]} label="🕵️ D" sub="학습 중" t={seg(t, 0.15, 0.25)} />
        <Arrow x1={176} y1={74} x2={248} y2={118} color={c.series[3]} t={seg(t, 0.25, 0.2)} />
        <Arrow x1={176} y1={180} x2={248} y2={150} color={c.series[2]} t={seg(t, 0.3, 0.2)} />
        <g opacity={seg(t, 0.45, 0.2)}>
          <Pill x={384} y={110} label="→ 1" color={c.pos} size={FS.m} />
          <Pill x={384} y={156} label="→ 0" color={c.neg} size={FS.m} />
          <T x={190} y={42} size={FS.m} weight={800} color={c.series[3]}>
            진짜
          </T>
          <T x={190} y={214} size={FS.m} weight={800} color={c.series[2]}>
            가짜 (detach)
          </T>
        </g>
        <g opacity={seg(t, 0.6, 0.25)}>
          <path d="M248 160 C 200 250, 120 260, 70 214" fill="none" stroke={c.series[0]} strokeWidth={2} strokeDasharray="5 5" />
          <T x={150} y={276} anchor="middle" size={FS.m} weight={800} color={c.series[0]}>
            기울기는 여기서 멈춤
          </T>
          <T x={110} y={244} anchor="middle" size={20}>
            ✂️
          </T>
        </g>
      </g>
    )
  },
  {
    title: 'G 스텝 — D가 "진짜"라고 믿게',
    text: rich('이번엔 G 차례. 가짜를 D에 보여 주고 **"진짜(1)"라고 판정받는 쪽**으로 G를 고쳐요(non-saturating 손실). 기울기가 D를 **통과해** G까지 흘러가지만, 갱신은 `opt_g`만 해요. 이 줄다리기를 반복하면 가짜가 점점 진짜 같아져요.'),
    code: dd(String.raw`            # --- G 스텝: D가 가짜를 1로 믿게 (non-saturating loss) ---
            z = torch.randn(B, Z, 1, 1, device=device)
            loss_g = bce(D(G(z)), ones)
            opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()`),
    highlight: [3, 4],
    render: (t, c) => {
      const prog = seg(t, 0.35, 0.6);
      return (
        <g>
          <Box x={14} y={46} w={80} h={56} color={c.series[2]} label="🎨 G" sub="학습 중" t={seg(t, 0, 0.2)} />
          <Arrow x1={96} y1={74} x2={150} y2={74} color={c.series[2]} t={seg(t, 0.08, 0.2)} />
          <Box x={152} y={46} w={88} h={56} color={c.series[0]} label="🕵️ D" sub="고정" t={seg(t, 0.12, 0.2)} />
          <Arrow x1={242} y1={74} x2={290} y2={74} color={c.series[0]} t={seg(t, 0.18, 0.2)} />
          <Pill x={350} y={74} label="1이라고 해 줘!" color={c.pos} t={seg(t, 0.22, 0.2)} size={FS.m} />
          <path d="M290 92 C 250 130, 120 130, 60 104" fill="none" stroke={c.series[2]} strokeWidth={2.5} strokeDasharray="6 5" opacity={seg(t, 0.28, 0.2)} />
          {seg(t, 0.28, 0.2) >= 1 && <Pulse path="M290 92 C 250 130, 120 130, 60 104" color={c.series[2]} dur={1.8} />}
          <T x={175} y={138} anchor="middle" size={FS.m} weight={800} color={c.series[2]} opacity={seg(t, 0.3, 0.2)}>
            기울기: D를 통과해 G로
          </T>
          <T x={14} y={180} size={FS.s} weight={850} color="var(--text-2)">
            에포크가 지날수록 (fixed_z로 관찰)
          </T>
          {[0, 1, 2, 3].map((i) => {
            // later epochs: less noise, more digit (illustration)
            const img = mix(NOISES[i], SAMPLES[i], [0.08, 0.4, 0.75, 1][i] * (0.6 + 0.4 * prog));
            return (
              <g key={i}>
                <Pic img={img} x={24 + i * 98} y={196} cell={5} color={c.series[2]} t={seg(t, 0.35 + i * 0.12, 0.2)} />
                <T x={52 + i * 98} y={272} anchor="middle" size={FS.m} weight={750} color="var(--text-3)" opacity={seg(t, 0.35 + i * 0.12, 0.2)}>
                  {['1', '5', '12', '25'][i]}에포크
                </T>
              </g>
            );
          })}
        </g>
      );
    }
  }
];

export default function Lab14VaeGan(_: VizProps) {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={2} />;
}
