import type { VizProps } from '../Viz';
import { CodeStory, type StoryStep } from '../core/story';
import type { VizColors } from '../core/colors';
import { VB, FS, seg, alpha, rich, dd, T, Box, Arrow, Pill, Cells } from './_LabKitB';

/* A short CartPole-like trajectory: reward 1 per step, falls (terminates) after step 5. */
const GAMMA = 0.99;
const LAM = 0.95;
const REW = [1, 1, 1, 1, 1, 1];
const VAL = [8.2, 7.5, 6.1, 4.8, 3.0, 1.2];
const DONE = [0, 0, 0, 0, 0, 1];
const GAE = (() => {
  const adv = new Array(REW.length).fill(0);
  const delta = new Array(REW.length).fill(0);
  let g = 0;
  for (let t = REW.length - 1; t >= 0; t--) {
    const nextV = t === REW.length - 1 ? 0 : VAL[t + 1];
    const nonterminal = 1 - DONE[t];
    delta[t] = REW[t] + GAMMA * nextV * nonterminal - VAL[t];
    g = delta[t] + GAMMA * LAM * nonterminal * g;
    adv[t] = g;
  }
  return { adv, delta };
})();

const RAW_ADV = [2.4, -0.6, 1.1, -3.2, 0.3, 4.1, -1.5, 0.8];
const MEAN = RAW_ADV.reduce((a, b) => a + b, 0) / RAW_ADV.length;
const STD = Math.sqrt(RAW_ADV.reduce((a, b) => a + (b - MEAN) ** 2, 0) / (RAW_ADV.length - 1));
const NORM_ADV = RAW_ADV.map((v) => (v - MEAN) / (STD + 1e-8));

function ClipPlot({ x, y, w, h, sign, c, t }: { x: number; y: number; w: number; h: number; sign: 1 | -1; c: VizColors; t: number }) {
  const eps = 0.2;
  const r0 = 0.4;
  const r1 = 1.6;
  const A = sign;
  const f = (r: number) => Math.min(r * A, Math.min(Math.max(r, 1 - eps), 1 + eps) * A);
  const yMin = A > 0 ? 0.3 : -1.7;
  const yMax = A > 0 ? 1.5 : -0.2;
  const px = (r: number) => x + ((r - r0) / (r1 - r0)) * w;
  const py = (v: number) => y + h - ((v - yMin) / (yMax - yMin)) * h;
  const pts: string[] = [];
  const raw: string[] = [];
  for (let i = 0; i <= 60; i++) {
    const r = r0 + ((r1 - r0) * i) / 60;
    pts.push(`${px(r).toFixed(1)},${py(f(r)).toFixed(1)}`);
    raw.push(`${px(r).toFixed(1)},${py(r * A).toFixed(1)}`);
  }
  const probe = r0 + (r1 - r0) * seg(t, 0.25, 0.7);
  const col = A > 0 ? c.pos : c.neg;
  return (
    <g>
      <rect x={px(1 - eps)} y={y} width={px(1 + eps) - px(1 - eps)} height={h} fill={alpha(c.series[3], 0.14)} />
      <line x1={x} y1={y + h} x2={x + w} y2={y + h} stroke="var(--viz-axis)" />
      <line x1={px(1)} y1={y} x2={px(1)} y2={y + h} stroke="var(--border-strong)" strokeDasharray="3 3" />
      <polyline points={raw.join(' ')} fill="none" stroke="var(--text-3)" strokeWidth={1.5} strokeDasharray="4 4" />
      <polyline points={pts.join(' ')} fill="none" stroke={col} strokeWidth={3.5} strokeLinejoin="round" />
      <circle cx={px(probe)} cy={py(f(probe))} r={6} fill={col} stroke="var(--surface-solid)" strokeWidth={2} />
      <T x={x + w / 2} y={y - 10} anchor="middle" size={FS.s} weight={850} color={col}>
        {A > 0 ? '이점 A > 0 (좋은 행동)' : '이점 A < 0 (나쁜 행동)'}
      </T>
      <T x={px(1 - eps)} y={y + h + 16} anchor="middle" size={11.5} weight={750} mono>
        0.8
      </T>
      <T x={px(1 + eps)} y={y + h + 16} anchor="middle" size={11.5} weight={750} mono>
        1.2
      </T>
      <T x={x + w} y={y + h + 16} anchor="end" size={11.5} weight={700} color="var(--text-3)">
        r
      </T>
    </g>
  );
}

const steps: StoryStep[] = [
  {
    title: '액터-크리틱 — 몸통 하나, 머리 둘',
    text: rich('같은 몸통에서 두 머리가 나와요. **액터(π)** 는 "왼쪽 40%, 오른쪽 60%"처럼 **행동의 확률**을, **크리틱(V)** 은 "이 상태면 평균 몇 점쯤"이라는 **기준선**을 내요. 크리틱 덕분에 "평균보다 나았나?"를 잴 수 있어요.'),
    code: dd(String.raw`class ActorCritic(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                      nn.Linear(64, 64), nn.Tanh())
        self.pi = nn.Linear(64, n_actions)     # 정책 로짓
        self.v = nn.Linear(64, 1)              # 상태 가치`),
    highlight: [6, 7],
    render: (t, c) => (
      <g>
        <g opacity={seg(t, 0, 0.2)}>
          <Cells x={20} y={104} rows={4} cols={1} cell={18} values={(r) => [0.3, 0.6, 0.45, 0.8][r]} color={c.series[3]} />
          <T x={29} y={196} anchor="middle" size={FS.m} weight={800}>
            s
          </T>
          <T x={29} y={92} anchor="middle" size={11.5} weight={700} color="var(--text-2)">
            상태 4칸
          </T>
        </g>
        <Arrow x1={44} y1={140} x2={74} y2={140} color={c.series[3]} t={seg(t, 0.12, 0.2)} />
        <Box x={78} y={100} w={98} h={80} color={c.series[0]} t={seg(t, 0.15, 0.25)} label="몸통" sub="64 → 64 (tanh)" />
        <Arrow x1={178} y1={124} x2={226} y2={80} color={c.series[2]} t={seg(t, 0.35, 0.2)} />
        <Arrow x1={178} y1={156} x2={226} y2={204} color={c.series[1]} t={seg(t, 0.4, 0.2)} />
        <g opacity={seg(t, 0.45, 0.25)}>
          <T x={232} y={42} size={FS.s} weight={850} color={c.series[2]}>
            🎭 액터 π(a|s)
          </T>
          {[
            { l: '← 왼쪽', p: 0.4 },
            { l: '→ 오른쪽', p: 0.6 }
          ].map((b, i) => (
            <g key={i}>
              <T x={232} y={74 + i * 28} size={FS.m} weight={750}>
                {b.l}
              </T>
              <rect x={300} y={62 + i * 28} width={100} height={14} rx={7} fill="var(--surface-3)" />
              <rect x={300} y={62 + i * 28} width={100 * b.p * seg(t, 0.5, 0.3)} height={14} rx={7} fill={c.series[2]} />
              <T x={396} y={74 + i * 28} anchor="end" size={11.5} weight={800} mono>
                {b.p.toFixed(1)}
              </T>
            </g>
          ))}
        </g>
        <g opacity={seg(t, 0.55, 0.25)}>
          <T x={232} y={186} size={FS.s} weight={850} color={c.series[1]}>
            📏 크리틱 V(s)
          </T>
          <rect x={232} y={200} width={170} height={44} rx={12} fill={alpha(c.series[1], 0.16)} stroke={c.series[1]} strokeWidth={2} />
          <T x={317} y={228} anchor="middle" size={FS.h} weight={900} mono>
            약 23.5점
          </T>
          <T x={317} y={266} anchor="middle" size={11.5} weight={700} color="var(--text-2)">
            "이 상태면 보통 이 정도"
          </T>
        </g>
      </g>
    )
  },
  {
    title: '롤아웃 — 지금 정책으로 2048걸음 모으기',
    text: rich('먼저 **현재 정책 그대로** 게임을 2048걸음 해 보며 (상태, 행동, 보상, 가치)를 모아요. 이때 고른 행동의 **로그확률 logp를 꼭 저장**해요 — 나중에 "정책이 얼마나 바뀌었나"를 잴 기준(π_old)이 되거든요.'),
    code: dd(String.raw`            st = torch.tensor(s, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist, v = model(st)
                a = dist.sample()
                logp = dist.log_prob(a)        # π_old의 로그확률 — 클리핑 비율의 분모`),
    highlight: [2, 4, 5],
    render: (t, c) => {
      const n = 12;
      const shown = Math.ceil(n * seg(t, 0.05, 0.7));
      return (
        <g>
          <T x={14} y={34} size={FS.s} weight={850}>
            경험 버퍼 (on-policy · 한 번 쓰고 버려요)
          </T>
          {Array.from({ length: n }, (_, i) => {
            const x = 14 + i * 33;
            const done = i === 6;
            const visible = i < shown;
            return (
              <g key={i} opacity={visible ? 1 : 0.15}>
                <rect x={x} y={56} width={28} height={140} rx={8} fill={alpha(done ? c.neg : c.series[0], 0.12)} stroke={done ? c.neg : c.series[0]} strokeWidth={1.5} />
                <T x={x + 14} y={78} anchor="middle" size={11.5} weight={800}>
                  s
                </T>
                <T x={x + 14} y={104} anchor="middle" size={14}>
                  {i % 3 === 1 ? '←' : '→'}
                </T>
                <T x={x + 14} y={130} anchor="middle" size={11.5} weight={800} color={c.pos}>
                  +1
                </T>
                <rect x={x + 4} y={142} width={20} height={20} rx={5} fill={alpha(c.series[2], 0.35)} stroke={c.series[2]} />
                <T x={x + 14} y={186} anchor="middle" size={11} weight={700} color="var(--text-2)">
                  V
                </T>
              </g>
            );
          })}
          <g opacity={seg(t, 0.55, 0.2)}>
            <rect x={14} y={140} width={392} height={24} rx={8} fill="none" stroke={c.series[2]} strokeWidth={2} strokeDasharray="5 4" />
            <T x={210} y={226} anchor="middle" size={FS.s} weight={850} color={c.series[2]}>
              logp(π_old) 저장 — 클리핑의 기준
            </T>
            <T x={14 + 6 * 33 + 14} y={214} anchor="middle" size={11.5} weight={800} color={c.neg}>
              넘어짐(done)
            </T>
          </g>
          <T x={210} y={262} anchor="middle" size={FS.m} weight={750} color="var(--text-2)">
            {Math.round(2048 * seg(t, 0.05, 0.7))} / 2048 걸음 수집
          </T>
        </g>
      );
    }
  },
  {
    title: 'GAE — 뒤에서부터 "평균보다 얼마나 좋았나"',
    text: rich('각 걸음의 **δ = r + γ·V(다음) - V(지금)** 은 "예상보다 얼마나 좋았나"예요. 이걸 **뒤에서부터** γλ씩 줄여 가며 더하면 이점 A가 돼요. 마지막 걸음에서 넘어졌으니(done) 그 뒤 가치는 0으로 끊어요.'),
    code: dd(String.raw`    adv = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = last_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_v * nonterminal - values[t]
        gae = delta + GAMMA * LAM * nonterminal * gae
        adv[t] = gae`),
    highlight: [3, 6, 7],
    render: (t, c) => {
      const n = REW.length;
      const sweep = seg(t, 0.1, 0.75);
      const cutoff = n - Math.floor(sweep * (n + 0.999));
      const maxA = Math.max(...GAE.adv.map(Math.abs));
      return (
        <g>
          {['r', 'V', 'δ', 'A'].map((lab, k) => (
            <T key={lab} x={22} y={66 + k * 52} anchor="middle" size={FS.s} weight={900} color={[c.pos, c.series[1], c.series[3], c.series[2]][k]}>
              {lab}
            </T>
          ))}
          {REW.map((r, i) => {
            const x = 46 + i * 60;
            const computed = i >= cutoff;
            return (
              <g key={i}>
                <T x={x + 24} y={36} anchor="middle" size={11.5} weight={700} color="var(--text-3)" mono>
                  t={i}
                </T>
                <rect x={x} y={50} width={48} height={24} rx={7} fill={alpha(c.pos, 0.14)} stroke={c.pos} />
                <T x={x + 24} y={67} anchor="middle" size={FS.m} weight={800} mono>
                  +{r}
                </T>
                <rect x={x} y={102} width={48} height={24} rx={7} fill={alpha(c.series[1], 0.14)} stroke={c.series[1]} />
                <T x={x + 24} y={119} anchor="middle" size={FS.m} weight={800} mono>
                  {VAL[i].toFixed(1)}
                </T>
                <g opacity={computed ? 1 : 0.15}>
                  <rect x={x} y={154} width={48} height={24} rx={7} fill={alpha(c.series[3], 0.14)} stroke={c.series[3]} />
                  <T x={x + 24} y={171} anchor="middle" size={FS.m} weight={800} mono>
                    {GAE.delta[i].toFixed(2)}
                  </T>
                  <rect x={x} y={206} width={48} height={24} rx={7} fill={alpha(c.series[2], 0.12 + 0.3 * (Math.abs(GAE.adv[i]) / maxA))} stroke={c.series[2]} />
                  <T x={x + 24} y={223} anchor="middle" size={FS.m} weight={850} mono>
                    {GAE.adv[i].toFixed(2)}
                  </T>
                </g>
                {DONE[i] === 1 && (
                  <T x={x + 24} y={252} anchor="middle" size={11.5} weight={850} color={c.neg}>
                    done ✗
                  </T>
                )}
              </g>
            );
          })}
          {sweep < 1 && cutoff < n && (
            <Arrow x1={46 + Math.min(n - 1, cutoff + 1) * 60 + 24} y1={196} x2={46 + cutoff * 60 + 30} y2={196} color={c.series[2]} width={3} />
          )}
          <T x={210} y={282} anchor="middle" size={FS.m} weight={750} color="var(--text-2)" opacity={seg(t, 0.85, 0.15)}>
            ← 뒤에서부터 계산 (γ=0.99, λ=0.95)
          </T>
        </g>
      );
    }
  },
  {
    title: '이점 정규화 — 크기를 고르게',
    text: rich('모은 이점들은 크기가 들쭉날쭉해요. **평균을 빼고 표준편차로 나눠** 평균 0, 퍼짐 1로 맞추면 학습이 훨씬 안정돼요. 부호(좋았나/나빴나)는 그대로 지켜져요.'),
    code: dd(String.raw`        adv, returns = compute_gae(rews, vals, dones_t, last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # 이점 정규화: 안정성의 표준 트릭`),
    highlight: [2],
    render: (t, c) => {
      const m = seg(t, 0.25, 0.55);
      const vals = RAW_ADV.map((v, i) => v + (NORM_ADV[i] - v) * m);
      const scale = 22;
      const base = 160;
      return (
        <g>
          <line x1={24} y1={base} x2={404} y2={base} stroke="var(--viz-axis)" strokeWidth={1.5} />
          <T x={404} y={base - 8} anchor="end" size={11.5} weight={700} color="var(--text-3)">
            0
          </T>
          {vals.map((v, i) => {
            const x = 40 + i * 46;
            const h = v * scale;
            const col = v >= 0 ? c.pos : c.neg;
            return (
              <g key={i}>
                <rect x={x} y={h >= 0 ? base - h : base} width={30} height={Math.abs(h)} rx={6} fill={alpha(col, 0.7)} />
                <T x={x + 15} y={h >= 0 ? base - h - 6 : base - h + 16} anchor="middle" size={11.5} weight={800} mono>
                  {v.toFixed(1)}
                </T>
              </g>
            );
          })}
          <Pill x={210} y={36} label={m < 1 ? '정규화 전: 크기가 제각각' : '정규화 후: 평균 0 · 표준편차 1'} color={m < 1 ? c.series[3] : c.series[1]} size={FS.m} />
        </g>
      );
    }
  },
  {
    title: '클리핑 — 좋은 쪽으로도 한 번에 조금만',
    text: rich('비율 **r = π_new / π_old** 가 1에서 너무 멀어지면(0.8~1.2 밖) 더 이상 이득을 주지 않아요. 좋은 행동은 1.2배까지만 늘리고, 나쁜 행동은 0.8배까지만 줄여요. 이 **보수성** 덕분에 같은 데이터로 여러 번 학습해도 정책이 망가지지 않아요.'),
    code: dd(String.raw`                ratio = (logp_new - logps_old[b]).exp()          # π_new / π_old

                # PPO-clip 목적: 이점 방향으로, 그러나 비율은 [1-ε, 1+ε] 안에서만
                surr1 = ratio * adv[b]
                surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv[b]
                policy_loss = -torch.min(surr1, surr2).mean()`),
    highlight: [1, 5, 6],
    render: (t, c) => (
      <g>
        <ClipPlot x={18} y={64} w={176} h={150} sign={1} c={c} t={t} />
        <ClipPlot x={226} y={64} w={176} h={150} sign={-1} c={c} t={t} />
        <g opacity={seg(t, 0.6, 0.25)}>
          <rect x={60} y={252} width={14} height={10} fill={alpha(c.series[3], 0.3)} />
          <T x={80} y={262} size={FS.m} weight={750} color="var(--text-2)">
            안전 구간 [1-ε, 1+ε]
          </T>
          <line x1={234} y1={257} x2={254} y2={257} stroke="var(--text-3)" strokeDasharray="4 4" strokeWidth={1.5} />
          <T x={260} y={262} size={FS.m} weight={750} color="var(--text-2)">
            클리핑 없을 때
          </T>
        </g>
      </g>
    )
  },
  {
    title: '손실 세 가지를 한 번에',
    text: rich('정책 손실에 **크리틱의 회귀 손실(×0.5)** 을 더하고, **엔트로피(×0.01)** 는 빼요. 엔트로피 보너스는 "너무 일찍 한 행동만 고집하지 말고 계속 탐색해"라는 장치예요. 마지막에 기울기 크기를 0.5로 잘라 폭주를 막아요.'),
    code: dd(String.raw`                value_loss = (v - returns[b]).pow(2).mean()      # 크리틱: 리턴 회귀
                entropy = dist.entropy().mean()                  # 탐색 유지 보너스

                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()`),
    highlight: [4, 7],
    render: (t, c) => {
      const parts = [
        { label: '정책 손실', coef: '× 1', v: 0.5, color: c.series[2], sign: '+' },
        { label: '가치 손실', coef: '× 0.5', v: 0.36, color: c.series[1], sign: '+' },
        { label: '엔트로피', coef: '× 0.01', v: 0.08, color: c.series[3], sign: '-' }
      ];
      return (
        <g>
          {parts.map((p, i) => {
            const a = seg(t, 0.08 + i * 0.15, 0.25);
            const y = 44 + i * 58;
            return (
              <g key={i} opacity={a}>
                <T x={20} y={y + 20} size={FS.s} weight={850} color={p.color}>
                  {p.sign} {p.label}
                </T>
                <T x={20} y={y + 38} size={11.5} weight={700} color="var(--text-2)" mono>
                  {p.coef}
                </T>
                <rect x={128} y={y + 8} width={270} height={22} rx={11} fill="var(--surface-3)" />
                <rect x={128} y={y + 8} width={270 * p.v} height={22} rx={11} fill={p.color} />
              </g>
            );
          })}
          <g opacity={seg(t, 0.6, 0.25)}>
            <line x1={20} y1={222} x2={400} y2={222} stroke="var(--border-strong)" />
            <T x={20} y={250} size={FS.s} weight={900}>
              = loss
            </T>
            <rect x={128} y={236} width={270} height={22} rx={11} fill="var(--surface-3)" />
            <rect x={128} y={236} width={270 * 0.78} height={22} rx={11} fill={c.series[0]} />
            <T x={210} y={284} anchor="middle" size={11.5} weight={750} color="var(--text-2)">
              ✂️ 기울기 크기는 최대 0.5까지 (clip_grad_norm_)
            </T>
          </g>
        </g>
      );
    }
  },
  {
    title: '같은 데이터로 10번 — 그리고 버리기',
    text: rich('2048개 경험을 섞어 64개씩 미니배치로 나누고(32묶음), 이걸 **10 에포크** 반복해요. 클리핑이 있어서 가능한 재사용이에요. 끝나면 경험을 **버리고** 바뀐 정책으로 새로 모아요. RLHF에서는 액터가 LLM, 행동이 다음 토큰, 보상이 보상 모델 점수예요.'),
    code: dd(String.raw`        idx = torch.arange(ROLLOUT_STEPS, device=device)
        for _ in range(EPOCHS_PER_UPDATE):
            for start in range(0, ROLLOUT_STEPS, MINIBATCH):
                b = idx[torch.randperm(ROLLOUT_STEPS, device=device)[start:start + MINIBATCH]]`),
    highlight: [2, 3, 4],
    render: (t, c) => {
      const p = seg(t, 0.05, 0.8);
      const total = 320;
      const done = Math.floor(total * p);
      const epoch = Math.min(10, Math.floor(done / 32) + 1);
      const inEpoch = done % 32;
      return (
        <g>
          <T x={14} y={34} size={FS.s} weight={850}>
            미니배치 32묶음 × 에포크 {epoch}/10
          </T>
          <Cells
            x={14}
            y={48}
            rows={4}
            cols={8}
            cell={22}
            gap={4}
            values={(r, k) => (p >= 1 ? 0.85 : r * 8 + k < inEpoch ? 0.85 : 0.15)}
            color={c.series[0]}
          />
          <g opacity={seg(t, 0.2, 0.2)}>
            {Array.from({ length: 10 }, (_, k) => (
              <rect key={k} x={250 + k * 15} y={58} width={11} height={70} rx={4} fill={k < epoch ? c.series[2] : 'var(--surface-3)'} />
            ))}
            <T x={250} y={148} size={11.5} weight={750} color="var(--text-2)">
              에포크 진행
            </T>
          </g>
          <g opacity={seg(t, 0.82, 0.18)}>
            <Pill x={210} y={180} label="🗑️ 다 쓴 경험은 버리고 새로 수집 (on-policy)" color={c.neg} size={FS.m} />
            <T x={210} y={222} anchor="middle" size={FS.m} weight={850} color={c.series[3]}>
              RLHF: 액터 = LLM · 행동 = 토큰 · 보상 = 보상 모델
            </T>
            <T x={210} y={246} anchor="middle" size={11.5} weight={700} color="var(--text-2)">
              에피소드 = 응답 하나, 구조는 이 코드 그대로
            </T>
          </g>
        </g>
      );
    }
  }
];

export default function Lab17Ppo(_: VizProps) {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={2} />;
}
