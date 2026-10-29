import { useEffect, useRef, useState } from 'react';
import type { VizProps } from '../Viz';
import type { AlgoMood } from '@/components/mascot/Algo';
import { CodeBlock } from '../core/story';
import { Button, Controls, Legend, Note, Segmented, Stat } from '../core/ui';
import { useCanvas, useElementSize, useInView, useRafLoop } from '../core/hooks';
import { mixHex, useVizColors, type VizColors } from '../core/colors';
import { alpha, dd, hx, rich } from './_LabKitB';
import { CartPole, DqnAgent, QNet, THETA_LIMIT, X_LIMIT, evaluate, mulberry } from './_dqnB';

type Mode = 'idle' | 'train' | 'demo';
type Speed = 'real' | 'fast' | 'max';
type Tab = 'eps' | 'td' | 'sync' | 'done';

/* The lab's recipe, scaled down to train in a browser tab (compared in the table at the bottom). */
const CFG = { hidden: 32, lr: 5e-4, batch: 32, bufferSize: 20000, learnStart: 500, epsDecay: 1500 };
const DEFAULT_SEED = 7;
const DEFAULT_SYNC = 500;
const MAX_TRAIN_STEPS = 60000;
const SOLVE = 475;
const RATE: Record<Exclude<Speed, 'max'>, number> = { real: 50, fast: 1000 };
const FRAME_BUDGET_MS = 12;
const COLS = 25;
const CELLS = 250;
const PER_CELL = CFG.bufferSize / CELLS; // 80 transitions per cell

const makeAgent = (targetSync: number, seed: number) => new DqnAgent({ ...CFG, targetSync, seed });

/** Short numbers for values that can explode (diverging Q, TD loss). */
const compact = (v: number, digits = 1) => {
  const a = Math.abs(v);
  if (!Number.isFinite(v)) return '∞';
  if (a >= 1e12) return v.toExponential(1);
  if (a >= 1e8) return `${(v / 1e8).toFixed(1)}억`;
  if (a >= 1e4) return `${(v / 1e4).toFixed(1)}만`;
  if (a >= 1000) return Math.round(v).toLocaleString('ko-KR');
  return v.toFixed(digits);
};

const CODE: Record<Tab, { label: string; code: string; hl: number[]; text: string }> = {
  eps: {
    label: 'ε-greedy',
    code: dd(`
            # ε-greedy: 탐색 확률을 스텝에 따라 지수 감쇠
            eps = EPS_END + (EPS_START - EPS_END) * (0.99 ** (global_step / EPS_DECAY * 100))
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = q(torch.tensor(s, dtype=torch.float32, device=device)).argmax().item()`),
    hl: [2, 3, 4, 7],
    text: '**ε의 확률**로는 무작위(🎲 화살표), 나머지는 Q값이 더 큰 쪽(🧠 화살표)을 골라요. ε는 1에서 0.05까지 점점 줄어요.'
  },
  td: {
    label: '버퍼·TD 타깃',
    code: dd(`
            # ---- 학습 스텝 ----
            if len(buffer) >= 1_000:
                bs, ba, br, bs2, bdone = buffer.sample(BATCH)
                # 현재 추정: Q(s, a) — 실제 취한 행동의 Q만 gather
                q_sa = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # TD 타깃: r + γ·max Q_target(s') — 타깃넷으로 계산 (자기추격 차단)
                    max_q2 = q_target(bs2).max(dim=1).values
                    target = br + GAMMA * max_q2 * (1 - bdone)
                loss = F.smooth_l1_loss(q_sa, target)   # Huber: 이상치 TD 오차에 견고
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()`),
    hl: [3, 5, 8, 9],
    text: "버퍼에서 **무작위로 한 묶음**을 뽑아(버퍼 그림 곳곳에 흩어진 칸) `Q(s,a)`를 `r + γ·max Q_target(s')` 쪽으로 당겨요. 연달아 일어난 비슷한 경험끼리 몰리지 않게 섞는 게 핵심이에요."
  },
  sync: {
    label: '타깃 동기화',
    code: `TARGET_SYNC = 500                                   # 타깃 네트워크 동기화 주기
# ...
    q, q_target = QNet().to(device), QNet().to(device)
    q_target.load_state_dict(q.state_dict())
# ...
            if global_step % TARGET_SYNC == 0:
                q_target.load_state_dict(q.state_dict())`,
    hl: [1, 6, 7],
    text: '정답(과녁)을 계산하는 `Q_target`은 **N스텝마다 한 번만** 복사돼요. 그동안 과녁이 가만히 있어서 학습이 안정돼요. 아래에서 1로 바꾸면 매 스텝 과녁이 따라 움직여요.'
  },
  done: {
    label: '종료 처리',
    code: dd(`
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            # truncated(시간 초과)는 실패가 아니다 — done으로 부트스트랩을 끊으면 안 된다
            buffer.push(s, a, r, s2, float(terminated))
            # ...
                    target = br + GAMMA * max_q2 * (1 - bdone)`),
    hl: [1, 4, 6],
    text: '막대가 쓰러지거나 선을 넘으면 `terminated`(진짜 실패), 500스텝을 버티면 `truncated`(시간 초과)예요. 시간 초과는 실패가 아니라서 `done=0`으로 저장하고 미래 가치를 계속 더해요.'
  }
};

interface Hist {
  steps: number[];
  rewards: number[];
  means: number[];
}
interface Ghost {
  label: string;
  pts: [number, number][];
}
interface Demo {
  env: CartPole;
  action: number;
  acc: number;
  hold: number;
  result: string | null;
  pose: number[] | null;
  t: number;
  runs: number;
  scores: number[];
}
interface Checkpoint {
  net: QNet;
  step: number;
  ep: number;
}
interface Hold {
  s: number[];
  t: number;
  left: number;
  text: string;
}
interface SceneView {
  s: number[];
  action: number;
  explore: boolean;
  showForce: boolean;
  q: [number, number];
  label: string;
  flash: { text: string; good: boolean } | null;
}

const endText = (s2: number[], truncated: boolean) =>
  truncated ? '500스텝 만점! 🎉' : Math.abs(s2[0]) > X_LIMIT ? '선 밖으로 나갔어요!' : '막대가 쓰러졌어요!';

export default function Lab16DqnCartpole({ expanded }: VizProps) {
  const c = useVizColors();
  const [box, { width }] = useElementSize<HTMLDivElement>();
  const [stage, { width: stageW }] = useElementSize<HTMLDivElement>();
  const inView = useInView(box, '0px');
  const [mode, setMode] = useState<Mode>('idle');
  const [speed, setSpeed] = useState<Speed>('fast');
  const [sync, setSync] = useState(DEFAULT_SYNC);
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [tab, setTab] = useState<Tab>('eps');
  const [useCkpt, setUseCkpt] = useState(true);
  const [tick, setTick] = useState(0);

  const agentRef = useRef<DqnAgent | null>(null);
  if (!agentRef.current) agentRef.current = makeAgent(DEFAULT_SYNC, DEFAULT_SEED);
  const ag = agentRef.current;
  const hist = useRef<Hist>({ steps: [], rewards: [], means: [] });
  const ghosts = useRef<Ghost[]>([]);
  const result = useRef<{ solvedAt: number | null; solvedEp: number; capped: boolean }>({ solvedAt: null, solvedEp: 0, capped: false });
  const demo = useRef<Demo | null>(null);
  /* Like real projects: when an episode hits 500, give the greedy policy a short test (ε = 0)
     and keep the weights that passed — the final weights may already have started to forget. */
  const ckpt = useRef<Checkpoint | null>(null);
  const passes = useRef<number[]>([]);
  const hold = useRef<Hold | null>(null);
  const acc = useRef(0);
  const uiAcc = useRef(0);
  const seenSyncs = useRef(0);
  const syncFlashAt = useRef(-1e9);
  const font = useRef('sans-serif');

  useEffect(() => {
    font.current = getComputedStyle(document.body).fontFamily || 'sans-serif';
  }, []);

  const wide = width > 720;
  const narrow = width > 0 && width < 520;
  const cw = Math.max(280, stageW);
  const scale = (cw - 32) / (2 * X_LIMIT * 1.16);
  const stageH = Math.round(Math.min(expanded ? 380 : 330, Math.max(cw < 380 ? 224 : 200, scale * visScale(cw) * 1.33 + 84)));

  const newDemo = (): Demo => ({
    env: new CartPole(mulberry(seed * 7919 + 17)),
    action: 0,
    acc: 0,
    hold: 0,
    result: null,
    pose: null,
    t: 0,
    runs: 0,
    scores: []
  });

  const demoNet = () => (useCkpt && ckpt.current ? ckpt.current.net : ag.q);

  const view = (): SceneView => {
    const d = demo.current;
    if (mode === 'demo' && d) {
      const s = d.pose ?? d.env.s;
      return {
        s,
        action: d.action,
        explore: false,
        showForce: d.t > 0,
        q: demoNet().forward(s),
        label: `🎬 시연${useCkpt && ckpt.current ? ' 🏆' : ''} · ${d.t}스텝`,
        flash: d.hold > 0 && d.result ? { text: d.result, good: d.t >= 500 } : null
      };
    }
    const h = hold.current;
    const s = h ? h.s : ag.env.s;
    const t = h ? h.t : ag.env.t;
    return {
      s,
      action: ag.lastAction,
      explore: ag.lastExplore,
      showForce: ag.step > 0,
      q: ag.q.forward(s),
      label: `에피소드 ${h ? ag.episode : ag.episode + 1} · ${t}스텝`,
      flash: h ? { text: h.text, good: h.t >= 500 } : null
    };
  };

  const [canvasRef, paint] = useCanvas((ctx, w, h) => drawScene(ctx, w, h, c, view(), font.current), [c, mode, stageH, tick]);

  const train = (n: number, deadline: number): 'ok' | 'solved' | 'capped' | 'hold' => {
    const H = hist.current;
    for (let i = 0; i < n; i++) {
      if ((i & 15) === 15 && performance.now() > deadline) break;
      const ended = ag.tick();
      if (ag.syncs !== seenSyncs.current) {
        seenSyncs.current = ag.syncs;
        syncFlashAt.current = performance.now();
      }
      if (ended) {
        const r = ag.episodeRewards[ag.episodeRewards.length - 1];
        const m = ag.recentMean(20);
        H.steps.push(ag.step);
        H.rewards.push(r);
        H.means.push(m);
        if (r >= 500 && evaluate(ag.q, 3, 90001 + ag.episode * 7).every((v) => v >= 500)) {
          const net = ckpt.current?.net ?? new QNet(CFG.hidden);
          net.copyFrom(ag.q);
          ckpt.current = { net, step: ag.step, ep: ag.episode };
          passes.current.push(ag.step);
        }
        if (ag.episodeRewards.length >= 20 && m >= SOLVE) {
          result.current = { solvedAt: ag.step, solvedEp: ag.episode, capped: false };
          return 'solved';
        }
        if (speed === 'real' && ag.last) {
          hold.current = { s: ag.last.s2, t: r, left: 0.8, text: endText(ag.last.s2, ag.last.truncated) };
          return 'hold';
        }
      }
      if (ag.step >= MAX_TRAIN_STEPS) {
        result.current = { ...result.current, capped: true };
        return 'capped';
      }
    }
    return 'ok';
  };

  const stepDemo = (dt: number) => {
    const d = demo.current ?? (demo.current = newDemo());
    if (d.hold > 0) {
      d.hold -= dt;
      if (d.hold <= 0) {
        d.env.reset();
        d.pose = null;
        d.result = null;
        d.t = 0;
      }
      return;
    }
    d.acc += dt * 50;
    let n = Math.floor(d.acc);
    d.acc -= n;
    const net = demoNet();
    while (n-- > 0) {
      const [q0, q1] = net.forward(d.env.s);
      const a = q1 > q0 ? 1 : 0;
      d.action = a;
      const res = d.env.step(a);
      d.t = d.env.t;
      if (res.terminated || res.truncated) {
        d.pose = res.s;
        d.result = endText(res.s, res.truncated);
        d.runs += 1;
        d.scores = [...d.scores.slice(-4), d.t];
        d.hold = 1.2;
        break;
      }
    }
  };

  useRafLoop(
    (dt) => {
      if (mode === 'demo') stepDemo(dt);
      else if (mode === 'train') {
        if (hold.current) {
          hold.current.left -= dt;
          if (hold.current.left <= 0) hold.current = null;
        } else {
          let n = Number.MAX_SAFE_INTEGER;
          if (speed !== 'max') {
            acc.current += dt * RATE[speed];
            n = Math.floor(acc.current);
            acc.current -= n;
          }
          const res = train(n, performance.now() + FRAME_BUDGET_MS);
          if (res === 'solved') {
            demo.current = newDemo();
            setMode('demo');
          } else if (res === 'capped') setMode('idle');
        }
      }
      paint();
      uiAcc.current += dt;
      if (uiAcc.current > 0.12) {
        uiAcc.current = 0;
        setTick((v) => v + 1);
      }
    },
    mode !== 'idle' && inView
  );

  const resetRun = (nextSync: number, nextSeed: number) => {
    const H = hist.current;
    if (ag.step >= 1000 && H.steps.length > 1) {
      const pts = bucketize(H.steps, H.means, ag.step, 160, 'last');
      ghosts.current = [...ghosts.current, { label: `sync=${sync} · 시드 ${seed}`, pts }].slice(-2);
    }
    agentRef.current = makeAgent(nextSync, nextSeed);
    hist.current = { steps: [], rewards: [], means: [] };
    result.current = { solvedAt: null, solvedEp: 0, capped: false };
    ckpt.current = null;
    passes.current = [];
    demo.current = null;
    hold.current = null;
    acc.current = 0;
    seenSyncs.current = 0;
    syncFlashAt.current = -1e9;
    setUseCkpt(true);
    setMode('idle');
    setTick((v) => v + 1);
  };

  const finished = result.current.solvedAt !== null || result.current.capped;
  const onPlay = () => {
    if (mode === 'train') return setMode('idle');
    if (finished) {
      resetRun(sync, seed);
      setMode('train');
      return;
    }
    hold.current = null;
    setMode('train');
  };
  const onDemo = () => {
    if (mode === 'demo') return setMode('idle');
    demo.current = newDemo();
    setMode('demo');
  };
  const onSync = (v: number) => {
    setSync(v);
    resetRun(v, seed);
  };
  const onSeed = (next: number) => {
    setSeed(next);
    resetRun(sync, next);
  };
  const randomSeed = () => {
    let next = 1 + Math.floor(Math.random() * 99);
    if (next === seed) next = (next % 99) + 1;
    onSeed(next);
  };

  const H = hist.current;
  const mean = ag.recentMean(20);
  const eps = ag.epsilon();
  const learning = ag.buffer.length >= CFG.learnStart;
  const q = ag.q.forward(ag.env.s);
  const diverging = Math.max(Math.abs(q[0]), Math.abs(q[1])) > 1000;
  const note = noteFor({ mode, ag, sync, seed, result: result.current, demo: demo.current, mean, learning, ckpt: ckpt.current, useCkpt, diverging });
  const flashing = mode === 'train' && (sync === 1 || performance.now() - syncFlashAt.current < 450);
  const playLabel = mode === 'train' ? '❚❚ 멈춤' : finished ? '↺ 다시 학습' : ag.step === 0 ? '▶ 학습 시작' : '▶ 이어서 학습';
  const code = CODE[tab];

  return (
    <div ref={box}>
      <div ref={stage} style={{ padding: '14px 14px 0' }}>
        <canvas
          ref={canvasRef}
          role="img"
          aria-label="CartPole: 수레 위의 막대를 쓰러뜨리지 않게 좌우로 미는 환경"
          style={{ display: 'block', width: '100%', height: stageH, borderRadius: 14, background: 'var(--viz-panel)', border: '1px solid var(--border)' }}
        />
      </div>
      <div className="vz-playbar" style={{ padding: '10px 14px 0' }}>
        <Button variant="primary" onClick={onPlay}>
          {playLabel}
        </Button>
        <Button onClick={onDemo} active={mode === 'demo'} title="학습된 Q로 ε=0 시연">
          {mode === 'demo' ? '■ 시연 끝' : '🎬 시연 보기'}
        </Button>
        <Button variant="ghost" onClick={() => resetRun(sync, seed)}>
          ↺ 처음부터
        </Button>
        {mode === 'demo' && ckpt.current && (
          <Segmented
            label="시연할 가중치"
            options={[
              { value: 'ckpt', label: '🏆 체크포인트', title: 'ε=0 시험을 통과한 마지막 순간의 가중치' },
              { value: 'now', label: '지금 가중치' }
            ]}
            value={useCkpt ? 'ckpt' : 'now'}
            onChange={(v) => {
              setUseCkpt(v === 'ckpt');
              demo.current = newDemo();
            }}
          />
        )}
        <Segmented
          label="속도"
          options={[
            { value: 'real', label: '1×', title: '실제 시간 (초당 50스텝)' },
            { value: 'fast', label: '×20' },
            { value: 'max', label: '최고속' }
          ]}
          value={speed}
          onChange={(v) => setSpeed(v as Speed)}
        />
      </div>

      <div className="vz-pad" style={{ display: 'grid', gap: 14 }}>
        <Note mood={note.mood} tone={note.tone}>
          {rich(note.text)}
        </Note>

        <div style={{ display: 'grid', gap: 14, gridTemplateColumns: wide ? 'minmax(0, 1.25fr) minmax(0, 1fr)' : '1fr' }}>
          <div className="vz-panel">
            <div className="vz-panel-title">점수 그래프 — 한 판 동안 버틴 스텝 수 (500이 만점)</div>
            <RewardChart c={c} hist={H} ghosts={ghosts.current} solvedAt={result.current.solvedAt} step={ag.step} passes={passes.current} W={narrow ? 340 : 420} />
            <Legend
              items={[
                { color: c.brand, label: '최근 20판 평균' },
                { color: alpha(c.series[3], 0.45), label: '한 판 점수' },
                ...(passes.current.length ? [{ color: c.pos, label: '🏆 ε=0 시험 통과' }] : []),
                ...ghosts.current.map((g, i) => ({ color: GHOST(c)[i], label: `이전: ${g.label}`, dashed: true }))
              ]}
            />
          </div>
          <div className="vz-stats" style={{ alignContent: 'start' }}>
            <Stat label="끝난 판" value={ag.episode.toLocaleString('ko-KR')} sub={`시드 ${seed}`} />
            <Stat label="총 스텝" value={ag.step.toLocaleString('ko-KR')} sub="최대 6만" />
            <Stat label="최근 20판 평균" value={mean.toFixed(0)} tone={mean >= SOLVE ? 'good' : 'brand'} sub="해결 기준 475" />
            <Stat label="ε (탐험 확률)" value={`${Math.round(eps * 100)}%`} sub="🎲 무작위로 움직일 확률" />
            <Stat label="리플레이 버퍼" value={ag.buffer.length.toLocaleString('ko-KR')} sub="최대 20,000" />
            <Stat label="TD 손실" value={learning ? (ag.lastLoss < 10 ? ag.lastLoss.toFixed(3) : compact(ag.lastLoss)) : '–'} sub="점수와 따로 놀 수 있어요" />
          </div>
        </div>

        <div style={{ display: 'grid', gap: 14, gridTemplateColumns: wide ? 'minmax(0, 1.25fr) minmax(0, 1fr)' : '1fr' }}>
          <div className="vz-panel">
            <div className="vz-panel-title">리플레이 버퍼 — 한 칸 = 경험 80개</div>
            <BufferGrid c={c} ag={ag} learning={learning} />
            <Legend
              items={[
                { color: c.series[3], label: '쌓인 경험 (진할수록 최근)' },
                { color: c.series[4], label: `이번에 뽑힌 ${CFG.batch}개` },
                { color: c.brand, label: '다음에 쓸 칸', dashed: true }
              ]}
            />
          </div>
          <div className="vz-panel">
            <div className="vz-panel-title">타깃 네트워크 — 과녁은 가끔만 움직여요</div>
            <SyncMeter c={c} sync={sync} step={ag.step} syncs={ag.syncs} flashing={flashing} W={narrow ? 340 : 420} />
          </div>
        </div>
      </div>

      <Controls>
        <Segmented
          label="타깃 동기화 주기 (TARGET_SYNC)"
          options={[
            { value: 1, label: '1 (매 스텝)' },
            { value: 500, label: '500 (랩 기본)' },
            { value: 10000, label: '10000' }
          ]}
          value={sync}
          onChange={(v) => onSync(Number(v))}
        />
        <div className="vz-seg-wrap">
          <span className="vz-label">시드 {seed}</span>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <Button size="sm" onClick={randomSeed} title="같은 코드, 다른 운">
              🎲 다른 시드
            </Button>
            {seed !== DEFAULT_SEED && (
              <Button size="sm" variant="ghost" onClick={() => onSeed(DEFAULT_SEED)}>
                기본 시드 {DEFAULT_SEED}로
              </Button>
            )}
          </div>
        </div>
      </Controls>

      <div className="vz-pad" style={{ display: 'grid', gap: 10, borderTop: '1px solid var(--border)' }}>
        <Segmented
          label="이 장면의 코드 (랩 16 원문)"
          options={(Object.keys(CODE) as Tab[]).map((k) => ({ value: k, label: CODE[k].label }))}
          value={tab}
          onChange={(v) => setTab(v as Tab)}
        />
        <div style={{ maxHeight: expanded ? 'none' : 360, overflow: 'auto', minWidth: 0 }}>
          <CodeBlock code={code.code} highlight={code.hl} c={c} />
        </div>
        <p style={{ margin: 0, fontSize: 14, lineHeight: 1.65, color: 'var(--text-2)' }}>{rich(code.text)}</p>
        <SettingsTable sync={sync} />
      </div>
    </div>
  );
}

/* ---------------- note ---------------- */

function noteFor({
  mode,
  ag,
  sync,
  seed,
  result,
  demo,
  mean,
  learning,
  ckpt,
  useCkpt,
  diverging
}: {
  mode: Mode;
  ag: DqnAgent;
  sync: number;
  seed: number;
  result: { solvedAt: number | null; solvedEp: number; capped: boolean };
  demo: Demo | null;
  mean: number;
  learning: boolean;
  ckpt: Checkpoint | null;
  useCkpt: boolean;
  diverging: boolean;
}): { text: string; mood: AlgoMood; tone?: 'good' | 'warn' } {
  const solved = result.solvedAt !== null;
  const scores = demo && demo.scores.length ? ` 시연 점수: ${demo.scores.join(', ')}` : '';
  if (mode === 'demo') {
    const head = solved ? `**해결!** ${result.solvedEp}판(${result.solvedAt!.toLocaleString('ko-KR')}스텝) 만에 최근 20판 평균이 475를 넘었어요. ` : '';
    if (ckpt && useCkpt)
      return {
        text: `${head}지금 시연은 **🏆 체크포인트** — ε=0 시험을 통과한 마지막 순간(${ckpt.step.toLocaleString('ko-KR')}스텝)의 가중치예요. "지금 가중치"로 바꾸면 그 뒤로 조금 잊어버린 Q를 볼 수도 있어요.${scores}`,
        mood: 'cheer',
        tone: solved ? 'good' : undefined
      };
    if (ckpt)
      return {
        text: `${head}지금은 **마지막 순간의 가중치**로 시연 중이에요. 학습 점수는 높았는데 혼자 하면 금방 쓰러진다면, 그게 랩 본문 3번의 "오르다 무너지는" 망각이에요.${scores}`,
        mood: 'think'
      };
    if (solved) return { text: `${head}지금은 무작위 없이(ε=0) Q만 보고 움직이는 시연이에요.${scores}`, mood: 'cheer', tone: 'good' };
    if (ag.step === 0)
      return { text: `아직 아무것도 배우지 않은 Q예요. 금방 쓰러지죠? 이제 ▶ 학습 시작을 눌러 보세요.${scores}`, mood: 'think' };
    return { text: `지금까지 배운 Q로 혼자 버텨 보는 시연이에요(ε=0, 학습은 잠시 멈춤).${scores}`, mood: 'smile' };
  }
  if (solved)
    return {
      text: `**해결!** ${result.solvedEp}판 만에 최근 20판 평균 ${mean.toFixed(0)}점. 🎬 시연 보기로 실력을 확인하거나, 타깃 동기화 주기를 바꿔 다시 학습해서 그래프를 비교해 보세요.`,
      mood: 'cheer',
      tone: 'good'
    };
  if (result.capped) {
    const why =
      sync === 1
        ? '매 스텝 복사하면 과녁이 쉬지 않고 따라 움직여요(자기추격). 그래서 배우기가 훨씬 어려워요.'
        : sync === 10000
          ? '과녁이 1만 스텝에 한 번만 바뀌니, 새로 배운 것이 정답에 늦게 반영돼요.'
          : seed !== DEFAULT_SEED
            ? '같은 코드, 같은 설정인데 **시드만 바꿔도** 결과가 이렇게 달라요. 랩 본문의 "시드에 따라 편차가 크다"가 바로 이거예요.'
            : '이번엔 운이 따르지 않았어요.';
    return { text: `6만 스텝 안에 해결하지 못했어요. ${why}`, mood: 'think', tone: 'warn' };
  }
  if (ag.step === 0)
    return {
      text: '**▶ 학습 시작**을 눌러 보세요. 처음엔 ε=100%라 거의 무작위로 밀어서 막대가 금방 쓰러져요. 수레를 좌우로 밀어 막대를 세워 두는 게 목표예요.',
      mood: 'smile'
    };
  if (sync === 1 && diverging)
    return {
      text: '과녁이 매 스텝 따라 움직이니 **Q값이 끝없이 커지고**(발산, 오른쪽 위 숫자) 점수는 바닥이에요. 진짜 가치는 아무리 커도 100 정도인데 말이죠(γ=0.99).',
      mood: 'wow',
      tone: 'warn'
    };
  if (sync === 1)
    return {
      text: '지금은 **매 스텝 복사(sync=1)** — 타깃 네트워크가 없는 것과 같아요. 과녁이 계속 움직이면 점수가 어떻게 되는지 그래프를 지켜보세요.',
      mood: 'think'
    };
  if (!learning)
    return { text: `경험을 모으는 중이에요(${ag.buffer.length}/${CFG.learnStart}). 버퍼에 ${CFG.learnStart}개가 쌓이기 전엔 학습하지 않아요.`, mood: 'smile' };
  if (mean < 40)
    return { text: 'ε가 줄면서 🧠 화살표(Q의 선택)가 늘고 있어요. 처음 몇십 판은 점수가 잘 안 올라요 — 정상이에요.', mood: 'think' };
  if (mean < 250) return { text: '점수가 오르기 시작했어요! 그래프의 굵은 선(최근 20판 평균)이 475 점선을 넘으면 해결이에요.', mood: 'happy' };
  return { text: '거의 다 왔어요! 오르다가 갑자기 무너지기도 하는데, 그것도 RL에서 흔한 일(망각)이에요.', mood: 'wow' };
}

/* ---------------- canvas scene ---------------- */

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function arrow(ctx: CanvasRenderingContext2D, x1: number, y: number, x2: number, color: string) {
  const d = Math.sign(x2 - x1) || 1;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.lineTo(x2 - d * 8, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y);
  ctx.lineTo(x2 - d * 11, y - 7);
  ctx.lineTo(x2 - d * 11, y + 7);
  ctx.closePath();
  ctx.fill();
  ctx.lineCap = 'butt';
}

/** Phones: draw the cart and pole larger than true scale so they stay readable (positions and angles stay exact). */
const visScale = (w: number) => (w < 560 ? 1.4 : 1);

function drawScene(ctx: CanvasRenderingContext2D, w: number, h: number, c: VizColors, v: SceneView, family: string) {
  const dark = c.theme === 'dark';
  const f = (size: number, weight = 700) => `${weight} ${size}px ${family}`;
  const span = X_LIMIT * 1.16;
  const scale = (w - 32) / (2 * span);
  const X = (x: number) => w / 2 + x * scale;
  const groundY = h - 34;
  const vis = visScale(w);
  const cartW = Math.max(40, scale * 0.5 * vis);
  const cartH = Math.max(20, scale * 0.26 * vis);
  const wheelR = Math.max(4, cartH * 0.24);
  const bodyBottom = groundY - wheelR;
  const bodyTop = bodyBottom - cartH;
  const L = Math.min(scale * vis, bodyTop - 50);
  const [x, , th] = v.s;
  const cx = X(x);
  const zoneTop = bodyTop - L - 10;

  // out-of-bounds zones beyond ±2.4
  ctx.fillStyle = alpha(c.neg, dark ? 0.14 : 0.09);
  ctx.fillRect(X(-span), zoneTop, X(-X_LIMIT) - X(-span), groundY - zoneTop);
  ctx.fillRect(X(X_LIMIT), zoneTop, X(span) - X(X_LIMIT), groundY - zoneTop);
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = alpha(c.neg, 0.75);
  ctx.lineWidth = 1.5;
  for (const lim of [-X_LIMIT, X_LIMIT]) {
    ctx.beginPath();
    ctx.moveTo(X(lim), zoneTop);
    ctx.lineTo(X(lim), groundY);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // track and ticks
  ctx.strokeStyle = c.axis;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(X(-span), groundY);
  ctx.lineTo(X(span), groundY);
  ctx.stroke();
  ctx.fillStyle = c.text;
  ctx.font = f(11, 650);
  ctx.textAlign = 'center';
  for (const tx of [-X_LIMIT, -1, 0, 1, X_LIMIT]) {
    ctx.beginPath();
    ctx.moveTo(X(tx), groundY);
    ctx.lineTo(X(tx), groundY + 5);
    ctx.stroke();
    ctx.fillText(tx === 0 ? '0' : tx > 0 ? `+${tx}` : `-${-tx}`, X(tx), groundY + 18);
  }

  // safe-angle wedge (±12°)
  const pivotY = bodyTop;
  const R = L * 1.1;
  ctx.beginPath();
  ctx.moveTo(cx, pivotY);
  ctx.arc(cx, pivotY, R, -Math.PI / 2 - THETA_LIMIT, -Math.PI / 2 + THETA_LIMIT);
  ctx.closePath();
  ctx.fillStyle = alpha(c.pos, dark ? 0.16 : 0.13);
  ctx.fill();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = alpha(c.neg, 0.6);
  ctx.lineWidth = 1.2;
  for (const sgn of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(cx, pivotY);
    ctx.lineTo(cx + R * Math.sin(sgn * THETA_LIMIT), pivotY - R * Math.cos(THETA_LIMIT));
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.fillStyle = c.text;
  ctx.font = f(11, 750);
  ctx.textAlign = 'left';
  ctx.fillText('12°', cx + R * Math.sin(THETA_LIMIT) + 4, pivotY - R * Math.cos(THETA_LIMIT) + 6);

  // pole: turns red as it nears the 12° limit
  const danger = Math.min(1, Math.abs(th) / THETA_LIMIT);
  ctx.strokeStyle = mixHex(hx(c.series[2]), hx(c.neg), Math.max(0, danger * 1.3 - 0.3));
  ctx.lineCap = 'round';
  ctx.lineWidth = Math.max(7, scale * 0.07);
  ctx.beginPath();
  ctx.moveTo(cx, pivotY);
  ctx.lineTo(cx + L * Math.sin(th), pivotY - L * Math.cos(th));
  ctx.stroke();
  ctx.lineCap = 'butt';

  // cart
  ctx.fillStyle = c.series[3];
  roundRect(ctx, cx - cartW / 2, bodyTop, cartW, cartH, Math.min(8, cartH / 3));
  ctx.fill();
  ctx.fillStyle = c.ink;
  for (const k of [-0.3, 0.3]) {
    ctx.beginPath();
    ctx.arc(cx + k * cartW, bodyBottom, wheelR, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.beginPath();
  ctx.arc(cx, pivotY, Math.max(3.5, wheelR * 0.8), 0, Math.PI * 2);
  ctx.fillStyle = c.surface;
  ctx.fill();
  ctx.strokeStyle = c.ink;
  ctx.lineWidth = 2;
  ctx.stroke();

  // push arrow: 🎲 explore (amber) or 🧠 exploit (purple)
  if (v.showForce) {
    const d = v.action === 1 ? 1 : -1;
    const col = v.explore ? c.series[4] : c.brand;
    const y = bodyTop + cartH / 2;
    const x1 = cx - d * (cartW / 2 + 40);
    const x2 = cx - d * (cartW / 2 + 5);
    arrow(ctx, x1, y, x2, col);
    ctx.font = f(14, 700);
    ctx.textAlign = 'center';
    ctx.fillText(v.explore ? '🎲' : '🧠', (x1 + x2) / 2, y - 11);
  }

  // header: episode / step counter
  ctx.textAlign = 'left';
  ctx.font = f(13, 800);
  ctx.fillStyle = c.ink;
  ctx.fillText(v.label, 12, 22);

  // Q-values for the current state
  const [q0, q1] = v.q;
  const qmax = Math.max(Math.abs(q0), Math.abs(q1), 1e-6);
  const bw = Math.min(70, w * 0.16);
  const bx = w - bw - 58;
  ctx.font = f(11, 750);
  const qRows: [string, number][] = [
    ['Q(←)', q0],
    ['Q(→)', q1]
  ];
  qRows.forEach(([label, q], i) => {
    const y = (w < 380 ? 32 : 12) + i * 17;
    ctx.fillStyle = c.text;
    ctx.textAlign = 'right';
    ctx.fillText(label, bx - 6, y + 9);
    ctx.fillStyle = alpha(c.brand, 0.14);
    roundRect(ctx, bx, y, bw, 10, 5);
    ctx.fill();
    ctx.fillStyle = q === Math.max(q0, q1) ? c.brand : alpha(c.brand, 0.4);
    const fw = (bw * Math.max(0, q)) / qmax;
    if (fw > 1) {
      roundRect(ctx, bx, y, fw, 10, 5);
      ctx.fill();
    }
    ctx.fillStyle = c.ink;
    ctx.textAlign = 'left';
    ctx.fillText(compact(q), bx + bw + 6, y + 9);
  });

  // end-of-episode banner
  if (v.flash) {
    ctx.font = f(15, 800);
    const tw = ctx.measureText(v.flash.text).width;
    const bxw = tw + 28;
    const by = 40;
    roundRect(ctx, w / 2 - bxw / 2, by, bxw, 30, 15);
    ctx.fillStyle = alpha(c.surface, 0.94);
    ctx.fill();
    ctx.strokeStyle = v.flash.good ? c.pos : c.neg;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = v.flash.good ? c.pos : c.neg;
    ctx.textAlign = 'center';
    ctx.fillText(v.flash.text, w / 2, by + 20);
  }
}

/* ---------------- reward chart ---------------- */

const GHOST = (c: VizColors) => [c.series[5], c.series[1]];

/** Downsample (step, value) pairs into at most `nb` buckets along the step axis. */
function bucketize(steps: number[], vals: number[], xMax: number, nb: number, how: 'mean' | 'last'): [number, number][] {
  if (steps.length <= nb) return steps.map((s, i) => [s, vals[i]]);
  const out: [number, number][] = [];
  const bw = xMax / nb;
  let bucket = -1;
  let sum = 0;
  let n = 0;
  let lastS = 0;
  let lastV = 0;
  for (let i = 0; i < steps.length; i++) {
    const k = Math.floor(steps[i] / bw);
    if (k !== bucket && n > 0) {
      out.push([lastS, how === 'mean' ? sum / n : lastV]);
      sum = 0;
      n = 0;
    }
    bucket = k;
    sum += vals[i];
    n += 1;
    lastS = steps[i];
    lastV = vals[i];
  }
  if (n) out.push([lastS, how === 'mean' ? sum / n : lastV]);
  return out;
}

function RewardChart({
  c,
  hist,
  ghosts,
  solvedAt,
  step,
  passes,
  W
}: {
  c: VizColors;
  hist: Hist;
  ghosts: Ghost[];
  solvedAt: number | null;
  step: number;
  passes: number[];
  W: number;
}) {
  const H = 190;
  const L = 34;
  const R = W - 10;
  const T = 12;
  const B = 160;
  const lastGhost = ghosts.reduce((m, g) => Math.max(m, g.pts.length ? g.pts[g.pts.length - 1][0] : 0), 0);
  const xMax = Math.min(MAX_TRAIN_STEPS, Math.max(20000, Math.ceil(Math.max(step, lastGhost) / 10000) * 10000));
  const X = (s: number) => L + (s / xMax) * (R - L);
  const Y = (v: number) => B - (v / 500) * (B - T);
  const line = (pts: [number, number][]) => pts.map(([s, v]) => `${X(s).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  const raw = bucketize(hist.steps, hist.rewards, xMax, 200, 'mean');
  const avg = bucketize(hist.steps, hist.means, xMax, 200, 'last');
  const xticks: number[] = [];
  for (let s = 0; s <= xMax; s += 10000) xticks.push(s);
  return (
    <svg className="vz-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="총 스텝에 따른 에피소드 점수와 최근 20판 평균">
      {[0, 100, 200, 300, 400, 500].map((v) => (
        <g key={v}>
          <line x1={L} x2={R} y1={Y(v)} y2={Y(v)} stroke={c.grid} />
          <text x={L - 6} y={Y(v) + 4} textAnchor="end" fontSize={11} fontWeight={650} fill={c.text}>
            {v}
          </text>
        </g>
      ))}
      {xticks.map((s) => (
        <text key={s} x={X(s)} y={B + 16} textAnchor="middle" fontSize={11} fontWeight={650} fill={c.text}>
          {s === 0 ? '0' : `${s / 10000}만`}
        </text>
      ))}
      <text x={R} y={B + 29} textAnchor="end" fontSize={11} fontWeight={700} fill={c.text}>
        총 스텝 →
      </text>
      <line x1={L} x2={R} y1={Y(SOLVE)} y2={Y(SOLVE)} stroke={c.pos} strokeWidth={1.5} strokeDasharray="6 4" />
      <text x={L + 6} y={Y(SOLVE) + 13} fontSize={11} fontWeight={800} fill={c.pos}>
        해결 기준 475
      </text>
      {ghosts.map((g, i) => (
        <polyline key={i} points={line(g.pts)} fill="none" stroke={GHOST(c)[i]} strokeWidth={2} strokeDasharray="5 4" opacity={0.85} />
      ))}
      {[...new Set(passes.map((p) => Math.round(X(p))))].map((px) => (
        <path key={px} d={`M${px} ${T + 1} l-4 -7 h8 Z`} fill={c.pos} transform="translate(0 7)" />
      ))}
      {raw.length > 1 && <polyline points={line(raw)} fill="none" stroke={alpha(c.series[3], 0.45)} strokeWidth={1.2} strokeLinejoin="round" />}
      {avg.length > 1 && <polyline points={line(avg)} fill="none" stroke={c.brand} strokeWidth={3} strokeLinejoin="round" strokeLinecap="round" />}
      {solvedAt !== null && (
        <g>
          <line x1={X(solvedAt)} x2={X(solvedAt)} y1={T} y2={B} stroke={c.pos} strokeWidth={1.5} />
          <text x={X(solvedAt) + (X(solvedAt) > R - 60 ? -6 : 6)} y={T + 26} textAnchor={X(solvedAt) > R - 60 ? 'end' : 'start'} fontSize={12} fontWeight={850} fill={c.pos}>
            해결! 🎉
          </text>
        </g>
      )}
      {hist.steps.length === 0 && (
        <text x={(L + R) / 2} y={(T + B) / 2} textAnchor="middle" fontSize={13} fontWeight={700} fill={c.text}>
          학습을 시작하면 여기에 점수가 그려져요
        </text>
      )}
    </svg>
  );
}

/* ---------------- replay buffer ---------------- */

function BufferGrid({ c, ag, learning }: { c: VizColors; ag: DqnAgent; learning: boolean }) {
  const n = ag.buffer.length;
  const filled = Math.ceil(n / PER_CELL);
  const size = CFG.bufferSize;
  const newest = Math.floor((((ag.bufIdx - 1) % size) + size) % size / PER_CELL);
  const next = Math.floor(ag.bufIdx / PER_CELL) % CELLS;
  const picked = new Set(learning ? ag.lastBatch.map((i) => Math.floor(i / PER_CELL)) : []);
  const cw = 14;
  const ch = 11;
  const gap = 2;
  const rows = CELLS / COLS;
  const W = COLS * (cw + gap) + 2;
  const H = rows * (ch + gap) + 2;
  return (
    <svg className="vz-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`리플레이 버퍼: 경험 ${n}개`}>
      {Array.from({ length: CELLS }, (_, k) => {
        const x = 1 + (k % COLS) * (cw + gap);
        const y = 1 + Math.floor(k / COLS) * (ch + gap);
        const has = k < filled;
        const age = has ? ((newest - k + CELLS) % CELLS) / Math.max(1, filled) : 1;
        const isPicked = picked.has(k);
        return (
          <rect
            key={k}
            x={x}
            y={y}
            width={cw}
            height={ch}
            rx={2.5}
            fill={isPicked ? c.series[4] : has ? alpha(c.series[3], 0.9 - 0.62 * age) : 'var(--surface-3)'}
            stroke={k === next && n > 0 ? c.brand : 'none'}
            strokeWidth={k === next ? 1.6 : 0}
            strokeDasharray={k === next ? '2.5 1.5' : undefined}
          />
        );
      })}
    </svg>
  );
}

/* ---------------- target network sync ---------------- */

function SyncMeter({ c, sync, step, syncs, flashing, W }: { c: VizColors; sync: number; step: number; syncs: number; flashing: boolean; W: number }) {
  const frac = sync === 1 ? 1 : (step % sync) / sync;
  const left = sync === 1 ? 0 : sync - (step % sync);
  const arrowCol = flashing ? c.brand : alpha(c.brand, 0.35);
  const bw = W < 400 ? 128 : 150;
  const rx = W - 6 - bw;
  const mid = (6 + bw + rx) / 2;
  return (
    <svg className="vz-svg" viewBox={`0 0 ${W} 118`} role="img" aria-label="Q 네트워크를 Q_target으로 복사하는 주기">
      <rect x={6} y={8} width={bw} height={62} rx={14} fill={alpha(c.brand, 0.13)} stroke={c.brand} strokeWidth={2} />
      <text x={6 + bw / 2} y={34} textAnchor="middle" fontSize={15} fontWeight={850} fill={c.ink}>
        Q ✏️
      </text>
      <text x={6 + bw / 2} y={55} textAnchor="middle" fontSize={12} fontWeight={700} fill={c.text}>
        매 스텝 배우는 중
      </text>
      <rect x={rx} y={8} width={bw} height={62} rx={14} fill={alpha(c.series[2], 0.13)} stroke={c.series[2]} strokeWidth={2} />
      <text x={rx + bw / 2} y={34} textAnchor="middle" fontSize={15} fontWeight={850} fill={c.ink}>
        Q_target 🎯
      </text>
      <text x={rx + bw / 2} y={55} textAnchor="middle" fontSize={12} fontWeight={700} fill={c.text}>
        과녁(TD 타깃) 계산
      </text>
      <line
        x1={12 + bw}
        y1={39}
        x2={rx - 14}
        y2={39}
        stroke={arrowCol}
        strokeWidth={flashing ? 5 : 3}
        strokeLinecap="round"
        strokeDasharray={flashing ? undefined : '6 5'}
      />
      <path d={`M${rx - 6} 39 L${rx - 18} 31 L${rx - 18} 47 Z`} fill={arrowCol} />
      <text x={mid} y={28} textAnchor="middle" fontSize={12} fontWeight={850} fill={flashing ? c.brand : c.text}>
        {flashing ? '복사!' : '복사'}
      </text>
      <rect x={6} y={88} width={W - 12} height={10} rx={5} fill="var(--surface-3)" />
      <rect x={6} y={88} width={Math.max(10, (W - 12) * frac)} height={10} rx={5} fill={alpha(c.brand, sync === 1 ? 0.9 : 0.7)} />
      <text x={6} y={115} fontSize={11.5} fontWeight={700} fill={c.text}>
        {sync === 1 ? (W < 400 ? '매 스텝 복사 중!' : '매 스텝 복사 중 — 과녁이 쉬지 않고 움직여요') : `다음 복사까지 ${left.toLocaleString('ko-KR')}스텝`}
      </text>
      <text x={W - 6} y={115} textAnchor="end" fontSize={11.5} fontWeight={700} fill={c.text}>
        복사 {syncs.toLocaleString('ko-KR')}번
      </text>
    </svg>
  );
}

/* ---------------- settings ---------------- */

function SettingsTable({ sync }: { sync: number }) {
  const rows: [string, string, string][] = [
    ['은닉층 크기', '128', String(CFG.hidden)],
    ['리플레이 버퍼', '50,000', CFG.bufferSize.toLocaleString('ko-KR')],
    ['배치 크기', '64', String(CFG.batch)],
    ['학습 시작', '1,000개 후', `${CFG.learnStart}개 후`],
    ['ε 감쇠 (EPS_DECAY)', '5,000', CFG.epsDecay.toLocaleString('ko-KR')],
    ['학습률', '1e-3 (AdamW)', '5e-4 (Adam)'],
    ['TARGET_SYNC', '500', sync.toLocaleString('ko-KR')],
    ['저장하는 가중치', '마지막 가중치', '🏆 ε=0 시험 통과 체크포인트']
  ];
  return (
    <details style={{ fontSize: 13.5, color: 'var(--text-2)' }}>
      <summary style={{ cursor: 'pointer', fontWeight: 750 }}>이 데모의 설정 — 브라우저에서 수십 초 안에 돌도록 줄였어요</summary>
      <table style={{ width: '100%', marginTop: 8, borderCollapse: 'collapse', fontVariantNumeric: 'tabular-nums' }}>
        <thead>
          <tr>
            {['', '랩 코드', '이 데모'].map((h) => (
              <th key={h} style={{ textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid var(--border)' }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, a, b]) => (
            <tr key={k}>
              <td style={{ padding: '4px 6px' }}>{k}</td>
              <td style={{ padding: '4px 6px' }}>{a}</td>
              <td style={{ padding: '4px 6px', fontWeight: 750 }}>{b}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ margin: '8px 0 0' }}>
        나머지(γ=0.99, Huber 손실, 기울기 자르기 10, terminated/truncated 구분, CartPole-v1 물리)는 랩 코드와 같아요. 계산은 모두 이 브라우저 안에서 진짜로
        일어나고, 같은 시드면 어느 브라우저에서나 같은 결과가 나와요.
      </p>
    </details>
  );
}
