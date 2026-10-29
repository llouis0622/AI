import type { VizProps } from '../Viz';
import { CodeStory, type StoryStep } from '../core/story';
import { VB, FS, seg, alpha, rich, dd, T, Box, Arrow, Pill, Card, Pulse } from './_LabKitB';

/* request arrival times (ms) for the batching timeline — fixed so it looks the same every time */
const ARRIVALS = [1, 3, 4, 9, 12, 13, 14, 22, 24, 31, 33, 34, 35, 37, 46, 51, 53, 58, 60, 61];
const WAIT = 10;
const BATCHES = (() => {
  const out: { start: number; end: number; items: number[] }[] = [];
  let i = 0;
  while (i < ARRIVALS.length) {
    const first = ARRIVALS[i];
    const deadline = first + WAIT;
    const items: number[] = [];
    while (i < ARRIVALS.length && ARRIVALS[i] <= deadline && items.length < 32) items.push(ARRIVALS[i++]);
    out.push({ start: deadline, end: deadline + 4, items });
  }
  return out;
})();
/* one-at-a-time serving: each request waits for the previous one (3ms per forward pass) */
const SERIAL = (() => {
  let end = 0;
  return ARRIVALS.map((a) => {
    const start = Math.max(a, end);
    end = start + 3;
    return { start, end };
  });
})();

const steps: StoryStep[] = [
  {
    title: '모델은 서버가 켜질 때 딱 한 번',
    text: rich('`lifespan`은 서버가 **시작할 때 한 번** 실행돼요. 여기서 모델을 읽어 메모리에 올려 두면, 이후 요청은 계산만 하면 돼요. 요청마다 모델을 읽는 건 **최악의 안티패턴** — 매번 몇 초씩 기다리게 돼요.'),
    code: dd(String.raw`@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델은 서버 시작 시 '한 번만' 로드한다. 요청마다 로드하는 것은 최악의 안티패턴.
    from train_cifar import build_model                # 02 코드랩의 모델 정의 재사용
    model = build_model()
    ckpt = torch.load("ckpt_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    state["model"] = model
    yield
    state.clear()`),
    highlight: [5, 6, 7, 9],
    render: (t, c) => {
      const p = seg(t, 0.05, 0.85);
      const X = (ms: number) => 20 + (ms / 4000) * 380;
      const goodReq = [1800, 2050, 2300, 2550, 2800, 3050, 3300, 3550];
      return (
        <g>
          <T x={20} y={42} size={FS.s} weight={850} color={c.pos}>
            ✅ 시작할 때 한 번 로드
          </T>
          <rect x={X(0)} y={54} width={(X(1500) - X(0)) * Math.min(1, p / 0.35)} height={30} rx={7} fill={alpha(c.series[0], 0.4)} />
          <T x={X(750)} y={74} anchor="middle" size={FS.m} weight={800} opacity={p > 0.2 ? 1 : 0}>
            🧠 모델 로드
          </T>
          {goodReq.map((ms, i) => (
            <rect key={i} x={X(ms)} y={54} width={9} height={30} rx={3} fill={c.pos} opacity={p > ms / 4000 ? 1 : 0} />
          ))}
          <T x={20} y={124} size={FS.s} weight={850} color={c.neg}>
            ❌ 요청마다 로드
          </T>
          {[0, 1, 2].map((k) => {
            const s = k * 1350;
            return (
              <g key={k} opacity={p > s / 4000 ? 1 : 0}>
                <rect x={X(s)} y={136} width={X(1300) - X(0)} height={30} rx={7} fill={alpha(c.neg, 0.25)} />
                <T x={X(s + 650)} y={156} anchor="middle" size={FS.m} weight={800}>
                  🧠 로드
                </T>
                <rect x={X(s + 1300)} y={136} width={9} height={30} rx={3} fill={c.neg} />
              </g>
            );
          })}
          <line x1={20} y1={190} x2={400} y2={190} stroke="var(--viz-axis)" />
          {[0, 1, 2, 3, 4].map((s) => (
            <T key={s} x={X(s * 1000)} y={208} anchor="middle" size={11.5} weight={700} color="var(--text-2)" mono>
              {s}초
            </T>
          ))}
          <g opacity={seg(t, 0.8, 0.2)}>
            <Pill x={110} y={250} label="4초 동안 8건 처리" color={c.pos} size={FS.m} />
            <Pill x={310} y={250} label="4초 동안 3건 처리" color={c.neg} size={FS.m} />
          </g>
        </g>
      );
    }
  },
  {
    title: '요청 하나의 일생 — /predict',
    text: rich('사진이 도착하면 ① **형식 검사**(아니면 415), ② **이미지 해석**(깨졌으면 400), ③ **학습 때와 똑같은 전처리**, ④ 모델 → softmax, ⑤ **상위 3개를 JSON**으로 돌려줘요. 응답 형식은 `Prediction` 모델이 보장해요.'),
    code: dd(String.raw`@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, f"지원하지 않는 형식: {file.content_type}")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 해석할 수 없습니다")

    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = state["model"](x).softmax(dim=1)[0]
    top = probs.topk(3)`),
    highlight: [3, 6, 10, 12, 13],
    render: (t, c) => {
      const stages = [
        { l: '① 형식', s: '415?', x: 74, y: 40 },
        { l: '② 해석', s: '400?', x: 170, y: 40 },
        { l: '③ 전처리', s: '학습과 동일', x: 266, y: 40 },
        { l: '④ 모델', s: 'softmax', x: 266, y: 122 },
        { l: '⑤ top-3', s: 'JSON', x: 170, y: 122 }
      ];
      const path = 'M44 66 L74 66 M154 66 L170 66 M250 66 L266 66 M306 92 L306 122 M266 148 L250 148 M170 148 L150 148';
      return (
        <g>
          <g opacity={seg(t, 0, 0.2)}>
            <rect x={8} y={44} width={36} height={44} rx={8} fill="var(--surface-solid)" stroke="var(--border-strong)" />
            <T x={26} y={74} anchor="middle" size={20}>
              📱
            </T>
          </g>
          {stages.map((s, i) => (
            <Box key={i} x={s.x} y={s.y} w={80} h={52} color={[c.series[3], c.series[3], c.series[0], c.series[1], c.series[2]][i]} t={seg(t, 0.08 + i * 0.1, 0.2)} label={s.l} sub={s.s} labelSize={FS.m} />
          ))}
          <path d={path} fill="none" stroke="var(--border-strong)" strokeWidth={2.5} opacity={seg(t, 0.5, 0.2)} />
          {seg(t, 0.55, 0.1) >= 1 && <Pulse path="M44 66 L74 66 L170 66 L266 66 L306 66 L306 148 L170 148 L120 148" color={c.series[2]} dur={2.4} r={5} />}
          <Card
            x={14}
            y={186}
            w={392}
            title="200 OK · application/json"
            lines={['{"label": "cat", "confidence": 0.9132,', ' "top3": {"cat": 0.9132, "dog": 0.0641,', '          "deer": 0.0118}}']}
            color={c.pos}
            t={seg(t, 0.7, 0.25)}
            size={12}
            mono
          />
        </g>
      );
    }
  },
  {
    title: '/health — "살아 있어요?"',
    text: rich('로드밸런서나 쿠버네티스는 몇 초마다 `/health`를 두드려 봐요. 대답이 없는 서버는 **손님(트래픽)을 보내지 않고 빼 버려요**. 작아 보여도 실무 서빙에 **반드시 있어야 하는** 엔드포인트예요.'),
    code: dd(String.raw`@app.get("/health")
def health():
    """로드밸런서·k8s가 살아있는지 확인하는 엔드포인트 — 반드시 있어야 한다."""
    return {"status": "ok", "device": device}`),
    highlight: [1, 4],
    render: (t, c) => {
      const servers = [
        { y: 46, ok: true },
        { y: 124, ok: false },
        { y: 202, ok: true }
      ];
      return (
        <g>
          <Box x={20} y={118} w={96} h={66} color={c.series[3]} t={seg(t, 0, 0.2)} label="⚖️ 로드밸런서" labelSize={FS.m} />
          {servers.map((s, i) => {
            const a = seg(t, 0.1 + i * 0.1, 0.2);
            const col = s.ok ? c.pos : c.neg;
            return (
              <g key={i} opacity={a}>
                <Box x={286} y={s.y} w={118} h={56} color={col} label={`서버 ${i + 1}`} sub={s.ok ? '{"status": "ok"}' : '응답 없음…'} labelSize={FS.m} />
                <Arrow x1={118} y1={151} x2={284} y2={s.y + 28} color={col} t={seg(t, 0.35 + i * 0.08, 0.2)} dashed={!s.ok} label={i === 1 ? 'GET /health' : undefined} labelDy={-10} />
                {!s.ok && seg(t, 0.7, 0.2) > 0 && (
                  <T x={200} y={s.y + 50} anchor="middle" size={FS.m} weight={850} color={c.neg} opacity={seg(t, 0.7, 0.2)}>
                    ✗ 손님 안 보냄
                  </T>
                )}
              </g>
            );
          })}
          {seg(t, 0.8, 0.1) >= 1 && (
            <>
              <Pulse path="M118 151 L284 74" color={c.pos} dur={1.2} />
              <Pulse path="M118 151 L284 230" color={c.pos} dur={1.2} delay={0.6} />
            </>
          )}
        </g>
      );
    }
  },
  {
    title: '학습과 서빙의 전처리는 똑같이',
    text: rich('서빙의 전처리가 학습 때와 조금만 달라도(예: Normalize를 빼먹음) 모델은 **에러 없이 조용히** 틀린 답을 내요. 이걸 **학습-서빙 왜곡**이라고 해요. 그래서 학습 코드와 같은 변환을 그대로 가져와요.'),
    code: dd(String.raw`# 학습 때와 '동일한' 전처리 — 다르면 조용히 성능이 깎인다 (학습-서빙 왜곡)
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])`),
    highlight: [1, 5],
    render: (t, c) => {
      const chain = ['Resize', 'ToTensor', 'Normalize'];
      const row = (y: number, missing: boolean, a: number) => (
        <g opacity={a}>
          {chain.map((ch, i) => {
            const gone = missing && i === 2;
            return (
              <g key={i}>
                <rect x={92 + i * 90} y={y} width={80} height={36} rx={10} fill={gone ? 'none' : alpha(c.series[0], 0.16)} stroke={gone ? c.neg : c.series[0]} strokeWidth={2} strokeDasharray={gone ? '5 4' : undefined} />
                <T x={132 + i * 90} y={y + 23} anchor="middle" size={12.5} weight={800} mono color={gone ? c.neg : 'var(--text)'}>
                  {ch}
                </T>
                {gone && (
                  <T x={132 + i * 90} y={y - 6} anchor="middle" size={FS.m} weight={900} color={c.neg}>
                    빠짐!
                  </T>
                )}
              </g>
            );
          })}
        </g>
      );
      return (
        <g>
          <T x={14} y={66} size={FS.s} weight={850}>
            학습
          </T>
          {row(44, false, seg(t, 0.05, 0.25))}
          <Pill x={378} y={62} label="95%" color={c.pos} t={seg(t, 0.3, 0.2)} size={FS.m} />
          <T x={14} y={146} size={FS.s} weight={850}>
            서빙
          </T>
          {row(124, true, seg(t, 0.35, 0.25))}
          <Pill x={378} y={142} label="62%" color={c.neg} t={seg(t, 0.6, 0.2)} size={FS.m} />
          <g opacity={seg(t, 0.72, 0.2)}>
            <rect x={40} y={196} width={340} height={64} rx={14} fill={alpha(c.neg, 0.1)} stroke={c.neg} strokeWidth={1.5} />
            <T x={210} y={222} anchor="middle" size={FS.s} weight={850} color={c.neg}>
              에러는 0개, 정확도만 조용히 추락 📉
            </T>
            <T x={210} y={246} anchor="middle" size={FS.m} weight={700} color="var(--text-2)">
              (숫자는 예시예요)
            </T>
          </g>
        </g>
      );
    }
  },
  {
    title: '동적 배칭 — 모아서 한 번에',
    text: rich('GPU에 한 건씩 넣으면 대부분의 시간을 놀아요. 첫 요청이 오면 **최대 10ms 동안(또는 32개가 찰 때까지)** 기다리며 모은 뒤, **한 번의 순전파**로 처리하고 각자의 future에 결과를 돌려줘요. Triton·vLLM이 하는 일의 원형이에요.'),
    code: dd(String.raw`    async def _loop(self):
        while True:
            x0, fut0 = await self.queue.get()  # 첫 요청이 올 때까지 대기
            batch, futs = [x0], [fut0]
            deadline = asyncio.get_event_loop().time() + MAX_WAIT_MS / 1000
            while len(batch) < MAX_BATCH:      # 마감까지 추가 요청을 모은다
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break`),
    highlight: [3, 5, 6],
    render: (t, c) => {
      const p = seg(t, 0.05, 0.85);
      const now = 70 * p;
      const X = (ms: number) => 24 + (ms / 70) * 380;
      return (
        <g>
          <T x={20} y={34} size={FS.s} weight={850}>
            요청 도착 (점) → 10ms씩 모아서
          </T>
          {ARRIVALS.map((ms, i) => (
            <circle key={i} cx={X(ms)} cy={62} r={5} fill={c.series[3]} opacity={ms <= now ? 1 : 0.12} />
          ))}
          {BATCHES.map((b, i) => (
            <g key={i} opacity={b.items[0] <= now ? 1 : 0}>
              <rect x={X(b.items[0]) - 5} y={50} width={X(b.start) - X(b.items[0]) + 4} height={24} rx={8} fill={alpha(c.series[0], 0.06)} stroke={c.series[0]} strokeWidth={1.5} strokeDasharray="4 3" />
              <line x1={X(b.start)} y1={76} x2={X(b.start)} y2={112} stroke={c.series[0]} strokeWidth={1.5} />
              {b.start <= now && <rect x={X(b.start)} y={112} width={X(b.end) - X(b.start)} height={26} rx={5} fill={c.series[0]} />}
              {b.start <= now && (
                <T x={X(b.start) + 2} y={156} size={11} weight={800} mono>
                  ×{b.items.length}
                </T>
              )}
            </g>
          ))}
          <T x={20} y={106} size={11.5} weight={800} color={c.series[0]}>
            GPU (배칭)
          </T>
          <T x={20} y={196} size={11.5} weight={800} color={c.neg}>
            GPU (한 건씩)
          </T>
          {SERIAL.map((r, i) => (
            <rect key={`n${i}`} x={X(r.start)} y={204} width={X(r.end) - X(r.start) - 1.5} height={26} rx={4} fill={alpha(c.neg, 0.7)} opacity={r.start <= now ? 1 : 0} />
          ))}
          <line x1={24} y1={250} x2={404} y2={250} stroke="var(--viz-axis)" />
          {[0, 20, 40, 60].map((ms) => (
            <T key={ms} x={X(ms)} y={268} anchor="middle" size={11.5} weight={700} color="var(--text-2)" mono>
              {ms}ms
            </T>
          ))}
          <T x={404} y={290} anchor="end" size={FS.m} weight={800} opacity={seg(t, 0.85, 0.15)}>
            순전파: 배칭 {BATCHES.length}번 vs 한 건씩 {ARRIVALS.length}번
          </T>
        </g>
      );
    }
  },
  {
    title: 'MAX_WAIT — 지연과 처리량의 다이얼',
    text: rich('오래 기다릴수록 배치가 커져 **처리량(초당 처리 수)** 은 오르지만, 각 요청이 기다리는 **지연(p95)** 도 늘어요. 그래서 평균이 아니라 **p95/p99 지연**을 보며 이 다이얼을 맞춰요. (곡선은 개념을 보여 주는 예시예요.)'),
    code: dd(String.raw`MAX_BATCH, MAX_WAIT_MS = 32, 10`),
    highlight: [1],
    render: (t, c) => {
      const P = { x: 48, y: 40, w: 320, h: 180 };
      const n = Math.max(2, Math.round(60 * seg(t, 0.05, 0.6)));
      const rps = (w: number) => 1600 * (1 - Math.exp(-w / 6)) + 120;
      const lat = (w: number) => 10 + 1.05 * w;
      const px = (w: number) => P.x + (w / 40) * P.w;
      const pyR = (v: number) => P.y + P.h - (v / 1800) * P.h;
      const pyL = (v: number) => P.y + P.h - (v / 60) * P.h;
      const ptsR: string[] = [];
      const ptsL: string[] = [];
      for (let i = 0; i < n; i++) {
        const w = (40 * i) / 59;
        ptsR.push(`${px(w).toFixed(1)},${pyR(rps(w)).toFixed(1)}`);
        ptsL.push(`${px(w).toFixed(1)},${pyL(lat(w)).toFixed(1)}`);
      }
      const mark = seg(t, 0.7, 0.2);
      return (
        <g>
          <line x1={P.x} y1={P.y + P.h} x2={P.x + P.w} y2={P.y + P.h} stroke="var(--viz-axis)" />
          <line x1={P.x} y1={P.y} x2={P.x} y2={P.y + P.h} stroke="var(--viz-axis)" />
          <line x1={P.x + P.w} y1={P.y} x2={P.x + P.w} y2={P.y + P.h} stroke="var(--viz-axis)" strokeDasharray="3 3" />
          {[0, 10, 20, 30, 40].map((w) => (
            <T key={w} x={px(w)} y={P.y + P.h + 17} anchor="middle" size={11.5} weight={700} color="var(--text-2)" mono>
              {w}
            </T>
          ))}
          <T x={P.x + P.w / 2} y={P.y + P.h + 36} anchor="middle" size={FS.m} weight={750} color="var(--text-2)">
            MAX_WAIT_MS
          </T>
          <polyline points={ptsR.join(' ')} fill="none" stroke={c.series[1]} strokeWidth={3.5} />
          <polyline points={ptsL.join(' ')} fill="none" stroke={c.neg} strokeWidth={3} strokeDasharray="7 4" />
          <T x={P.x + 6} y={P.y + 14} size={FS.m} weight={850} color={c.series[1]}>
            처리량 (요청/초)
          </T>
          <T x={P.x + P.w - 6} y={P.y + 14} anchor="end" size={FS.m} weight={850} color={c.neg}>
            p95 지연 (ms)
          </T>
          <g opacity={mark}>
            <line x1={px(10)} y1={P.y} x2={px(10)} y2={P.y + P.h} stroke={c.series[3]} strokeWidth={2} />
            <circle cx={px(10)} cy={pyR(rps(10))} r={6} fill={c.series[1]} />
            <circle cx={px(10)} cy={pyL(lat(10))} r={6} fill={c.neg} />
            <Pill x={px(10) + 70} y={P.y + P.h - 40} label={`10ms: ${Math.round(rps(10))}건/초 · ${lat(10).toFixed(1)}ms`} color={c.series[3]} size={11.5} />
          </g>
        </g>
      );
    }
  },
  {
    title: 'Docker — 레이어 캐시로 빠르게',
    text: rich('Dockerfile은 위에서부터 **레이어**를 쌓아요. 잘 안 바뀌는 **의존성 설치를 코드보다 먼저** 두면, 코드만 고쳤을 때 아래 레이어는 캐시를 그대로 쓰고 위만 다시 만들어 **수 초 만에** 빌드가 끝나요.'),
    code: dd(String.raw`FROM python:3.12-slim

WORKDIR /app
# 의존성 레이어를 코드와 분리 — 코드만 바뀌면 캐시로 빌드가 수 초에 끝난다
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py train_cifar.py ckpt_best.pt ./`),
    highlight: [5, 6, 8],
    render: (t, c) => {
      const layers = [
        { l: 'FROM python:3.12-slim', cached: true },
        { l: 'RUN pip install -r requirements.txt', cached: true },
        { l: 'COPY serve.py … ./', cached: false },
        { l: 'CMD uvicorn serve:app', cached: false }
      ];
      return (
        <g>
          <T x={20} y={34} size={FS.s} weight={850}>
            serve.py만 고친 뒤 docker build
          </T>
          {layers.map((ly, i) => {
            const y = 230 - i * 50;
            const a = seg(t, 0.05 + i * 0.12, 0.2);
            const col = ly.cached ? c.pos : c.series[2];
            const rebuilding = !ly.cached && seg(t, 0.55, 0.3) < 1;
            return (
              <g key={i} opacity={a}>
                <rect x={20} y={y} width={300} height={40} rx={10} fill={alpha(col, 0.16)} stroke={col} strokeWidth={2} />
                <T x={32} y={y + 25} size={12} weight={800} mono>
                  {ly.l}
                </T>
                <T x={404} y={y + 25} anchor="end" size={FS.m} weight={850} color={col}>
                  {ly.cached ? '캐시 ✓' : rebuilding ? '다시 빌드…' : '완료 ✓'}
                </T>
              </g>
            );
          })}
          <Pill x={210} y={286} label="빌드 수 초 · 의존성은 그대로" color={c.pos} t={seg(t, 0.85, 0.15)} size={FS.m} />
        </g>
      );
    }
  }
];

export default function Lab20Serving(_: VizProps) {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={2} />;
}
