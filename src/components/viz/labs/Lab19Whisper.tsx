import type { VizProps } from '../Viz';
import { CodeStory, type StoryStep } from '../core/story';
import type { VizColors } from '../core/colors';
import { VB, FS, seg, alpha, rich, dd, T, Box, Arrow, Pill } from './_LabKitB';
import { SIGNAL, SR, SPEC, MEL, MEL_FB, NFFT, WIN, HOP, N_MELS, normalise, heatmapURL, hexToRgb } from './_audioB';

const SPEC_N = normalise(SPEC.map((r) => r.map((v) => Math.sqrt(v))), false);
const MEL_N = normalise(MEL.map((r) => r.map((v) => Math.sqrt(v))), false);
const LOGMEL_N = normalise(MEL, true);
const FRAMES = SPEC.length;

/* color ramp: surface → series[0] → series[2] */
function rampFn(c: VizColors) {
  const a = hexToRgb(c.surface, c.theme === 'dark' ? [16, 22, 58] : [255, 255, 255]);
  const b = hexToRgb(c.series[0]);
  const d = hexToRgb(c.series[2]);
  return (v: number): [number, number, number] => {
    const x = Math.min(1, Math.max(0, v));
    const [p, q, k] = x < 0.5 ? [a, b, x * 2] : [b, d, (x - 0.5) * 2];
    return [Math.round(p[0] + (q[0] - p[0]) * k), Math.round(p[1] + (q[1] - p[1]) * k), Math.round(p[2] + (q[2] - p[2]) * k)];
  };
}

function Heat({ id, mat, x, y, w, h, c, reveal = 1, label }: { id: string; mat: number[][]; x: number; y: number; w: number; h: number; c: VizColors; reveal?: number; label?: string }) {
  const url = heatmapURL(`${id}-${c.theme}-${c.series[0]}`, mat, rampFn(c));
  const clip = `lab19clip-${id}`;
  return (
    <g>
      <defs>
        <clipPath id={clip}>
          <rect x={x} y={y} width={w * reveal} height={h} />
        </clipPath>
      </defs>
      <rect x={x} y={y} width={w} height={h} rx={6} fill="var(--surface-solid)" stroke="var(--border-strong)" />
      {url && <image href={url} x={x} y={y} width={w} height={h} preserveAspectRatio="none" clipPath={`url(#${clip})`} style={{ imageRendering: 'pixelated' }} />}
      {label && (
        <T x={x} y={y - 8} size={FS.m} weight={850}>
          {label}
        </T>
      )}
    </g>
  );
}

function Wave({ x, y, w, h, c, from = 0, to = 1, color }: { x: number; y: number; w: number; h: number; c: VizColors; from?: number; to?: number; color?: string }) {
  const n = SIGNAL.length;
  const i0 = Math.floor(from * n);
  const i1 = Math.floor(to * n);
  const cols = Math.round(w);
  const step = (i1 - i0) / cols;
  const pts: string[] = [];
  let peak = 0;
  for (let i = i0; i < i1; i++) peak = Math.max(peak, Math.abs(SIGNAL[i]));
  for (let k = 0; k < cols; k++) {
    let lo = 0;
    let hi = 0;
    const s = i0 + Math.floor(k * step);
    const e = Math.max(s + 1, i0 + Math.floor((k + 1) * step));
    for (let i = s; i < e; i++) {
      lo = Math.min(lo, SIGNAL[i]);
      hi = Math.max(hi, SIGNAL[i]);
    }
    pts.push(`${(x + k).toFixed(1)},${(y + h / 2 - (hi / (peak || 1)) * (h / 2)).toFixed(1)} ${(x + k).toFixed(1)},${(y + h / 2 - (lo / (peak || 1)) * (h / 2)).toFixed(1)}`);
  }
  return (
    <g>
      <line x1={x} y1={y + h / 2} x2={x + w} y2={y + h / 2} stroke="var(--border-strong)" />
      <polyline points={pts.join(' ')} fill="none" stroke={color ?? c.series[0]} strokeWidth={1.1} />
    </g>
  );
}

const MEL_CODE = dd(String.raw`mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=400,          # 25ms 창 — 이 안에서는 소리가 '정상적'이라고 가정
    hop_length=160,     # 10ms 스트라이드 — 초당 100 프레임
    n_mels=80,          # 멜 주파수 빈 수 (Whisper도 80/128을 쓴다)
)(wav)`);

const steps: StoryStep[] = [
  {
    title: '소리 = 1초에 16,000개의 숫자',
    text: rich('마이크가 잡은 공기의 떨림은 **1차원 숫자 줄(파형)** 이에요. 스테레오면 평균 내서 모노로, 샘플링 속도는 **16kHz**로 맞춰요. 그림은 "아 · 이 · 스"를 흉내 낸 합성음이에요(가볍게 8kHz로 계산).'),
    code: dd(String.raw`wav, sr = torchaudio.load("speech.wav")          # (채널, 샘플 수)
wav = wav.mean(dim=0, keepdim=True)              # 모노로
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000`),
    highlight: [2, 4],
    render: (t, c) => {
      const zoom = seg(t, 0.45, 0.3);
      const z0 = 0.15;
      const z1 = 0.16;
      const zs = Math.round(z0 * SR);
      const ze = Math.round(z1 * SR);
      const pk = Math.max(...Array.from(SIGNAL.slice(zs, ze)).map(Math.abs)) || 1;
      return (
        <g>
          <g opacity={seg(t, 0, 0.3)}>
            <Wave x={14} y={46} w={392} h={110} c={c} />
            {[
              { t: 0.17, l: '아' },
              { t: 0.5, l: '이' },
              { t: 0.81, l: '스' }
            ].map((s) => (
              <T key={s.l} x={14 + 392 * s.t} y={40} anchor="middle" size={FS.s} weight={850} color={c.series[2]}>
                {s.l}
              </T>
            ))}
            <T x={14} y={176} size={11.5} weight={700} color="var(--text-2)" mono>
              0초
            </T>
            <T x={406} y={176} anchor="end" size={11.5} weight={700} color="var(--text-2)" mono>
              1초
            </T>
          </g>
          <g opacity={zoom}>
            <rect x={14 + 392 * z0} y={44} width={392 * (z1 - z0) + 2} height={114} fill="none" stroke={c.neg} strokeWidth={2} />
            <rect x={60} y={196} width={300} height={84} rx={10} fill="var(--surface-solid)" stroke={c.neg} strokeWidth={1.5} />
            {Array.from({ length: ze - zs }, (_, k) => {
              const v = SIGNAL[zs + k] / pk;
              const px = 72 + (k * 276) / (ze - zs - 1);
              return (
                <g key={k}>
                  <line x1={px} y1={238} x2={px} y2={238 - v * 32} stroke={alpha(c.series[0], 0.6)} strokeWidth={1.5} />
                  <circle cx={px} cy={238 - v * 32} r={2.6} fill={c.series[0]} />
                </g>
              );
            })}
            <T x={210} y={293} anchor="middle" size={11.5} weight={750} color="var(--text-2)">
              확대: 10ms 안에도 숫자가 80개 (16kHz면 160개)
            </T>
          </g>
        </g>
      );
    }
  },
  {
    title: '짧은 창으로 잘라 보기',
    text: rich('긴 파형은 구조가 안 보여요. **25ms(400샘플) 창**을 **10ms(160샘플)씩** 밀며 잘라요 — 초당 100조각. 이렇게 짧은 구간 안에서는 소리가 거의 변하지 않는다고 볼 수 있어요.'),
    code: MEL_CODE,
    highlight: [3, 4],
    render: (t, c) => {
      const pos = seg(t, 0.1, 0.8);
      const wx = 14 + 392 * (0.05 + 0.8 * pos);
      const ww = (392 * WIN) / SIGNAL.length;
      const hop = (392 * HOP) / SIGNAL.length;
      return (
        <g>
          <Wave x={14} y={60} w={392} h={110} c={c} />
          {[0, 1, 2].map((k) => (
            <rect key={k} x={wx - k * hop} y={56} width={ww} height={118} rx={4} fill={alpha(c.series[2], k === 0 ? 0.28 : 0.12)} stroke={c.series[2]} strokeWidth={k === 0 ? 2 : 1} strokeDasharray={k === 0 ? undefined : '3 3'} />
          ))}
          <T x={wx + ww / 2} y={48} anchor="middle" size={11.5} weight={850} color={c.series[2]}>
            25ms 창
          </T>
          {Array.from({ length: 40 }, (_, k) => (
            <rect key={k} x={14 + k * 9.8} y={196} width={7.8} height={18} rx={2.5} fill={k / 40 <= 0.05 + 0.8 * pos ? alpha(c.series[2], 0.7) : 'var(--surface-3)'} />
          ))}
          <T x={210} y={240} anchor="middle" size={FS.s} weight={800}>
            한 칸 = 한 프레임 (10ms마다 하나)
          </T>
          <T x={210} y={266} anchor="middle" size={FS.m} weight={750} color="var(--text-2)">
            1초 → 약 100 프레임
          </T>
        </g>
      );
    }
  },
  {
    title: '조각마다 FFT → 스펙트로그램',
    text: rich('각 조각을 **푸리에 변환**하면 "어떤 높이(주파수)의 소리가 얼마나 센가"가 나와요. 이 막대들을 시간 순서로 옆에 붙이면 **스펙트로그램** — 가로는 시간, 세로는 주파수, 밝기는 세기예요. 진짜로 계산한 결과예요.'),
    code: MEL_CODE,
    highlight: [1, 6],
    render: (t, c) => {
      const rev = seg(t, 0.2, 0.7);
      const fi = Math.min(FRAMES - 1, Math.floor(FRAMES * 0.17));
      const spec = SPEC_N[fi];
      return (
        <g>
          <T x={14} y={30} size={FS.m} weight={850}>
            "아"의 한 조각
          </T>
          {spec.map((v, k) =>
            k % 2 === 0 ? (
              <rect key={k} x={14} y={256 - (k / spec.length) * 220} width={Math.max(1, v * 96)} height={3} fill={c.series[0]} opacity={seg(t, 0, 0.3)} />
            ) : null
          )}
          <T x={14} y={276} size={11} weight={700} color="var(--text-2)">
            ↑ 높은 소리
          </T>
          <Arrow x1={118} y1={150} x2={142} y2={150} color={c.series[0]} t={seg(t, 0.15, 0.15)} />
          <Heat id="spec" mat={SPEC_N} x={146} y={36} w={262} h={220} c={c} reveal={rev} label="스펙트로그램 (선형 주파수)" />
          <T x={146} y={276} size={11} weight={700} color="var(--text-2)" mono>
            0초
          </T>
          <T x={408} y={276} anchor="end" size={11} weight={700} color="var(--text-2)" mono>
            1초 →
          </T>
        </g>
      );
    }
  },
  {
    title: '멜 필터뱅크 — 사람 귀처럼 보기',
    text: rich('사람 귀는 **낮은 소리의 차이에 민감**하고 높은 소리는 뭉뚱그려 들어요. 그래서 낮은 쪽엔 촘촘하게, 높은 쪽엔 넓게 **삼각형 필터**를 깔아 주파수를 묶어요. Whisper는 80개(그림은 40개)를 써요.'),
    code: MEL_CODE,
    highlight: [5],
    render: (t, c) => {
      const nBins = NFFT / 2 + 1;
      const fx = (k: number) => 14 + (k / (nBins - 1)) * 392;
      const shown = Math.ceil(N_MELS * seg(t, 0.05, 0.55));
      return (
        <g>
          <T x={14} y={30} size={FS.m} weight={850}>
            삼각형 필터 {N_MELS}개 (가로 = 주파수 0 → 4kHz)
          </T>
          {MEL_FB.slice(0, shown).map((w, m) => {
            const pts = w.map((v, k) => (v > 0 || (w[k - 1] ?? 0) > 0 || (w[k + 1] ?? 0) > 0 ? `${fx(k).toFixed(1)},${(130 - v * 80).toFixed(1)}` : null)).filter(Boolean);
            return <polyline key={m} points={pts.join(' ')} fill={alpha(m % 2 ? c.series[0] : c.series[1], 0.12)} stroke={m % 2 ? c.series[0] : c.series[1]} strokeWidth={1.2} />;
          })}
          <line x1={14} y1={130} x2={406} y2={130} stroke="var(--viz-axis)" />
          <T x={14} y={148} size={11} weight={700} color="var(--text-2)">
            낮은 소리: 촘촘
          </T>
          <T x={406} y={148} anchor="end" size={11} weight={700} color="var(--text-2)">
            높은 소리: 넓게
          </T>
          <Heat id="spec2" mat={SPEC_N} x={14} y={176} w={180} h={96} c={c} label="129칸" />
          <Arrow x1={200} y1={224} x2={224} y2={224} color={c.series[1]} t={seg(t, 0.6, 0.2)} />
          <g opacity={seg(t, 0.65, 0.25)}>
            <Heat id="mel" mat={MEL_N} x={228} y={176} w={180} h={96} c={c} label={`멜 ${N_MELS}칸`} />
          </g>
        </g>
      );
    }
  },
  {
    title: '로그 — (1, 80, T) 이미지 완성',
    text: rich('마지막으로 **로그**를 씌워요. 사람이 소리 크기를 느끼는 방식도 로그라서, 작은 소리의 디테일이 살아나요. 이렇게 만든 **(1, 80, 시간)** 행렬이 음성 인식·화자 인식·오디오 분류의 표준 입력 — 사실상 **흑백 이미지**예요.'),
    code: dd(String.raw`log_mel = torch.log(mel + 1e-9)                  # 로그: 사람의 소리 크기 지각도 로그다

print(log_mel.shape)                             # (1, 80, 시간 프레임)`),
    highlight: [1, 3],
    render: (t, c) => (
      <g>
        <Heat id="mel3" mat={MEL_N} x={14} y={50} w={180} h={150} c={c} label="멜 (로그 전)" />
        <Arrow x1={200} y1={125} x2={224} y2={125} color={c.series[2]} t={seg(t, 0.2, 0.2)} label="log" labelDy={-8} />
        <g opacity={seg(t, 0.25, 0.3)}>
          <Heat id="logmel" mat={LOGMEL_N} x={228} y={50} w={180} h={150} c={c} label="로그 멜 = 이미지!" />
        </g>
        <g opacity={seg(t, 0.6, 0.25)}>
          <Pill x={210} y={238} label="shape: (1, 80, 시간 프레임)" color={c.series[2]} size={FS.m} mono />
          <T x={210} y={276} anchor="middle" size={FS.m} weight={750} color="var(--text-2)">
            이제 CNN·Transformer를 그대로 쓸 수 있어요
          </T>
        </g>
      </g>
    )
  },
  {
    title: 'Whisper — 로그 멜을 읽고 글자로',
    text: rich('Whisper는 **인코더-디코더 Transformer**예요. 인코더가 로그 멜 "이미지"를 읽고, 디코더가 글자 토큰을 하나씩 써요. `vad_filter`로 **무음 구간을 빼면** 조용한 곳에서 문장을 지어내는 환각을 줄일 수 있어요.'),
    code: dd(String.raw`segments, info = model.transcribe(
    "meeting.mp3",
    language="ko",                # 지정하면 언어 감지 오류 방지
    vad_filter=True,              # 무음 구간 제거 — 환각(무음에서 문장 생성) 억제
    beam_size=5,
)`),
    highlight: [3, 4],
    render: (t, c) => (
      <g>
        <Heat id="logmel2" mat={LOGMEL_N} x={14} y={46} w={150} h={110} c={c} label="로그 멜" />
        <g opacity={seg(t, 0.15, 0.25)}>
          {[
            { a: 0, b: 0.04 },
            { a: 0.31, b: 0.37 },
            { a: 0.63, b: 0.69 },
            { a: 0.93, b: 1 }
          ].map((s, i) => (
            <rect key={i} x={14 + 150 * s.a} y={46} width={150 * (s.b - s.a)} height={110} fill={alpha(c.neg, 0.35)} />
          ))}
          <T x={89} y={176} anchor="middle" size={11.5} weight={800} color={c.neg}>
            VAD: 빨간 곳(무음)은 건너뛰기
          </T>
        </g>
        <Arrow x1={168} y1={100} x2={190} y2={100} color={c.series[0]} t={seg(t, 0.3, 0.15)} />
        <Box x={192} y={70} w={86} h={60} color={c.series[0]} t={seg(t, 0.32, 0.2)} label="인코더" sub="Transformer" labelSize={FS.m} />
        <Arrow x1={280} y1={100} x2={300} y2={100} color={c.series[1]} t={seg(t, 0.45, 0.15)} />
        <Box x={302} y={70} w={104} h={60} color={c.series[1]} t={seg(t, 0.47, 0.2)} label="디코더" sub="토큰을 하나씩" labelSize={FS.m} />
        {[
          { time: '[0.05 → 0.30]', text: '아' },
          { time: '[0.38 → 0.62]', text: '이' },
          { time: '[0.70 → 0.92]', text: '스' }
        ].map((s, i) => (
          <g key={i} opacity={seg(t, 0.6 + i * 0.1, 0.15)}>
            <rect x={196} y={152 + i * 40} width={210} height={30} rx={9} fill={alpha(c.series[1], 0.14)} stroke={c.series[1]} />
            <T x={206} y={172 + i * 40} size={FS.m} weight={750} mono>
              {s.time}
            </T>
            <T x={396} y={172 + i * 40} anchor="end" size={FS.s} weight={850}>
              {s.text}
            </T>
          </g>
        ))}
        <T x={14} y={220} size={FS.m} weight={750} color="var(--text-2)" opacity={seg(t, 0.85, 0.15)}>
          누가 말했는지(화자 분리)는
        </T>
        <T x={14} y={240} size={FS.m} weight={750} color="var(--text-2)" opacity={seg(t, 0.85, 0.15)}>
          Whisper 밖의 문제예요
        </T>
      </g>
    )
  },
  {
    title: '스펙트로그램은 이미지 — CNN으로 키워드 분류',
    text: rich('"yes", "no", "stop" 같은 짧은 명령어는 1초짜리 로그 멜(64×101)을 **2D CNN**에 넣어 분류해요. CIFAR 사진 분류 코드와 거의 같아요 — **표현만 바꾸면 도구는 그대로 옮겨 와요**.'),
    code: dd(String.raw`model = nn.Sequential(                                  # 입력 (B, 1, 64, 101)
    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, len(LABELS)),
).to(device)`),
    highlight: [1, 2, 3, 4, 5],
    render: (t, c) => {
      const labels = ['yes', 'no', 'up', 'down', 'stop', 'go'];
      const probs = [0.86, 0.04, 0.03, 0.02, 0.03, 0.02];
      return (
        <g>
          <Heat id="logmel3" mat={LOGMEL_N} x={14} y={56} w={92} h={92} c={c} label="64×101" />
          {[
            { x: 122, s: 70, l: '32ch' },
            { x: 186, s: 52, l: '64ch' },
            { x: 236, s: 38, l: '128ch' }
          ].map((b, i) => (
            <g key={i} opacity={seg(t, 0.12 + i * 0.12, 0.2)}>
              <rect x={b.x} y={102 - b.s / 2} width={b.s * 0.75} height={b.s} rx={6} fill={alpha(c.series[0], 0.15 + i * 0.1)} stroke={c.series[0]} strokeWidth={2} />
              <T x={b.x + (b.s * 0.75) / 2} y={102 + b.s / 2 + 16} anchor="middle" size={11.5} weight={750} mono>
                {b.l}
              </T>
            </g>
          ))}
          <g opacity={seg(t, 0.5, 0.3)}>
            {labels.map((l, i) => (
              <g key={l}>
                <T x={290} y={50 + i * 34} size={FS.m} weight={800} mono>
                  {l}
                </T>
                <rect x={330} y={39 + i * 34} width={76} height={14} rx={7} fill="var(--surface-3)" />
                <rect x={330} y={39 + i * 34} width={76 * probs[i]} height={14} rx={7} fill={i === 0 ? c.pos : c.series[0]} />
              </g>
            ))}
          </g>
          <T x={14} y={250} size={FS.m} weight={750} color="var(--text-2)" opacity={seg(t, 0.75, 0.2)}>
            8 에포크면 정확도 약 95% · 코드는 CIFAR CNN과 거의 같아요
          </T>
        </g>
      );
    }
  }
];

export default function Lab19Whisper(_: VizProps) {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={2} />;
}
