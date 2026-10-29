import { CodeStory, type StoryStep } from '../core/story';
import { softmax } from '../core/math';
import { Arrow, Cells, Chip, Curve, Defs, Emoji, Txt, VB, mix, seg, soft } from './_labsA';

const SAMPLE = '안녕하세요.반가워';
const CHARS = [...new Set(SAMPLE)].sort();
const STOI = Object.fromEntries(CHARS.map((ch, i) => [ch, i]));
const SEQ = [...SAMPLE];

const CANDS = ['요', '상', '계', '다', '기', '서', '를', '에'];
const LOGITS = [3.2, 1.4, 1.1, 0.6, 0.2, -0.2, -0.6, -1.0];
const TEMP = 0.8;
const TOPK = 5;

const steps: StoryStep[] = [
  {
    title: '글자 → 번호, 그리고 "한 칸 민 것이 정답"',
    text: (
      <>
        문자 단위 GPT는 텍스트의 서로 다른 글자들에 번호를 붙여요(<code>stoi</code>). 학습 데이터는 연속된 글자 묶음 <strong>x</strong>, 정답은 그걸 <strong>한 칸 민 y</strong>예요. 그러면 모든 자리에서 동시에 "다음 글자 맞히기" 문제가 생겨요.
      </>
    ),
    code: `chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

x = torch.stack([d[i : i + CTX] for i in ix])
y = torch.stack([d[i + 1 : i + CTX + 1] for i in ix])   # 한 칸 민 것이 정답`,
    highlight: [2, 6, 7],
    render: (t, c) => {
      const cx = (i: number) => 48 + i * 48;
      return (
        <g>
          <Defs />
          {SEQ.map((ch, i) => (
            <g key={i}>
              <Chip x={cx(i)} y={70} text={ch} w={38} h={34} fs={17} color={c.series[0]} t={seg(t, i * 0.03, 0.15 + i * 0.03)} c={c} />
              <Txt x={cx(i)} y={104} size={12} weight={800} mono color="var(--text-2)" opacity={seg(t, 0.2 + i * 0.03, 0.35 + i * 0.03)}>
                {STOI[ch]}
              </Txt>
            </g>
          ))}
          <Txt x={14} y={40} size={12} weight={800} anchor="start" color="var(--text-2)">
            글자와 번호 (stoi)
          </Txt>
          <g opacity={seg(t, 0.4, 0.55)}>
            <rect x={cx(0) - 22} y={128} width={cx(4) - cx(0) + 44} height={34} rx={10} fill={soft(c.series[0], 0.12)} stroke={c.series[0]} strokeWidth={2} />
            <Txt x={cx(0) - 30} y={150} size={13} weight={900} anchor="end" color={c.series[0]}>
              x
            </Txt>
            {SEQ.slice(0, 5).map((ch, i) => (
              <Txt key={i} x={cx(i)} y={151} size={15} weight={800}>
                {ch}
              </Txt>
            ))}
          </g>
          <g opacity={seg(t, 0.55, 0.7)}>
            <rect x={cx(1) - 22} y={206} width={cx(5) - cx(1) + 44} height={34} rx={10} fill={soft(c.series[2], 0.12)} stroke={c.series[2]} strokeWidth={2} />
            <Txt x={cx(1) - 30} y={228} size={13} weight={900} anchor="end" color={c.series[2]}>
              y
            </Txt>
            {SEQ.slice(1, 6).map((ch, i) => (
              <Txt key={i} x={cx(i + 1)} y={229} size={15} weight={800}>
                {ch}
              </Txt>
            ))}
          </g>
          {SEQ.slice(0, 5).map((_, i) => (
            <Arrow key={i} x1={cx(i) + 4} y1={164} x2={cx(i + 1) - 4} y2={204} color={c.series[2]} width={2} t={seg(t, 0.65 + i * 0.05, 0.8 + i * 0.05)} />
          ))}
          <Txt x={240} y={276} size={12} weight={750} color="var(--text-2)" opacity={seg(t, 0.85, 1)}>
            "안" 다음은 "녕", "안녕" 다음은 "하"… 5문제를 한 번에! (실제 CTX = 256)
          </Txt>
        </g>
      );
    }
  },
  {
    title: '인과 어텐션: 각 자리는 과거만 본다',
    text: (
      <>
        GPT는 Transformer의 <strong>디코더만</strong> 쌓은 모델이에요. 각 글자는 <strong>자기와 앞 글자들만</strong> 참고할 수 있어요(인과 마스크). PyTorch의 <code>scaled_dot_product_attention(is_causal=True)</code> 한 줄이 이 마스크와 빠른 커널을 한꺼번에 처리해 줘요.
      </>
    ),
    code: `q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
q = q.view(B, L, H, D // H).transpose(1, 2)          # (B,H,L,dk)
k = k.view(B, L, H, D // H).transpose(1, 2)
v = v.view(B, L, H, D // H).transpose(1, 2)

# is_causal은 '정사각 마스크'를 가정하므로 캐시 사용 시(L=1)는 마스크 불필요
y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))`,
    highlight: [1, 7],
    render: (t, c) => {
      const xs = SEQ.slice(0, 5).map((_, i) => 40 + i * 52);
      const last = 4;
      return (
        <g>
          <Defs />
          {xs.map((x, i) => (
            <Chip key={i} x={x} y={220} text={SEQ[i]} w={38} h={34} fs={17} color={i === last ? c.series[2] : c.series[0]} active={i === last} c={c} />
          ))}
          {xs.slice(0, last).map((x, i) => {
            const w = [0.15, 0.2, 0.3, 0.9][i];
            const mid = (x + xs[last]) / 2;
            const p = seg(t, 0.1 + i * 0.1, 0.35 + i * 0.1);
            return (
              <path
                key={i}
                d={`M${xs[last]} 200 Q${mid} ${200 - 30 - (xs[last] - x) * 0.45} ${x} 200`}
                fill="none"
                stroke={c.series[2]}
                strokeWidth={1.5 + w * 5}
                opacity={0.3 + 0.7 * w}
                strokeLinecap="round"
                pathLength={1}
                strokeDasharray={`${p} 1`}
              />
            );
          })}
          <Txt x={xs[last]} y={262} size={12} weight={850} color={c.series[2]}>
            "요"를 계산할 때
          </Txt>
          <Txt x={150} y={60} size={12.5} weight={800} color="var(--text-2)" opacity={seg(t, 0.5, 0.7)}>
            앞 글자들만 바라봐요 (뒤는 아직 없어요)
          </Txt>
          <g opacity={seg(t, 0.4, 0.7)}>
            <Txt x={392} y={40} size={12} weight={850}>
              인과 마스크
            </Txt>
            <Cells x={330} y={52} rows={5} cols={5} cell={22} gap={3} color={c.series[3]} t={seg(t, 0.4, 0.9)} order="diag" value={(r, k) => (k <= r ? 0.75 : 0)} hidden={(r, k) => k > r} />
            <Txt x={392} y={196} size={11.5} weight={700} color="var(--text-2)">
              위쪽 삼각형 = 미래 ✗
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '가중치 공유: 입력 임베딩 = 출력 헤드',
    text: (
      <>
        입력 임베딩은 "글자 → 벡터", 출력 헤드는 "벡터 → 각 글자 점수"예요. 둘 다 같은 <strong>글자 ↔ 벡터 대응표</strong>라서 <strong>행렬 하나를 같이 써요</strong>(weight tying). 어휘가 클수록 파라미터가 크게 줄고, 성능도 대체로 좋아져요.
      </>
    ),
    code: `self.tok = nn.Embedding(vocab, D)
self.pos = nn.Embedding(CTX, D)
self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
self.ln_f = nn.LayerNorm(D)
self.head = nn.Linear(D, vocab, bias=False)
self.head.weight = self.tok.weight               # 가중치 공유(weight tying)`,
    highlight: [1, 5, 6],
    render: (t, c) => {
      const rows = 7;
      const mx = 190;
      const my = 60;
      const rowH = 22;
      const hi = 2;
      return (
        <g>
          <Defs />
          <Txt x={mx + 50} y={my - 14} size={12.5} weight={850}>
            행렬 하나 (어휘 × D)
          </Txt>
          {Array.from({ length: rows }, (_, r) => (
            <g key={r}>
              <rect x={mx} y={my + r * rowH} width={100} height={rowH - 4} rx={5} fill={r === hi ? soft(c.series[0], 0.7) : soft(c.series[0], 0.18 + (r % 3) * 0.08)} stroke={r === hi ? c.series[0] : 'none'} strokeWidth={2} />
              <Txt x={mx - 8} y={my + r * rowH + 14} size={11.5} weight={800} anchor="end" color="var(--text-2)">
                {CHARS[r] ?? ''}
              </Txt>
            </g>
          ))}
          <g opacity={seg(t, 0.05, 0.4)}>
            <Txt x={70} y={90} size={12.5} weight={850} color={c.series[0]}>
              ① 입력: "{CHARS[hi]}"
            </Txt>
            <Arrow x1={110} y1={my + hi * rowH + 9} x2={mx - 26} y2={my + hi * rowH + 9} color={c.series[0]} t={seg(t, 0.1, 0.35)} label="행 꺼내기" labelDy={-8} />
            <Txt x={70} y={150} size={11.5} weight={700} color="var(--text-2)">
              = 그 글자의 벡터
            </Txt>
          </g>
          <g opacity={seg(t, 0.45, 0.8)}>
            <Txt x={402} y={70} size={12.5} weight={850} color={c.series[2]}>
              ② 출력: 마지막 벡터 h
            </Txt>
            <rect x={372} y={82} width={60} height={14} rx={4} fill={soft(c.series[2], 0.6)} />
            {Array.from({ length: rows }, (_, r) => (
              <path key={r} d={`M372 89 L${mx + 104} ${my + r * rowH + 9}`} stroke={c.series[2]} strokeWidth={1.2} opacity={0.5 * seg(t, 0.5 + r * 0.03, 0.7 + r * 0.03)} fill="none" />
            ))}
            <Txt x={402} y={128} size={11.5} weight={700} color="var(--text-2)">
              모든 행과 내적
            </Txt>
            <Txt x={402} y={146} size={11.5} weight={700} color="var(--text-2)">
              = 글자별 점수(로짓)
            </Txt>
          </g>
          <g opacity={seg(t, 0.8, 1)}>
            <rect x={60} y={232} width={360} height={46} rx={14} fill={soft(c.series[1], 0.12)} stroke={soft(c.series[1], 0.6)} />
            <Txt x={240} y={254} size={13} weight={850}>
              같은 표를 앞에서 한 번, 뒤에서 한 번!
            </Txt>
            <Txt x={240} y={272} size={11.5} weight={700} color="var(--text-2)">
              어휘 5만 × 4096이라면 2억 개 파라미터 절약
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '샘플링: 온도로 날카롭게, top-k로 자르고, 주사위',
    text: (
      <>
        "안녕하세" 다음 글자의 점수(로짓)를 <strong>온도(0.8)로 나누면</strong> 1등이 더 두드러지고, <strong>top-k(5)</strong>는 6등부터를 아예 지워요. 남은 것으로 확률을 만들어 <strong>주사위를 굴려(multinomial)</strong> 한 글자를 뽑아요. 그래서 같은 질문에도 답이 조금씩 달라져요.
      </>
    ),
    code: `logits_last = logits[:, -1] / temperature
if top_k:
    thresh = logits_last.topk(top_k).values[:, -1:]
    logits_last[logits_last < thresh] = float("-inf")
nxt = torch.multinomial(logits_last.softmax(-1), 1)
idx = torch.cat([idx, nxt], dim=1)`,
    highlight: [1, 3, 4, 5],
    render: (t, c) => {
      const temp = seg(t, 0.15, 0.4);
      const cut = seg(t, 0.4, 0.6);
      const prob = seg(t, 0.6, 0.85);
      const scaled = LOGITS.map((v) => v / mix(1, TEMP, temp));
      const kept = scaled.map((_, i) => i < TOPK);
      const probs = softmax(scaled.filter((_, i) => kept[i]));
      const base = 150;
      return (
        <g>
          <Defs />
          <Txt x={14} y={30} size={12.5} weight={850} anchor="start" color="var(--text-2)">
            "안녕하세" → 다음 글자는?
          </Txt>
          <line x1={30} x2={450} y1={base} y2={base} stroke="var(--viz-axis)" strokeWidth={1.5} />
          {CANDS.map((ch, i) => {
            const x = 40 + i * 52;
            const removed = !kept[i] && cut > 0.5;
            const v = prob > 0 && kept[i] ? mix(scaled[i] * 22, probs[i] * 110, prob) : scaled[i] * 22;
            const h = v * (removed ? 1 - cut : 1);
            return (
              <g key={ch} opacity={removed ? 0.25 : 1}>
                <rect x={x} y={h >= 0 ? base - h : base} width={34} height={Math.abs(h)} rx={5} fill={i === 0 ? c.series[0] : v >= 0 ? soft(c.series[0], 0.55) : soft(c.neg, 0.6)} />
                <Txt x={x + 17} y={204} size={15} weight={850}>
                  {ch}
                </Txt>
                {prob > 0.8 && kept[i] && (
                  <Txt x={x + 17} y={base - h - 6} size={11} weight={800} color="var(--text-2)">
                    {Math.round(probs[i] * 100)}%
                  </Txt>
                )}
                {removed && (
                  <Txt x={x + 17} y={base - 8} size={14} weight={900} color={c.neg}>
                    ✗
                  </Txt>
                )}
              </g>
            );
          })}
          <Txt x={240} y={236} size={12.5} weight={850} color={prob > 0 ? c.series[3] : cut > 0 ? c.neg : c.series[0]} opacity={seg(t, 0.12, 0.2)}>
            {prob > 0 ? '③ 남은 5개로 확률 만들기 (softmax)' : cut > 0 ? '② top-5 밖은 지우기 (-inf)' : '① 점수 ÷ 온도 0.8 → 1등이 더 뚜렷'}
          </Txt>
          <g opacity={seg(t, 0.4, 0.6)}>
            <rect x={40 + TOPK * 52 - 9} y={50} width={2} height={150} fill={c.neg} opacity={0.6} />
            <Txt x={40 + TOPK * 52 - 4} y={62} size={11.5} weight={850} anchor="start" color={c.neg}>
              top-5 경계
            </Txt>
          </g>
          <g opacity={seg(t, 0.85, 1)}>
            <Emoji x={140} y={270} size={22} className="labA-bob">
              🎲
            </Emoji>
            <Txt x={166} y={268} size={13.5} weight={850} anchor="start">
              multinomial → "요" (가장 흔하지만, 매번은 아니에요)
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: 'KV 캐시: 프리필 한 번, 그다음엔 한 글자씩',
    text: (
      <>
        생성은 두 단계예요. <strong>프리필</strong>은 프롬프트 전체를 한 번에 통과시키며 각 층의 K, V를 저장하고, <strong>디코드</strong>는 새 글자 <strong>하나만</strong> 넣어 저장된 K, V에 이어 붙여요. 캐시가 없으면 매 글자마다 처음부터 다시 계산해서 점점 느려져요.
      </>
    ),
    code: `if kv_cache is not None:                             # 추론: 과거 K,V 재사용
    pk, pv = kv_cache
    k = torch.cat([pk, k], dim=2)
    v = torch.cat([pv, v], dim=2)

logits, caches = self(idx)
logits, caches = self(nxt, caches, pos_offset=idx.shape[1] - 1)`,
    highlight: [3, 4, 6, 7],
    render: (t, c) => {
      const prompt = ['안', '녕', '하'];
      const gen = ['세', '요'];
      const stage = t < 0.35 ? 0 : t < 0.65 ? 1 : 2;
      const cols = prompt.length + stage;
      const colW = 26;
      return (
        <g>
          <Defs />
          <Txt x={110} y={30} size={12.5} weight={850} color="var(--text-2)">
            {stage === 0 ? '① 프리필: 프롬프트 전체를 한 번에' : `② 디코드 ${stage}: 새 글자 하나만`}
          </Txt>
          {[...prompt, ...gen].map((ch, i) => {
            const isNew = i === cols - 1 && stage > 0;
            const visible = i < cols;
            const inInput = stage === 0 ? i < 3 : i === cols - 1;
            return (
              <g key={i} opacity={visible ? 1 : 0.15}>
                <Chip x={40 + i * 44} y={70} text={ch} w={36} h={32} fs={16} color={i < 3 ? c.series[0] : c.series[2]} active={inInput && visible} c={c} />
                {isNew && (
                  <Txt x={40 + i * 44} y={112} size={11} weight={850} color={c.series[2]}>
                    새 입력
                  </Txt>
                )}
              </g>
            );
          })}
          <rect x={20} y={136} width={240} height={110} rx={14} fill={soft(c.series[1], 0.06)} stroke={soft(c.series[1], 0.5)} strokeDasharray="5 4" />
          <Txt x={140} y={156} size={12} weight={850} color={c.series[1]}>
            KV 캐시 (층마다)
          </Txt>
          {Array.from({ length: cols }, (_, i) => {
            const fresh = stage > 0 && i === cols - 1;
            const p = fresh ? seg(t, stage === 1 ? 0.4 : 0.7, stage === 1 ? 0.6 : 0.9) : seg(t, 0.05 + i * 0.06, 0.25 + i * 0.06);
            return (
              <g key={i} opacity={p}>
                <rect x={36 + i * (colW + 6)} y={170} width={colW} height={28} rx={5} fill={soft(c.series[1], fresh ? 0.8 : 0.4)} />
                <rect x={36 + i * (colW + 6)} y={204} width={colW} height={28} rx={5} fill={soft(c.series[3], fresh ? 0.8 : 0.4)} />
              </g>
            );
          })}
          <Txt x={30} y={189} size={11} weight={850} anchor="end" color={c.series[1]}>
            K
          </Txt>
          <Txt x={30} y={223} size={11} weight={850} anchor="end" color={c.series[3]}>
            V
          </Txt>
          <Txt x={370} y={146} size={12} weight={850}>
            글자 하나당 계산량
          </Txt>
          {[1, 2, 3, 4, 5].map((n, i) => (
            <g key={n}>
              <rect x={300 + i * 30} y={250 - n * 16} width={11} height={n * 16} rx={3} fill={soft(c.neg, 0.55)} />
              <rect x={312 + i * 30} y={250 - 16} width={11} height={16} rx={3} fill={c.pos} />
            </g>
          ))}
          <line x1={294} x2={454} y1={250} y2={250} stroke="var(--viz-axis)" strokeWidth={1.5} />
          <Txt x={318} y={270} size={11} weight={800} anchor="start" color={c.neg}>
            캐시 없음
          </Txt>
          <Txt x={388} y={270} size={11} weight={800} anchor="start" color={c.pos}>
            캐시 있음
          </Txt>
        </g>
      );
    }
  },
  {
    title: '손실 읽기: ln(어휘)에서 1.5까지',
    text: (
      <>
        처음 손실은 <strong>ln(어휘 수)</strong> 근처예요 — 글자 65종이면 약 4.2(아무거나 찍는 수준). 학습 후 손실 1.5는 퍼플렉시티 <strong>e^1.5 ≈ 4.5</strong>, 즉 "다음 글자 후보를 평균 4~5개로 좁혔다"는 뜻이에요. val이 train과 벌어지기 시작하면 과적합의 신호예요.
      </>
    ),
    code: `logits, _ = model(x)
loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
opt.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step(); sched.step()`,
    highlight: [2],
    render: (t, c) => {
      const train = (s: number) => 1.32 + 2.85 * Math.exp(-s * 1.2) - 0.06 * s;
      const val = (s: number) => 1.42 + 2.75 * Math.exp(-s * 1.2) + 0.012 * s * s;
      return (
        <g>
          <Defs />
          <Curve x={52} y={36} w={250} h={180} fn={train} xDomain={[0, 5]} yDomain={[0, 4.5]} t={seg(t, 0.05, 0.75)} color={c.series[0]} xLabel="스텝 (천 단위)" yLabel="손실" />
          <Curve x={52} y={36} w={250} h={180} fn={val} xDomain={[0, 5]} yDomain={[0, 4.5]} t={seg(t, 0.1, 0.8)} color={c.series[2]} />
          <line x1={52} x2={302} y1={36 + 180 - (4.17 / 4.5) * 180} y2={36 + 180 - (4.17 / 4.5) * 180} stroke={c.series[4]} strokeDasharray="4 5" opacity={seg(t, 0, 0.2)} />
          <Txt x={306} y={36 + 180 - (4.17 / 4.5) * 180 + 4} size={11.5} weight={800} anchor="start" color="var(--text-2)" opacity={seg(t, 0, 0.2)}>
            ln 65 ≈ 4.17
          </Txt>
          <g opacity={seg(t, 0.6, 0.8)}>
            <Txt x={70} y={258} size={11.5} weight={800} anchor="start" color={c.series[0]}>
              ━ train
            </Txt>
            <Txt x={140} y={258} size={11.5} weight={800} anchor="start" color={c.series[2]}>
              ━ val
            </Txt>
          </g>
          <g opacity={seg(t, 0.75, 1)}>
            <rect x={324} y={84} width={146} height={150} rx={16} fill={soft(c.series[3], 0.1)} stroke={soft(c.series[3], 0.6)} />
            <Txt x={397} y={108} size={12.5} weight={850}>
              손실 1.5면
            </Txt>
            <Txt x={397} y={128} size={11.5} weight={700} color="var(--text-2)">
              후보가 약 4.5개
            </Txt>
            {['요', '다', '서', '고'].map((ch, i) => (
              <Chip key={ch} x={350 + (i % 2) * 46} y={154 + Math.floor(i / 2) * 34} text={ch} w={36} h={28} fs={15} color={c.series[3]} c={c} />
            ))}
            <Txt x={397} y={226} size={11.5} weight={800} color="var(--text-2)">
              + 반쯤 하나 더
            </Txt>
          </g>
        </g>
      );
    }
  }
];

/** Code lab 09 walkthrough: char-level GPT — data, causal attention, weight tying, sampling, KV cache. */
export default function Lab09Gpt() {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={2} />;
}
