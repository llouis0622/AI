import { CodeStory, type StoryStep } from '../core/story';
import type { VizColors } from '../core/colors';
import { Arrow, Box, Cells, Chip, Defs, Txt, VB, seg, soft } from './_labsA';

const SRC = ['h', 'e', 'l', 'l', 'o'];
const TGT_IN = ['BOS', 'o', 'l', 'l', 'e', 'h'];
const TGT_OUT = ['o', 'l', 'l', 'e', 'h', 'EOS'];

function Tok({ x, y, s, c, t = 1, kind = 'src', w }: { x: number; y: number; s: string; c: VizColors; t?: number; kind?: 'src' | 'tgt' | 'out' | 'pad'; w?: number }) {
  const col = kind === 'src' ? c.series[0] : kind === 'tgt' ? c.series[3] : kind === 'out' ? c.series[2] : '#9aa0b4';
  const special = s === 'BOS' || s === 'EOS' || s === 'PAD';
  return <Chip x={x} y={y} text={s} w={w ?? (special ? 40 : 28)} h={26} fs={special ? 11 : 14} mono color={col} t={t} c={c} dim={kind === 'pad'} />;
}

/** Attention grid with row/column token labels. */
function AttnGrid({
  x,
  y,
  rows,
  cols,
  cell,
  t,
  value,
  hidden,
  color,
  order = 'row'
}: {
  x: number;
  y: number;
  rows: string[];
  cols: string[];
  cell: number;
  t: number;
  value: (r: number, k: number) => number;
  hidden?: (r: number, k: number) => boolean;
  color: string;
  order?: 'row' | 'diag';
}) {
  return (
    <g>
      {cols.map((s, k) => (
        <Txt key={`c${k}`} x={x + k * (cell + 3) + cell / 2} y={y - 8} size={11} weight={800} mono color="var(--text-2)">
          {s}
        </Txt>
      ))}
      {rows.map((s, r) => (
        <Txt key={`r${r}`} x={x - 8} y={y + r * (cell + 3) + cell / 2 + 4} size={11} weight={800} mono anchor="end" color="var(--text-2)">
          {s}
        </Txt>
      ))}
      <Cells x={x} y={y} rows={rows.length} cols={cols.length} cell={cell} gap={3} color={color} t={t} value={value} hidden={hidden} order={order} />
    </g>
  );
}

const steps: StoryStep[] = [
  {
    title: '구조 한눈에: 인코더가 읽고, 디코더가 쓴다',
    text: (
      <>
        <strong>인코더</strong>는 소스 문장 전체를 서로 바라보며(마스크 없이) 이해하고, <strong>디코더</strong>는 지금까지 쓴 타깃 토큰만 보면서(인과 마스크) 다음 토큰을 골라요. 디코더는 중간에 <strong>cross-attention</strong>으로 인코더의 출력을 참고해요.
      </>
    ),
    code: `def forward(self, src, tgt):
    src_mask, tgt_mask = self.make_masks(src, tgt)
    e = self.emb(src) + self.pe[: src.shape[1]]
    for layer in self.encoder:
        e = layer(e, src_mask)
    d = self.emb(tgt) + self.pe[: tgt.shape[1]]
    for layer in self.decoder:
        d = layer(d, e, tgt_mask, src_mask)
    return self.head(self.ln_f(d))`,
    highlight: [4, 5, 7, 8, 9],
    render: (t, c) => (
      <g>
        <Defs />
        {SRC.map((s, i) => (
          <Tok key={i} x={56 + i * 32} y={264} s={s} c={c} t={seg(t, i * 0.03, 0.15 + i * 0.03)} />
        ))}
        <Txt x={120} y={292} size={11.5} weight={800} color={c.series[0]}>
          소스 토큰
        </Txt>
        <Arrow x1={120} y1={248} x2={120} y2={206} color={c.series[0]} t={seg(t, 0.15, 0.3)} flow />
        <Txt x={130} y={232} size={11.5} weight={750} anchor="start" color="var(--text-2)" opacity={seg(t, 0.2, 0.3)}>
          임베딩 + 위치
        </Txt>
        <Box x={40} y={104} w={160} h={96} label="인코더 ×N" sub="self-attn · 마스크 없음" subFs={11} fs={15} color={c.series[0]} t={seg(t, 0.2, 0.4)} c={c} />
        {TGT_IN.map((s, i) => (
          <Tok key={i} x={290 + i * 33} y={264} s={s} c={c} kind="tgt" t={seg(t, 0.3 + i * 0.03, 0.45 + i * 0.03)} w={s === 'BOS' ? 40 : 28} />
        ))}
        <Txt x={366} y={292} size={11.5} weight={800} color={c.series[3]}>
          타깃 토큰 (한 칸 밀림)
        </Txt>
        <Arrow x1={366} y1={248} x2={366} y2={214} color={c.series[3]} t={seg(t, 0.45, 0.58)} flow />
        <Txt x={376} y={235} size={11.5} weight={750} anchor="start" color="var(--text-2)" opacity={seg(t, 0.5, 0.58)}>
          임베딩 + 위치
        </Txt>
        <Box x={282} y={84} w={168} h={124} c={c} color={c.series[3]} t={seg(t, 0.5, 0.65)} />
        <g opacity={seg(t, 0.55, 0.7)}>
          <Txt x={366} y={106} size={15} weight={850}>
            디코더 ×N
          </Txt>
          <rect x={296} y={118} width={140} height={22} rx={6} fill={soft(c.series[3], 0.2)} />
          <Txt x={366} y={133} size={11.5} weight={750}>
            인과 self-attn
          </Txt>
          <rect x={296} y={146} width={140} height={22} rx={6} fill={soft(c.series[2], 0.25)} />
          <Txt x={366} y={161} size={11.5} weight={800}>
            cross-attn
          </Txt>
          <rect x={296} y={174} width={140} height={22} rx={6} fill={soft(c.series[1], 0.2)} />
          <Txt x={366} y={189} size={11.5} weight={750}>
            FFN
          </Txt>
        </g>
        <Arrow x1={204} y1={152} x2={292} y2={157} color={c.series[2]} width={3} t={seg(t, 0.68, 0.85)} label="K, V" labelDy={-10} flow />
        <Arrow x1={366} y1={80} x2={366} y2={50} color={c.series[2]} t={seg(t, 0.82, 0.95)} />
        <g opacity={seg(t, 0.88, 1)}>
          <rect x={300} y={18} width={132} height={28} rx={14} fill={soft(c.series[2], 0.16)} stroke={c.series[2]} />
          <Txt x={366} y={37} size={12.5} weight={850}>
            다음 토큰 로짓
          </Txt>
        </g>
      </g>
    )
  },
  {
    title: '과제: 문자열 뒤집기 + teacher forcing',
    text: (
      <>
        무작위 문자열을 넣으면 거꾸로 뒤집어 내는 장난감 과제예요. 학습 때 디코더 입력은 <code>[BOS, 정답…]</code>, 맞힐 목표는 <code>[정답…, EOS]</code>로 <strong>한 칸씩 밀려</strong> 있어요. 각 자리가 "다음 토큰"을 맞히는 거죠 (teacher forcing).
      </>
    ),
    code: `s = torch.randint(3, VOCAB, (L,))                # 특수 토큰 제외
r = s.flip(0)
pad = max_len - L
src.append(torch.cat([s, torch.zeros(pad, dtype=torch.long)]))
# teacher forcing: 디코더 입력은 [BOS, r], 정답은 [r, EOS]
tgt_in.append(torch.cat([torch.tensor([BOS]), r, torch.zeros(pad, dtype=torch.long)]))
tgt_out.append(torch.cat([r, torch.tensor([EOS]), torch.zeros(pad, dtype=torch.long)]))`,
    highlight: [2, 6, 7],
    render: (t, c) => (
      <g>
        <Defs />
        <Txt x={84} y={52} size={12.5} weight={850} anchor="end" color={c.series[0]}>
          src
        </Txt>
        {SRC.map((s, i) => (
          <Tok key={i} x={120 + i * 44} y={46} s={s} c={c} t={seg(t, i * 0.04, 0.15 + i * 0.04)} />
        ))}
        <Txt x={360} y={52} size={11.5} weight={700} anchor="start" color="var(--text-2)" opacity={seg(t, 0.2, 0.35)}>
          ↩ 뒤집으면
        </Txt>
        <Txt x={84} y={142} size={12.5} weight={850} anchor="end" color={c.series[3]}>
          tgt_in
        </Txt>
        {TGT_IN.map((s, i) => (
          <Tok key={i} x={120 + i * 44} y={136} s={s} c={c} kind="tgt" t={seg(t, 0.25 + i * 0.04, 0.4 + i * 0.04)} w={s === 'BOS' ? 40 : 28} />
        ))}
        <Txt x={84} y={232} size={12.5} weight={850} anchor="end" color={c.series[2]}>
          tgt_out
        </Txt>
        {TGT_OUT.map((s, i) => (
          <Tok key={i} x={120 + i * 44} y={226} s={s} c={c} kind="out" t={seg(t, 0.45 + i * 0.04, 0.6 + i * 0.04)} w={s === 'EOS' ? 40 : 28} />
        ))}
        {TGT_IN.map((_, i) => (
          <Arrow key={i} x1={120 + i * 44} y1={152} x2={120 + i * 44} y2={208} color={c.series[2]} width={2} t={seg(t, 0.6 + i * 0.05, 0.75 + i * 0.05)} />
        ))}
        <Txt x={400} y={186} size={12} weight={800} anchor="start" color={c.series[2]} opacity={seg(t, 0.8, 1)}>
          ↓ 각 자리의
        </Txt>
        <Txt x={400} y={204} size={12} weight={800} anchor="start" color={c.series[2]} opacity={seg(t, 0.8, 1)}>
          "다음 토큰"
        </Txt>
        <Txt x={240} y={284} size={11.5} weight={700} color="var(--text-2)" opacity={seg(t, 0.85, 1)}>
          (실제 데이터는 무작위 문자 id예요 — 그림은 알아보기 쉽게 글자로)
        </Txt>
      </g>
    )
  },
  {
    title: '인코더: 패딩 마스크로 빈칸은 안 본다',
    text: (
      <>
        길이가 다른 문장을 묶으려면 뒤를 <strong>PAD</strong>로 채워요. 어텐션이 PAD를 진짜 데이터로 착각하지 않게, <strong>PAD 열을 가리는 마스크</strong>를 모든 쿼리에 똑같이 씌워요. 인코더는 그 외엔 마스크가 없어서 소스끼리 자유롭게 바라봐요.
      </>
    ),
    code: `# 패딩 마스크: (B, 1, 1, L) — 브로드캐스트로 모든 쿼리에 적용
src_mask = (src != PAD)[:, None, None, :]

x = x + self.sa(self.ln1(x), self.ln1(x), self.ln1(x), src_mask)
return x + self.ff(self.ln2(x))`,
    highlight: [2, 4],
    render: (t, c) => {
      const toks = [...SRC, 'PAD', 'PAD'];
      return (
        <g>
          <Defs />
          <AttnGrid
            x={110}
            y={52}
            rows={toks}
            cols={toks}
            cell={26}
            t={seg(t, 0.05, 0.75)}
            color={c.series[0]}
            value={(r, k) => (r >= 5 ? 0.12 : r === k ? 0.85 : 0.35 + 0.3 * Math.abs(Math.sin(r * 2 + k)))}
            hidden={(_r, k) => k >= 5}
          />
          <g opacity={seg(t, 0.6, 0.85)}>
            <rect x={251} y={44} width={63} height={212} rx={8} fill="none" stroke={c.neg} strokeWidth={2} strokeDasharray="5 4" />
            <Txt x={282} y={274} size={12} weight={850} color={c.neg}>
              PAD 열 ✗
            </Txt>
          </g>
          <g opacity={seg(t, 0.75, 1)}>
            <Txt x={392} y={110} size={13} weight={850}>
              진짜 토큰끼리만
            </Txt>
            <Txt x={392} y={132} size={13} weight={850}>
              서로 바라봐요
            </Txt>
            <Txt x={392} y={164} size={11.5} weight={700} color="var(--text-2)">
              행 = 바라보는 쪽 (Q)
            </Txt>
            <Txt x={392} y={182} size={11.5} weight={700} color="var(--text-2)">
              열 = 바라봐지는 쪽 (K)
            </Txt>
            <Txt x={392} y={214} size={11.5} weight={700} color="var(--text-2)">
              (PAD 행 결과는 버려요)
            </Txt>
          </g>
        </g>
      );
    }
  },
  {
    title: '디코더 self-attn: 미래는 못 본다',
    text: (
      <>
        디코더가 학습 중에 정답의 뒷부분을 훔쳐보면 안 되겠죠? <code>torch.tril</code>로 만든 <strong>아래 삼각형 마스크</strong> 덕분에 각 자리는 자기와 그 앞만 볼 수 있어요. 그래서 모든 자리를 한 번에 병렬로 학습할 수 있어요.
      </>
    ),
    code: `tgt_pad = (tgt != PAD)[:, None, None, :]
L = tgt.shape[1]
causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=tgt.device))
return src_mask, tgt_pad & causal                # (B,1,L,L)`,
    highlight: [3, 4],
    render: (t, c) => (
      <g>
        <Defs />
        <AttnGrid
          x={110}
          y={52}
          rows={TGT_IN}
          cols={TGT_IN}
          cell={30}
          t={seg(t, 0.05, 0.8)}
          color={c.series[3]}
          order="diag"
          value={(r, k) => (k <= r ? 0.35 + 0.55 * ((k + 1) / (r + 1)) : 0)}
          hidden={(r, k) => k > r}
        />
        <g opacity={seg(t, 0.6, 0.9)}>
          <Txt x={316} y={72} size={12.5} weight={850} color={c.neg} anchor="start">
            ← 미래 차단
          </Txt>
          <Txt x={390} y={150} size={13} weight={850}>
            첫 "l" 줄은
          </Txt>
          <Txt x={390} y={172} size={13} weight={850}>
            BOS·o·l까지만
          </Txt>
          <Txt x={390} y={204} size={11.5} weight={700} color="var(--text-2)">
            그래서 모든 자리를
          </Txt>
          <Txt x={390} y={222} size={11.5} weight={700} color="var(--text-2)">
            한 번에 병렬 학습!
          </Txt>
        </g>
      </g>
    )
  },
  {
    title: 'cross-attn: 뒤집기라면 역대각선을 본다',
    text: (
      <>
        cross-attention에서 Q는 디코더(타깃), K·V는 <strong>인코더 출력</strong>이에요. 잘 학습된 뒤집기 모델은 출력의 첫 글자가 소스의 마지막 글자를, 두 번째 글자가 끝에서 두 번째를… 바라봐요. 그래서 가중치가 <strong>역대각선</strong> 모양이 돼요.
      </>
    ),
    code: `def forward(self, x, enc, tgt_mask, src_mask):
    x = x + self.sa(self.ln1(x), self.ln1(x), self.ln1(x), tgt_mask)   # 미래 차단
    x = x + self.ca(self.ln2(x), enc, enc, src_mask)                    # 소스 참조
    return x + self.ff(self.ln3(x))`,
    highlight: [3],
    render: (t, c) => (
      <g>
        <Defs />
        <Txt x={190} y={22} size={12} weight={800} color={c.series[0]}>
          K, V = 인코더 출력 (소스)
        </Txt>
        <AttnGrid
          x={124}
          y={54}
          rows={TGT_OUT}
          cols={SRC}
          cell={30}
          t={seg(t, 0.1, 0.8)}
          color={c.series[2]}
          value={(r, k) => (r < 5 && k === 4 - r ? 0.95 : r === 5 ? 0.25 : 0.06)}
        />
        <Txt x={66} y={160} size={12} weight={800} color={c.series[3]} anchor="middle">
          Q = 디코더
        </Txt>
        <g opacity={seg(t, 0.75, 1)}>
          <path d="M 271 69 L 139 201" stroke={c.series[2]} strokeWidth={2} strokeDasharray="5 5" fill="none" />
          <Txt x={390} y={124} size={13} weight={850}>
            역대각선 = 뒤집기!
          </Txt>
          <Txt x={390} y={150} size={11.5} weight={700} color="var(--text-2)">
            복사 과제라면 대각선이
          </Txt>
          <Txt x={390} y={168} size={11.5} weight={700} color="var(--text-2)">
            보일 거예요
          </Txt>
        </g>
      </g>
    )
  },
  {
    title: '생성: 한 토큰씩, 가장 확률 높은 것으로',
    text: (
      <>
        추론 때는 정답이 없으니 <strong>BOS 하나로 시작</strong>해 매번 마지막 자리의 로짓에서 <code>argmax</code>로 다음 토큰을 골라 이어 붙여요(greedy). EOS가 나오면 멈춰요. 학습은 병렬, 추론은 한 글자씩 — 이 비대칭이 LLM 서빙 문제의 뿌리예요.
      </>
    ),
    code: `ys = torch.full((src.shape[0], 1), BOS, dtype=torch.long, device=src.device)
for _ in range(max_len):
    logits = self(src, ys)[:, -1]                # 마지막 위치의 다음 토큰
    nxt = logits.argmax(-1, keepdim=True)
    ys = torch.cat([ys, nxt], dim=1)
    if (nxt == EOS).all():
        break
return ys`,
    highlight: [3, 4, 5, 6],
    render: (t, c) => {
      const out = ['BOS', ...TGT_OUT];
      const q = seg(t, 0.05, 0.92) * (out.length - 1);
      const n = Math.min(out.length, 1 + Math.floor(q));
      const local = n >= out.length ? 1 : q - Math.floor(q);
      return (
        <g>
          <Defs />
          <Txt x={70} y={60} size={12.5} weight={850} anchor="end" color={c.series[0]}>
            src
          </Txt>
          {SRC.map((s, i) => (
            <Tok key={i} x={100 + i * 36} y={54} s={s} c={c} />
          ))}
          <Txt x={70} y={150} size={12.5} weight={850} anchor="end" color={c.series[3]}>
            ys
          </Txt>
          {out.slice(0, n).map((s, i) => (
            <Tok key={`${i}-${s}`} x={110 + i * 46} y={144} s={s} c={c} kind={i === 0 ? 'tgt' : 'out'} w={s.length > 1 ? 40 : 30} t={i === n - 1 && i > 0 ? Math.min(1, local * 1.6 + 0.15) : 1} />
          ))}
          {n < out.length && (
            <g>
              <Arrow x1={110 + (n - 1) * 46} y1={162} x2={110 + n * 46 - 12} y2={162} color={c.series[2]} width={2} t={1} bend={20} />
              <Txt x={110 + n * 46 - 4} y={202} size={11.5} weight={850} color={c.series[2]}>
                argmax
              </Txt>
            </g>
          )}
          <g opacity={n >= out.length ? 1 : 0}>
            <rect x={130} y={226} width={220} height={40} rx={14} fill={soft(c.pos, 0.14)} stroke={c.pos} />
            <Txt x={240} y={252} size={14} weight={850}>
              EOS → 멈춤! "olleh" 완성
            </Txt>
          </g>
        </g>
      );
    }
  }
];

/** Code lab 08 overview: encoder-decoder map, masks, cross-attention and greedy decoding. */
export default function Lab08TransformerMap() {
  return <CodeStory steps={steps} viewBox={VB} stepSeconds={2.2} />;
}
