# 09. GPT 밑바닥 구현과 학습

**만드는 것**: 디코더 전용(decoder-only) GPT를 구현해 텍스트 코퍼스로 학습하고, 온도·top-k 샘플링으로 생성하고, KV 캐시로 추론을 가속한다. 모든 현대 LLM의 최소 완전체다.

**선행 지식**: [GPT와 인과 언어 모델링](/handbook/08-sequence-nlp/12-gpt-and-causal-lm), [LLM의 본질](/curriculum/ch06/lecture13), [08. Transformer](/practice/08-transformer-from-scratch)

## 전체 코드

문자 단위 언어 모델이다. 어휘가 작아 임베딩이 가볍고, 토크나이저 없이 원리에 집중할 수 있다. `input.txt`에 아무 텍스트(소설, 위키 덤프, 코드 등 1MB+)를 넣는다.

```python
"""GPT from scratch — 문자 단위 LM. 실행: python gpt.py"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 하이퍼파라미터 ----------
CTX = 256          # 컨텍스트 길이
D, H, LAYERS = 384, 6, 6
BATCH, STEPS, LR = 64, 5000, 3e-4


# ---------- 데이터: 문자 → 정수 ----------
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(len(data) * 0.9)
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - CTX - 1, (BATCH,))
    x = torch.stack([d[i : i + CTX] for i in ix])
    y = torch.stack([d[i + 1 : i + CTX + 1] for i in ix])   # 한 칸 민 것이 정답
    return x.to(device), y.to(device)


# ---------- 모델 ----------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def attn(self, x, kv_cache=None):
        B, L, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q = q.view(B, L, H, D // H).transpose(1, 2)          # (B,H,L,dk)
        k = k.view(B, L, H, D // H).transpose(1, 2)
        v = v.view(B, L, H, D // H).transpose(1, 2)
        if kv_cache is not None:                             # 추론: 과거 K,V 재사용
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        # is_causal은 '정사각 마스크'를 가정하므로 캐시 사용 시(L=1)는 마스크 불필요
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))
        y = y.transpose(1, 2).reshape(B, L, D)
        return self.proj(y), (k, v)

    def forward(self, x, kv_cache=None):
        a, new_cache = self.attn(x, kv_cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)
        self.head.weight = self.tok.weight               # 가중치 공유(weight tying)

    def forward(self, idx, caches=None, pos_offset=0):
        B, L = idx.shape
        pos = torch.arange(pos_offset, pos_offset + L, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, caches[i] if caches else None)
            new_caches.append(c)
        return self.head(self.ln_f(x)), new_caches

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=1.0, top_k=50):
        self.eval()
        # 1) 프리필: 프롬프트 전체를 한 번에 통과시키며 캐시 구축
        logits, caches = self(idx)
        for step in range(max_new):
            logits_last = logits[:, -1] / temperature
            if top_k:
                thresh = logits_last.topk(top_k).values[:, -1:]
                logits_last[logits_last < thresh] = float("-inf")
            nxt = torch.multinomial(logits_last.softmax(-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if idx.shape[1] >= CTX:                       # 컨텍스트 초과 방지
                break
            # 2) 디코드: 새 토큰 '하나'만 넣는다 — 캐시가 과거를 기억한다
            logits, caches = self(nxt, caches, pos_offset=idx.shape[1] - 1)
        return idx


# ---------- 학습 ----------
def main():
    torch.manual_seed(1337)
    model = GPT(len(chars)).to(device)
    print(f"파라미터: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, 어휘: {len(chars)}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    for step in range(STEPS):
        x, y = get_batch("train")
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if (step + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch("val")
                vlogits, _ = model(vx)
                vloss = F.cross_entropy(vlogits.reshape(-1, vlogits.shape[-1]), vy.reshape(-1))
            print(f"step {step+1:5d}  train {loss.item():.4f}  val {vloss.item():.4f}")
            model.train()

    # 생성 데모
    prompt = torch.tensor([[stoi[c] for c in text[:32]]], device=device)
    out = model.generate(prompt, max_new=200, temperature=0.8)
    print("".join(itos[i] for i in out[0].tolist()))
    torch.save(model.state_dict(), "gpt_char.pt")


if __name__ == "__main__":
    main()
```

GPU에서 5000 스텝 ~10분. 한국어 소설을 넣으면 그럴듯한 한국어 문체가, 코드를 넣으면 들여쓰기와 문법 구조가 나오기 시작한다. **이 작은 모델이 하는 일과 최신 LLM이 하는 일은 스케일만 다르고 본질이 같다.**

## 반드시 이해할 세 지점

**1. 프리필과 디코드의 분리.** `generate`는 두 단계다 — 프롬프트 전체를 병렬 처리하는 프리필(compute-bound), 토큰 하나씩 뽑는 디코드(memory-bound). KV 캐시가 없으면 매 토큰마다 전체 시퀀스를 재계산해 생성이 $O(L^2)$이 된다. `caches=None`으로 고정해 속도 차이를 직접 측정해 보라. 이 구분이 [vLLM과 서빙 최적화](/handbook/10-llm-engineering/14-inference-serving-and-batching) 전체의 출발점이다.

**2. 가중치 공유(weight tying).** 입력 임베딩과 출력 헤드는 둘 다 "토큰 ↔ 벡터" 대응이므로 행렬을 공유한다. 파라미터가 크게 줄고 성능도 대체로 좋아진다 — GPT-2부터 최신 모델까지 쓰는 표준 기법이다.

**3. 손실 값 읽는 법.** 초기 손실은 $\ln(\text{vocab})$ 근처여야 한다. 학습 후 문자 단위 LM의 손실 1.5는 "다음 문자를 평균적으로 $e^{1.5} \approx 4.5$개 후보로 좁혔다"는 뜻이다(퍼플렉시티). val 손실이 train과 벌어지기 시작하는 지점이 과적합의 시작이다.

## 확장 과제

1. **RoPE로 교체** — 학습형 위치 임베딩을 RoPE로 바꿔라. 학습 길이보다 긴 시퀀스에서 두 방식의 성능 차이를 관찰하면 [긴 컨텍스트 문제](/handbook/10-llm-engineering/05-long-context)가 체감된다.
2. **BPE 연결** — [07의 토크나이저](/practice/07-bpe-tokenizer)로 서브워드 단위 학습으로 전환하라. 같은 컴퓨트에서 문자 단위보다 얼마나 좋은가?
3. **스케일링 실험** — D=128/256/512로 파라미터를 4배씩 키우며 val 손실을 기록하라. 로그-로그 그래프에서 [스케일링 법칙](/handbook/10-llm-engineering/01-scaling-laws)의 직선이 보이는가?
4. **top-p 샘플링 추가** — top-k 대신 누적 확률 기반 nucleus 샘플링을 구현하라.

## 다음

밑바닥 구현에서 사전학습 모델 활용으로 → [10. Hugging Face로 BERT 파인튜닝](/practice/10-bert-finetuning-hf)
