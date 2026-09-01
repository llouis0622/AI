# 07. BPE 토크나이저 구현

**만드는 것**: GPT 계열이 쓰는 Byte Pair Encoding 토크나이저를 밑바닥부터 — 학습(merge 규칙 추출), 인코딩, 디코딩까지 구현하고, Hugging Face `tokenizers`로 실전 학습법을 잇는다.

**왜 중요한가**: LLM의 입력은 텍스트가 아니라 토큰이다. 컨텍스트 길이도, API 비용도, "strawberry에 r이 몇 개냐"를 틀리는 이유도 모두 토크나이저에서 시작된다.

**선행 지식**: [토큰화 이론](/handbook/08-sequence-nlp/01-tokenization)

## BPE의 아이디어

문자(또는 바이트) 단위에서 시작해, **가장 자주 등장하는 인접 쌍을 병합**하는 규칙을 원하는 어휘 크기가 될 때까지 반복 학습한다. 자주 쓰는 단어는 통째로 한 토큰이 되고, 처음 보는 단어도 서브워드 조각으로 항상 표현 가능하다 — OOV(어휘 밖 단어) 문제가 사라진다.

## 전체 코드

```python
"""BPE 토크나이저 밑바닥 구현. 실행: python bpe.py"""
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}   # (a, b) → 새 토큰 id
        self.vocab: dict[int, bytes] = {}              # id → 바이트열

    # ---------- 학습 ----------
    def train(self, text: str, vocab_size: int):
        assert vocab_size >= 256
        # 바이트 수준에서 시작: 유니코드 무엇이 와도 0~255로 표현된다.
        ids = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}

        for new_id in range(256, vocab_size):
            pairs = Counter(zip(ids, ids[1:]))         # 인접 쌍 빈도
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]          # 최빈 쌍
            ids = self._merge(ids, best, new_id)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id); i += 2
            else:
                out.append(ids[i]); i += 1
        return out

    # ---------- 인코딩: 학습된 merge를 '학습 순서대로' 적용 ----------
    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            # 적용 가능한 merge 중 가장 먼저 학습된 것(id가 작은 것)부터
            candidates = [p for p in pairs if p in self.merges]
            if not candidates:
                break
            best = min(candidates, key=lambda p: self.merges[p])
            ids = self._merge(ids, best, self.merges[best])
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        # 잘못된 토큰 경계로 바이트가 깨질 수 있으니 replace로 안전하게
        return data.decode("utf-8", errors="replace")


if __name__ == "__main__":
    corpus = open("corpus.txt", encoding="utf-8").read()  # 아무 텍스트나 수 MB
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=1000)

    s = "인공지능은 데이터로부터 학습한다. Machine learning!"
    ids = tok.encode(s)
    print("토큰 수:", len(ids), "vs 바이트 수:", len(s.encode("utf-8")))
    print("복원 일치:", tok.decode(ids) == s)
    # 어떤 조각으로 잘렸는지 보기
    print([tok.vocab[i].decode("utf-8", errors="replace") for i in ids])
```

핵심 성질 세 가지를 코드에서 직접 확인하라.

1. **무손실**: `decode(encode(s)) == s`가 항상 성립한다(바이트 수준이므로).
2. **압축**: 학습 코퍼스와 비슷한 텍스트일수록 토큰 수가 준다. 코퍼스에 없던 언어는 거의 바이트 단위로 쪼개진다 — 한국어 성능이 토크나이저 학습 데이터에 좌우되는 이유다.
3. **인코딩 순서**: merge는 학습된 순서대로 적용해야 한다. 순서를 무시하면 학습 때와 다른 분할이 나온다.

## 실전에서는: Hugging Face tokenizers

프로덕션 토크나이저는 Rust 구현인 `tokenizers`로 학습한다. 위 원리 그대로, 정규화·사전분할·특수 토큰이 더해진 것뿐이다.

```python
# pip install tokenizers
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tokenizer = Tokenizer(models.BPE(unk_token=None))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<|endoftext|>", "<|pad|>"],
)
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("my_tokenizer.json")

enc = tokenizer.encode("안녕하세요, BPE 토크나이저입니다.")
print(enc.tokens, enc.ids)
```

기존 모델의 토크나이저를 쓸 때는 반드시 그 모델의 것을 그대로 로드한다 — 토크나이저와 임베딩 행렬은 한 몸이다.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.encode("hello world"), tok.tokenize("hello world"))
```

## 토크나이저가 만드는 실무 이슈들

- **비용과 컨텍스트**: API 과금과 컨텍스트 한도는 토큰 단위다. 같은 의미라도 언어에 따라 토큰 수가 2~3배 차이 난다.
- **숫자와 철자**: "12345"가 한 토큰인지 다섯 토큰인지에 따라 산술 능력이 달라진다. 철자 질문에 약한 것도 문자가 아닌 서브워드를 보기 때문이다.
- **어휘 크기 트레이드오프**: 크면 시퀀스가 짧아지지만 임베딩 행렬이 커지고 희귀 토큰 학습이 어렵다. 현대 LLM은 3만~25만 사이를 쓴다.

## 확장 과제

1. **속도 개선** — 위 구현은 학습이 $O(\text{vocab} \times \text{corpus})$다. 쌍 빈도를 증분 갱신하는 방식으로 개선하라(우선순위 큐 힌트).
2. **어휘 크기 실험** — 500/2000/8000으로 학습해 같은 문장의 토큰 수와, 조각들의 "의미 단위다움"을 비교하라.
3. **GPT-2 재현** — tiktoken(`pip install tiktoken`)으로 GPT-2 인코딩을 로드해, 같은 문장을 내 BPE와 비교하라. 사전분할 정규식이 왜 필요한지 관찰된다.

## 다음

토큰 시퀀스를 처리하는 아키텍처의 완성형 → [08. Transformer 밑바닥 구현](/practice/08-transformer-from-scratch)
