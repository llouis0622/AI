# 06. ViT 밑바닥 구현

**만드는 것**: Vision Transformer를 밑바닥부터 구현해 CIFAR-10을 학습한다. 여기서 짠 Multi-Head Attention은 [08. Transformer](/practice/08-transformer-from-scratch)와 [09. GPT](/practice/09-gpt-from-scratch)에서 그대로 재사용된다 — "어텐션은 하나다"를 몸으로 확인한다.

**핵심 아이디어**: 이미지를 16×16(여기서는 4×4) 패치로 잘라 "단어"처럼 취급하면, 이미지 분류가 시퀀스 분류 문제가 된다. 합성곱의 귀납 편향(지역성, 평행이동 등변성)을 버리는 대신, 데이터가 충분하면 어텐션이 전역 관계를 직접 학습한다.

**선행 지식**: [ViT와 Swin](/handbook/07-computer-vision/10-vision-transformer), [Multi-Head Attention](/handbook/08-sequence-nlp/07-multi-head-attention)

## 전체 코드

```python
"""ViT from scratch — CIFAR-10. 실행: python vit.py"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


class PatchEmbed(nn.Module):
    """이미지 → 패치 시퀀스. stride=patch인 Conv2d 하나가 '자르기+선형투영'을 동시에 한다."""
    def __init__(self, img_size=32, patch=4, d_model=192):
        super().__init__()
        self.n_patches = (img_size // patch) ** 2          # 8×8 = 64개
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)

    def forward(self, x):                                  # (B, 3, 32, 32)
        x = self.proj(x)                                   # (B, D, 8, 8)
        return x.flatten(2).transpose(1, 2)                # (B, 64, D) — 패치 시퀀스


class MultiHeadAttention(nn.Module):
    """표준 MHA. 이 클래스는 08/09 코드랩에서 마스크만 바꿔 재사용된다."""
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):                       # x: (B, L, D)
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)               # 각 (B, H, L, d_head)
        att = (q @ k.transpose(-2, -1)) / self.d_head**0.5  # (B, H, L, L)
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        att = self.drop(att.softmax(dim=-1))
        y = (att @ v).transpose(1, 2).reshape(B, L, D)     # 헤드 합치기
        return self.out(y)


class Block(nn.Module):
    """Pre-LN Transformer 블록: LN → MHA → 잔차, LN → MLP → 잔차."""
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                     # 잔차: 그래디언트의 고속도로
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, n_classes=10, d_model=192, depth=6, n_heads=6):
        super().__init__()
        self.patch = PatchEmbed(d_model=d_model)
        # [CLS] 토큰: 시퀀스 전체를 요약하는 학습 가능한 토큰
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        # 학습형 위치 임베딩: 패치 64개 + CLS 1개
        self.pos = nn.Parameter(torch.zeros(1, self.patch.n_patches + 1, d_model))
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        x = self.patch(x)                                  # (B, 64, D)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos          # (B, 65, D)
        x = self.blocks(x)
        return self.head(self.ln(x[:, 0]))                 # CLS 토큰만 분류에 사용


def main():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_dl = DataLoader(datasets.CIFAR10("./data", True, download=True, transform=train_tf),
                          batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(datasets.CIFAR10("./data", False, download=True, transform=test_tf),
                         batch_size=512, num_workers=4)

    model = ViT().to(device)
    epochs = 30
    # ViT는 AdamW + 웜업이 사실상 필수다. SGD로는 학습이 잘 안 붙는다.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                                                total_steps=epochs * len(train_dl), pct_start=0.1)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval()
        correct = sum((model(x.to(device)).argmax(1).cpu() == y).sum().item()
                      for x, y in test_dl)
        print(f"epoch {epoch+1:2d}  acc {correct/10000:.4f}")


if __name__ == "__main__":
    main()
```

30 에포크에 ~85% 부근. 같은 조건의 ResNet(~93%)보다 낮다 — 이것이 정상이다.

## 반드시 이해할 세 지점

**1. 작은 데이터에서 ViT가 CNN에 지는 이유.** ViT에는 지역성이라는 귀납 편향이 없다. "가까운 픽셀이 관련 있다"조차 데이터에서 배워야 하므로, CIFAR-10 5만 장으로는 부족하다. ImageNet-21k급 사전학습 후 파인튜닝하면 역전된다 — 귀납 편향과 데이터량의 트레이드오프를 보여주는 대표 사례다.

**2. Conv2d 하나로 패치 임베딩이 되는 이유.** kernel=stride=patch인 합성곱은 겹치지 않게 자른 각 패치에 같은 선형변환을 적용하는 것과 정확히 같다. `unfold`로 잘라 `Linear`를 거는 것과 수학적으로 동일하지만 훨씬 빠르다.

**3. Pre-LN.** 원조 Transformer는 잔차 후에 LN(Post-LN)이었지만, 깊은 모델에서는 LN을 먼저 통과시키는 Pre-LN이 그래디언트 흐름이 안정적이라 현대 구현의 표준이다. ([정규화 위치](/handbook/08-sequence-nlp/09-transformer-block-and-masking))

## 확장 과제

1. **어텐션 시각화** — 마지막 블록에서 CLS 토큰의 어텐션 가중치(64패치)를 8×8로 reshape해 원본 이미지 위에 히트맵으로 얹어라. 모델이 어디를 보는가?
2. **CLS vs 평균 풀링** — CLS 토큰 대신 패치 토큰 평균(`x[:, 1:].mean(1)`)으로 분류해 성능을 비교하라. 현대 ViT 다수가 평균 풀링을 쓴다.
3. **패치 크기 실험** — patch=2(256 토큰)와 patch=8(16 토큰)로 바꿔 정확도와 학습 시간을 비교하라. 시퀀스 길이의 제곱 비용을 체감할 수 있다.
4. **`F.scaled_dot_product_attention`으로 교체** — 수동 어텐션을 PyTorch 내장 SDPA로 바꾸고 속도를 비교하라(FlashAttention 커널이 자동 선택된다).

## 다음

시퀀스 모델링의 시작, 텍스트를 숫자로 바꾸는 법부터 → [07. BPE 토크나이저 구현](/practice/07-bpe-tokenizer)
