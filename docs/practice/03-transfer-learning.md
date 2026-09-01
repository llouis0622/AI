# 03. 전이학습과 파인튜닝

**만드는 것**: ImageNet 사전학습 모델을 내 데이터셋(꽃 102종)에 파인튜닝한다. 선형 프로브 → 부분 해동 → 전체 파인튜닝의 3단계 전략을 코드로 비교한다.

**왜 중요한가**: 실무 CV 문제의 90%는 밑바닥 학습이 아니라 파인튜닝이다. 데이터 수천 장으로도 사전학습 백본 위에서는 높은 정확도가 나온다. 언제 어디까지 해동(unfreeze)할지가 실력이다.

**선행 지식**: [전이학습과 파인튜닝 전략](/handbook/07-computer-vision/06-transfer-learning)

## 전체 코드

```python
"""Flowers-102 파인튜닝 — 3가지 전략 비교.

실행: python finetune.py --mode linear|partial|full
의존성: torch torchvision
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_loaders(img_size=224, batch_size=64):
    # 사전학습 모델을 쓸 때는 반드시 그 모델의 전처리(입력 크기·정규화 상수)를 따른다.
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    norm = weights.transforms()  # 공식 전처리: resize 232 → crop 224 → ImageNet 정규화
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm.mean, norm.std),
    ])
    train_ds = datasets.Flowers102("./data", split="train", download=True, transform=train_tf)
    val_ds = datasets.Flowers102("./data", split="val", download=True, transform=norm)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True))


def build_model(mode: str, num_classes=102):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # 분류 헤드 교체: 1000 클래스 → 102 클래스. 새 층은 무작위 초기화된다.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if mode == "linear":
        # 선형 프로브: 백본 전체 동결, 헤드만 학습.
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc")
    elif mode == "partial":
        # 부분 해동: 마지막 스테이지(layer4)와 헤드만 학습.
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(("layer4", "fc"))
    elif mode == "full":
        pass  # 전체 학습
    return model.to(device)


def build_optimizer(model, mode: str):
    if mode == "full":
        # 차등 학습률: 사전학습층은 이미 좋은 지점에 있으므로 작게,
        # 새로 초기화된 헤드는 크게 움직인다.
        head_params = [p for n, p in model.named_parameters() if n.startswith("fc")]
        body_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
        return torch.optim.AdamW([
            {"params": body_params, "lr": 1e-4},
            {"params": head_params, "lr": 1e-3},
        ], weight_decay=1e-4)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["linear", "partial", "full"], default="partial")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    train_dl, val_dl = build_loaders()
    model = build_model(args.mode)
    optimizer = build_optimizer(model, args.mode)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"mode={args.mode}  학습 파라미터: {n_train/1e6:.1f}M")

    for epoch in range(args.epochs):
        model.train()
        # 주의: 동결층이 있어도 model.train()은 그 층의 BatchNorm 통계를 갱신한다.
        # 선형 프로브에서 백본 통계까지 완전히 고정하려면 백본을 eval로 둔다.
        if args.mode == "linear":
            model.eval()
            model.fc.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"epoch {epoch+1:2d}  val acc {evaluate(model, val_dl):.4f}")


if __name__ == "__main__":
    main()
```

10 에포크 기준 대략적 결과(GPU): 선형 프로브 ~88%, 부분 해동 ~93%, 전체 파인튜닝(차등 lr) ~95%. 훈련 데이터가 클래스당 10장뿐인데도 이 정도가 나오는 것이 전이학습의 힘이다.

## 전략 선택 기준

| 상황 | 전략 | 이유 |
| --- | --- | --- |
| 데이터가 아주 적다 (클래스당 <20장) | 선형 프로브 | 백본을 건드리면 과적합. 특징 추출기로만 쓴다 |
| 데이터가 적당하다 | 부분 해동 | 상위층(과제 특화 특징)만 적응시킨다 |
| 데이터가 많다 (수만 장+) | 전체 파인튜닝 | 낮은 lr + 차등 lr로 전체를 조정 |
| 도메인이 크게 다르다 (의료영상, 위성) | 전체 파인튜닝 | 저수준 특징부터 재조정이 필요할 수 있다 |

세 가지 함정:

1. **전처리 불일치** — 사전학습 시 정규화 상수와 다르게 넣으면 성능이 조용히 몇 %p 깎인다. `weights.transforms()`로 공식 전처리를 가져오는 습관을 들인다.
2. **동결층의 BatchNorm** — `requires_grad=False`는 가중치만 동결한다. BN의 이동평균 통계는 `train()` 모드에서 계속 갱신된다. 코드의 주석 참고.
3. **새 헤드에 큰 lr, 백본에 작은 lr** — 무작위 초기화된 헤드의 큰 그래디언트가 백본까지 흔드는 것을 막는다. 헤드만 몇 에포크 먼저 학습한 뒤 해동하는 워밍업도 흔한 레시피다.

## 확장 과제

1. **점진적 해동** — 3에포크마다 layer4 → layer3 → layer2를 순차 해동하는 스케줄을 구현하고 full과 비교하라.
2. **백본 교체** — `models.convnext_tiny`, `models.vit_b_16`으로 바꿔 성능·속도를 비교하라. ViT는 어떤 lr이 필요한가?
3. **특징 캐싱** — 선형 프로브에서 백본 출력(2048차원)을 미리 계산해 디스크에 저장하면 학습이 왜, 얼마나 빨라지는지 측정하라.

## 다음

분류를 넘어 픽셀 단위 예측으로 → [04. U-Net 세그멘테이션](/practice/04-unet-segmentation)
