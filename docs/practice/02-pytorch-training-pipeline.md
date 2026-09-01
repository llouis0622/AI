# 02. PyTorch 학습 파이프라인

**만드는 것**: CIFAR-10을 ResNet으로 분류하는, 실무 수준의 완전한 학습 스크립트. 데이터 증강, 혼합 정밀도(AMP), 학습률 스케줄, 체크포인트, 검증 루프까지 — 어떤 프로젝트에도 재사용할 수 있는 템플릿이다.

**선행 지식**: [표준 학습 루프](/handbook/05-pytorch/05-training-loop-template), [Dataset과 DataLoader](/handbook/05-pytorch/04-dataset-and-dataloader)

## 전체 코드

```python
"""CIFAR-10 학습 파이프라인 — 재사용 가능한 템플릿.

실행: python train_cifar.py
GPU 기준 ~10분(20 에포크), 테스트 정확도 ~92%. CPU도 동작(느림).
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ---------- 설정: 딕셔너리 하나에 모은다. 나중에 argparse/YAML로 빼기 쉽다. ----------
CFG = {
    "epochs": 20,
    "batch_size": 256,
    "lr": 0.1,
    "weight_decay": 5e-4,
    "num_workers": 4,
    "ckpt_path": "./ckpt_best.pt",
    "seed": 42,
}


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_loaders():
    # 증강은 훈련에만. 검증/테스트는 결정적이어야 성능 비교가 가능하다.
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False,
                         num_workers=CFG["num_workers"], pin_memory=True)
    return train_dl, test_dl


def build_model():
    # torchvision ResNet-18을 CIFAR(32×32)용으로 수정:
    # 7×7 stride-2 conv와 maxpool은 224×224용이라 작은 이미지를 너무 일찍 뭉갠다.
    m = models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def evaluate(model, loader, device):
    model.eval()                      # BatchNorm/Dropout을 추론 모드로
    correct = total = 0
    with torch.no_grad():             # 그래프 생성 중단 — eval()과는 별개 기능
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    set_seed(CFG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, test_dl = build_loaders()
    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=CFG["lr"],
                                momentum=0.9, weight_decay=CFG["weight_decay"],
                                nesterov=True)
    # 웜업 없이 코사인 감쇠. 총 스텝 수 기준으로 스케줄한다.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG["lr"],
        total_steps=CFG["epochs"] * len(train_dl),
        pct_start=0.1,               # 앞 10%는 웜업
    )
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_acc = 0.0
    for epoch in range(CFG["epochs"]):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # AMP: 순전파를 fp16/bf16으로 — 메모리 절반, 속도 1.5~3배
            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                       # 클리핑 전 언스케일 필수
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()                                 # OneCycle은 스텝 단위
            running += loss.item()

        acc = evaluate(model, test_dl, device)
        lr_now = scheduler.get_last_lr()[0]
        print(f"epoch {epoch+1:2d}  loss {running/len(train_dl):.4f}  "
              f"acc {acc:.4f}  lr {lr_now:.4f}")

        if acc > best_acc:                                   # 최고 성능만 저장
            best_acc = acc
            torch.save({"model": model.state_dict(),
                        "acc": acc, "epoch": epoch, "cfg": CFG}, CFG["ckpt_path"])

    print(f"best test acc: {best_acc:.4f}  (saved to {CFG['ckpt_path']})")


if __name__ == "__main__":
    main()
```

## 파이프라인의 결정들 — 왜 이렇게 짰는가

**증강은 훈련에만.** 검증에 무작위 증강이 섞이면 에포크마다 다른 데이터를 평가하는 셈이라 성능 비교가 무의미해진다. 흔한 실수다.

**`zero_grad(set_to_none=True)`.** 0으로 채우는 대신 None으로 만들어 메모리 쓰기를 아낀다. 현재 PyTorch의 기본값이기도 하다.

**AMP와 GradScaler.** fp16 순전파는 그래디언트가 언더플로할 수 있어, 손실을 키워서(backward) 그래디언트를 계산한 뒤 되돌리는(step 전) 것이 GradScaler다. bf16을 쓸 수 있는 GPU(Ampere+)라면 `autocast(dtype=torch.bfloat16)`에 스케일러 없이도 안정적이다.

**클리핑 전 `unscale_`.** 스케일된 그래디언트에 클리핑하면 임계값의 의미가 없어진다. 순서가 중요하다: `scale → backward → unscale → clip → step`.

**체크포인트에 설정을 함께 저장.** `state_dict`만 저장하면 3주 뒤에 "이 모델이 어떤 하이퍼파라미터였지?"를 알 수 없다. 재현성의 최소 단위는 (가중치 + 설정 + 성능)이다. ([실험 추적](/handbook/12-mlops/01-experiment-tracking))

**성능이 좋을 때만 저장.** 마지막 에포크가 최고 성능이라는 보장이 없다 — 사실상의 조기 종료다.

## 학습이 잘 되는지 확인하는 법

첫 실행에서 볼 것 세 가지:

1. **첫 배치 손실** — 10 클래스 분류의 초기 손실은 $-\log(1/10) \approx 2.30$이어야 한다. 크게 다르면 초기화나 손실 정의가 잘못됐다.
2. **작은 데이터 과적합 테스트** — 배치 하나(256장)만으로 학습해 손실이 ~0으로 가는지 확인한다. 안 가면 모델/루프에 버그가 있다.
3. **학습률 곡선** — 손실이 처음부터 발산하면 lr 10배 감소, 너무 느리면 10배 증가부터 시도한다. ([진단 순서](/handbook/04-deep-learning/10-training-failure-diagnosis))

## 확장 과제

1. **TensorBoard 연동** — `torch.utils.tensorboard.SummaryWriter`로 손실·정확도·학습률을 기록하고 곡선을 관찰하라.
2. **재개(resume) 기능** — 체크포인트에서 optimizer/scheduler 상태까지 복원해 중단 지점부터 이어 학습하도록 확장하라(무엇을 추가로 저장해야 하는가?).
3. **MixUp 구현** — 두 이미지와 레이블을 섞는 MixUp 증강을 추가하고 정확도 변화를 측정하라.
4. **`torch.compile`** — `model = torch.compile(model)` 한 줄을 추가하고 처리량 변화를 측정하라. ([원리](/handbook/05-pytorch/07-torch-compile))

## 다음

밑바닥부터 학습하는 대신, 사전학습된 지식을 빌려온다 → [03. 전이학습과 파인튜닝](/practice/03-transfer-learning)
