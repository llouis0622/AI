# 04. U-Net 세그멘테이션

**만드는 것**: U-Net을 밑바닥부터 구현해 Oxford-IIIT Pet 데이터셋에서 "픽셀마다" 반려동물/배경/경계를 분할한다.

**핵심 개념**: 분류는 이미지 → 레이블 하나지만, 세그멘테이션은 이미지 → 같은 크기의 레이블 맵이다. 공간 정보를 잃지 않으면서 넓은 문맥을 보는 것이 과제이고, U-Net의 인코더-디코더 + 스킵 연결이 그 답이다.

**선행 지식**: [세그멘테이션 이론](/handbook/07-computer-vision/09-segmentation), [합성곱 연산](/handbook/07-computer-vision/01-convolution-arithmetic)

## 전체 코드

```python
"""U-Net 세그멘테이션 — Oxford-IIIT Pet (3 클래스: 전경/배경/경계).

실행: python unet_pets.py
의존성: torch torchvision
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128


# ---------- 모델 ----------
class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2 — U-Net의 기본 블록."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, n_classes=3, base=32):
        super().__init__()
        chs = [base, base*2, base*4, base*8]          # 32, 64, 128, 256
        # 인코더: 해상도 절반 ↓, 채널 2배 ↑ — "무엇이 있는가"를 압축
        self.enc1 = DoubleConv(3, chs[0])
        self.enc2 = DoubleConv(chs[0], chs[1])
        self.enc3 = DoubleConv(chs[1], chs[2])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[2], chs[3])
        # 디코더: 업샘플 후, 같은 해상도의 인코더 특징을 이어붙인다(스킵 연결)
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = DoubleConv(chs[3], chs[2])        # 입력 = up(256→128) + skip(128)
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = DoubleConv(chs[2], chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = DoubleConv(chs[1], chs[0])
        self.head = nn.Conv2d(chs[0], n_classes, 1)   # 1×1 conv: 픽셀별 분류기

    def forward(self, x):
        s1 = self.enc1(x)                             # (B, 32, 128, 128)
        s2 = self.enc2(self.pool(s1))                 # (B, 64, 64, 64)
        s3 = self.enc3(self.pool(s2))                 # (B, 128, 32, 32)
        b = self.bottleneck(self.pool(s3))            # (B, 256, 16, 16)
        d3 = self.dec3(torch.cat([self.up3(b), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.head(d1)                          # (B, 3, 128, 128) — 픽셀별 로짓


# ---------- 손실: CE + Dice ----------
def dice_loss(logits, target, eps=1e-6):
    """Dice = 2|A∩B| / (|A|+|B|). 클래스 불균형(작은 전경)에 강하다."""
    probs = logits.softmax(dim=1)
    target_1h = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * target_1h).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_1h.sum(dim=(2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


# ---------- 데이터 ----------
def get_loaders(batch_size=32):
    def tf(img, mask):
        img = v2.functional.resize(img, [IMG_SIZE, IMG_SIZE])
        # 마스크는 최근접 보간! 쌍선형 보간은 레이블 값을 섞어버린다.
        mask = v2.functional.resize(mask, [IMG_SIZE, IMG_SIZE],
                                    interpolation=v2.InterpolationMode.NEAREST)
        img = v2.functional.to_dtype(v2.functional.to_image(img), torch.float32, scale=True)
        mask = torch.as_tensor(__import__("numpy").array(mask), dtype=torch.long) - 1  # 1..3 → 0..2
        return img, mask

    def make(split):
        return datasets.OxfordIIITPet("./data", split=split, target_types="segmentation",
                                      download=True, transforms=tf)
    return (DataLoader(make("trainval"), batch_size=batch_size, shuffle=True, num_workers=4),
            DataLoader(make("test"), batch_size=batch_size, shuffle=False, num_workers=4))


@torch.no_grad()
def mean_iou(model, loader, n_classes=3):
    model.eval()
    inter = torch.zeros(n_classes); union = torch.zeros(n_classes)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for c in range(n_classes):
            inter[c] += ((pred == c) & (y == c)).sum()
            union[c] += ((pred == c) | (y == c)).sum()
    return (inter / union.clamp(min=1)).mean().item()


def main():
    train_dl, test_dl = get_loaders()
    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(15):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
            loss.backward()
            opt.step()
        print(f"epoch {epoch+1:2d}  mIoU {mean_iou(model, test_dl):.4f}")

if __name__ == "__main__":
    main()
```

15 에포크에 mIoU ~0.80 부근이 나온다.

## 반드시 이해할 세 지점

**1. 스킵 연결이 없으면 어떻게 되는가.** 인코더가 해상도를 16×16까지 줄이며 "무엇"은 남기지만 "어디"의 정밀도를 잃는다. 디코더가 업샘플만으로 복원하면 경계가 뭉개진다. 같은 해상도의 인코더 특징을 이어붙여(concat) 위치 정보를 되살리는 것이 U-Net의 본질이다. `torch.cat` 세 줄을 지우고 mIoU가 얼마나 떨어지는지 직접 확인해 보라.

**2. 마스크 리사이즈는 최근접 보간.** 레이블 1(전경)과 2(배경)를 쌍선형 보간하면 1.5라는 무의미한 값이 생긴다. 세그멘테이션 데이터 파이프라인의 단골 버그다.

**3. Dice 손실을 더하는 이유.** 픽셀별 CE는 넓은 배경이 지배해, 작은 전경을 다 틀려도 손실이 낮을 수 있다. Dice는 클래스별 겹침 비율이라 불균형에 강하다. 실무에서는 CE+Dice 합이 기본기다.

## 확장 과제

1. **사전학습 인코더** — 인코더를 ResNet-18의 스테이지들로 교체하고(스킵 연결 지점을 맞춰야 한다) 수렴 속도를 비교하라. `segmentation_models_pytorch` 라이브러리가 실무 표준이니 결과를 대조해 보라.
2. **증강 추가** — 이미지와 마스크에 **같은** 무작위 변환(좌우 반전, 회전)을 적용하는 증강을 구현하라. 왜 같은 시드/파라미터여야 하는가?
3. **경계 클래스 가중치** — 경계(3번 클래스)는 픽셀 수가 적다. `F.cross_entropy(weight=...)`로 가중치를 주고 경계 IoU 변화를 관찰하라.

## 다음

픽셀 분류에서 박스 예측으로 → [05. 객체 탐지](/practice/05-object-detection)
