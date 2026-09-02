# 14. VAE와 GAN 구현

**만드는 것**: MNIST에서 VAE와 DCGAN을 각각 구현해 숫자 이미지를 생성한다. 생성 모델의 두 축 — "가능도 하한을 최대화"(VAE)와 "판별자를 속이기"(GAN) — 를 코드 수준에서 대비한다.

**선행 지식**: [VAE: ELBO와 재파라미터화](/book/19-generative-models), [GAN: 미니맥스](/book/19-generative-models), [0장]()

## 1. VAE

```python
"""VAE — MNIST. 실행: python vae.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"
Z = 16   # 잠재 차원


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(784, 400), nn.ReLU(),
            nn.Linear(400, 200), nn.ReLU())
        self.fc_mu = nn.Linear(200, Z)        # 인코더는 분포(μ, logσ²)를 출력한다
        self.fc_logvar = nn.Linear(200, Z)
        self.dec = nn.Sequential(
            nn.Linear(Z, 200), nn.ReLU(),
            nn.Linear(200, 400), nn.ReLU(),
            nn.Linear(400, 784))               # 로짓 출력 (BCEWithLogits 사용)

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        # 재파라미터화 트릭: z = μ + σ·ε.
        # '샘플링'을 미분 가능한 연산 + 외부 노이즈로 바꾼다 — 이게 없으면 역전파 불가.
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.dec(z), mu, logvar


def vae_loss(recon_logits, x, mu, logvar):
    # 재구성 항: 픽셀별 베르누이 로그가능도
    recon = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum")
    # KL 항: q(z|x)=N(μ,σ²) 와 p(z)=N(0,1) 사이 — 가우시안이라 닫힌 형태
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon + kl) / x.shape[0]


def main():
    dl = DataLoader(datasets.MNIST("./data", True, download=True,
                                   transform=transforms.ToTensor()),
                    batch_size=256, shuffle=True, num_workers=2)
    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        total = 0.0
        for x, _ in dl:
            x = x.view(-1, 784).to(device)
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += loss.item() * x.shape[0]
        print(f"epoch {epoch+1:2d}  loss {total/len(dl.dataset):.2f}")

    # 생성: 사전분포 N(0,1)에서 z를 뽑아 디코딩 — 인코더는 이제 필요 없다
    with torch.no_grad():
        z = torch.randn(64, Z, device=device)
        samples = torch.sigmoid(model.dec(z)).view(-1, 1, 28, 28)
        save_image(samples, "vae_samples.png", nrow=8)


if __name__ == "__main__":
    main()
```

## 2. DCGAN

```python
"""DCGAN — MNIST. 실행: python gan.py"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"
Z = 64


class Generator(nn.Module):
    """노이즈 (B, Z, 1, 1) → 이미지 (B, 1, 28, 28). 전치 합성곱으로 업샘플."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(Z, 256, 7, 1, 0, bias=False),   # → (256, 7, 7)
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # → (128, 14, 14)
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),               # → (1, 28, 28)
            nn.Tanh())                                          # 출력 [-1, 1]
    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1), nn.LeakyReLU(0.2, True),   # BN 없이 시작
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 7, 1, 0))                             # 진짜/가짜 로짓
    def forward(self, x):
        return self.net(x).view(-1)


def main():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(0.5, 0.5)])       # [-1,1]로 — Tanh와 짝
    dl = DataLoader(datasets.MNIST("./data", True, download=True, transform=tf),
                    batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    G, D = Generator().to(device), Discriminator().to(device)
    # DCGAN 표준 설정: Adam(2e-4, β1=0.5). β1=0.9는 진동을 키운다.
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed_z = torch.randn(64, Z, 1, 1, device=device)   # 진행 관찰용 고정 노이즈

    for epoch in range(25):
        for real, _ in dl:
            real = real.to(device)
            B = real.shape[0]
            ones, zeros = torch.ones(B, device=device), torch.zeros(B, device=device)

            # --- D 스텝: 진짜→1, 가짜→0 ---
            z = torch.randn(B, Z, 1, 1, device=device)
            fake = G(z).detach()                 # detach: D 스텝에서 G로 그래디언트 차단
            loss_d = bce(D(real), ones) + bce(D(fake), zeros)
            opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()

            # --- G 스텝: D가 가짜를 1로 믿게 (non-saturating loss) ---
            z = torch.randn(B, Z, 1, 1, device=device)
            loss_g = bce(D(G(z)), ones)
            opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()

        print(f"epoch {epoch+1:2d}  D {loss_d.item():.3f}  G {loss_g.item():.3f}")
        with torch.no_grad():
            save_image(G(fixed_z) * 0.5 + 0.5, f"gan_epoch{epoch+1:02d}.png", nrow=8)


if __name__ == "__main__":
    main()
```

## 두 모델의 대비 — 코드에서 보이는 것

| | VAE | GAN |
| --- | --- | --- |
| 학습 신호 | 명시적 손실(재구성 + KL) — 안정적 단조 감소 | 두 네트워크의 균형 — 손실 값이 품질을 말해주지 않음 |
| 생성 품질 | 흐릿함(픽셀 평균으로 수렴하는 경향) | 선명하지만 모드 붕괴 위험 |
| 잠재 공간 | 사전분포로 정규화됨 — 보간·산술이 자연스러움 | 구조가 보장되지 않음 |
| 디버깅 | 손실 곡선을 보면 됨 | 샘플을 계속 눈으로 봐야 함 (fixed_z의 존재 이유) |

GAN 코드의 세 가지 급소:

1. **`detach()`** — D 스텝에서 빼먹으면 D의 손실이 G까지 갱신해 학습이 꼬인다.
2. **non-saturating G 손실** — 이론의 $\min \log(1-D(G(z)))$ 대신 $\max \log D(G(z))$를 쓴다. 초반에 D가 쉽게 이길 때 그래디언트가 살아있게 하는 표준 트릭이다.
3. **D와 G의 균형** — D 손실이 0 근처로 붕괴하면 G가 배울 신호가 없다. lr·구조로 균형을 맞추는 것이 GAN 튜닝의 본질이고, 이 고통이 [WGAN](/book/19-generative-models)과 확산 모델로 이어지는 동기다.

## 확장 과제

1. **잠재 공간 보간** — VAE에서 두 숫자의 z를 구해 선형 보간하며 디코딩하라. 3이 8로 "변형"되는 과정이 보인다. GAN의 z 보간과 비교하라.
2. **β-VAE** — KL 항에 β=4를 곱해 학습하라. 재구성 품질과 잠재 차원의 해석 가능성이 어떻게 변하는가?
3. **조건부 생성(cGAN)** — 레이블을 G와 D 양쪽에 임베딩으로 주입해 "원하는 숫자"를 생성하게 하라.
4. **모드 붕괴 유발** — G의 lr을 10배 키워 모드 붕괴(같은 숫자만 생성)를 직접 관찰하라.

## 다음

현대 생성 모델의 주류 — 노이즈에서 데이터로 → [15. Diffusion(DDPM) 밑바닥 구현](/practice/15-ddpm-from-scratch)
