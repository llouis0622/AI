# 15. Diffusion(DDPM) 밑바닥 구현

**만드는 것**: DDPM을 밑바닥부터 — 노이즈 스케줄, 학습(노이즈 예측), 샘플링(점진적 복원)까지 구현해 MNIST 숫자를 생성한다. Stable Diffusion·이미지 생성 서비스의 뿌리가 되는 알고리즘이다.

**선행 지식**: [확산 순방향 과정](/handbook/09-generative-models/06-diffusion-forward-process), [DDPM 손실 유도](/handbook/09-generative-models/07-ddpm-objective)

## 알고리즘 요약

**순방향(고정)**: 데이터 $x_0$에 $T$단계에 걸쳐 가우시안 노이즈를 더한다. 닫힌 형태로 임의 시점을 바로 얻는다:

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

**학습**: 네트워크 $\epsilon_\theta(x_t, t)$가 "더해진 노이즈"를 맞추는 회귀 문제다. 손실은 그냥 MSE:

$$
\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

**샘플링(역방향)**: 순수 노이즈 $x_T$에서 시작해, 매 단계 예측 노이즈를 빼며 $x_0$까지 되돌아간다.

## 전체 코드

```python
"""DDPM — MNIST. 실행: python ddpm.py (GPU 권장, ~20분)"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"
T = 1000    # 확산 단계 수


# ---------- 노이즈 스케줄: 미리 전부 계산해 버퍼로 ----------
class Schedule:
    def __init__(self):
        self.beta = torch.linspace(1e-4, 0.02, T, device=device)     # 선형 스케줄
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)            # ᾱ_t

    def add_noise(self, x0, t, eps):
        """순방향 닫힌 형태: x_t = √ᾱ·x0 + √(1-ᾱ)·ε"""
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps


# ---------- 시간 임베딩: 스텝 t를 사인 임베딩으로 ----------
def time_embed(t, dim=128):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------- 간단한 U-Net (노이즈 예측기) ----------
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, c_out)         # 시간 정보를 채널에 주입
        self.norm1 = nn.GroupNorm(8, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]  # 어느 시점의 노이즈인지 알려준다
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class MiniUNet(nn.Module):
    """28×28 → 14×14 → 7×7 → 14×14 → 28×28, 스킵 연결 포함."""
    def __init__(self, ch=64):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.inc = ResBlock(1, ch)
        self.down1 = ResBlock(ch, ch * 2)
        self.down2 = ResBlock(ch * 2, ch * 4)
        self.mid = ResBlock(ch * 4, ch * 4)
        self.up2 = ResBlock(ch * 4 + ch * 4, ch * 2)
        self.up1 = ResBlock(ch * 2 + ch * 2, ch)
        self.out = nn.Conv2d(ch + ch, 1, 3, padding=1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, t):
        temb = self.t_mlp(time_embed(t))
        h1 = self.inc(x, temb)                                   # (B, 64, 28, 28)
        h2 = self.down1(self.pool(h1), temb)                     # (B, 128, 14, 14)
        h3 = self.down2(self.pool(h2), temb)                     # (B, 256, 7, 7)
        m = self.mid(h3, temb)
        u2 = self.up2(torch.cat([m, h3], 1), temb)
        u2 = F.interpolate(u2, scale_factor=2)                   # (B, 128, 14, 14)
        u1 = self.up1(torch.cat([u2, h2], 1), temb)
        u1 = F.interpolate(u1, scale_factor=2)                   # (B, 64, 28, 28)
        return self.out(torch.cat([u1, h1], 1))                  # 예측 노이즈


# ---------- 샘플링: T → 0으로 되돌아간다 ----------
@torch.no_grad()
def sample(model, sched, n=64):
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)                 # x_T: 순수 노이즈
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, tt)
        a, ab, b = sched.alpha[t], sched.alpha_bar[t], sched.beta[t]
        # 평균: 예측 노이즈를 빼서 한 단계 덜 노이즈한 상태로
        x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
        if t > 0:                                                # 마지막 단계 빼고 노이즈 재주입
            x = x + b.sqrt() * torch.randn_like(x)
    return (x.clamp(-1, 1) + 1) / 2


def main():
    torch.manual_seed(0)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    dl = DataLoader(datasets.MNIST("./data", True, download=True, transform=tf),
                    batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    model = MiniUNet().to(device)
    sched = Schedule()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(30):
        for x0, _ in dl:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.shape[0],), device=device)   # 시점 무작위
            eps = torch.randn_like(x0)
            xt = sched.add_noise(x0, t, eps)
            loss = F.mse_loss(model(xt, t), eps)                     # 노이즈 맞추기 — 이게 전부
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        print(f"epoch {epoch+1:2d}  loss {loss.item():.4f}")
        if (epoch + 1) % 5 == 0:
            save_image(sample(model, sched), f"ddpm_epoch{epoch+1:02d}.png", nrow=8)
            model.train()


if __name__ == "__main__":
    main()
```

에포크 10쯤부터 숫자 형태가 보이고, 30 에포크면 또렷한 샘플이 나온다.

## 반드시 이해할 세 지점

**1. 학습이 GAN보다 압도적으로 안정적인 이유.** 손실이 단순 MSE 회귀다 — 두 네트워크의 균형도, 모드 붕괴도 없다. [14의 GAN](/practice/14-vae-gan)을 튜닝해 본 직후라면 이 안정성이 왜 확산 모델이 주류가 됐는지를 설명해 준다. 대가는 샘플링 비용: 생성 한 번에 순전파 $T=1000$번.

**2. 시간 임베딩의 역할.** 하나의 네트워크가 "거의 깨끗한 $x_{10}$"과 "거의 노이즈인 $x_{990}$"을 모두 처리한다. $t$를 모르면 어느 강도의 노이즈를 빼야 할지 알 수 없다 — `t_proj` 주입을 지우면 학습이 망가지는 것으로 확인할 수 있다.

**3. 샘플링 시 노이즈 재주입.** 역방향 각 단계는 결정적 복원이 아니라 분포에서의 샘플링이다. $t>0$에서 노이즈를 다시 더하는 항을 지우면 샘플이 흐릿하고 다양성이 죽는다 — 직접 지워서 비교해 보라.

## 확장 과제

1. **DDIM 샘플러** — 50스텝만으로 샘플링하는 DDIM을 구현해 1000스텝 DDPM과 품질·속도를 비교하라. ([DDIM 원리](/handbook/09-generative-models/08-ddim-and-fast-sampling))
2. **코사인 스케줄** — 선형 β 대신 코사인 스케줄로 바꿔 저해상도에서의 품질 차이를 관찰하라.
3. **클래스 조건부 + CFG** — 레이블 임베딩을 시간 임베딩에 더해 조건부로 학습하고(10%는 무조건부로), 샘플링에서 classifier-free guidance를 구현하라. guidance 배율에 따라 품질·다양성이 어떻게 변하는가? ([CFG](/handbook/09-generative-models/09-latent-diffusion-and-guidance))
4. **x0 예측으로 변경** — ε 대신 x0를 예측하도록 파라미터화를 바꿔 보라. 어느 쪽이 잘 되고, 왜 실무에서는 ε(또는 v) 예측을 쓰는가?

## 다음

지도 신호 없이 보상으로 배우는 패러다임 → [16. DQN으로 CartPole 정복](/practice/16-dqn-cartpole)
