# 16. DQN으로 CartPole 정복

**만드는 것**: DQN(Deep Q-Network)을 밑바닥부터 구현해 CartPole-v1(막대 세우기)을 만점(500점)으로 풀어낸다. 리플레이 버퍼와 타깃 네트워크 — 딥 강화학습을 실제로 돌아가게 만든 두 장치를 코드로 이해한다.

**선행 지식**: [MDP와 Q-learning](/handbook/11-other-domains/05-rl-foundations), [강화학습의 기본 개념](/curriculum/ch08/lecture17)

## 왜 그냥 Q-learning + 신경망은 실패하는가

Q-테이블을 신경망으로 바꾸면 두 가지가 무너진다.

1. **상관된 데이터** — 연속된 스텝은 거의 같은 상태다. i.i.d.를 가정하는 SGD에 상관 샘플을 먹이면 최근 경험에 과적합하며 진동한다. → **리플레이 버퍼**: 경험을 쌓아두고 무작위로 뽑아 상관을 깬다.
2. **움직이는 타깃** — 갱신 목표 $r + \gamma \max Q(s')$ 자체가 학습 중인 네트워크로 계산된다. 내가 움직이면 목표도 움직이는 자기추격이 발산을 부른다. → **타깃 네트워크**: 목표 계산용 복사본을 두고 가끔만 동기화한다.

## 전체 코드

```python
"""DQN — CartPole-v1. 실행: python dqn.py
의존성: pip install torch gymnasium
"""
import random
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

GAMMA = 0.99
LR = 1e-3
BATCH = 64
BUFFER_SIZE = 50_000
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 5_000   # ε-greedy 감쇠 (스텝 기준)
TARGET_SYNC = 500                                   # 타깃 네트워크 동기화 주기
EPISODES = 600


class QNet(nn.Module):
    """상태(4차원) → 각 행동의 Q값(2개). 작은 MLP면 충분하다."""
    def __init__(self, obs_dim=4, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions))
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, size):
        self.buf = deque(maxlen=size)
    def push(self, *transition):                    # (s, a, r, s', done)
        self.buf.append(transition)
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.tensor(s, dtype=torch.float32, device=device),
                torch.tensor(a, dtype=torch.long, device=device),
                torch.tensor(r, dtype=torch.float32, device=device),
                torch.tensor(s2, dtype=torch.float32, device=device),
                torch.tensor(d, dtype=torch.float32, device=device))
    def __len__(self):
        return len(self.buf)


def main():
    env = gym.make("CartPole-v1")
    q, q_target = QNet().to(device), QNet().to(device)
    q_target.load_state_dict(q.state_dict())
    q_target.eval()
    opt = torch.optim.AdamW(q.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    global_step = 0
    recent = deque(maxlen=20)                       # 최근 20 에피소드 평균으로 진행 판단

    for ep in range(EPISODES):
        s, _ = env.reset(seed=ep)
        ep_reward = 0.0
        done = False
        while not done:
            # ε-greedy: 탐색 확률을 스텝에 따라 지수 감쇠
            eps = EPS_END + (EPS_START - EPS_END) * (0.99 ** (global_step / EPS_DECAY * 100))
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = q(torch.tensor(s, dtype=torch.float32, device=device)).argmax().item()

            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            # truncated(시간 초과)는 실패가 아니다 — done으로 부트스트랩을 끊으면 안 된다
            buffer.push(s, a, r, s2, float(terminated))
            s = s2
            ep_reward += r
            global_step += 1

            # ---- 학습 스텝 ----
            if len(buffer) >= 1_000:
                bs, ba, br, bs2, bdone = buffer.sample(BATCH)
                # 현재 추정: Q(s, a) — 실제 취한 행동의 Q만 gather
                q_sa = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # TD 타깃: r + γ·max Q_target(s') — 타깃넷으로 계산 (자기추격 차단)
                    max_q2 = q_target(bs2).max(dim=1).values
                    target = br + GAMMA * max_q2 * (1 - bdone)
                loss = F.smooth_l1_loss(q_sa, target)   # Huber: 이상치 TD 오차에 견고
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()

            if global_step % TARGET_SYNC == 0:
                q_target.load_state_dict(q.state_dict())

        recent.append(ep_reward)
        if (ep + 1) % 20 == 0:
            print(f"ep {ep+1:4d}  최근 20 평균 {sum(recent)/len(recent):6.1f}  ε {eps:.3f}")
        if len(recent) == 20 and sum(recent) / 20 >= 475:
            print(f"해결! (episode {ep+1})")
            break

    torch.save(q.state_dict(), "dqn_cartpole.pt")


if __name__ == "__main__":
    main()
```

보통 300~500 에피소드 안에 평균 475+로 "해결"된다. RL 특성상 시드에 따라 편차가 크다 — 이것 자체가 배울 점이다.

## 반드시 이해할 세 지점

**1. `gather`와 `max`의 비대칭.** 현재 Q는 "실제 취한 행동"의 값(`gather`), 타깃은 "다음 상태의 최선 행동"의 값(`max`)이다. 이 비대칭이 Q-learning의 오프폴리시 성질 그 자체다.

**2. terminated vs truncated.** CartPole은 500스텝에서 시간 초과(truncated)로 끝난다. 이것은 "그 상태의 가치가 0"이라는 뜻이 아니므로, 부트스트랩을 끊는 done 플래그에 truncated를 포함하면 가치 추정이 체계적으로 왜곡된다. Gymnasium이 둘을 분리해 반환하는 이유이고, RL 구현의 단골 버그다.

**3. RL 디버깅은 지도학습과 다르다.** 손실이 내려가도 정책이 나쁠 수 있고, 손실이 요동쳐도 잘 배우는 중일 수 있다. 믿을 지표는 **에피소드 보상의 이동평균**뿐이다. 보상이 오르다 무너지면(catastrophic forgetting) 타깃 동기화 주기·버퍼 크기·lr을 의심한다.

## 확장 과제

1. **Double DQN** — 타깃 계산에서 행동 선택은 온라인 넷, 평가는 타깃 넷으로 분리하라(`q(bs2).argmax` → `q_target(bs2).gather`). 과대추정이 줄어드는 이유를 설명할 수 있는가?
2. **하이퍼파라미터 민감도** — TARGET_SYNC를 1(매 스텝)과 10000으로 바꿔 학습 곡선을 비교하라. 타깃 네트워크의 존재 이유가 그래프로 보인다.
3. **LunarLander 도전** — `gym.make("LunarLander-v3")`(8차원 상태, 4행동)로 같은 코드를 돌려 보라. 무엇을 조정해야 하는가?
4. **Dueling DQN** — Q를 V(상태 가치) + A(행동 이점)로 분해하는 구조로 바꿔 비교하라.

## 다음

가치 기반에서 정책 기반으로 — 현대 RL(그리고 RLHF)의 주력 → [17. PPO 구현](/practice/17-ppo-from-scratch)
