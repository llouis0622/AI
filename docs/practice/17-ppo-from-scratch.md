# 17. PPO 구현

**만드는 것**: PPO(Proximal Policy Optimization)를 액터-크리틱 + GAE + 클리핑까지 밑바닥부터 구현해 CartPole을 푼다. PPO는 로봇 제어부터 RLHF까지 쓰이는 현대 RL의 사실상 표준 알고리즘이다 — 이 코드를 이해하면 [RLHF](/book/22-alignment)의 학습 루프가 그대로 읽힌다.

**선행 지식**: [정책 경사와 PPO](/book/26-policy-gradient), [강화학습의 주요 방법론](/book/25-reinforcement-learning)

## DQN과 무엇이 다른가

DQN은 가치를 배우고 정책은 argmax로 유도한다. PPO는 **정책 자체** $\pi_\theta(a|s)$를 확률분포로 파라미터화해 기대 보상의 그래디언트로 직접 밀어 올린다. 세 부품이 필요하다:

1. **액터-크리틱** — 액터(정책)와 크리틱(상태 가치 V). 크리틱은 "이 상태에서 평균적으로 얼마나 잘하나"의 기준선이 되어 그래디언트 분산을 줄인다.
2. **GAE** — 이점(advantage) $A_t$ = "이 행동이 평균보다 얼마나 좋았나"를 편향-분산 절충($\lambda$)으로 추정.
3. **클리핑** — 확률비 $r_t = \pi_{new}/\pi_{old}$를 $[1-\epsilon, 1+\epsilon]$로 잘라, 한 번의 갱신으로 정책이 너무 멀리 가는 것을 막는다. 덕분에 같은 데이터로 여러 에포크 재학습이 가능하다(샘플 효율).

## 전체 코드

```python
"""PPO — CartPole-v1. 실행: python ppo.py
의존성: pip install torch gymnasium
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym

device = "cuda" if torch.cuda.is_available() else "cpu"

GAMMA, LAM = 0.99, 0.95          # 할인율, GAE λ
CLIP_EPS = 0.2
ROLLOUT_STEPS = 2048             # 한 번에 모으는 경험량
EPOCHS_PER_UPDATE = 10           # 같은 데이터로 재학습 횟수 (클리핑이 있어 가능)
MINIBATCH = 64
LR = 3e-4
ENT_COEF, VF_COEF = 0.01, 0.5
TOTAL_UPDATES = 100


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                      nn.Linear(64, 64), nn.Tanh())
        self.pi = nn.Linear(64, n_actions)     # 정책 로짓
        self.v = nn.Linear(64, 1)              # 상태 가치

    def forward(self, x):
        h = self.backbone(x)
        return Categorical(logits=self.pi(h)), self.v(h).squeeze(-1)


def compute_gae(rewards, values, dones, last_value):
    """GAE: δ_t = r + γV(s') - V(s),  A_t = δ_t + γλ·A_{t+1}"""
    adv = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = last_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_v * nonterminal - values[t]
        gae = delta + GAMMA * LAM * nonterminal * gae
        adv[t] = gae
    returns = adv + values                     # 크리틱의 회귀 타깃
    return adv, returns


def main():
    torch.manual_seed(0)
    env = gym.make("CartPole-v1")
    model = ActorCritic().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    s, _ = env.reset(seed=0)
    ep_reward, ep_rewards = 0.0, []

    for update in range(TOTAL_UPDATES):
        # ---------- 1) 롤아웃: 현재 정책으로 경험 수집 ----------
        obs, acts, logps, rews, dones, vals = [], [], [], [], [], []
        for _ in range(ROLLOUT_STEPS):
            st = torch.tensor(s, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist, v = model(st)
                a = dist.sample()
                logp = dist.log_prob(a)        # π_old의 로그확률 — 클리핑 비율의 분모
            s2, r, term, trunc, _ = env.step(a.item())
            done = term or trunc
            obs.append(st); acts.append(a); logps.append(logp)
            rews.append(r); dones.append(float(term)); vals.append(v)
            ep_reward += r
            s = s2
            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                s, _ = env.reset()

        obs = torch.stack(obs); acts = torch.stack(acts)
        logps_old = torch.stack(logps)
        vals = torch.stack(vals)
        rews = torch.tensor(rews, dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, last_v = model(torch.tensor(s, dtype=torch.float32, device=device))
        adv, returns = compute_gae(rews, vals, dones_t, last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # 이점 정규화: 안정성의 표준 트릭

        # ---------- 2) 갱신: 같은 데이터로 여러 에포크 ----------
        idx = torch.arange(ROLLOUT_STEPS, device=device)
        for _ in range(EPOCHS_PER_UPDATE):
            for start in range(0, ROLLOUT_STEPS, MINIBATCH):
                b = idx[torch.randperm(ROLLOUT_STEPS, device=device)[start:start + MINIBATCH]]
                dist, v = model(obs[b])
                logp_new = dist.log_prob(acts[b])
                ratio = (logp_new - logps_old[b]).exp()          # π_new / π_old

                # PPO-clip 목적: 이점 방향으로, 그러나 비율은 [1-ε, 1+ε] 안에서만
                surr1 = ratio * adv[b]
                surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv[b]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (v - returns[b]).pow(2).mean()      # 크리틱: 리턴 회귀
                entropy = dist.entropy().mean()                  # 탐색 유지 보너스

                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        if ep_rewards:
            avg = sum(ep_rewards[-20:]) / len(ep_rewards[-20:])
            print(f"update {update+1:3d}  평균 보상(최근 20 에피소드) {avg:6.1f}")
            if avg >= 475:
                print("해결!")
                break


if __name__ == "__main__":
    main()
```

보통 30~60 업데이트(6만~12만 스텝) 안에 해결된다.

## 반드시 이해할 세 지점

**1. `logps_old`를 롤아웃 때 저장하는 이유.** 클리핑 비율의 분모는 "데이터를 모을 때의 정책"이다. 갱신 중 재계산하면 비율이 항상 1이 되어 클리핑이 무력화된다. `with torch.no_grad()`로 수집하는 것까지 포함해, on-policy 데이터의 "신선도"를 관리하는 것이 PPO 구현의 핵심 규율이다.

**2. `torch.min(surr1, surr2)`의 의미.** 이점이 양수일 때는 비율이 $1+\epsilon$을 넘는 이득을 무시하고(더 밀어붙일 유인 제거), 음수일 때는 비율이 $1-\epsilon$ 아래로 내려가는 이득을 무시한다. 요약하면 "좋은 방향으로도 한 번에 조금만" — 이 보수성이 같은 데이터 10 에포크 재사용을 가능하게 한다.

**3. RLHF와의 대응.** LLM의 RLHF에서는: 액터=LLM, 행동=다음 토큰, 크리틱=가치 헤드, 보상=보상 모델 점수(+참조 모델 KL 페널티), 롤아웃=응답 생성. 위 코드의 구조가 거의 그대로 확장된다. 차이는 스케일과, 에피소드가 "응답 하나"라는 점뿐이다.

## 확장 과제

1. **연속 행동 공간** — `Pendulum-v1`을 풀도록 정책을 가우시안(`Normal` 분포, 평균·표준편차 출력)으로 바꿔라. PPO가 DQN과 달리 연속 행동을 자연스럽게 다루는 이유가 코드로 보인다.
2. **클리핑 제거 실험** — `surr2`를 빼고 순수 정책 경사로 10 에포크 재학습을 시도하라. 성능이 무너지는 과정을 관찰하면 클리핑의 존재 이유가 명확해진다.
3. **병렬 환경** — `gym.vector.SyncVectorEnv`로 8개 환경을 동시에 굴려 롤아웃을 모아라. 벽시계 시간이 얼마나 줄어드는가?
4. **λ 스윕** — GAE λ를 0(TD)과 1(몬테카를로)로 바꿔 학습 곡선을 비교하라. 편향-분산 트레이드오프의 실물이다.

## 다음

이미지와 텍스트를 한 공간에서 — 멀티모달 → [18. CLIP과 멀티모달](/practice/18-clip-multimodal)
