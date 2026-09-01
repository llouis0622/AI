# Lecture 22. 확률적 모델링

## 개요

**핵심 질문**

- 확률 분포는 AI에서 무엇을 표현하는가?
- MLE는 손실 함수와 어떻게 연결되는가?
- KL 발산과 엔트로피는 어디에 쓰이는가?
- 베이즈 추론은 딥러닝과 어떤 관계인가?

**학습 목표**

- 확률 공리·조건부 확률·베이즈 정리를 설명할 수 있다.
- 주요 확률 분포의 정의·평균·분산과 AI 적용처를 연결할 수 있다.
- MLE가 손실 함수 유도의 기반임을 수식으로 이해한다.
- KL 발산·엔트로피·크로스 엔트로피의 관계를 설명할 수 있다.

---

## 핵심 개념

### 1. 확률의 기초

**확률 공리 (Kolmogorov)**

$$
0 \leq P(A) \leq 1, \quad P(\Omega) = 1
$$

$$
P\!\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i) \quad (\text{서로소인 사건})
$$

**결합·주변·조건부 확률**

$$
P(A \cap B) = P(A|B)P(B) = P(B|A)P(A) \quad \text{(곱 법칙)}
$$

$$
P(A) = \sum_B P(A \cap B) = \sum_B P(A|B)P(B) \quad \text{(주변화, 합 법칙)}
$$

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

**독립과 배반**

- 독립: $P(A \cap B) = P(A)P(B)$
- 배반(서로소): $P(A \cap B) = 0 \Rightarrow P(A \cup B) = P(A) + P(B)$

**전확률 공식**

$$
P(A) = P(A|B)P(B) + P(A|B^C)P(B^C) = \sum_k P(A|B_k)P(B_k)
$$

**베이즈 정리 (Bayes' Theorem)**

$$
P(\theta|x) = \frac{P(x|\theta)P(\theta)}{P(x)} \propto P(x|\theta)P(\theta)
$$

- $P(\theta)$: 사전 확률 (Prior) — 데이터 관측 전 파라미터에 대한 믿음
- $P(x|\theta)$: 우도 (Likelihood) — 파라미터가 주어졌을 때 데이터가 나올 확률
- $P(\theta|x)$: 사후 확률 (Posterior) — 데이터를 관측한 후 업데이트된 믿음
- $P(x)$: 증거 (Evidence) — 정규화 상수

**AI에서의 사용**

- 판별 모델: $P(y|\mathbf{x})$를 직접 모델링
- 생성 모델: $P(\mathbf{x}|y)P(y)$로 분해 → 베이즈 정리로 연결
- 베이지안 최적화: 하이퍼파라미터 탐색 시 사후 확률 업데이트

---

### 2. 확률 변수와 확률 분포

**확률 변수 (Random Variable)**

$$
P(X = x), \quad F_X(x) = P(X \leq x)
$$

- 이산형: 가능한 값의 종류를 셀 수 있음 → 확률 질량 함수(PMF)
- 연속형: 가능한 값의 종류를 셀 수 없음 → 확률 밀도 함수(PDF)

**이산형 PMF 조건**

$$
P_X(x) \geq 0, \quad \sum_x P_X(x) = 1
$$

**연속형 PDF 조건**

$$
f_X(x) \geq 0, \quad \int_{-\infty}^{\infty} f_X(x)\,dx = 1, \quad P(a < X < b) = \int_a^b f_X(x)\,dx
$$

**독립 항등 분포 (i.i.d.)**

머신러닝에서 훈련 데이터는 대부분 i.i.d.를 가정한다. 각 샘플이 독립이고 동일한 분포를 따르면 결합 확률을 곱으로 분해할 수 있다.

$$
p(x_1, x_2, \ldots, x_n) = \prod_{i=1}^n p(x_i)
$$

---

### 3. 이산형 확률 분포

**베르누이 분포 (Bernoulli)**

$$
X \sim \text{Bernoulli}(p), \quad P(X=x) = p^x(1-p)^{1-x}, \quad x \in \{0,1\}
$$

$$
E(X) = p, \quad \text{Var}(X) = p(1-p)
$$

**AI 사용**: 이진 분류의 출력층. 시그모이드 활성 함수 → 베르누이 파라미터 $\mu$ 추정.

**이항 분포 (Binomial)**

$$
X \sim \text{Binomial}(n, p), \quad P(X=x) = \binom{n}{x}p^x(1-p)^{n-x}
$$

$$
E(X) = np, \quad \text{Var}(X) = np(1-p)
$$

**AI 사용**: 베르누이 시행의 합. 이진 분류를 $n$번 시행하는 모델링.

**카테고리 분포 (Categorical)**

$$
P(\mathbf{x}|\boldsymbol{\mu}) = \prod_{k=1}^K \mu_k^{x_k}, \quad \sum_k \mu_k = 1, \quad x_k \in \{0,1\}
$$

$$
E(x_k) = \mu_k, \quad \text{Var}(x_k) = \mu_k(1-\mu_k)
$$

**AI 사용**: 다중 분류의 출력층. 소프트맥스 활성 함수 → 카테고리 분포 파라미터 $\boldsymbol{\mu}$ 추정.

**포아송 분포 (Poisson)**

$$
X \sim \text{Poisson}(\lambda), \quad P(X=x) = \frac{e^{-\lambda}\lambda^x}{x!}, \quad x = 0,1,2,\ldots
$$

$$
E(X) = \lambda, \quad \text{Var}(X) = \lambda
$$

**AI 사용**: 단위 시간 내 이벤트 발생 횟수 모델링. 이상치 탐지, 카운트 데이터.

이항 분포의 포아송 근사: $n \to \infty$, $p \to 0$, $np = \lambda$ 고정 시 $\text{Binomial}(n,p) \to \text{Poisson}(\lambda)$.

---

### 4. 연속형 확률 분포

**가우시안 분포 (Normal/Gaussian)**

$$
X \sim \mathcal{N}(\mu, \sigma^2), \quad f_X(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

$$
E(X) = \mu, \quad \text{Var}(X) = \sigma^2
$$

표준 정규 분포: $\mathcal{N}(0,1)$.

**다변량 가우시안**

$$
\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

공분산 행렬 $\Sigma$는 반드시 양반정치(PSD)이어야 한다.

**AI 사용**: 회귀 노이즈 모델 → MLE = MSE 유도 (아래 §6 참고). VAE의 잠재 사전 분포 $p(\mathbf{z}) = \mathcal{N}(0, I)$. 배치 정규화 목표 분포.

**정규 근사**: $n$이 커지면 분포에 무관하게 표본 평균이 정규 분포에 수렴 (중심 극한 정리).

$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)
$$

**연속형 균일 분포 (Uniform)**

$$
X \sim U(a,b), \quad f_X(x) = \frac{1}{b-a}, \quad x \in [a,b]
$$

$$
E(X) = \frac{a+b}{2}, \quad \text{Var}(X) = \frac{(b-a)^2}{12}
$$

**AI 사용**: 가중치 초기화(Xavier 초기화), 하이퍼파라미터 랜덤 서치.

**라플라스 분포 (Laplace)**

$$
f_X(x) = \frac{1}{2b}\exp\!\left(-\frac{|x-\mu|}{b}\right)
$$

$$
E(X) = \mu, \quad \text{Var}(X) = 2b^2
$$

**AI 사용**: $L_1$ 정규화의 사전 분포. MAP 추정에서 라플라스 사전 분포 → Lasso 손실.

**감마 분포 (Gamma)**

$$
X \sim \text{Gamma}(\alpha, \beta), \quad f_X(x) = \frac{1}{\Gamma(\alpha)\beta^\alpha}x^{\alpha-1}e^{-x/\beta}
$$

$$
E(X) = \alpha\beta, \quad \text{Var}(X) = \alpha\beta^2
$$

- 지수 분포: $\alpha=1$인 특수 케이스 ($E(X)=\beta$, $\text{Var}(X)=\beta^2$)
- 카이제곱 분포: $\alpha=p/2$, $\beta=2$인 특수 케이스 ($E(X)=p$, $\text{Var}(X)=2p$)

**베타 분포 (Beta)**

$$
X \sim \text{Beta}(\alpha, \beta), \quad f_X(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, \quad x \in (0,1)
$$

$$
E(X) = \frac{\alpha}{\alpha+\beta}, \quad \text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
$$

**AI 사용**: 베르누이/이항 분포의 켤레 사전 분포. 베이지안 추론에서 확률 파라미터의 분포 표현.

---

### 5. 기댓값, 분산, 공분산

**기댓값 (Expectation)**

$$
E(X) = \sum_x x\,P(X=x) \quad (\text{이산}), \qquad E(X) = \int_{-\infty}^\infty x\,f_X(x)\,dx \quad (\text{연속})
$$

$$
E(aX + bY) = aE(X) + bE(Y) \quad \text{(선형성)}
$$

**분산 (Variance)**

$$
\text{Var}(X) = E[(X-\mu)^2] = E(X^2) - [E(X)]^2
$$

$$
\text{Var}(aX + b) = a^2\text{Var}(X)
$$

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)
$$

**공분산 (Covariance)**

$$
\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)] = E(XY) - E(X)E(Y)
$$

$$
\text{Cov}(X,X) = \text{Var}(X), \quad \text{Cov}(aX, bY) = ab\,\text{Cov}(X,Y)
$$

**상관 계수 (Correlation)**

$$
\text{Corr}(X,Y) = \frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y}, \quad -1 \leq \text{Corr}(X,Y) \leq 1
$$

**표본 통계량**

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i, \quad s^2 = \frac{1}{n-1}\sum_{i=1}^n(x_i-\bar{x})^2
$$

$$
E(\bar{X}) = \mu, \quad \text{Var}(\bar{X}) = \frac{\sigma^2}{n}, \quad E(s^2) = \sigma^2
$$

**AI에서의 사용**

- **배치 정규화**: 미니배치의 평균·분산으로 활성화 정규화
- **공분산 행렬**: PCA의 핵심 — 특성 간 선형 관계 파악
- **분산 감소**: Actor-Critic에서 어드밴티지 함수로 Policy Gradient 분산 감소

---

### 6. 최대우도추정 (MLE)

**정의**

주어진 데이터가 나올 확률(우도)을 최대화하는 파라미터를 찾는 것.

$$
\hat{\theta} = \argmax_\theta L(\theta|\mathbf{x}) = \argmax_\theta \prod_{i=1}^N f(x_i|\theta)
$$

수치 안정성과 계산 편의를 위해 로그 우도 사용:

$$
\hat{\theta} = \argmax_\theta \sum_{i=1}^N \log f(x_i|\theta) = \argmin_\theta \left[-\sum_{i=1}^N \log f(x_i|\theta)\right]
$$

**MLE → 손실 함수 유도**

**회귀 (가우시안 노이즈 가정)**

$$
p(t|\mathbf{x};\theta) = \mathcal{N}(t\,|\,y(\mathbf{x};\theta),\,\sigma^2)
$$

$$
-\log p(\mathcal{D}|\theta) \propto \frac{1}{N}\sum_{i=1}^N(t_i - y(\mathbf{x}_i;\theta))^2 = \text{MSE}
$$

가우시안 노이즈 가정 → MLE = MSE 최소화.

**이진 분류 (베르누이 가정)**

$$
p(t|\mathbf{x};\theta) = \mu(\mathbf{x};\theta)^t(1-\mu(\mathbf{x};\theta))^{1-t}
$$

$$
-\log p(\mathcal{D}|\theta) = -\sum_{i=1}^N\left[t_i\log\mu_i + (1-t_i)\log(1-\mu_i)\right] = \text{Binary Cross Entropy}
$$

**다중 분류 (카테고리 가정)**

$$
p(t|\mathbf{x};\theta) = \prod_{k=1}^K \mu(\mathbf{x};\theta)_k^{t_k}
$$

$$
-\log p(\mathcal{D}|\theta) = -\sum_{i=1}^N\sum_{k=1}^K t_{ik}\log\mu(\mathbf{x}_i;\theta)_k = \text{Cross Entropy}
$$

**MLE = KL 발산 최소화**

$$
\hat{\theta}_{\text{MLE}} = \argmin_\theta D_{\text{KL}}(p_{\text{data}} \| p_\theta) = \argmin_\theta\left[-\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}\log p_\theta(\mathbf{x})\right]
$$

MLE는 데이터 분포와 모델 분포 사이의 KL 발산을 최소화하는 것과 수학적으로 동치다.

**최대 사후 추정 (MAP)**

MLE에 사전 분포를 추가한 형태:

$$
\hat{\theta}_{\text{MAP}} = \argmax_\theta p(\theta|\mathbf{x}) \propto \argmax_\theta p(\mathbf{x}|\theta)p(\theta)
$$

$$
-\log p(\theta|\mathbf{x}) \propto -\log p(\mathbf{x}|\theta) - \log p(\theta)
$$

- 가우시안 사전 분포 $p(\theta) \sim \mathcal{N}(0, \tau^2 I)$ → MAP = $L_2$ 정규화 (Ridge)
- 라플라스 사전 분포 $p(\theta) \sim \text{Laplace}(0, b)$ → MAP = $L_1$ 정규화 (Lasso)

---

### 7. 정보 이론

**정보량 (Self-Information)**

$$
I(x) = -\log p(x)
$$

사건 확률이 낮을수록 정보량이 크다.

**엔트로피 (Entropy)**

확률 분포의 불확실성(랜덤성)을 측정:

$$
H(p) = \mathbb{E}_{x \sim p(x)}[-\log p(x)] = -\sum_x p(x)\log p(x)
$$

- 모든 사건이 동일 확률: 엔트로피 최대
- 하나의 사건이 확실: 엔트로피 = 0

**크로스 엔트로피 (Cross Entropy)**

분포 $p$를 기준으로 분포 $q$의 정보량에 대한 기댓값:

$$
H(p,q) = -\mathbb{E}_{x \sim p(x)}\log q(x) = -\sum_x p(x)\log q(x)
$$

**KL 발산 (Kullback-Leibler Divergence)**

두 분포 사이의 비유사도(비대칭):

$$
D_{\text{KL}}(p \| q) = \mathbb{E}_{x \sim p}\left[\log\frac{p(x)}{q(x)}\right] = H(p,q) - H(p) \geq 0
$$

$D_{\text{KL}}(p\|q) = 0 \Leftrightarrow p = q$.

$D_{\text{KL}}(p\|q) \neq D_{\text{KL}}(q\|p)$ (비대칭).

세 개념의 관계:

$$
H(p,q) = H(p) + D_{\text{KL}}(p\|q)
$$

**AI에서의 사용**

- **분류 손실**: 레이블 분포 $p$ = 원-핫, 예측 분포 $q$ = 소프트맥스 출력 → Cross Entropy 최소화 = KL 발산 최소화
- **VAE**: ELBO의 KL 항 $D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$ — 잠재 분포를 표준 가우시안에 가깝게 유지
- **RLHF**: KL 페널티 $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ — 정렬 과정에서 정책이 기준 모델에서 너무 멀리 벗어나지 않도록 제한
- **지식 증류**: 학생 모델과 교사 모델의 출력 분포 간 KL 발산 최소화
- **언어 모델 Perplexity**: $\text{PPL} = e^{H(p,q)}$ — 모델이 실제 분포를 얼마나 잘 예측하는지의 지표

---

### 8. 조건부 확률과 마르코프 연쇄

**조건부 독립**

$$
P(A|B,C) = P(A|C) \Leftrightarrow P(A,B|C) = P(A|C)P(B|C)
$$

**마르코프 성질 (Markov Property)**

$$
P(X_{n+1}=j \mid X_0,X_1,\ldots,X_n) = P(X_{n+1}=j \mid X_n)
$$

현재 상태만으로 미래를 결정한다 — 과거 이력 불필요.

**전이 확률 행렬**

$$
P = [P_{ij}], \quad P_{ij} = P(X_{n+1}=j \mid X_n=i), \quad \sum_j P_{ij} = 1
$$

$n$-step 전이: $P^{(n)} = P^n$.

**AI에서의 사용**

- **강화학습 MDP**: 마르코프 성질이 핵심 가정 — 현재 상태 $s_t$만으로 다음 상태 결정
- **언어 모델 (N-gram)**: N-1차 마르코프 가정으로 다음 단어 확률 추정
- **MCMC 샘플링**: 마르코프 체인으로 복잡한 사후 분포에서 샘플 추출

---

### 9. 몬테카를로 방법

**기본 원리**

기댓값을 랜덤 샘플의 평균으로 추정:

$$
E(f(X)) = \int f(x)p(x)\,dx \approx \frac{1}{n}\sum_{i=1}^n f(x_i), \quad x_i \sim p(x)
$$

대수의 법칙에 의해 $n \to \infty$이면 추정값이 실제 기댓값에 수렴한다.

$$
\text{Var}(\hat{\mu}_n) = \frac{\sigma^2}{n} \to 0
$$

**MCMC (Markov Chain Monte Carlo)**

직접 샘플링이 어려운 복잡한 분포 $\pi(x)$에서 마르코프 체인을 이용해 샘플 추출.

메트로폴리스-헤이스팅스 채택 비율:

$$
R = \frac{\pi(y)Q(y \to x)}{\pi(x)Q(x \to y)}
$$

$U \sim U(0,1)$을 추출하여 $U \leq R$이면 $y$ 채택, 아니면 현재 $x$ 유지.

**AI에서의 사용**

- **베이지안 추론**: 해석적으로 계산 불가능한 사후 분포 $p(\theta|\mathbf{x})$에서 샘플 추출
- **데이터 증강**: 학습 분포에서 새로운 샘플 생성
- **적분 추정**: 고차원 적분을 샘플링으로 근사 — 강화학습 기댓값 추정

---

## 수식 정리

**베이즈 정리**

$$
P(\theta|x) = \frac{P(x|\theta)P(\theta)}{P(x)} \propto P(x|\theta)P(\theta)
$$

**주요 분포 요약**

| 분포 | $E(X)$ | $\text{Var}(X)$ | AI 적용 |
|---|---|---|---|
| $\text{Bernoulli}(p)$ | $p$ | $p(1-p)$ | 이진 분류 출력 |
| $\text{Categorical}(\boldsymbol{\mu})$ | $\mu_k$ | $\mu_k(1-\mu_k)$ | 다중 분류 출력 |
| $\mathcal{N}(\mu,\sigma^2)$ | $\mu$ | $\sigma^2$ | 회귀, VAE 잠재 분포 |
| $\text{Laplace}(\mu,b)$ | $\mu$ | $2b^2$ | $L_1$ 정규화 사전 분포 |
| $\text{Gamma}(\alpha,\beta)$ | $\alpha\beta$ | $\alpha\beta^2$ | 대기 시간 모델링 |

**MLE → 손실 함수**

$$
\mathcal{N} \Rightarrow \text{MSE}, \quad \text{Bernoulli} \Rightarrow \text{BCE}, \quad \text{Categorical} \Rightarrow \text{CE}
$$

**정보 이론**

$$
H(p) = -\sum_x p(x)\log p(x)
$$

$$
H(p,q) = -\sum_x p(x)\log q(x)
$$

$$
D_{\text{KL}}(p\|q) = H(p,q) - H(p) \geq 0
$$

$$
H(p,q) = H(p) + D_{\text{KL}}(p\|q)
$$

**MAP와 정규화**

$$
\hat{\theta}_{\text{MAP}} = \argmin_\theta \left[-\log p(\mathbf{x}|\theta) - \log p(\theta)\right]
$$

$$
p(\theta) \sim \mathcal{N}(0,\tau^2 I) \Rightarrow L_2 \text{ 정규화}, \quad p(\theta) \sim \text{Laplace}(0,b) \Rightarrow L_1 \text{ 정규화}
$$

**중심 극한 정리**

$$
\frac{\bar{X}-\mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)
$$
