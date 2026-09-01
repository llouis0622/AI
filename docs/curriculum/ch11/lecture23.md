# Lecture 23. 최적화와 학습

## 개요

**핵심 질문**

- 미분은 함수의 변화를 어떻게 포착하는가?
- 다변수 함수에서 그래디언트는 무엇을 의미하는가?
- 경사 하강법은 왜 최적화의 기본 방법인가?
- 딥러닝의 학습은 최적화 관점에서 어떻게 이해할 수 있는가?

**학습 목표**

- 도함수·편미분·그래디언트의 수학적 정의와 기하학적 의미를 설명할 수 있다.
- 연쇄 법칙이 역전파의 수학적 기반임을 이해한다.
- 경사 하강법 및 주요 옵티마이저(SGD, Adam)의 원리를 설명할 수 있다.
- 볼록·비볼록 함수, 안장점, 지역 최소의 차이를 이해한다.

---

## 핵심 개념

### 1. 함수와 변화량

**함수의 정의**

$$
f: X \to Y, \quad y = f(x)
$$

어떤 집합의 각 원소를 다른 집합의 유일한 원소에 대응시키는 규칙.

**변화량과 평균 변화율**

입력 $x$가 $\Delta x$만큼 변할 때 출력의 변화량:

$$
\Delta y = f(x + \Delta x) - f(x)
$$

평균 변화율 (두 점을 잇는 할선의 기울기):

$$
\frac{\Delta y}{\Delta x} = \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

**AI에서의 의미**

학습에서 "변화량"은 핵심이다. 파라미터 $\theta$를 $\Delta\theta$만큼 바꿨을 때 손실이 얼마나 변하는지($\Delta J$)를 알아야 올바른 방향으로 파라미터를 업데이트할 수 있다.

---

### 2. 미분 (Differentiation)

**도함수 (Derivative)**

평균 변화율에서 $\Delta x \to 0$으로 보낸 극한값 — 특정 점에서의 순간 변화율이자 접선의 기울기:

$$
f'(x) = \frac{df}{dx} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

**기하학적 의미**

- $f'(x) > 0$: 함수가 증가하는 방향
- $f'(x) < 0$: 함수가 감소하는 방향
- $f'(x) = 0$: 임계점(critical point) — 극값·안장점 후보

**주요 미분 공식**

$$
(x^n)' = nx^{n-1}, \quad (e^x)' = e^x, \quad (\ln x)' = \frac{1}{x}
$$

$$
(\sin x)' = \cos x, \quad (\cos x)' = -\sin x
$$

$$
(f + g)' = f' + g', \quad (cf)' = cf', \quad (fg)' = f'g + fg'
$$

$$
\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}
$$

**활성 함수의 미분**

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x)), \quad \sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$
\tanh'(x) = 1 - \tanh^2(x)
$$

$$
\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

---

### 3. 연쇄 법칙 (Chain Rule)

합성 함수의 미분:

$$
\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)
$$

표기법으로:

$$
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
$$

**역전파는 연쇄 법칙의 반복 적용이다.**

신경망 $J = \ell(f^{(L)}(\cdots f^{(1)}(\mathbf{x})))$에서:

$$
\frac{\partial J}{\partial W^{(1)}} = \frac{\partial J}{\partial \mathbf{a}^{(L)}} \cdot \frac{\partial \mathbf{a}^{(L)}}{\partial \mathbf{z}^{(L)}} \cdots \frac{\partial \mathbf{z}^{(1)}}{\partial W^{(1)}}
$$

각 층에서 지역 미분(local gradient)을 계산하고, 역방향으로 전달받은 전역 미분(upstream gradient)과 곱하면 해당 파라미터의 그래디언트를 얻는다.

$$
\frac{\partial J}{\partial W^{(l)}} = \delta^{(l)} \cdot (\mathbf{a}^{(l-1)})^\top, \quad \delta^{(l)} = \frac{\partial J}{\partial \mathbf{z}^{(l)}}
$$

**AI에서의 사용**

역전파 알고리즘은 연쇄 법칙을 계산 그래프(Computational Graph) 위에서 효율적으로 수행하는 알고리즘이다. 공통 부분(오차 $\delta$)을 한 번만 계산하고 재사용하기 때문에 나이브한 수치 미분 대비 계산 비용이 크게 절감된다.

---

### 4. 편미분과 그래디언트

**편미분 (Partial Derivative)**

다변수 함수 $f(x_1, x_2, \ldots, x_n)$에서 하나의 변수만 변화시키고 나머지는 고정한 미분:

$$
\frac{\partial f}{\partial x_i} = \lim_{\Delta x_i \to 0} \frac{f(x_1, \ldots, x_i + \Delta x_i, \ldots, x_n) - f(x_1, \ldots, x_n)}{\Delta x_i}
$$

**그래디언트 (Gradient)**

모든 편미분을 모아 벡터로 표현한 것 — 함수가 **가장 가파르게 증가하는 방향**:

$$
\nabla_\mathbf{x} f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}
$$

**기하학적 의미**

- $\nabla f(\mathbf{x})$는 점 $\mathbf{x}$에서 함수값이 가장 빠르게 증가하는 방향
- $-\nabla f(\mathbf{x})$는 함수값이 가장 빠르게 감소하는 방향 → **경사 하강법의 이동 방향**
- $\|\nabla f(\mathbf{x})\|$는 그 방향으로의 변화 속도

**야코비안 (Jacobian)**

벡터 함수 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$의 모든 편미분을 행렬로 표현:

$$
J_f = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix} \in \mathbb{R}^{m \times n}
$$

역전파에서 층과 층 사이의 그래디언트 전달은 야코비안 곱으로 표현된다.

**헤시안 (Hessian)**

스칼라 함수 $f: \mathbb{R}^n \to \mathbb{R}$의 2차 편미분 행렬:

$$
H_f = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & & \ddots
\end{pmatrix}
$$

헤시안은 손실 함수의 **곡률** 정보를 담고 있다. 헤시안의 고윳값 분포로 임계점의 종류(극소·극대·안장점)를 판별할 수 있다.

**주요 그래디언트 공식**

$$
\nabla_\mathbf{x}(\mathbf{w}^\top \mathbf{x}) = \mathbf{w}
$$

$$
\nabla_\mathbf{x}(\mathbf{x}^\top A\mathbf{x}) = 2A\mathbf{x} \quad (A \text{: 대칭})
$$

$$
\nabla_\mathbf{w}\|\mathbf{y} - X\mathbf{w}\|_2^2 = -2X^\top(\mathbf{y} - X\mathbf{w})
$$

---

### 5. 최적화 문제의 정의

**최적화 문제의 표준 형태**

$$
\min_{\theta \in \mathcal{D}} J(\theta)
$$

- $J(\theta)$: 목적 함수(손실 함수) — 최소화 대상
- $\theta$: 최적화 변수(신경망 파라미터)
- $\mathcal{D}$: 탐색 공간

최대화 문제는 $\max_\theta f(\theta) = \min_\theta -f(\theta)$로 변환 가능.

**임계점의 종류**

$\nabla J(\theta) = \mathbf{0}$을 만족하는 점:

| 종류 | 헤시안 고윳값 | 특징 |
|---|---|---|
| 전역 최소 (Global Minimum) | 모두 양수 | 함수 전체에서 가장 낮은 점 |
| 지역 최소 (Local Minimum) | 모두 양수 | 주변에서만 가장 낮은 점 |
| 안장점 (Saddle Point) | 양수·음수 혼재 | 어떤 방향은 극소, 어떤 방향은 극대 |
| 전역 최대 | 모두 음수 | 함수 전체에서 가장 높은 점 |

$$
P(\text{안장점}) = 1 - \frac{1}{2^{n-1}}, \quad P(\text{지역 최소}) = \frac{1}{2^n} \quad (n\text{: 차원})
$$

고차원에서 임계점 대부분은 안장점이다. 딥러닝에서 지역 최소에 갇히는 문제보다 안장점에서 그래디언트가 0에 가까워 학습이 느려지는 문제가 더 자주 발생한다.

**볼록 함수 (Convex Function)**

$$
f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y}), \quad \forall \lambda \in [0,1]
$$

볼록 함수에서는 지역 최소 = 전역 최소. 딥러닝의 손실 함수는 대부분 비볼록(non-convex)이다.

---

### 6. 경사 하강법 (Gradient Descent)

**핵심 아이디어**

현재 위치에서 손실이 가장 빠르게 감소하는 방향($-\nabla J$)으로 한 걸음 이동하기를 반복한다.

$$
\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)
$$

- $\alpha$: 학습률(Learning Rate, Step Size) — 한 걸음의 크기
- $\nabla_\theta J$: 현재 파라미터에서의 손실 그래디언트

**학습률의 역할**

- $\alpha$ 너무 작음: 수렴은 하지만 학습 속도 매우 느림
- $\alpha$ 너무 큼: 최소점을 지나쳐 발산할 수 있음
- 최적 $\alpha$: 안정적이고 빠른 수렴

**훈련 데이터 단위에 따른 분류**

| 방법 | 배치 크기 | 그래디언트 | 특징 |
|---|---|---|---|
| 배치 GD (Batch GD) | 전체 데이터 | 정확 | 부드러운 경로, 느린 업데이트 |
| 미니배치 SGD | $B$개 ($32 \sim 512$) | 근사 | 실제로 가장 많이 사용 |
| 확률적 GD (SGD) | 1개 | 잡음 많음 | 빠르지만 불안정 |

미니배치 방식이 선호되는 이유: GPU 병렬 처리 효율성 + 잡음이 정규화 효과를 주어 일반화 성능 향상.

---

### 7. 주요 옵티마이저

**SGD + Momentum**

이전 업데이트 방향을 관성으로 유지하여 안장점 탈출과 진동 감소:

$$
v_{t+1} = \rho v_t + \nabla J(\theta_t), \quad \theta_{t+1} = \theta_t - \alpha v_{t+1}
$$

- $\rho$: 모멘텀 계수 (보통 0.9)
- 이전 속도가 누적되어 가속 효과 → 안장점 탈출 용이

**Nesterov Momentum**

현재 속도로 미리 한 걸음 나간 위치에서 그래디언트를 계산 → 오버슈팅 억제:

$$
v_{t+1} = \rho v_t - \alpha \nabla J(\theta_t + \rho v_t), \quad \theta_{t+1} = \theta_t + v_{t+1}
$$

**AdaGrad**

파라미터별로 과거 그래디언트 제곱합을 누적하여 적응적 학습률 적용:

$$
r_{t+1} = r_t + (\nabla J(\theta_t))^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{r_{t+1}} + \epsilon} \odot \nabla J(\theta_t)
$$

많이 업데이트된 파라미터는 학습률이 작아지고, 드물게 업데이트된 파라미터는 학습률이 유지된다. 단점: 학습이 진행될수록 $r$이 단조 증가 → 학습률이 0에 수렴하여 학습 중단.

**RMSProp**

AdaGrad의 학습 중단 문제 해결 — 지수 가중 이동 평균으로 최근 그래디언트만 반영:

$$
r_{t+1} = \beta r_t + (1-\beta)(\nabla J(\theta_t))^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{r_{t+1}} + \epsilon} \odot \nabla J(\theta_t)
$$

**Adam (Adaptive Moment Estimation)**

SGD Momentum + RMSProp의 결합. 현재 가장 널리 사용되는 옵티마이저.

1차 모멘텀 (속도):
$$
v_{t+1} = \beta_1 v_t + (1-\beta_1)\nabla J(\theta_t)
$$

2차 모멘텀 (적응적 학습률):
$$
r_{t+1} = \beta_2 r_t + (1-\beta_2)(\nabla J(\theta_t))^2
$$

초기 편향 보정:
$$
\hat{v} = \frac{v_{t+1}}{1-\beta_1^t}, \quad \hat{r} = \frac{r_{t+1}}{1-\beta_2^t}
$$

파라미터 업데이트:
$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{r}} + \epsilon} \odot \hat{v}
$$

기본값: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

---

### 8. 학습을 최적화로 이해하기

**신경망 학습 = 손실 함수 최소화**

$$
\theta^* = \argmin_\theta J(\theta) = \argmin_\theta \frac{1}{N}\sum_{i=1}^N \mathcal{L}(f(\mathbf{x}_i;\theta), t_i)
$$

**훈련 루프 (Training Loop)**

1. 미니배치 $(\mathbf{x}, \mathbf{t})$ 추출
2. 순전파: $\hat{y} = f(\mathbf{x};\theta)$
3. 손실 계산: $J = \mathcal{L}(\hat{y}, \mathbf{t})$
4. 역전파: $\nabla_\theta J$ 계산 (연쇄 법칙)
5. 파라미터 업데이트: $\theta \leftarrow \theta - \alpha\nabla_\theta J$
6. 반복

**학습 과정을 '움직임'으로 이해하기**

파라미터 공간($\theta$ 공간)에서 손실 함수 $J(\theta)$는 하나의 지형(landscape)을 이룬다.

- 초기화: 이 지형의 임의 위치에서 출발
- 각 업데이트: 현재 위치에서 가장 가파른 내리막($-\nabla J$) 방향으로 $\alpha$ 만큼 이동
- 모멘텀: 이전 방향의 관성을 유지하여 협곡에서 진동 없이 전진
- 적응적 학습률: 지형의 곡률에 따라 걸음 크기를 자동 조절
- 학습 완료: 손실이 충분히 낮은 지역 최소 근방에 도달

**그레이디언트 소실 (Vanishing Gradient)**

깊은 네트워크에서 역전파 중 그래디언트가 층을 거슬러 올라가며 0에 수렴:

$$
\frac{\partial J}{\partial W^{(1)}} = \prod_{l=2}^{L} \frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{z}^{(l)}} \cdot \frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{a}^{(l-1)}} \cdot \frac{\partial J}{\partial \mathbf{a}^{(L)}}
$$

시그모이드·tanh는 포화 영역에서 미분값이 0에 가까워 곱이 0으로 수렴한다.

해결책: ReLU 활성 함수, 잔차 연결, 배치 정규화.

**그레이디언트 클리핑 (Gradient Clipping)**

RNN 등에서 그래디언트 폭발 방지:

$$
g \leftarrow \frac{g}{\|g\|} \cdot v, \quad \text{if } \|g\| > v
$$

**학습률 스케줄링**

- Step Decay: 일정 에포크마다 학습률을 $\gamma$배 감소
- Cosine Annealing: 코사인 함수 형태로 학습률을 주기적으로 감소

$$
\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\frac{\pi t}{T}\right)
$$

- Warmup: 초기에 학습률을 천천히 올린 뒤 감소 — Transformer 학습에서 표준

---

### 9. 최적화와 일반화의 관계

최적화의 목표는 훈련 손실 최소화이지만, 실제 목표는 **일반화** — 새로운 데이터에서의 성능이다.

$$
\underbrace{J_{\text{test}}(\theta)}_{\text{실제 목표}} = \underbrace{J_{\text{train}}(\theta)}_{\text{최적화 대상}} + \underbrace{(J_{\text{test}} - J_{\text{train}})}_{\text{일반화 갭}}
$$

- 과적합: 훈련 손실은 낮지만 일반화 갭이 큼
- 미니배치 SGD의 잡음: 정규화 효과 → 평평한(flat) 최소로 수렴 → 일반화 성능 향상
- 배치 정규화: 최적화 지형을 평탄하게 만들어 큰 학습률 사용 가능 + 일반화 개선

---

## 수식 정리

**도함수 정의**

$$
f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

**연쇄 법칙**

$$
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
$$

**그래디언트**

$$
\nabla_\mathbf{x} f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)^\top
$$

**주요 그래디언트 공식**

$$
\nabla_\mathbf{x}(\mathbf{w}^\top \mathbf{x}) = \mathbf{w}, \quad \nabla_\mathbf{x}(\mathbf{x}^\top A\mathbf{x}) = 2A\mathbf{x}
$$

**경사 하강법**

$$
\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)
$$

**Adam 업데이트**

$$
v \leftarrow \beta_1 v + (1-\beta_1)\nabla J, \quad r \leftarrow \beta_2 r + (1-\beta_2)(\nabla J)^2
$$

$$
\theta \leftarrow \theta - \frac{\alpha}{\sqrt{\hat{r}} + \epsilon}\hat{v}
$$

**역전파 오차**

$$
\delta^{(l)} = \frac{\partial J}{\partial \mathbf{z}^{(l)}}, \quad \frac{\partial J}{\partial W^{(l)}} = \delta^{(l)} (\mathbf{a}^{(l-1)})^\top
$$

**그래디언트 클리핑**

$$
g \leftarrow \frac{g}{\|g\|} \cdot v \quad \text{if } \|g\| > v
$$

**Cosine Annealing 학습률**

$$
\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\frac{\pi t}{T}\right)
$$
