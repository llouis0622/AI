# Lecture 21. 선형대수와 표현

## 개요

**핵심 질문**

- 벡터·행렬·텐서는 AI에서 무엇을 표현하는가?
- 행렬 곱은 신경망의 어떤 연산에 대응하는가?
- 고윳값 분해와 SVD는 AI에서 어떻게 활용되는가?
- 선형 변환의 기하학적 의미는 표현 학습과 어떻게 연결되는가?

**학습 목표**

- 스칼라·벡터·행렬·텐서의 수학적 정의와 연산을 설명할 수 있다.
- 내적·노름·행렬 곱의 기하학적 의미를 이해한다.
- 고윳값 분해와 SVD의 원리와 AI 응용을 설명할 수 있다.
- 선형 변환이 딥러닝에서 어떤 역할을 하는지 연결할 수 있다.

---

## 핵심 개념

### 1. 스칼라, 벡터, 행렬, 텐서

**수학적 정의**

| 구조 | 차원 | 표기 | 예시 |
|---|---|---|---|
| 스칼라 | 0차 | $x \in \mathbb{R}$ | 학습률 $\alpha = 0.01$ |
| 벡터 | 1차 | $\mathbf{x} \in \mathbb{R}^n$ | 단어 임베딩, 특성 벡터 |
| 행렬 | 2차 | $A \in \mathbb{R}^{m \times n}$ | 가중치 행렬, 이미지 채널 |
| 텐서 | N차 | $\mathcal{X} \in \mathbb{R}^{d_1 \times \cdots \times d_N}$ | 이미지 배치, 멀티헤드 어텐션 |

벡터는 크기(노름)와 방향을 동시에 표현하며, 행 벡터와 열 벡터의 구분은 행렬 연산에서 중요하다.

**AI에서의 사용**

- **스칼라**: 손실값, 학습률, 온도 파라미터
- **벡터**: 하나의 데이터 샘플(특성 벡터), 단어 임베딩, 은닉 상태 $\mathbf{h}_t$
- **행렬**: 가중치 $W$, 어텐션 행렬, 공분산 행렬
- **텐서**: 이미지 배치 $(N, H, W, C)$, 멀티헤드 어텐션 Q/K/V

GPU는 고차원 텐서 연산에 최적화된 하드웨어이기 때문에, 딥러닝 전체 연산을 텐서로 통일하면 하드웨어 가속을 그대로 활용할 수 있다.

---

### 2. 벡터 연산

**덧셈·뺄셈**

크기가 동일한 벡터끼리만 가능하며 교환 법칙이 성립한다.

$$
\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}
$$

**스칼라 곱**

벡터의 길이를 변경하고, 음수이면 방향도 반전된다.

$$
c\mathbf{u} = (cu_1, cu_2, \ldots, cu_n)
$$

**내적 (Inner Product)**

$$
\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^\top \mathbf{v} = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta
$$

- 내적 > 0: 두 벡터 사이 각도 < 90°
- 내적 = 0: 직교 (수직)
- 내적 < 0: 두 벡터 사이 각도 > 90°

**외적 (Outer Product)**

$$
\mathbf{u} \otimes \mathbf{v} = \mathbf{u}\mathbf{v}^\top \in \mathbb{R}^{m \times n}
$$

결과가 행렬이 된다. 주의할 점은 내적과 달리 외적은 스칼라가 아닌 행렬을 반환한다는 것이다.

**AI에서의 사용**

- **Attention Score**: $\text{score}(q, k) = q^\top k$ — Scaled Dot-Product Attention의 핵심 연산
- **코사인 유사도**: $\cos\theta = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ — 임베딩 검색, RAG 유사도 계산
- **선형 뉴런**: $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$ — 뉴런의 가중 합산

---

### 3. 노름 (Norm)

**수학적 정의**

$$
\|\mathbf{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}, \quad p \geq 1
$$

| 노름 | 수식 | 기하학적 의미 |
|---|---|---|
| $L_1$ | $\sum_i |x_i|$ | 맨하탄 거리 |
| $L_2$ | $\sqrt{\sum_i x_i^2}$ | 유클리드 거리 |
| $L_\infty$ | $\max_i |x_i|$ | 체비쇼프 거리 |

**AI에서의 사용**

- **$L_2$ 정규화 (Weight Decay)**: $\tilde{J}(\theta) = J(\theta) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$ — 가중치 크기 제한
- **$L_1$ 정규화 (Lasso)**: $\tilde{J}(\theta) = J(\theta) + \lambda\|\mathbf{w}\|_1$ — 희소성 유도, 불필요한 가중치를 정확히 0으로 만듦
- **Gradient Clipping**: $\|\nabla\theta\| > v \Rightarrow g \leftarrow \frac{g}{\|g\|} v$ — 그레이디언트 폭발 방지

$L_1$이 희소성을 만드는 이유는 $L_1$의 제약 영역이 다이아몬드 형태로, 손실 함수의 등고선과 축 위(모서리)에서 먼저 접촉하기 때문이다.

---

### 4. 행렬 연산

**행렬 곱**

$$
(AB)_{ij} = \sum_{k=1}^p A_{ik} B_{kj}, \quad A \in \mathbb{R}^{m \times p},\ B \in \mathbb{R}^{p \times n} \Rightarrow AB \in \mathbb{R}^{m \times n}
$$

교환 법칙 불성립: $AB \neq BA$.

**전치 행렬**

$$
(A^\top)_{ij} = A_{ji}, \quad (AB)^\top = B^\top A^\top
$$

**대각합 (Trace)**

$$
\text{tr}(A) = \sum_i a_{ii} = \sum_i \lambda_i
$$

대각합은 고윳값의 합과 같다.

**행렬식 (Determinant)**

$$
\det(A) = \sum_{\sigma} \text{sgn}(\sigma) \prod_i a_{i,\sigma(i)}
$$

- $\det(A) = 0$: 역행렬 없음 (특이 행렬), 열 벡터들이 선형 종속
- $|\det(A)|$: 선형 변환이 단위 부피를 얼마나 늘리거나 줄이는지의 배수

**역행렬**

$$
AA^{-1} = A^{-1}A = I, \quad A^{-1} = \frac{1}{\det(A)} \text{adj}(A)
$$

**주요 행렬 종류**

| 종류 | 조건 | 성질 |
|---|---|---|
| 대칭 행렬 | $A = A^\top$ | 고윳값 분해 가능, 실수 고윳값 |
| 직교 행렬 | $AA^\top = I$ | $A^{-1} = A^\top$, $\|\det\| = 1$ |
| 대각 행렬 | 비대각 원소 = 0 | 역행렬은 대각 원소의 역수 |
| 단위 행렬 | $I$: 대각 원소 = 1 | $AI = IA = A$ |

**AI에서의 사용**

- **순전파**: $\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ — 신경망 각 층의 본체
- **역전파**: $W^\top \delta$ 형태로 그레이디언트 역방향 전파
- **Attention**: $\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ — 전체가 행렬 곱의 연쇄
- **배치 연산**: 미니배치를 행렬로 묶어 한 번에 처리 → GPU 병렬화

---

### 5. 선형 독립, 기저, 차원, 랭크

**선형 독립 (Linear Independence)**

벡터 집합 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$가 선형 독립이면:

$$
c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \Rightarrow c_1 = \cdots = c_k = 0
$$

**기저 (Basis)와 차원 (Dimension)**

- 기저: 벡터 공간을 생성하는 선형 독립 벡터들의 집합
- 차원: 기저 벡터의 개수. $n$차원 공간은 정확히 $n$개의 기저 벡터가 필요

**랭크 (Rank)**

$$
\text{rank}(A) + \text{nullity}(A) = n \quad (\text{열 수 기준})
$$

- **랭크**: 행 공간(= 열 공간)의 차원 — 행렬이 실제로 표현하는 독립 정보의 수
- **풀 랭크**: $\text{rank}(A) = \min(m, n)$ — 역행렬 또는 최소 제곱해 존재 조건

**AI에서의 사용**

- **저랭크 가정**: 실제 가중치 행렬의 유효 랭크는 낮다 → LoRA, SVD로 압축 가능
- **임베딩 차원 선택**: 정보를 충분히 표현하면서 과적합을 막을 최소 차원 탐색

---

### 6. 내적 공간과 직교성

**정규 직교 기저 (Orthonormal Basis)**

$$
\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}
$$

**그람-슈미트 과정**: 임의의 기저를 정규 직교 기저로 변환.

$$
\mathbf{u}_k = \mathbf{s}_k - \sum_{j=1}^{k-1} \frac{\langle \mathbf{s}_k, \mathbf{u}_j \rangle}{\|\mathbf{u}_j\|^2} \mathbf{u}_j
$$

**직교 행렬**

$$
AA^\top = A^\top A = I \Rightarrow A^{-1} = A^\top, \quad |\det(A)| = 1
$$

직교 행렬은 내적을 보존하는 변환이다 — 회전과 반사.

**QR 분해**

$$
A = QR, \quad Q \in \mathbb{R}^{m \times m} \text{: 직교 행렬},\ R \in \mathbb{R}^{m \times n} \text{: 상 삼각 행렬}
$$

**AI에서의 사용**

- **직교 초기화**: 가중치를 직교 행렬로 초기화 → 그레이디언트 소실/폭발 방지
- **정규화 기법**: 배치 정규화·레이어 정규화의 수학적 기반

---

### 7. 고윳값과 고유 벡터

**수학적 정의**

$$
A\mathbf{x} = \lambda\mathbf{x}
$$

- $\mathbf{x}$: 고유 벡터 — 선형 변환 후 **방향이 변하지 않는** 벡터
- $\lambda$: 고윳값 — 변환 후 크기 변화 배수

**특성 방정식**으로 고윳값 계산:

$$
\det(A - \lambda I) = 0
$$

**대각합·행렬식과 고윳값의 관계**

$$
\text{tr}(A) = \sum_i \lambda_i, \quad \det(A) = \prod_i \lambda_i
$$

**고윳값 분해 (Eigenvalue Decomposition)**

대칭 행렬 $A = A^\top$에 대해:

$$
A = PDP^\top, \quad D = \text{diag}(\lambda_1, \ldots, \lambda_n), \quad P^\top P = I
$$

**양정치 행렬 (Positive Definite)**

$$
\mathbf{x}^\top A\mathbf{x} > 0 \text{ for all } \mathbf{x} \neq \mathbf{0} \Leftrightarrow \text{모든 고윳값} > 0
$$

준양정치(PSD): 고윳값 $\geq 0$.

**AI에서의 사용**

- **PCA**: 공분산 행렬 $\Sigma$의 고유 벡터 = 데이터 분산이 최대인 방향 → 차원 축소
- **헤시안 분석**: 손실 함수의 2차 도함수 행렬의 고윳값으로 곡률 파악 → 안장점·극소 판별
- **공분산·커널 행렬**: 반드시 양반정치(PSD) — SVR, Gaussian Process에서 중요

---

### 8. 특이값 분해 (SVD)

**수학적 정의**

임의의 $m \times n$ 행렬을 세 행렬의 곱으로 분해:

$$
A = U\Sigma V^\top
$$

- $U \in \mathbb{R}^{m \times m}$: 좌 특이 벡터 — $AA^\top$의 고유 벡터로 구성된 직교 행렬
- $\Sigma \in \mathbb{R}^{m \times n}$: 특이값 대각 행렬 ($\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$)
- $V \in \mathbb{R}^{n \times n}$: 우 특이 벡터 — $A^\top A$의 고유 벡터로 구성된 직교 행렬

고윳값과의 관계: $\sigma_i = \sqrt{\lambda_i}$ ($\lambda_i$: $A^\top A$의 고윳값)

$$
AA^\top = U\Sigma\Sigma^\top U^\top, \quad A^\top A = V\Sigma^\top\Sigma V^\top
$$

**저랭크 근사 (Eckart-Young 정리)**

상위 $k$개 특이값만 사용하는 것이 프로베니우스 노름 기준 최적 랭크-$k$ 근사다:

$$
A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = \argmin_{\text{rank}(B)=k} \|A - B\|_F
$$

**AI에서의 사용**

- **PCA**: SVD의 직접 응용 — $V$의 열벡터가 주성분 방향, $\Sigma$의 대각 원소가 분산 크기
- **LoRA**: $W \approx W_0 + AB$, $A \in \mathbb{R}^{m \times r}$, $B \in \mathbb{R}^{r \times n}$, $r \ll \min(m,n)$ — 저랭크 가정으로 LLM 효율적 미세조정
- **추천 시스템**: 사용자-아이템 행렬 SVD → 잠재 요인 분해 (Matrix Factorization)
- **LSA**: 문서-단어 행렬 SVD → 잠재 의미 분석

고윳값 분해는 정사각 대칭 행렬에만 적용 가능하지만, SVD는 임의 크기의 직사각 행렬 모두에 적용된다. 딥러닝의 가중치 행렬은 대부분 직사각 행렬이므로 SVD가 더 범용적이다.

---

### 9. 선형 변환과 딥러닝

**선형 변환 (Linear Transformation)**

$$
f(\mathbf{x}) = W\mathbf{x}: \quad \mathbb{R}^n \to \mathbb{R}^m
$$

덧셈과 스칼라 곱을 보존한다: $f(\mathbf{u} + \mathbf{v}) = f(\mathbf{u}) + f(\mathbf{v})$.

**아핀 변환 (Affine Transformation)**

$$
f(\mathbf{x}) = W\mathbf{x} + \mathbf{b}
$$

신경망의 각 층은 **아핀 변환 + 비선형 활성 함수**의 반복이다. 활성 함수 없이 층을 아무리 쌓아도 결국 하나의 선형 변환이기 때문에 비선형 활성 함수가 필수다.

**벡터 미분**

$$
\frac{\partial (\mathbf{w}^\top \mathbf{x})}{\partial \mathbf{x}} = \mathbf{w}, \quad \frac{\partial (\mathbf{x}^\top A\mathbf{x})}{\partial \mathbf{x}} = 2A\mathbf{x} \quad (A \text{: 대칭})
$$

역전파에서 그레이디언트를 계산하는 직접적인 기반이다.

깊은 신경망은 고차원 데이터 매니폴드를 층마다 선형 변환 + 비선형 왜곡으로 점진적으로 변환하여, 최종적으로 클래스가 선형 분리 가능한 표현 공간을 만든다.

---

## 수식 정리

**내적과 코사인 유사도**

$$
\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^\top \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta, \quad \cos\theta = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}
$$

**$L_p$ 노름**

$$
\|\mathbf{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}
$$

**행렬 곱 (순전파)**

$$
\mathbf{z} = W\mathbf{x} + \mathbf{b}, \quad W \in \mathbb{R}^{m \times n}
$$

**고윳값 방정식과 고윳값 분해**

$$
A\mathbf{x} = \lambda\mathbf{x}, \quad \det(A - \lambda I) = 0, \quad A = PDP^\top
$$

$$
\text{tr}(A) = \sum_i \lambda_i, \quad \det(A) = \prod_i \lambda_i
$$

**SVD와 저랭크 근사**

$$
A = U\Sigma V^\top, \quad A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

**LoRA**

$$
W \approx W_0 + AB, \quad A \in \mathbb{R}^{m \times r},\ B \in \mathbb{R}^{r \times n},\ r \ll \min(m,n)
$$

**벡터 미분**

$$
\frac{\partial (\mathbf{w}^\top \mathbf{x})}{\partial \mathbf{x}} = \mathbf{w}, \quad \frac{\partial (\mathbf{x}^\top A\mathbf{x})}{\partial \mathbf{x}} = 2A\mathbf{x}
$$
