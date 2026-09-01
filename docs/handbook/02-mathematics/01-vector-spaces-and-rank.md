# 벡터 공간, 기저, 차원, 랭크

## 한 줄 정의

벡터 공간은 덧셈과 스칼라 곱이 닫혀 있는 집합이고, 기저는 그 공간을 낭비 없이 생성하는 최소 벡터 집합이며, 랭크는 행렬이 실제로 도달할 수 있는 공간의 차원이다.

## 문제 상황

신경망의 선형 계층 $W \in \mathbb{R}^{d_{out} \times d_{in}}$은 입력 공간의 벡터를 출력 공간으로 보낸다. 이때 자연스럽게 나오는 질문들이 있다. 파라미터가 $d_{out} \times d_{in}$개인데 이 계층이 실제로 표현할 수 있는 변환의 자유도는 그만큼인가. 두 선형 계층을 연달아 쌓으면 표현력이 늘어나는가. 임베딩 행렬의 열들이 서로 얼마나 독립적인가.

이 질문들은 모두 랭크로 답한다. 그리고 답이 실무 결정을 바꾼다.

구체적으로 세 가지 상황이 있다. 첫째, LoRA는 $\Delta W$를 $BA$ 형태의 저랭크 행렬로 근사한다. 왜 랭크 8이면 충분한지, 언제 부족한지를 판단하려면 랭크가 무엇을 제한하는지 알아야 한다. 둘째, 임베딩 붕괴 문제다. 대조학습이 실패하면 모든 임베딩이 낮은 차원 부분공간에 몰리는데, 이를 진단하려면 유효 랭크를 측정해야 한다. 셋째, 선형회귀에서 특징 행렬의 랭크가 부족하면 정규방정식이 유일해를 갖지 않는다.

## 직관적 이해

벡터 공간은 "출발점에서 갈 수 있는 모든 곳"이다. 몇 개의 방향(기저 벡터)이 주어지면, 그 방향들을 적당히 섞어 도달할 수 있는 지점 전체가 공간이 된다.

기저는 중복 없는 방향들이다. 세 개의 방향이 주어졌는데 셋째가 앞의 둘의 조합이라면 셋째는 새로운 곳으로 데려다주지 못한다. 그런 중복을 제거하고 남은 것이 기저이고, 그 개수가 차원이다.

랭크는 행렬을 "변환"으로 볼 때 그 변환의 출력이 채우는 공간의 차원이다. $3 \times 3$ 행렬이 3차원 공간의 벡터를 받아 3차원 공간으로 보내지만, 랭크가 2라면 출력은 항상 어떤 평면 위에만 있다. 부피가 있는 물체를 평면에 눌러 붙인 것과 같다. 눌린 정보는 되돌릴 수 없다.

## 형식화

### 벡터 공간

체 $\mathbb{F}$(여기서는 $\mathbb{R}$) 위의 벡터 공간 $V$는 덧셈 $+: V \times V \to V$와 스칼라 곱 $\cdot : \mathbb{F} \times V \to V$를 갖추고 다음 여덟 공리를 만족하는 집합이다.

$$u + v = v + u, \quad (u+v)+w = u+(v+w), \quad \exists\, 0 : v + 0 = v, \quad \exists\, (-v) : v + (-v) = 0$$

$$a(bv) = (ab)v, \quad 1 \cdot v = v, \quad a(u+v) = au + av, \quad (a+b)v = av + bv$$

이 공리들이 보장하는 것은 "선형 조합이 항상 공간 안에 머문다"는 성질이다. 실무에서 이것이 의미하는 바는, 은닉 표현들의 가중 평균이 여전히 같은 표현 공간에 있다는 것이다. Mixup 증강과 임베딩 보간이 성립하는 근거다.

### 선형 독립과 기저

벡터 집합 $\{v_1, \dots, v_k\} \subset V$가 선형 독립이라는 것은 다음을 뜻한다.

$$\sum_{i=1}^{k} c_i v_i = 0 \implies c_1 = \cdots = c_k = 0$$

즉 자명한 조합으로만 영벡터를 만들 수 있다. 만약 어떤 $c_j \neq 0$인 조합이 0을 만든다면

$$v_j = -\frac{1}{c_j}\sum_{i \neq j} c_i v_i$$

이므로 $v_j$가 나머지의 조합으로 표현된다. 중복이 있다는 뜻이다.

생성(span)은 모든 선형 조합의 집합이다.

$$\text{span}\{v_1, \dots, v_k\} = \left\{ \sum_{i=1}^{k} c_i v_i : c_i \in \mathbb{R} \right\}$$

기저는 선형 독립이면서 공간 전체를 생성하는 집합이고, 차원 $\dim V$는 기저의 원소 개수다. 이 개수가 기저 선택과 무관하게 일정하다는 것이 차원 정리의 내용이다.

### 네 가지 기본 부분공간

행렬 $A \in \mathbb{R}^{m \times n}$은 네 개의 부분공간을 정의한다.

열공간은 출력이 도달할 수 있는 곳이다.

$$\text{Col}(A) = \{Ax : x \in \mathbb{R}^n\} \subseteq \mathbb{R}^m$$

영공간은 0으로 눌리는 입력들이다.

$$\text{Null}(A) = \{x \in \mathbb{R}^n : Ax = 0\} \subseteq \mathbb{R}^n$$

행공간과 좌영공간은 $A^\top$에 대해 같은 방식으로 정의된다.

랭크는 열공간의 차원이며, 행공간의 차원과 항상 같다.

$$\text{rank}(A) = \dim \text{Col}(A) = \dim \text{Row}(A)$$

이 등식이 자명하지 않다는 점을 짚어둔다. 열의 독립성과 행의 독립성이 같은 수라는 것은 증명이 필요한 사실이며, 뒤에 나올 SVD가 그 이유를 가장 명확히 보여준다.

### 랭크-영도 정리

가장 중요한 관계다.

$$\text{rank}(A) + \dim \text{Null}(A) = n$$

유도는 다음과 같다. $\text{Null}(A)$의 기저를 $\{u_1, \dots, u_k\}$라 하자($k = \dim \text{Null}(A)$). 이를 $\mathbb{R}^n$ 전체의 기저 $\{u_1, \dots, u_k, w_1, \dots, w_{n-k}\}$로 확장한다. 임의의 $x \in \mathbb{R}^n$은

$$x = \sum_{i=1}^{k} a_i u_i + \sum_{j=1}^{n-k} b_j w_j$$

로 쓰이고, $Au_i = 0$이므로

$$Ax = \sum_{j=1}^{n-k} b_j Aw_j$$

이다. 따라서 $\{Aw_1, \dots, Aw_{n-k}\}$가 $\text{Col}(A)$를 생성한다. 이들이 선형 독립임을 보이면 된다. $\sum_j c_j Aw_j = 0$이라 하면 $A(\sum_j c_j w_j) = 0$이므로 $\sum_j c_j w_j \in \text{Null}(A)$이고, 따라서 어떤 $a_i$에 대해

$$\sum_j c_j w_j = \sum_i a_i u_i$$

인데 $\{u_i\} \cup \{w_j\}$가 기저이므로 모든 계수가 0이다. 특히 $c_j = 0$이다. 결론적으로 $\dim \text{Col}(A) = n - k$이며 이것이 정리의 내용이다.

이 정리가 실무에서 말하는 바는 명확하다. 입력 차원 $n$이 고정일 때 랭크가 낮을수록 영공간이 크다. 영공간이 크다는 것은 서로 다른 입력이 같은 출력으로 뭉개진다는 뜻이다. 정보가 손실된다.

### 랭크의 부등식

곱의 랭크에 대한 기본 부등식이다.

$$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$$

증명은 열공간 포함 관계에서 나온다. $\text{Col}(AB) \subseteq \text{Col}(A)$는 $ABx = A(Bx)$이므로 자명하고, $\text{rank}(AB) = \text{rank}((AB)^\top) = \text{rank}(B^\top A^\top) \leq \text{rank}(B^\top) = \text{rank}(B)$가 두 번째 부등식이다.

이 부등식이 LoRA의 수학적 기반이다. $\Delta W = BA$에서 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$이면

$$\text{rank}(\Delta W) \leq r$$

이므로, 랭크 $r$을 고르는 것은 곧 업데이트가 도달할 수 있는 부분공간의 차원을 고르는 것이다. 파라미터 수는 $r(d+k)$로 $dk$보다 훨씬 작다.

부분가법성도 유용하다.

$$\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$$

사전학습 가중치 $W_0$의 랭크가 이미 $d$(전랭크)라면, 저랭크 업데이트를 더해도 랭크는 늘지 않는다. LoRA가 표현력을 제한하는 것은 랭크 자체가 아니라 업데이트 방향의 자유도다.

## 구현

### 랭크 계산

수치적으로 랭크를 구하는 방법은 특이값의 크기를 세는 것이다. 이론적 정의(선형 독립인 열의 최대 개수)를 그대로 구현하면 부동소수점 오차 때문에 거의 항상 전랭크가 나온다.

```python
import numpy as np


def numerical_rank(matrix: np.ndarray, tol: float | None = None):
    singular = np.linalg.svd(matrix, compute_uv=False)
    if tol is None:
        tol = max(matrix.shape) * np.finfo(matrix.dtype).eps * singular[0]
    return int((singular > tol).sum()), singular


rng = np.random.default_rng(0)

full = rng.normal(size=(6, 4))
rank_full, sv_full = numerical_rank(full)

left = rng.normal(size=(6, 2))
right = rng.normal(size=(2, 4))
low = left @ right
rank_low, sv_low = numerical_rank(low)

noisy = low + rng.normal(scale=1e-10, size=low.shape)
rank_noisy, sv_noisy = numerical_rank(noisy)

print(f"full-rank matrix: rank={rank_full}, singular={np.round(sv_full, 4)}")
print(f"rank-2 product:   rank={rank_low}, singular={np.round(sv_low, 4)}")
print(f"with tiny noise:  rank={rank_noisy}, singular={np.array2string(sv_noisy, precision=2)}")
```

노이즈를 넣으면 특이값이 정확히 0이 아니게 되지만 허용 오차 아래이므로 랭크는 그대로 2로 판정된다. 허용 오차의 기본값 $\max(m,n) \cdot \epsilon \cdot \sigma_1$은 반올림 오차의 크기 추정에서 나온 것이다.

앞의 랭크 부등식을 검증한다.

```python
import numpy as np


def rank_of(matrix: np.ndarray):
    return int(np.linalg.matrix_rank(matrix))


rng = np.random.default_rng(0)
a = rng.normal(size=(8, 3)) @ rng.normal(size=(3, 10))
b = rng.normal(size=(10, 4)) @ rng.normal(size=(4, 6))

print(f"rank(A)={rank_of(a)} rank(B)={rank_of(b)}")
print(f"rank(AB)={rank_of(a @ b)} <= min = {min(rank_of(a), rank_of(b))}")

c = rng.normal(size=(8, 2)) @ rng.normal(size=(2, 10))
print(f"rank(A+C)={rank_of(a + c)} <= {rank_of(a) + rank_of(c)}")
```

### 랭크-영도 정리 확인

정리를 수치적으로 검증한다.

```python
import numpy as np
from scipy.linalg import null_space


def verify_rank_nullity(matrix: np.ndarray):
    n = matrix.shape[1]
    r = int(np.linalg.matrix_rank(matrix))
    kernel = null_space(matrix)
    nullity = kernel.shape[1]
    return {"n": n, "rank": r, "nullity": nullity, "sum": r + nullity}


rng = np.random.default_rng(0)
cases = {
    "full column rank (8x4)": rng.normal(size=(8, 4)),
    "rank deficient (8x6, rank 3)": rng.normal(size=(8, 3)) @ rng.normal(size=(3, 6)),
    "wide (4x9)": rng.normal(size=(4, 9)),
}
for label, matrix in cases.items():
    print(f"{label}: {verify_rank_nullity(matrix)}")
```

넓은 행렬($m < n$)은 반드시 비자명한 영공간을 갖는다. 랭크가 최대 $m$인데 $n > m$이므로 영도가 최소 $n - m$이다. 입력 차원을 줄이는 모든 계층이 정보를 잃는다는 뜻이며, 오토인코더의 병목이 정확히 이 성질을 이용한다.

영공간 벡터가 실제로 0으로 매핑되는지 확인한다.

```python
import numpy as np
from scipy.linalg import null_space


rng = np.random.default_rng(0)
matrix = rng.normal(size=(5, 3)) @ rng.normal(size=(3, 8))
kernel = null_space(matrix)

print(f"nullspace shape: {kernel.shape}")
for j in range(min(3, kernel.shape[1])):
    residual = np.linalg.norm(matrix @ kernel[:, j])
    print(f"||A v_{j}|| = {residual:.2e}")

x = rng.normal(size=(8,))
x_shifted = x + 3.7 * kernel[:, 0]
print(f"||Ax - Ax'|| = {np.linalg.norm(matrix @ x - matrix @ x_shifted):.2e}")
```

영공간 방향으로 아무리 움직여도 출력이 변하지 않는다. 이것이 정보 손실의 구체적 형태다.

### 유효 랭크와 표현 붕괴

임베딩 붕괴 진단에 쓰는 지표들이다. 이산적 랭크는 노이즈에 민감해서 실무에서는 연속적 지표를 쓴다.

```python
import numpy as np


def effective_rank_metrics(embeddings: np.ndarray):
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    total = energy.sum()
    ratios = energy / total

    stable_rank = total / (singular[0] ** 2)
    nonzero = ratios[ratios > 1e-12]
    entropy = -(nonzero * np.log(nonzero)).sum()
    erank = float(np.exp(entropy))
    participation = float((total ** 2) / (energy ** 2).sum())
    cumulative = np.cumsum(ratios)
    dims_90 = int(np.searchsorted(cumulative, 0.90) + 1)

    return {
        "ambient_dim": embeddings.shape[1],
        "stable_rank": round(float(stable_rank), 3),
        "entropy_rank": round(erank, 3),
        "participation_ratio": round(participation, 3),
        "dims_for_90pct": dims_90,
    }


rng = np.random.default_rng(0)
dim = 64
n = 2000

healthy = rng.normal(size=(n, dim))

collapsed = rng.normal(size=(n, 3)) @ rng.normal(size=(3, dim))
collapsed = collapsed + rng.normal(scale=0.01, size=collapsed.shape)

anisotropic = rng.normal(size=(n, dim)) * np.exp(-np.arange(dim) / 8.0)

for label, emb in [("healthy", healthy), ("collapsed", collapsed), ("anisotropic", anisotropic)]:
    print(f"{label:14s} {effective_rank_metrics(emb)}")
```

세 지표가 각각 다른 것을 잡는다.

안정 랭크 $\|A\|_F^2 / \|A\|_2^2 = \sum_i \sigma_i^2 / \sigma_1^2$는 첫 방향이 얼마나 지배적인지를 본다. 값이 1에 가까우면 사실상 한 방향만 있다는 뜻이다.

엔트로피 랭크는 특이값 에너지 분포의 지수 엔트로피다. 에너지가 $k$개 방향에 균등하면 정확히 $k$가 나온다.

참여율 $(\sum \sigma_i^2)^2 / \sum \sigma_i^4$도 같은 계열이며 물리학에서 왔다.

대조학습 중 이 값들이 계속 떨어지면 붕괴가 진행 중이다. 손실만 보고 있으면 알 수 없다.

### 그람-슈미트와 QR

기저를 직교화하는 과정을 직접 구현해 수치 안정성 차이를 본다.

```python
import numpy as np


def gram_schmidt_classical(vectors: np.ndarray):
    n, k = vectors.shape
    basis = np.zeros((n, k))
    for j in range(k):
        v = vectors[:, j].copy()
        for i in range(j):
            v = v - (basis[:, i] @ vectors[:, j]) * basis[:, i]
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            continue
        basis[:, j] = v / norm
    return basis


def gram_schmidt_modified(vectors: np.ndarray):
    n, k = vectors.shape
    basis = np.zeros((n, k))
    work = vectors.copy().astype(np.float64)
    for j in range(k):
        norm = np.linalg.norm(work[:, j])
        if norm < 1e-12:
            continue
        basis[:, j] = work[:, j] / norm
        for i in range(j + 1, k):
            work[:, i] -= (basis[:, j] @ work[:, i]) * basis[:, j]
    return basis


def orthogonality_error(basis: np.ndarray):
    gram = basis.T @ basis
    return float(np.abs(gram - np.eye(gram.shape[0])).max())


rng = np.random.default_rng(0)
scale = np.diag(np.logspace(0, -8, 6))
ill_conditioned = rng.normal(size=(12, 6)) @ scale

classical = gram_schmidt_classical(ill_conditioned)
modified = gram_schmidt_modified(ill_conditioned)
q, _ = np.linalg.qr(ill_conditioned)

print(f"classical GS error: {orthogonality_error(classical):.3e}")
print(f"modified GS error:  {orthogonality_error(modified):.3e}")
print(f"householder QR:     {orthogonality_error(q):.3e}")
```

고전 그람-슈미트는 조건이 나쁜 행렬에서 직교성을 잃는다. 실무에서는 항상 `np.linalg.qr`을 쓴다. 이 함수는 하우스홀더 반사를 사용해 훨씬 안정적이다.

### 부분공간 사이의 각도

두 표현 공간이 얼마나 겹치는지를 측정한다. 층별 표현 비교나 파인튜닝 전후 비교에 쓴다.

```python
import numpy as np


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray):
    qa, _ = np.linalg.qr(basis_a)
    qb, _ = np.linalg.qr(basis_b)
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    clipped = np.clip(singular, -1.0, 1.0)
    return np.arccos(clipped)


def subspace_overlap(basis_a: np.ndarray, basis_b: np.ndarray):
    angles = principal_angles(basis_a, basis_b)
    return float(np.cos(angles).mean())


rng = np.random.default_rng(0)
shared = rng.normal(size=(50, 3))

space_a = np.hstack([shared, rng.normal(size=(50, 2))])
space_b = np.hstack([shared, rng.normal(size=(50, 2))])
space_c = rng.normal(size=(50, 5))

print(f"angles(A, B) = {np.round(np.degrees(principal_angles(space_a, space_b)), 2)}")
print(f"angles(A, C) = {np.round(np.degrees(principal_angles(space_a, space_c)), 2)}")
print(f"overlap(A, B) = {subspace_overlap(space_a, space_b):.4f}")
print(f"overlap(A, C) = {subspace_overlap(space_a, space_c):.4f}")
```

공유 방향이 3개인 두 공간은 처음 세 주각이 0에 가깝다. 완전히 무관한 공간은 모든 각이 크다.

### 저랭크 파라미터화

LoRA의 랭크 제약을 직접 확인한다.

```python
import numpy as np


def lora_update(d_out: int, d_in: int, rank: int, alpha: float, rng: np.random.Generator):
    a = rng.normal(scale=1.0 / np.sqrt(d_in), size=(rank, d_in))
    b = np.zeros((d_out, rank))
    b += rng.normal(scale=0.01, size=(d_out, rank))
    return (alpha / rank) * (b @ a)


rng = np.random.default_rng(0)
d_out, d_in = 512, 512

for rank in [1, 4, 8, 64, 512]:
    delta = lora_update(d_out, d_in, rank, alpha=16.0, rng=rng)
    params_full = d_out * d_in
    params_lora = rank * (d_out + d_in)
    print(
        f"r={rank:4d}  rank(ΔW)={int(np.linalg.matrix_rank(delta)):4d}  "
        f"params={params_lora:8d} ({params_lora / params_full:.3%} of full)"
    )
```

$B$를 정확히 0으로 초기화하면 처음에 $\Delta W = 0$이 되어 사전학습 모델과 동일하게 시작한다. 위 코드에서는 랭크를 확인하려고 작은 노이즈를 넣었다.

랭크 $r$에 필요한 파라미터가 $r(d_{out} + d_{in})$이므로, $r$이 $\min(d_{out}, d_{in})/2$를 넘으면 저랭크 분해가 이득이 없다. 512차원에서 랭크 256이 손익분기점이다. 실무에서 쓰는 8이나 16은 이보다 훨씬 작다.

## 실무 관점

랭크 판정에서 허용 오차 선택이 중요하다. `np.linalg.matrix_rank`의 기본값은 앞에서 본 $\max(m,n)\epsilon\sigma_1$인데, 데이터에 실제 노이즈가 있으면 이보다 훨씬 큰 값을 써야 한다. 노이즈 표준편차가 $\eta$인 $m \times n$ 행렬에서 노이즈만으로 생기는 특이값의 크기는 대략 $\eta(\sqrt{m} + \sqrt{n})$이므로, 이 값을 임계로 삼는 것이 합리적이다.

```python
import numpy as np


def noise_aware_rank(matrix: np.ndarray, noise_std: float):
    m, n = matrix.shape
    threshold = noise_std * (np.sqrt(m) + np.sqrt(n))
    singular = np.linalg.svd(matrix, compute_uv=False)
    return int((singular > threshold).sum()), threshold, singular[:6]


rng = np.random.default_rng(0)
signal = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 100))
noise_std = 0.5
observed = signal + rng.normal(scale=noise_std, size=signal.shape)

naive = int(np.linalg.matrix_rank(observed))
aware, threshold, top = noise_aware_rank(observed, noise_std)
print(f"naive rank: {naive}")
print(f"noise-aware rank: {aware} (threshold={threshold:.2f})")
print(f"top singular values: {np.round(top, 2)}")
```

기본 판정은 전랭크를 주지만 실제 신호 랭크는 5다.

임베딩 붕괴를 학습 중 감시한다. 배치마다 계산하면 비싸므로 주기적으로 부분 샘플에 대해 계산한다.

```python
import numpy as np


class RankMonitor:
    def __init__(self, sample_size: int = 512, every: int = 200):
        self.sample_size = sample_size
        self.every = every
        self.history: list[tuple[int, float]] = []

    def maybe_record(self, step: int, embeddings: np.ndarray, rng: np.random.Generator):
        if step % self.every != 0:
            return None
        n = embeddings.shape[0]
        idx = rng.choice(n, size=min(self.sample_size, n), replace=False)
        sample = embeddings[idx]
        centered = sample - sample.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        energy = singular ** 2
        ratios = energy / energy.sum()
        nonzero = ratios[ratios > 1e-12]
        erank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
        self.history.append((step, erank))
        return erank


rng = np.random.default_rng(0)
monitor = RankMonitor(every=100)
dim = 128
for step in range(0, 1000, 100):
    collapse_factor = max(1, int(dim * (1 - step / 1200)))
    latent = rng.normal(size=(1024, collapse_factor))
    emb = latent @ rng.normal(size=(collapse_factor, dim))
    value = monitor.maybe_record(step, emb, rng)
    print(f"step={step:4d} effective_rank={value:.2f}")
```

유효 랭크가 단조 감소하면 붕괴가 진행 중이다. 대응책은 대조학습에서 음성 샘플 수를 늘리거나, 분산 정규화 항을 추가하거나, 프로젝션 헤드 구조를 바꾸는 것이다.

선형 계층을 여러 개 쌓을 때 활성함수가 없으면 표현력이 늘지 않는다는 사실도 랭크 부등식의 결과다.

```python
import numpy as np


rng = np.random.default_rng(0)
w1 = rng.normal(size=(128, 256))
w2 = rng.normal(size=(64, 128))
w3 = rng.normal(size=(32, 64))

composed = w3 @ w2 @ w1
print(f"rank(W3 W2 W1) = {int(np.linalg.matrix_rank(composed))}")
print(f"bounded by min(32, 64, 128, 256) = {min(32, 64, 128, 256)}")

direct = rng.normal(size=(32, 256))
print(f"single layer rank = {int(np.linalg.matrix_rank(direct))}")
```

세 층의 합성이 단일 층 하나와 같은 표현력을 갖는다. 비선형성이 필요한 이유의 절반이 여기 있고, 나머지 절반은 Part 04에서 다룬다.

메모리와 계산 비용도 랭크로 정리된다. $m \times n$ 행렬을 랭크 $r$로 근사하면 저장 공간이 $mn$에서 $r(m+n)$으로, 벡터 곱 비용이 $O(mn)$에서 $O(r(m+n))$으로 줄어든다. $r \ll \min(m,n)$일 때만 이득이므로, 압축률은

$$\frac{r(m+n)}{mn}$$

로 계산한다. $4096 \times 4096$ 행렬에 랭크 16이면 0.78퍼센트다.

수치 계산 시 조건수를 함께 본다. 랭크가 형식적으로 전랭크여도 조건수가 크면 실질적으로 랭크 부족과 같은 문제가 생긴다.

```python
import numpy as np


rng = np.random.default_rng(0)
for exponent in [0, 4, 8, 12, 16]:
    scale = np.diag(np.logspace(0, -exponent, 5))
    matrix = rng.normal(size=(5, 5)) @ scale @ rng.normal(size=(5, 5))
    cond = np.linalg.cond(matrix)
    rank = int(np.linalg.matrix_rank(matrix))
    print(f"spread=1e-{exponent:2d}  cond={cond:.2e}  rank={rank}")
```

float64의 유효 자릿수가 약 16자리이므로 조건수가 $10^{16}$을 넘으면 해가 무의미해진다. float32에서는 $10^7$이 한계다. 딥러닝에서 float32나 bfloat16을 쓴다는 것은 이 여유가 훨씬 적다는 뜻이며, 정규화 계층이 조건수를 낮추는 역할을 겸한다는 관점이 여기서 나온다.

## 핵심 정리

랭크는 행렬이 도달할 수 있는 출력 공간의 차원이며, 열 랭크와 행 랭크가 항상 같다.

랭크-영도 정리 $\text{rank}(A) + \dim\text{Null}(A) = n$이 정보 손실을 정량화한다. 영공간 방향으로 움직여도 출력이 변하지 않는다.

$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$가 LoRA의 근거다. 랭크 $r$을 고르는 것은 업데이트가 도달할 부분공간의 차원을 고르는 것이며, 파라미터는 $dk$에서 $r(d+k)$로 줄어든다.

활성함수 없이 선형 계층을 쌓으면 랭크가 늘지 않는다. 합성이 단일 계층과 같은 표현력을 갖는다.

수치적 랭크는 특이값을 임계와 비교해 판정한다. 데이터에 노이즈가 있으면 기본 허용 오차가 아니라 $\eta(\sqrt{m}+\sqrt{n})$을 임계로 쓴다.

임베딩 붕괴 진단에는 이산 랭크가 아니라 안정 랭크, 엔트로피 랭크, 참여율 같은 연속 지표를 쓴다. 학습 중 단조 감소하면 붕괴다.

직교화는 항상 `np.linalg.qr`을 쓴다. 고전 그람-슈미트는 조건이 나쁜 입력에서 직교성을 잃는다.

조건수가 float 정밀도의 역수를 넘으면 형식적으로 전랭크여도 해가 무의미하다. float32에서는 $10^7$이 한계다.
