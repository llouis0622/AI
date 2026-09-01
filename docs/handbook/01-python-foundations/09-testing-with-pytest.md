# pytest 기반 테스트

## 한 줄 정의

pytest는 평범한 함수와 `assert` 문만으로 테스트를 작성하게 하고, 픽스처로 준비와 정리를 선언적으로 주입하며, 파라미터화로 같은 검증을 여러 입력에 확장하는 테스트 프레임워크다.

## 문제 상황

머신러닝 코드는 테스트하기 어렵다는 인식이 널리 퍼져 있다. 결과가 확률적이고, 학습에 시간이 오래 걸리며, GPU가 필요하고, "정답"이 무엇인지 명확하지 않기 때문이다. 그래서 테스트 없이 실험을 돌리고 성능 지표로만 판단하는 관행이 굳는다.

문제는 성능 지표가 버그를 잡아내지 못한다는 것이다. 데이터 증강이 레이블을 뒤섞고 있어도 정확도는 그럴듯하게 나온다. 마스킹이 한 칸 어긋나 미래 토큰이 새고 있어도 손실은 잘 떨어진다. 오히려 더 잘 떨어진다. 검증 세트에 학습 데이터가 섞여 있으면 지표가 좋아진다. 이런 버그는 지표를 보고 있으면 절대 발견되지 않는다.

두 번째 문제는 리팩터링이다. 전처리 함수를 벡터화해서 5배 빠르게 만들었는데 결과가 같은지 확인할 방법이 없으면 그 변경을 신뢰할 수 없다. 결국 아무도 코드를 고치지 않고 복사만 늘어난다.

세 번째는 회귀다. 어제 고친 버그가 다음 주에 되살아나도 아무도 모른다.

## 직관적 이해

ML 코드에서 테스트할 수 있는 것과 없는 것을 나누는 기준은 결정성이다. "이 모델이 92퍼센트 정확도를 낸다"는 테스트할 수 없지만 "동일 시드로 두 번 실행하면 같은 결과가 나온다"는 테스트할 수 있다. "이 어텐션 구현이 좋다"는 테스트할 수 없지만 "인과 마스크를 적용하면 미래 위치의 어텐션 가중치가 정확히 0이다"는 테스트할 수 있다.

즉 성능이 아니라 성질을 테스트한다. 형상 보존, 불변량, 수학적 항등식, 경계 조건, 결정성이 대상이다. 이것들은 모두 결정적이고 빠르며 명확한 정답이 있다.

픽스처는 재료 준비대다. 테스트 함수가 인자 이름으로 재료를 요청하면 pytest가 준비해서 건네주고, 끝나면 치운다. 앞 문서의 컨텍스트 매니저와 정확히 같은 구조이며 실제로 `yield` 픽스처가 그 문법을 쓴다.

## 형식화

테스트 하나가 주는 정보량을 생각해보자. 코드 변경이 버그를 유발할 확률을 $p$, 테스트가 그 버그를 잡아낼 확률을 $q$라 하면, 테스트를 통과했을 때 코드가 옳을 사후 확률은 베이즈 규칙으로 다음과 같다.

$$P(\text{정상} \mid \text{통과}) = \frac{(1-p)}{(1-p) + p(1-q)}$$

$q$가 1에 가까울수록 통과가 강한 증거가 된다. 반대로 $q$가 낮은 테스트(예: 함수가 예외 없이 실행되는지만 보는 테스트)는 통과해도 정보가 거의 없다. 테스트를 설계할 때 "이 테스트가 잡아낼 수 있는 버그가 무엇인가"를 먼저 묻는 이유다.

수치 비교의 허용 오차도 형식화해둘 가치가 있다. 부동소수점 연산 $n$번을 거친 결과의 상대 오차 상한은 대략

$$\frac{|\hat{y} - y|}{|y|} \lesssim n \cdot \epsilon_{\text{machine}} \cdot \kappa$$

이며 $\epsilon_{\text{machine}}$은 float32에서 약 $1.19 \times 10^{-7}$, float64에서 약 $2.22 \times 10^{-16}$이고 $\kappa$는 문제의 조건수다. 따라서 float32 행렬 곱 결과를 `atol=1e-9`로 비교하면 반드시 실패한다. 누적 연산 수와 정밀도에 맞춰 허용 오차를 정해야 한다.

수치 그래디언트 검증에서도 같은 계산이 필요하다. 중심 차분의 오차는

$$\frac{f(x+h) - f(x-h)}{2h} = f'(x) + \frac{h^2}{6}f'''(\xi) + O\left(\frac{\epsilon}{h}\right)$$

로 절단 오차 $O(h^2)$와 반올림 오차 $O(\epsilon/h)$의 합이다. 둘의 합을 최소화하는 $h$는

$$h^* \approx \epsilon^{1/3}$$

이며 float64에서 약 $6 \times 10^{-6}$이다. $h$를 무작정 작게 하면 오히려 나빠진다는 것이 여기서 나온다.

## 구현

### 첫 테스트

전처리 함수와 그 성질 테스트를 함께 만든다.

```python
import numpy as np


def standardize(values: np.ndarray, axis: int = 0, eps: float = 1e-8):
    mean = values.mean(axis=axis, keepdims=True)
    std = values.std(axis=axis, keepdims=True)
    return (values - mean) / (std + eps)


def train_val_split(indices: np.ndarray, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    cut = int(len(shuffled) * (1.0 - val_ratio))
    return shuffled[:cut], shuffled[cut:]
```

테스트는 `tests/test_preprocess.py`에 둔다.

```python
import numpy as np
import pytest

from handbook.preprocess import standardize, train_val_split


def test_standardize_preserves_shape():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(64, 8))
    assert standardize(values).shape == values.shape


def test_standardize_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=5.0, scale=3.0, size=(1000, 4))
    result = standardize(values)
    assert np.allclose(result.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(result.std(axis=0), 1.0, atol=1e-4)


def test_standardize_constant_column_does_not_explode():
    values = np.ones((10, 3))
    result = standardize(values)
    assert np.isfinite(result).all()
    assert np.allclose(result, 0.0)


def test_split_is_disjoint_and_complete():
    indices = np.arange(100)
    train, val = train_val_split(indices, val_ratio=0.2, seed=0)
    assert len(train) + len(val) == 100
    assert len(np.intersect1d(train, val)) == 0
    assert np.array_equal(np.sort(np.concatenate([train, val])), indices)


def test_split_is_deterministic():
    indices = np.arange(100)
    a = train_val_split(indices, 0.2, seed=42)
    b = train_val_split(indices, 0.2, seed=42)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_split_differs_across_seeds():
    indices = np.arange(100)
    a, _ = train_val_split(indices, 0.2, seed=1)
    b, _ = train_val_split(indices, 0.2, seed=2)
    assert not np.array_equal(a, b)
```

여섯 개 테스트가 각각 다른 버그를 겨냥한다. 형상 변경, 통계 오류, 0 나눗셈, 분할 누수, 시드 무시, 시드 고장이다. 앞의 형식화에서 말한 $q$를 높이는 방식이다.

허용 오차가 `mean`은 `1e-6`, `std`는 `1e-4`인 것이 자의적으로 보이지만 근거가 있다. 표본 1000개의 표준편차 추정 오차가 $1/\sqrt{2n} \approx 0.022$ 수준이고, `eps`를 더한 나눗셈이 값을 아주 약간 줄이기 때문이다.

### 픽스처

준비 코드를 픽스처로 뺀다. `tests/conftest.py`에 두면 모든 테스트에서 쓸 수 있다.

```python
import json
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def sample_matrix(rng):
    return rng.normal(size=(128, 16))


@pytest.fixture
def corpus_file(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"id": i, "text": "sample " * (i + 1)}) + "\n")
    return path


@pytest.fixture
def clean_env(monkeypatch):
    for key in ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONHASHSEED"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    return os.environ


@pytest.fixture
def deterministic():
    import random
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(0)
    np.random.seed(0)
    yield 0
    random.setstate(py_state)
    np.random.set_state(np_state)
```

`tmp_path`와 `monkeypatch`는 pytest 내장 픽스처다. `tmp_path`는 테스트마다 새 디렉터리를 만들고 자동 정리하며, `monkeypatch`는 환경 변수나 속성을 임시로 바꾸고 테스트 종료 시 복원한다. 앞 문서에서 손으로 만든 `env_override` 컨텍스트 매니저와 같은 일을 한다.

`deterministic` 픽스처가 `yield` 이후 상태를 복원하는 것이 중요하다. 복원하지 않으면 다음 테스트의 무작위성이 오염되어, 테스트 실행 순서에 따라 결과가 달라지는 최악의 상황이 된다.

스코프 선택 기준도 명확히 한다. `function`이 기본이며 격리가 완전하다. `session`은 비싼 준비(모델 로드, DB 연결)에만 쓰고, 반드시 불변 객체여야 한다. 가변 객체를 세션 스코프로 두면 한 테스트가 다른 테스트를 오염시킨다.

### 파라미터화

같은 성질을 여러 입력에서 확인한다.

```python
import numpy as np
import pytest

from handbook.preprocess import standardize


@pytest.mark.parametrize("shape", [(1, 1), (10, 1), (1, 10), (256, 64), (3, 3, 3)])
def test_standardize_shape_preserved(shape, rng):
    values = rng.normal(size=shape)
    assert standardize(values).shape == shape


@pytest.mark.parametrize(
    "dtype,tolerance",
    [
        (np.float32, 1e-4),
        (np.float64, 1e-10),
    ],
)
def test_standardize_precision(dtype, tolerance, rng):
    values = rng.normal(size=(500, 4)).astype(dtype)
    result = standardize(values)
    assert np.abs(result.mean(axis=0)).max() < tolerance


@pytest.mark.parametrize("val_ratio", [0.0, 0.1, 0.5, 0.9])
def test_split_ratio(val_ratio):
    from handbook.preprocess import train_val_split
    indices = np.arange(1000)
    train, val = train_val_split(indices, val_ratio, seed=0)
    assert abs(len(val) / 1000 - val_ratio) < 0.01
```

`dtype`별 허용 오차가 다른 것이 앞의 형식화에서 유도한 내용이다. float32에 float64용 허용 오차를 쓰면 반드시 실패한다.

여러 파라미터를 조합하면 곱집합이 된다.

```python
import pytest


@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.parametrize("seq_len", [1, 16, 512])
@pytest.mark.parametrize("heads", [1, 8])
def test_attention_shape(batch, seq_len, heads):
    import numpy as np
    dim = 64
    q = np.zeros((batch, heads, seq_len, dim))
    scores = np.einsum("bhqd,bhkd->bhqk", q, q)
    assert scores.shape == (batch, heads, seq_len, seq_len)
```

18개 조합이 생성된다. 경계값(1)을 반드시 포함하는 것이 중요하다. 배치 크기 1이나 시퀀스 길이 1에서 차원이 사라지는 버그가 매우 흔하다.

`id`를 붙이면 실패 메시지가 읽기 좋아진다.

```python
import pytest


@pytest.mark.parametrize(
    "reduction,expected",
    [
        pytest.param("mean", 2.0, id="mean-reduces-to-scalar"),
        pytest.param("sum", 6.0, id="sum-reduces-to-scalar"),
    ],
)
def test_reduction(reduction, expected):
    import numpy as np
    values = np.array([1.0, 2.0, 3.0])
    result = values.mean() if reduction == "mean" else values.sum()
    assert result == pytest.approx(expected)
```

### 수치 검증

수학적 항등식은 강력한 테스트다. 앞의 소프트맥스 구현을 검증한다.

```python
import numpy as np
import pytest


def softmax(scores: np.ndarray, axis: int = -1):
    shifted = scores - scores.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def test_softmax_sums_to_one():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(32, 10)) * 5
    probs = softmax(scores)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6)


def test_softmax_shift_invariance():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(8, 5))
    shift = rng.normal(size=(8, 1)) * 100
    assert np.allclose(softmax(scores), softmax(scores + shift), atol=1e-6)


def test_softmax_no_overflow_on_large_inputs():
    scores = np.array([[1000.0, 1001.0, 999.0]])
    probs = softmax(scores)
    assert np.isfinite(probs).all()
    assert probs.argmax() == 1


def test_softmax_uniform_on_equal_scores():
    scores = np.zeros((4, 7))
    assert np.allclose(softmax(scores), 1.0 / 7)
```

이동 불변성 테스트가 특히 유용하다. 이 성질이 깨졌다면 최댓값 빼기가 빠졌거나 축을 잘못 잡은 것이다.

그래디언트 검증은 형식화에서 유도한 최적 $h$를 쓴다.

```python
import numpy as np


def numeric_grad(fn, x: np.ndarray, h: float = 6e-6):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + h
        plus = fn(x)
        x[idx] = original - h
        minus = fn(x)
        x[idx] = original
        grad[idx] = (plus - minus) / (2 * h)
        it.iternext()
    return grad


def mse_loss(pred: np.ndarray, target: np.ndarray):
    return float(((pred - target) ** 2).mean())


def mse_grad(pred: np.ndarray, target: np.ndarray):
    return 2.0 * (pred - target) / pred.size


def test_mse_gradient_matches_numeric():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(6, 3))
    target = rng.normal(size=(6, 3))
    analytic = mse_grad(pred, target)
    numeric = numeric_grad(lambda p: mse_loss(p, target), pred.copy())
    rel = np.abs(analytic - numeric) / (np.abs(analytic) + np.abs(numeric) + 1e-12)
    assert rel.max() < 1e-6
```

절대 오차가 아니라 상대 오차를 쓴 것이 핵심이다. 그래디언트 크기가 성분마다 크게 다를 때 절대 오차 기준은 의미가 없다.

### 마커와 선택 실행

느리거나 GPU가 필요한 테스트를 분리한다.

```python
import pytest


@pytest.mark.slow
def test_full_epoch_runs():
    total = sum(i * i for i in range(5_000_000))
    assert total > 0


@pytest.mark.gpu
def test_cuda_available():
    pytest.importorskip("torch")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no cuda device")
    x = torch.randn(4, 4, device="cuda")
    assert x.sum().item() == pytest.approx(x.cpu().sum().item(), abs=1e-4)


@pytest.mark.parametrize("n", [10, 100])
def test_fast(n):
    assert sum(range(n)) == n * (n - 1) // 2
```

실행은 다음과 같이 나눈다.

```bash
pytest -m "not slow and not gpu"
pytest -m slow
pytest -m gpu
```

CI에서 커밋마다 도는 것은 첫 번째이고, 야간 빌드에서 나머지를 돈다. 개발 중 반복 실행되는 테스트 묶음이 10초를 넘으면 아무도 돌리지 않게 된다.

`pytest.importorskip`은 선택적 의존성이 없을 때 실패가 아니라 건너뛰기로 처리한다.

### 예외와 경고 테스트

실패 경로도 명세의 일부다.

```python
import numpy as np
import pytest


class ShapeMismatchError(ValueError):
    pass


def batched_dot(a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        raise ShapeMismatchError(f"expected same shape, got {a.shape} and {b.shape}")
    if a.ndim != 2:
        raise ShapeMismatchError(f"expected 2D, got {a.ndim}D")
    return (a * b).sum(axis=1)


def test_raises_on_shape_mismatch():
    a = np.zeros((4, 3))
    b = np.zeros((4, 5))
    with pytest.raises(ShapeMismatchError, match=r"got \(4, 3\) and \(4, 5\)"):
        batched_dot(a, b)


def test_raises_on_wrong_ndim():
    a = np.zeros((4, 3, 2))
    with pytest.raises(ShapeMismatchError, match="expected 2D"):
        batched_dot(a, a)


def test_warns_on_deprecated():
    import warnings

    def legacy():
        warnings.warn("use batched_dot", DeprecationWarning, stacklevel=2)
        return 0

    with pytest.warns(DeprecationWarning, match="batched_dot"):
        legacy()
```

`match`로 메시지까지 확인하는 것이 중요하다. 타입만 맞으면 통과하는 테스트는 엉뚱한 이유로 난 예외도 통과시킨다.

### 학습 코드 특유의 테스트

ML 코드에서 실제로 값이 큰 테스트들이다.

```python
import numpy as np
import pytest


def causal_mask(seq_len: int):
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


def masked_attention(scores: np.ndarray):
    seq_len = scores.shape[-1]
    mask = causal_mask(seq_len)
    scores = np.where(mask, -np.inf, scores)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def test_causal_mask_blocks_future():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(4, 4))
    weights = masked_attention(scores)
    for i in range(4):
        for j in range(i + 1, 4):
            assert weights[i, j] == 0.0


def test_causal_attention_rows_sum_to_one():
    rng = np.random.default_rng(0)
    weights = masked_attention(rng.normal(size=(6, 6)))
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_future_tokens_do_not_affect_past():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(5, 5))
    baseline = masked_attention(scores)
    perturbed = scores.copy()
    perturbed[:, 4] += 100.0
    changed = masked_attention(perturbed)
    assert np.allclose(baseline[:4, :4], changed[:4, :4])
```

세 번째 테스트가 정보 누수를 직접 겨냥한다. 마지막 위치의 점수를 크게 바꿔도 앞 위치들의 출력이 변하지 않아야 한다. 마스킹이 한 칸 어긋나면 이 테스트가 실패한다. 성능 지표로는 절대 발견되지 않는 버그다.

한 배치 과적합 테스트는 모델 정의와 손실이 연결되었는지 확인하는 가장 강력한 통합 테스트다.

```python
import numpy as np
import pytest


def train_tiny_model(steps: int = 300, lr: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(8, 4))
    labels = rng.integers(0, 2, size=(8,))
    weights = np.zeros((4,))
    bias = 0.0
    losses = []
    for _ in range(steps):
        logits = features @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        loss = -np.mean(labels * np.log(probs + 1e-12) + (1 - labels) * np.log(1 - probs + 1e-12))
        losses.append(loss)
        error = probs - labels
        weights -= lr * (features.T @ error) / len(labels)
        bias -= lr * error.mean()
    return losses


def test_overfits_single_batch():
    losses = train_tiny_model()
    assert losses[-1] < 0.05
    assert losses[-1] < losses[0]


def test_loss_decreases_monotonically_on_average():
    losses = train_tiny_model()
    first_half = np.mean(losses[:150])
    second_half = np.mean(losses[150:])
    assert second_half < first_half
```

작은 배치를 과적합시키지 못하는 모델은 큰 데이터에서도 학습되지 않는다. 이 테스트가 실패하면 학습률, 손실 정의, 그래디언트 계산 중 하나가 잘못된 것이다.

### 회귀 테스트

성능 저하를 잡는 테스트는 기준값을 저장해두고 비교한다.

```python
import json
from pathlib import Path

import numpy as np
import pytest


def compute_metrics(seed: int = 0):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(200,))
    labels = (scores + rng.normal(scale=0.5, size=(200,)) > 0).astype(int)
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)
    auc = float(np.trapezoid(tpr, fpr))
    return {"auc": round(auc, 6), "positives": int(labels.sum())}


def test_metrics_match_baseline(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(compute_metrics()), encoding="utf-8")

    current = compute_metrics()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert current["positives"] == baseline["positives"]
    assert abs(current["auc"] - baseline["auc"]) < 1e-6
```

실제로는 기준값 파일을 저장소에 커밋한다. 값이 정당하게 바뀌었으면 명시적으로 갱신하고, 그 갱신이 리뷰 대상이 된다.

## 실무 관점

테스트 실행 시간이 개발 습관을 결정한다. 저장할 때마다 돌리는 묶음은 5초 이내여야 한다. `pytest-xdist`로 병렬 실행하면 코어 수만큼 줄어들지만, 픽스처 스코프가 세션인 경우 워커마다 중복 생성된다는 점을 고려한다.

```bash
pytest -n auto -m "not slow"
```

실패 시 빠르게 멈추는 옵션도 유용하다.

```bash
pytest -x --ff
```

`-x`는 첫 실패에서 중단, `--ff`는 지난번 실패한 테스트를 먼저 실행한다. 수정과 확인의 반복 주기가 짧아진다.

커버리지는 참고 지표이지 목표가 아니다. 100퍼센트 커버리지가 버그 없음을 뜻하지 않는다. 앞의 형식화에서 $q$가 낮은 테스트를 많이 써도 커버리지는 올라간다. 다만 커버리지 리포트로 아예 실행되지 않는 코드 경로를 찾는 것은 가치가 있다.

```bash
pytest --cov=handbook --cov-report=term-missing --cov-fail-under=70
```

무작위성 처리는 명확한 규칙을 둔다. 테스트 안에서 시드를 고정하되, 시드에 의존하지 않는 성질을 검증한다. 특정 시드에서만 통과하는 테스트는 성질이 아니라 우연을 검증하는 것이다. 여러 시드로 파라미터화하면 이를 확인할 수 있다.

```python
import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_property_holds_across_seeds(seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(100, 5))
    centered = values - values.mean(axis=0)
    assert np.abs(centered.mean(axis=0)).max() < 1e-12
```

부동소수점 비교에 `==`를 쓰지 않는다. `pytest.approx` 또는 `np.allclose`를 쓰고, 허용 오차는 정밀도와 연산 수를 근거로 정한다. `np.allclose`의 기본값은 `rtol=1e-5, atol=1e-8`이며 float32 결과에는 적절하지만 float64 정밀 검증에는 느슨하다.

파일 시스템과 네트워크에 의존하지 않는다. `tmp_path`로 격리하고 외부 호출은 대역으로 대체한다.

```python
import pytest


class FakeStorage:
    def __init__(self):
        self.saved: dict[str, bytes] = {}

    def put(self, key: str, blob: bytes):
        self.saved[key] = blob

    def get(self, key: str):
        if key not in self.saved:
            raise KeyError(key)
        return self.saved[key]


@pytest.fixture
def storage():
    return FakeStorage()


def save_checkpoint(storage, step: int, payload: bytes):
    storage.put(f"ckpt/step-{step:08d}.bin", payload)
    return f"ckpt/step-{step:08d}.bin"


def test_checkpoint_key_is_zero_padded(storage):
    key = save_checkpoint(storage, 42, b"weights")
    assert key == "ckpt/step-00000042.bin"
    assert storage.get(key) == b"weights"
```

제로 패딩을 테스트한 이유는 사전순 정렬이 필요하기 때문이다. `step-9`와 `step-10`은 사전순으로 뒤집힌다. 체크포인트 목록에서 최신 것을 고를 때 실제로 문제가 된다.

테스트가 실패했을 때 원인을 즉시 알 수 있게 만든다. pytest는 `assert` 표현식을 재작성해 중간값을 보여주지만, 복잡한 조건은 여전히 읽기 어렵다. 단언을 쪼개고 메시지를 붙인다.

```python
import numpy as np


def test_with_helpful_message():
    rng = np.random.default_rng(0)
    result = rng.normal(size=(4, 8))
    assert result.ndim == 2, f"expected 2D, got shape {result.shape}"
    assert result.shape[1] == 8, f"expected 8 features, got {result.shape[1]}"
    assert np.isfinite(result).all(), f"found {(~np.isfinite(result)).sum()} non-finite values"
```

pytest 실행 자체를 CI 게이트로 만든다. 테스트가 실패해도 병합이 가능하면 테스트는 곧 방치된다.

## 핵심 정리

ML 코드에서 테스트하는 것은 성능이 아니라 성질이다. 형상 보존, 불변량, 수학적 항등식, 경계 조건, 결정성이 대상이며 모두 결정적이고 빠르다.

테스트를 설계할 때 "이 테스트가 잡아낼 수 있는 버그가 무엇인가"를 먼저 묻는다. 예외 없이 실행되는지만 보는 테스트는 통과해도 정보가 거의 없다.

허용 오차는 정밀도와 누적 연산 수를 근거로 정한다. float32에 float64용 오차를 쓰면 반드시 실패한다. 수치 그래디언트의 최적 스텝은 $\epsilon^{1/3}$이며 float64에서 약 $6 \times 10^{-6}$이다.

픽스처는 상태를 반드시 복원한다. 복원하지 않으면 테스트 실행 순서에 따라 결과가 달라진다.

파라미터화에는 경계값(크기 1, 빈 입력, 단일 클래스)을 반드시 포함한다. 차원이 사라지는 버그가 여기서 잡힌다.

마스킹 검증은 "미래 위치를 바꿔도 과거 출력이 변하지 않는다"로 한다. 정보 누수를 직접 겨냥하며 성능 지표로는 발견되지 않는다.

한 배치 과적합 테스트는 모델과 손실의 연결을 확인하는 가장 강력한 통합 테스트다.

빠른 테스트와 느린 테스트를 마커로 분리한다. 개발 중 반복 실행되는 묶음이 10초를 넘으면 아무도 돌리지 않는다.
