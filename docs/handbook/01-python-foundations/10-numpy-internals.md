# NumPy: 메모리 레이아웃과 브로드캐스팅

## 한 줄 정의

ndarray는 연속된 메모리 블록 하나와 그 블록을 어떻게 읽을지 지시하는 형상, 스트라이드, dtype의 조합이며, NumPy의 거의 모든 성능 특성과 버그가 이 세 메타데이터에서 나온다.

## 문제 상황

순수 Python 리스트로 수치 계산을 하면 두 종류의 비용이 발생한다. 첫째, 각 원소가 별개의 Python 객체라서 값 하나당 헤더 오버헤드가 붙는다. `float` 하나가 24바이트, 그 포인터가 8바이트다. 실제 데이터는 8바이트인데 32바이트를 쓴다. 둘째, 연산마다 객체 언박싱, 타입 확인, 재박싱이 일어난다. 100만 개 덧셈이 100만 번의 동적 디스패치가 된다.

NumPy가 이를 해결한 뒤에도 남는 문제가 있다. NumPy를 쓰면서도 파이썬 루프로 원소를 하나씩 건드리면 이득이 사라진다. 오히려 리스트보다 느려지는데, 인덱싱마다 스칼라를 배열에서 꺼내 Python 객체로 감싸기 때문이다.

세 번째 문제는 조용한 오류다. 뷰와 복사본을 구분하지 못하면 원본이 예기치 않게 수정된다. 브로드캐스팅 규칙을 잘못 이해하면 `(B,)`와 `(B, 1)`의 뺄셈이 `(B, B)`가 되어 손실이 틀려도 예외가 나지 않는다. 형상이 맞아떨어져 버리는 경우가 최악이다.

## 직관적 이해

ndarray를 두 부분으로 나눠 생각한다. 하나는 값이 죽 늘어선 창고(버퍼)이고, 다른 하나는 "몇 번 칸부터 몇 칸씩 건너뛰며 읽어라"는 지시서다. 지시서만 바꾸면 창고를 건드리지 않고도 전치, 슬라이싱, 차원 추가가 된다. 그래서 이런 연산이 공짜다.

반대로 지시서만으로 표현할 수 없는 연산은 창고를 새로 만들어야 한다. 팬시 인덱싱으로 임의 순서의 원소를 뽑아내는 것이 그렇다. 규칙적인 건너뛰기로 표현할 수 없으므로 복사가 일어난다.

브로드캐스팅은 크기가 1인 축을 무한 반복하는 지시로 해석하는 것이다. 실제로 메모리를 복제하지 않고 스트라이드를 0으로 두면 같은 값을 계속 읽게 된다. 공짜로 확장되는 이유가 이것이다.

## 형식화

ndarray의 원소 접근은 다음 주소 계산으로 정의된다. 형상 $(n_0, \dots, n_{k-1})$, 스트라이드 $(s_0, \dots, s_{k-1})$인 배열의 인덱스 $(i_0, \dots, i_{k-1})$ 위치는

$$\text{addr}(i) = \text{base} + \sum_{d=0}^{k-1} i_d \cdot s_d$$

이며 스트라이드 단위는 바이트다. 이 식 하나로 여러 성질이 설명된다.

C 순서(행 우선)의 스트라이드는 다음과 같이 결정된다.

$$s_{k-1} = \text{itemsize}, \qquad s_d = s_{d+1} \cdot n_{d+1}$$

Fortran 순서(열 우선)는 반대다.

$$s_0 = \text{itemsize}, \qquad s_d = s_{d-1} \cdot n_{d-1}$$

전치는 형상과 스트라이드를 함께 뒤집는 것으로 끝난다. 데이터는 그대로다.

$$\text{shape}' = (n_{k-1}, \dots, n_0), \qquad \text{stride}' = (s_{k-1}, \dots, s_0)$$

슬라이싱 `a[start:stop:step]`은 base를 $\text{base} + \text{start} \cdot s_d$로 옮기고 스트라이드를 $s_d \cdot \text{step}$으로 바꾼다. 역시 복사가 없다.

브로드캐스팅 규칙은 두 형상을 오른쪽 정렬한 뒤 각 축에 대해 판정한다. 축 $d$에서

$$n_d^{(A)} = n_d^{(B)} \quad \text{또는} \quad n_d^{(A)} = 1 \quad \text{또는} \quad n_d^{(B)} = 1$$

중 하나가 성립해야 하고, 결과 축 크기는 $\max(n_d^{(A)}, n_d^{(B)})$다. 크기 1인 축은 스트라이드를 0으로 설정해 확장한다. 스트라이드 0이면 인덱스가 변해도 주소가 그대로이므로 같은 값을 반복해서 읽는다. 메모리 복제가 없는 이유가 이것이다.

성능 측면에서 메모리 바운드 연산의 시간은 이동해야 할 바이트 수로 결정된다. 원소 $n$개, 원소당 $b$바이트인 배열의 원소별 연산은

$$T \approx \frac{n \cdot b \cdot (r + w)}{\text{BW}}$$

이며 $r$은 읽는 배열 수, $w$는 쓰는 배열 수, BW는 메모리 대역폭이다. 이 식에서 두 가지가 따라온다. 첫째, 임시 배열을 만들 때마다 $w$가 늘어나므로 `a + b + c`가 `np.add(np.add(a, b), c)`보다 느리지 않게 하려면 in-place 연산이 필요하다. 둘째, float64 대신 float32를 쓰면 $b$가 절반이라 대략 2배 빨라진다.

캐시 효율도 스트라이드로 설명된다. 캐시 라인이 64바이트이므로, 마지막 축을 따라 순회하면(스트라이드가 itemsize) 한 번 읽을 때 8개 float64가 함께 온다. 첫 축을 따라 순회하면(스트라이드가 크면) 라인마다 하나만 쓰고 버린다. 순회 순서와 메모리 순서가 어긋나면 실측 대역폭이 8분의 1로 떨어진다.

## 구현

### 메타데이터 직접 확인

앞의 형식화를 코드로 확인한다.

```python
import numpy as np


def describe(arr: np.ndarray, label: str):
    print(f"{label}: shape={arr.shape} strides={arr.strides} "
          f"itemsize={arr.itemsize} contiguous(C)={arr.flags['C_CONTIGUOUS']} "
          f"contiguous(F)={arr.flags['F_CONTIGUOUS']} owns={arr.flags['OWNDATA']}")


base = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
describe(base, "base")
describe(base.T, "transpose")
describe(base[:, ::2, :], "strided slice")
describe(base.reshape(6, 4), "reshape")
describe(np.asfortranarray(base), "fortran")
```

`base`의 스트라이드가 `(96, 32, 8)`인 것이 형식화의 C 순서 공식과 일치한다. 마지막 축이 8(itemsize), 그 앞이 $8 \times 4 = 32$, 그 앞이 $32 \times 3 = 96$이다.

전치는 스트라이드가 뒤집힐 뿐 `OWNDATA`가 `False`다. 데이터를 공유한다.

주소 계산을 직접 해보면 이해가 굳는다.

```python
import numpy as np


def manual_address(arr: np.ndarray, index: tuple[int, ...]):
    offset = sum(i * s for i, s in zip(index, arr.strides))
    return offset // arr.itemsize


base = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
flat = base.ravel()
for idx in [(0, 0, 0), (0, 1, 2), (1, 2, 3)]:
    linear = manual_address(base, idx)
    print(f"{idx} -> flat[{linear}] = {flat[linear]}, actual = {base[idx]}")
```

### 뷰와 복사본

어느 연산이 복사를 만드는지가 실무 버그의 핵심이다.

```python
import numpy as np


def shares_memory_with(result: np.ndarray, source: np.ndarray):
    return np.shares_memory(result, source)


source = np.arange(20, dtype=np.float64).reshape(4, 5)

cases = {
    "basic slice a[1:3]": source[1:3],
    "step slice a[::2]": source[::2],
    "transpose a.T": source.T,
    "reshape a.reshape(5,4)": source.reshape(5, 4),
    "ravel a.ravel()": source.ravel(),
    "flatten a.flatten()": source.flatten(),
    "fancy index a[[0,2]]": source[[0, 2]],
    "boolean mask a[a>10]": source[source > 10],
    "astype same dtype": source.astype(np.float64),
    "astype copy=False": source.astype(np.float64, copy=False),
    "expand_dims": np.expand_dims(source, 0),
    "broadcast_to": np.broadcast_to(source, (2, 4, 5)),
}

for label, result in cases.items():
    print(f"{label:35s} view={shares_memory_with(result, source)}")
```

기본 슬라이싱, 전치, reshape(가능한 경우), ravel, 차원 조작, broadcast_to는 뷰다. 팬시 인덱싱, 불리언 마스킹, flatten은 복사다.

경계 사례가 있다. `reshape`은 스트라이드로 표현 가능할 때만 뷰다.

```python
import numpy as np


source = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
transposed = source.transpose(1, 0, 2)

as_view = source.reshape(6, 4)
print(f"contiguous reshape is view: {np.shares_memory(as_view, source)}")

forced = transposed.reshape(6, 4)
print(f"non-contiguous reshape is view: {np.shares_memory(forced, source)}")
print(f"transposed is contiguous: {transposed.flags['C_CONTIGUOUS']}")
```

전치된 배열을 reshape하면 스트라이드로 표현할 수 없어 복사가 일어난다. `ravel`도 마찬가지이며, 반드시 뷰여야 한다면 `reshape(-1)` 대신 형상을 확인하거나 `np.ascontiguousarray`를 명시적으로 부른다.

뷰가 만드는 버그를 재현한다.

```python
import numpy as np


def normalize_inplace_wrong(matrix: np.ndarray):
    rows = matrix[:2]
    rows -= rows.mean(axis=0)
    return rows


original = np.arange(12, dtype=np.float64).reshape(4, 3)
snapshot = original.copy()
normalize_inplace_wrong(original)
print(f"original modified: {not np.array_equal(original, snapshot)}")
print(original)
```

함수가 반환값만 쓸 것처럼 보이지만 원본을 수정했다. 전처리 파이프라인에서 이런 함수가 하나 있으면 이후 단계가 오염된다. 방어는 명시적 복사다.

```python
import numpy as np


def normalize_safe(matrix: np.ndarray, rows: int):
    selected = matrix[:rows].copy()
    selected -= selected.mean(axis=0)
    return selected


original = np.arange(12, dtype=np.float64).reshape(4, 3)
snapshot = original.copy()
normalize_safe(original, 2)
print(f"original preserved: {np.array_equal(original, snapshot)}")
```

원본을 보호하려면 쓰기 금지 플래그를 걸 수도 있다.

```python
import numpy as np


frozen = np.arange(6, dtype=np.float64)
frozen.flags.writeable = False
try:
    frozen[0] = 1.0
except ValueError as exc:
    print(f"caught: {exc}")
```

캐시된 데이터셋 배열에 이 플래그를 걸어두면 실수로 수정하는 경로가 즉시 드러난다.

### 브로드캐스팅

규칙을 코드로 검증한다.

```python
import numpy as np


def broadcast_report(shape_a: tuple, shape_b: tuple):
    try:
        result = np.broadcast_shapes(shape_a, shape_b)
        return f"{shape_a} + {shape_b} -> {result}"
    except ValueError as exc:
        return f"{shape_a} + {shape_b} -> error: {exc}"


pairs = [
    ((4, 3), (3,)),
    ((4, 3), (4, 1)),
    ((4, 3), (1, 3)),
    ((4, 1), (1, 3)),
    ((32,), (32, 1)),
    ((8, 1, 6, 1), (7, 1, 5)),
    ((4, 3), (4,)),
]
for a, b in pairs:
    print(broadcast_report(a, b))
```

다섯 번째 항목이 앞에서 말한 사고다. `(32,)`와 `(32, 1)`을 더하면 `(32, 32)`가 나온다. 손실 계산에서 예측과 타깃의 형상이 이렇게 어긋나면 예외 없이 틀린 값이 만들어진다.

스트라이드 0으로 확장되는 것을 직접 확인한다.

```python
import numpy as np


column = np.arange(4, dtype=np.float64).reshape(4, 1)
expanded = np.broadcast_to(column, (4, 3))
print(f"shape={expanded.shape} strides={expanded.strides}")
print(f"shares memory: {np.shares_memory(expanded, column)}")
print(f"writeable: {expanded.flags.writeable}")
print(expanded)
```

마지막 축의 스트라이드가 0이다. 인덱스를 바꿔도 같은 주소를 읽는다. 쓰기가 금지된 이유도 명확하다. 한 위치에 쓰면 여러 인덱스가 동시에 바뀌기 때문이다.

브로드캐스팅 실수를 막는 방어 코드를 만든다.

```python
import numpy as np


def safe_subtract(left: np.ndarray, right: np.ndarray):
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")
    return left - right


predictions = np.arange(8, dtype=np.float64)
targets = np.arange(8, dtype=np.float64).reshape(8, 1)
print((predictions - targets).shape)
try:
    safe_subtract(predictions, targets)
except ValueError as exc:
    print(f"caught: {exc}")
```

손실 함수 진입부에 이런 검사를 두는 것이 앞의 타입 힌트 문서에서 말한 형상 단언이다.

### 벡터화

파이썬 루프와 벡터 연산의 차이를 측정한다.

```python
import time

import numpy as np


def timeit(fn, repeat: int = 3):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


rng = np.random.default_rng(0)
values = rng.normal(size=(1_000_000,))


def loop_version():
    total = 0.0
    for v in values:
        total += v * v
    return total


def list_comp_version():
    return sum(v * v for v in values)


def vector_version():
    return float((values * values).sum())


def einsum_version():
    return float(np.einsum("i,i->", values, values))


for name, fn in [
    ("python loop", loop_version),
    ("generator sum", list_comp_version),
    ("numpy vector", vector_version),
    ("numpy einsum", einsum_version),
]:
    print(f"{name:16s} {timeit(fn):.4f}s")
```

벡터 버전이 두 자릿수 배 빠르다. `einsum`은 임시 배열을 만들지 않아 벡터 버전보다도 빠른 경우가 많다. 형식화의 $w$가 줄기 때문이다.

임시 배열 제거 효과를 직접 본다.

```python
import time

import numpy as np


def timeit(fn, repeat: int = 5):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


rng = np.random.default_rng(0)
a = rng.normal(size=(4_000_000,))
b = rng.normal(size=(4_000_000,))
out = np.empty_like(a)


def with_temporaries():
    return a * 2.0 + b * 3.0


def in_place():
    np.multiply(a, 2.0, out=out)
    out += b * 3.0
    return out


def fully_fused():
    np.multiply(a, 2.0, out=out)
    np.add(out, b * 3.0, out=out)
    return out


print(f"temporaries: {timeit(with_temporaries):.4f}s")
print(f"partial in-place: {timeit(in_place):.4f}s")
print(f"out= everywhere: {timeit(fully_fused):.4f}s")
```

배열이 클수록 차이가 커진다. 4백만 개 float64면 32MB이므로 L3 캐시를 넘어 메인 메모리 대역폭이 병목이 된다.

dtype 선택 효과도 확인한다.

```python
import time

import numpy as np


def timeit(fn, repeat: int = 5):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


rng = np.random.default_rng(0)
for dtype in [np.float64, np.float32, np.float16]:
    arr = rng.normal(size=(8_000_000,)).astype(dtype)
    elapsed = timeit(lambda: float(arr.sum()))
    print(f"{np.dtype(dtype).name:9s} {arr.nbytes / 1024**2:6.1f} MiB  {elapsed:.4f}s")
```

float32가 float64의 절반 시간에 가깝다. float16은 하드웨어 지원이 없으면 오히려 느릴 수 있다. 딥러닝 데이터 파이프라인에서 float32를 기본으로 쓰는 이유가 여기에 있다.

### 순회 순서와 캐시

형식화에서 말한 캐시 효과를 측정한다.

```python
import time

import numpy as np


def timeit(fn, repeat: int = 3):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


matrix = np.zeros((4000, 4000), dtype=np.float64)

print(f"sum axis=1 (rows, cache friendly): {timeit(lambda: matrix.sum(axis=1)):.4f}s")
print(f"sum axis=0 (cols, cache hostile):  {timeit(lambda: matrix.sum(axis=0)):.4f}s")

fortran = np.asfortranarray(matrix)
print(f"F-order sum axis=0: {timeit(lambda: fortran.sum(axis=0)):.4f}s")
print(f"F-order sum axis=1: {timeit(lambda: fortran.sum(axis=1)):.4f}s")
```

C 순서에서는 축 1 방향 축약이, Fortran 순서에서는 축 0 방향이 빠르다. 배열 레이아웃과 접근 패턴을 맞추는 것이 무료로 얻는 성능이다.

### einsum과 고급 연산

배치 어텐션 점수 계산을 여러 방식으로 써본다.

```python
import numpy as np


def attention_scores_loop(queries: np.ndarray, keys: np.ndarray):
    batch, heads, seq, dim = queries.shape
    out = np.empty((batch, heads, seq, seq), dtype=queries.dtype)
    for b in range(batch):
        for h in range(heads):
            out[b, h] = queries[b, h] @ keys[b, h].T
    return out


def attention_scores_matmul(queries: np.ndarray, keys: np.ndarray):
    return queries @ np.swapaxes(keys, -1, -2)


def attention_scores_einsum(queries: np.ndarray, keys: np.ndarray):
    return np.einsum("bhqd,bhkd->bhqk", queries, keys)


rng = np.random.default_rng(0)
q = rng.normal(size=(4, 8, 32, 64)).astype(np.float32)
k = rng.normal(size=(4, 8, 32, 64)).astype(np.float32)

a = attention_scores_loop(q, k)
b = attention_scores_matmul(q, k)
c = attention_scores_einsum(q, k)
print(np.allclose(a, b, atol=1e-4), np.allclose(a, c, atol=1e-4))
print(a.shape)
```

`einsum`의 첨자 표기가 형상 변환 의도를 명시적으로 드러낸다. `swapaxes`와 `@`의 조합보다 읽기 쉽고, 축을 헷갈릴 여지가 적다.

`np.einsum`에 `optimize=True`를 주면 다중 항 축약의 계약 순서를 최적화한다.

```python
import numpy as np


rng = np.random.default_rng(0)
a = rng.normal(size=(100, 200))
b = rng.normal(size=(200, 300))
c = rng.normal(size=(300, 50))

path, info = np.einsum_path("ij,jk,kl->il", a, b, c, optimize="optimal")
print(path)
print(info.split("\n")[2].strip())
```

세 행렬 곱의 순서에 따라 연산량이 크게 달라진다는 것이 여기서 드러난다.

### 고급 인덱싱

권장 패턴과 함정을 정리한다.

```python
import numpy as np


logits = np.arange(20, dtype=np.float64).reshape(4, 5)
labels = np.array([1, 0, 4, 2])

picked = logits[np.arange(len(labels)), labels]
print(picked)

picked_alt = np.take_along_axis(logits, labels[:, None], axis=1).squeeze(1)
print(picked_alt)

one_hot = np.zeros_like(logits)
one_hot[np.arange(len(labels)), labels] = 1.0
print(one_hot)
```

교차 엔트로피 구현에서 정답 클래스 로짓을 뽑는 표준 패턴이다. `np.arange(len(labels))`가 행 인덱스를 제공한다.

중복 인덱스에 대한 대입은 예상과 다르게 동작한다.

```python
import numpy as np


counts = np.zeros(5, dtype=np.float64)
indices = np.array([0, 0, 1, 1, 1, 3])

counts[indices] += 1.0
print(f"naive: {counts}")

counts_correct = np.zeros(5, dtype=np.float64)
np.add.at(counts_correct, indices, 1.0)
print(f"add.at: {counts_correct}")

counts_bincount = np.bincount(indices, minlength=5).astype(np.float64)
print(f"bincount: {counts_bincount}")
```

`counts[indices] += 1`은 읽기, 더하기, 쓰기가 분리되어 중복 인덱스에서 마지막 쓰기만 남는다. 스캐터 연산이 필요하면 `np.add.at`이나 `np.bincount`를 쓴다. `bincount`가 훨씬 빠르므로 가능하면 그쪽을 쓴다.

### 메모리 매핑

메모리보다 큰 배열을 다룬다.

```python
import os
import tempfile

import numpy as np


tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "features.npy")

rows, cols = 100_000, 64
writer = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, cols))
rng = np.random.default_rng(0)
for start in range(0, rows, 10_000):
    writer[start:start + 10_000] = rng.normal(size=(10_000, cols)).astype(np.float32)
writer.flush()
del writer

reader = np.load(path, mmap_mode="r")
print(f"shape={reader.shape} dtype={reader.dtype} nbytes={reader.nbytes / 1024**2:.1f} MiB")
print(f"row 5000 mean: {reader[5000].mean():.4f}")
print(f"slice mean: {reader[1000:2000].mean():.4f}")
```

파일 전체를 메모리에 올리지 않고 필요한 페이지만 읽는다. 임베딩 테이블이나 사전 계산된 특징을 다룰 때 표준 방법이다. 다만 무작위 접근이 많으면 페이지 폴트가 병목이 되므로, 접근 순서를 순차에 가깝게 만드는 것이 중요하다.

## 실무 관점

전처리 코드에서 파이썬 루프를 발견하면 벡터화를 검토한다. 다만 모든 루프가 제거 가능한 것은 아니며, 벡터화가 중간 배열을 크게 만들면 오히려 손해다. 배치 크기 $B$, 시퀀스 길이 $L$인 상황에서 $(B, L, L)$ 배열을 만드는 벡터화는 $L$이 커지면 메모리를 폭발시킨다. 청크로 나눠 처리하는 것이 답이다.

```python
import numpy as np


def chunked_pairwise_distance(points: np.ndarray, chunk: int = 512):
    n = points.shape[0]
    squared = (points ** 2).sum(axis=1)
    out = np.empty((n, n), dtype=points.dtype)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = points[start:stop]
        cross = block @ points.T
        out[start:stop] = squared[start:stop, None] + squared[None, :] - 2 * cross
    np.maximum(out, 0, out=out)
    return np.sqrt(out, out=out)


rng = np.random.default_rng(0)
pts = rng.normal(size=(1500, 32)).astype(np.float32)
dist = chunked_pairwise_distance(pts)
print(dist.shape, float(dist.diagonal().max()))
```

`np.maximum(out, 0, out=out)`이 필요한 이유는 부동소수점 오차로 거리 제곱이 음수가 될 수 있기 때문이다. 이를 빠뜨리면 `sqrt`에서 NaN이 나온다. 실제로 자주 발생하는 버그다.

dtype 관리 규칙을 정한다. 딥러닝 파이프라인에서는 float32를 기본으로 하고, 정수 레이블은 int64를 쓴다. PyTorch가 `nll_loss`에 int64를 요구하기 때문이다. NumPy 기본이 float64와 int64이므로, 명시하지 않으면 float64가 흘러들어와 GPU 전송량이 두 배가 된다.

```python
import numpy as np


def to_model_dtype(features: np.ndarray, labels: np.ndarray):
    return features.astype(np.float32, copy=False), labels.astype(np.int64, copy=False)


rng = np.random.default_rng(0)
f, l = to_model_dtype(rng.normal(size=(4, 8)), rng.integers(0, 3, size=(4,)))
print(f.dtype, l.dtype)
```

`copy=False`가 이미 맞는 dtype이면 복사를 건너뛴다.

정수 오버플로를 경계한다. NumPy 정수는 고정 폭이라 Python int와 달리 조용히 순환한다.

```python
import numpy as np


small = np.array([120, 100], dtype=np.int8)
print(f"int8 sum: {small.sum()}")
print(f"int8 add: {small[0] + small[1]}")
print(f"int64 sum: {small.astype(np.int64).sum()}")
```

토큰 카운트나 인덱스 계산에서 int32를 쓰면 21억을 넘는 순간 음수가 된다. 대규모 코퍼스 처리에서 실제로 발생한다.

NaN 처리도 명확히 한다. NaN은 자기 자신과도 같지 않으므로 `==` 비교가 실패한다.

```python
import numpy as np


values = np.array([1.0, np.nan, 3.0, np.inf, -np.inf])
print(f"isnan: {np.isnan(values)}")
print(f"isfinite: {np.isfinite(values)}")
print(f"nanmean: {np.nanmean(values[:3])}")
print(f"mean: {values[:3].mean()}")


def find_bad_values(arr: np.ndarray, name: str):
    bad = ~np.isfinite(arr)
    if bad.any():
        idx = np.argwhere(bad)[:5]
        return f"{name}: {bad.sum()} non-finite values, first at {idx.tolist()}"
    return f"{name}: clean"


rng = np.random.default_rng(0)
sample = rng.normal(size=(10, 4))
sample[3, 2] = np.nan
print(find_bad_values(sample, "features"))
```

학습 데이터 로딩 직후에 이 검사를 한 번 돌리면 NaN 손실의 원인 중 상당수가 사전에 잡힌다.

무작위성은 `np.random.default_rng`를 쓴다. 레거시 `np.random.seed`는 전역 상태를 바꾸므로 라이브러리 코드에서 쓰면 호출자의 무작위성을 오염시킨다. 제너레이터 객체를 명시적으로 전달하는 방식이 테스트와 병렬화 모두에서 안전하다.

```python
import numpy as np


def augment(images: np.ndarray, rng: np.random.Generator):
    flips = rng.random(images.shape[0]) < 0.5
    result = images.copy()
    result[flips] = result[flips, :, ::-1]
    return result


rng = np.random.default_rng(0)
batch = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
print(augment(batch, rng).shape)
```

워커별로 독립적인 스트림이 필요하면 `spawn`을 쓴다.

```python
import numpy as np


parent = np.random.default_rng(42)
children = parent.spawn(4)
print([int(c.integers(0, 1000)) for c in children])
```

같은 시드에서 파생되었지만 통계적으로 독립인 스트림이 만들어진다. 앞 문서의 DataLoader 워커 샤딩과 결합하면 워커마다 다른 증강을 적용하면서도 재현 가능한 파이프라인이 된다.

디버깅 도구를 정리한다. 형상이 예상과 다를 때는 중간 단계마다 찍는다.

```python
import numpy as np


def trace_shapes(**arrays):
    for name, arr in arrays.items():
        print(f"{name:12s} shape={arr.shape} dtype={arr.dtype} "
              f"min={float(arr.min()):.4f} max={float(arr.max()):.4f} "
              f"finite={bool(np.isfinite(arr).all())}")


rng = np.random.default_rng(0)
logits = rng.normal(size=(8, 5))
shifted = logits - logits.max(axis=1, keepdims=True)
probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
trace_shapes(logits=logits, shifted=shifted, probs=probs)
```

## 핵심 정리

ndarray는 버퍼 하나와 형상, 스트라이드, dtype으로 구성된다. 원소 주소는 $\text{base} + \sum_d i_d s_d$로 계산되며 이 식 하나가 뷰, 전치, 브로드캐스팅을 모두 설명한다.

기본 슬라이싱, 전치, 연속 배열의 reshape, `broadcast_to`는 뷰다. 팬시 인덱싱, 불리언 마스킹, `flatten`은 복사다. 뷰를 in-place로 수정하면 원본이 바뀐다.

브로드캐스팅은 크기 1인 축의 스트라이드를 0으로 만들어 확장한다. `(B,)`와 `(B, 1)`이 `(B, B)`가 되는 사고는 예외 없이 조용히 틀린 값을 만든다. 손실 함수 진입부에 형상 단언을 둔다.

원소별 연산은 메모리 바운드다. 임시 배열을 줄이고(`out=`), float32를 쓰면 대략 배수만큼 빨라진다.

순회 방향과 메모리 레이아웃을 맞춘다. C 순서 배열은 마지막 축 방향 접근이 빠르다.

중복 인덱스에 `+=`를 쓰면 마지막 쓰기만 남는다. 스캐터가 필요하면 `np.add.at` 또는 `np.bincount`를 쓴다.

거리 제곱을 확장식으로 계산할 때 음수 클리핑을 빠뜨리면 `sqrt`에서 NaN이 난다.

정수는 고정 폭이라 조용히 오버플로한다. 대규모 카운트에는 int64를 명시한다.

무작위성은 `default_rng` 제너레이터를 명시적으로 전달한다. 전역 `np.random.seed`는 호출자를 오염시킨다.
