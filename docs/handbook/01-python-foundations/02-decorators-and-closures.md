# 데코레이터와 클로저, functools

## 한 줄 정의

데코레이터는 함수를 받아 함수를 돌려주는 고차 함수이며, 그 안에서 원본 함수를 붙잡아 두는 장치가 클로저다.

## 문제 상황

학습 코드에는 본질적 로직이 아니면서 여러 함수에 반복되는 관심사가 있다. 실행 시간 측정, 재시도, 결과 캐싱, 입력 검증, 실험 추적 로깅이 대표적이다. 데코레이터가 없다면 이런 코드는 함수마다 복사된다.

```python
import time


def load_shard(shard_id):
    start = time.perf_counter()
    result = _read_parquet(f"shard_{shard_id}.parquet")
    print(f"load_shard took {time.perf_counter() - start:.3f}s")
    return result


def tokenize_shard(rows):
    start = time.perf_counter()
    result = _tokenize(rows)
    print(f"tokenize_shard took {time.perf_counter() - start:.3f}s")
    return result
```

측정 방식을 `perf_counter`에서 `monotonic`으로 바꾸거나, 출력을 로거로 보내거나, 누적 통계를 내려면 모든 함수를 고쳐야 한다. 관심사가 코드 전체에 흩어져 있어 한 곳에서 바꿀 수 없다는 것이 문제다.

또 하나의 결핍은 함수 자체를 값으로 다루면서 상태를 붙이는 일이었다. 호출 횟수를 세거나, 마지막 결과를 기억하거나, 무거운 계산을 캐싱하려면 전역 딕셔너리를 만들고 함수 이름을 키로 쓰는 식으로 우회했다. 전역 상태는 테스트를 오염시키고 멀티스레드에서 깨진다.

## 직관적 이해

클로저는 함수가 자기가 태어난 방의 열쇠를 들고 나오는 것이다. 방은 이미 비었지만(외부 함수는 반환되어 스택 프레임이 사라졌지만) 열쇠를 가진 함수는 그 방의 물건에 계속 접근할 수 있다. Python은 이 "물건"을 셀(cell) 객체로 힙에 남겨두고 함수의 `__closure__`에 붙여둔다.

데코레이터는 원본 함수를 포장지로 한 겹 싸는 것이다. 밖에서 보면 이름과 호출 방식이 같지만, 안으로 들어가면 포장지가 먼저 무언가를 하고 원본을 부른 뒤 다시 무언가를 한다. 문제는 포장지가 원본의 이름표를 가린다는 것이다. `functools.wraps`는 원본의 이름표를 포장지 겉면에 옮겨 붙이는 작업이다.

## 형식화

데코레이터의 타입은 앞 문서의 함수 타입 표기로 정확히 쓸 수 있다. 함수 공간을 $\mathcal{F} = \{f : A \to B\}$라 하면, 데코레이터 $D$는 다음 사상이다.

$$D : \mathcal{F} \to \mathcal{F}, \quad D(f) = g$$

시그니처를 보존하는 데코레이터는 $g : A \to B$로 $f$와 같은 타입을 갖는다. 이때 `@D`가 붙은 정의

```python
@D
def f(x):
    ...
```

는 정확히 `f = D(f)`와 같다. 데코레이터가 여러 개 쌓이면 적용 순서는 안쪽부터다.

$$\texttt{@}D_1 \ \texttt{@}D_2 \ f \quad \equiv \quad f = D_1(D_2(f))$$

즉 소스 코드에서 아래에 붙은 것이 먼저 적용되고, 실행 시에는 위에 붙은 것이 먼저 진입한다. 이 순서를 헷갈리면 `@lru_cache`와 `@retry`를 잘못 조합해 실패한 결과를 캐싱하는 사고가 난다.

인자를 받는 데코레이터는 한 단계가 더 필요하다. 인자 공간을 $P$라 하면

$$D : P \to (\mathcal{F} \to \mathcal{F})$$

이고, `@D(p)`는 `f = D(p)(f)`로 전개된다. 함수 정의 구문 세 겹이 나오는 이유가 여기 있다.

클로저의 자유 변수는 다음처럼 정의된다. 함수 $g$의 본문에서 참조되는 이름 중 $g$의 지역 변수도 전역도 아닌 것의 집합이 자유 변수 $FV(g)$이며, Python은 각 자유 변수마다 셀 하나를 만들어 $g.\_\_closure\_\_$에 튜플로 담는다. 셀은 값이 아니라 참조를 담으므로, 외부에서 값이 바뀌면 클로저가 보는 값도 바뀐다. 이것이 뒤에 나올 반복문 클로저 함정의 원인이다.

## 구현

### 클로저의 실체 확인

먼저 셀 객체를 직접 들여다본다.

```python
def make_ema(alpha: float):
    state = {"value": None}

    def update(x: float):
        if state["value"] is None:
            state["value"] = x
        else:
            state["value"] = alpha * state["value"] + (1.0 - alpha) * x
        return state["value"]

    return update


ema = make_ema(0.9)
for loss in [1.0, 0.8, 0.9, 0.6, 0.55]:
    print(round(ema(loss), 4))

print([c.cell_contents for c in ema.__closure__])
```

`__closure__`에 `alpha`와 `state`가 셀로 잡혀 있다. `make_ema`는 이미 반환되었지만 두 값은 살아 있다.

값을 딕셔너리에 담은 이유는 재바인딩 때문이다. 클로저 안에서 외부 이름에 대입하려면 그 이름은 지역 변수로 간주되어 `UnboundLocalError`가 난다. `nonlocal`로 해결할 수도 있다.

```python
def make_counter():
    count = 0

    def step():
        nonlocal count
        count += 1
        return count

    return step


counter = make_counter()
print(counter(), counter(), counter())
```

`nonlocal`이 없으면 `count += 1`이 `count`를 지역 변수로 만들고, 대입 전에 읽으므로 오류가 난다.

### 반복문 클로저 함정

셀이 참조를 담는다는 성질이 만드는 대표적 버그다.

```python
broken = []
for lr in [1e-3, 1e-4, 1e-5]:
    broken.append(lambda: lr)

print([f() for f in broken])

fixed = []
for lr in [1e-3, 1e-4, 1e-5]:
    fixed.append(lambda captured=lr: captured)

print([f() for f in fixed])
```

첫 리스트는 세 함수 모두 `1e-05`를 반환한다. 세 람다가 같은 셀을 공유하고 루프가 끝난 뒤 그 셀의 값이 마지막 값이기 때문이다. 기본 인자는 정의 시점에 평가되므로 두 번째 방식이 값을 고정한다. 스케줄러 팩토리나 콜백 등록 루프에서 실제로 자주 나온다.

### 기본 데코레이터와 wraps

앞의 문제 상황을 데코레이터로 정리한다.

```python
import functools
import time


def timed(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"{fn.__qualname__} took {elapsed:.4f}s")

    return wrapper


@timed
def build_vocab(tokens: list[str], min_freq: int = 2):
    """토큰 리스트에서 어휘를 만든다."""
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return sorted(t for t, c in counts.items() if c >= min_freq)


print(build_vocab(["a", "b", "a", "c", "b", "a"]))
print(build_vocab.__name__)
print(build_vocab.__doc__)
```

`finally`에 측정을 둔 이유는 예외가 나도 시간이 찍혀야 하기 때문이다. 실패한 호출이 얼마나 오래 걸렸는지가 디버깅에 필요하다.

`functools.wraps`를 빼면 정확히 무엇이 깨지는지 확인한다.

```python
import functools


def no_wraps(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def with_wraps(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def original(x: int, y: int = 3):
    """원본 문서 문자열."""
    return x + y


a = no_wraps(original)
b = with_wraps(original)
print(a.__name__, a.__doc__, getattr(a, "__wrapped__", None))
print(b.__name__, b.__doc__, b.__wrapped__ is original)

import inspect
print(inspect.signature(a))
print(inspect.signature(b))
```

`wraps`가 없으면 이름이 `wrapper`가 되고, 문서 문자열이 사라지며, `inspect.signature`가 `(*args, **kwargs)`를 반환한다. 마지막 항목이 실무에서 치명적이다. pytest의 픽스처 주입, FastAPI의 의존성 해석, `torch.jit.script`의 시그니처 분석이 모두 `inspect`에 의존한다. 데코레이터 하나 때문에 프레임워크가 인자를 못 찾는 사고가 여기서 나온다.

`__wrapped__` 속성도 `wraps`가 붙여준다. 원본에 도달하는 경로가 남아 있어야 디버깅과 언래핑이 가능하다.

### 인자를 받는 데코레이터

앞의 형식화에서 본 3단 구조다.

```python
import functools
import time


def retry(times: int = 3, delay: float = 0.5, exceptions: tuple = (OSError,)):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_error = exc
                    if attempt < times - 1:
                        time.sleep(delay * (2 ** attempt))
            raise last_error

        return wrapper

    return decorator


attempts = {"n": 0}


@retry(times=4, delay=0.01)
def flaky_download(url: str):
    attempts["n"] += 1
    if attempts["n"] < 3:
        raise OSError("connection reset")
    return f"payload from {url}"


print(flaky_download("s3://bucket/shard-0"))
print(attempts["n"])
```

`delay * (2 ** attempt)`가 지수 백오프다. 고정 지연으로 재시도하면 장애 상황에서 모든 클라이언트가 동시에 재요청해 부하를 키운다. 실무에서는 여기에 무작위 지터를 더한다.

인자 없이도 호출 가능한 데코레이터를 만들려면 첫 인자가 함수인지 판별한다.

```python
import functools


def logged(fn=None, *, prefix: str = "call"):
    if fn is None:
        return functools.partial(logged, prefix=prefix)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"{prefix}: {fn.__qualname__}")
        return fn(*args, **kwargs)

    return wrapper


@logged
def step_a():
    return 1


@logged(prefix="trace")
def step_b():
    return 2


print(step_a(), step_b())
```

`*`로 키워드 전용 인자를 강제한 것이 핵심이다. 이렇게 하지 않으면 `@logged("trace")` 같은 호출이 `fn="trace"`로 해석되어 조용히 망가진다.

### functools의 나머지 도구

`lru_cache`와 `cache`는 메모이제이션이다. 순수 함수에만 쓴다.

```python
import functools


@functools.lru_cache(maxsize=None)
def positional_encoding_row(pos: int, dim: int):
    import math
    return tuple(
        math.sin(pos / (10000 ** (2 * (i // 2) / dim))) if i % 2 == 0
        else math.cos(pos / (10000 ** (2 * (i // 2) / dim)))
        for i in range(dim)
    )


print(len(positional_encoding_row(5, 8)))
print(positional_encoding_row.cache_info())
positional_encoding_row(5, 8)
print(positional_encoding_row.cache_info())
```

`cache_info()`로 히트율을 확인할 수 있다. 인자가 해시 가능해야 하므로 `np.ndarray`나 `list`를 받는 함수에는 쓸 수 없다.

`maxsize=None`은 무한 캐시다. 학습 루프에서 배치마다 다른 인자로 호출하면 메모리가 계속 증가한다. 메서드에 `lru_cache`를 붙이면 `self`가 키에 포함되어 인스턴스가 영원히 해제되지 않는 누수가 발생한다. 이 조합은 피한다.

`cached_property`는 인스턴스 단위 지연 계산이다.

```python
import functools
import numpy as np


class Dataset:
    def __init__(self, values: np.ndarray):
        self.values = values

    @functools.cached_property
    def stats(self):
        return {
            "mean": float(self.values.mean()),
            "std": float(self.values.std()),
            "n": int(self.values.shape[0]),
        }


ds = Dataset(np.arange(1000, dtype=np.float64))
print(ds.stats)
print("stats" in ds.__dict__)
```

첫 접근 후 결과가 인스턴스 `__dict__`에 저장되어 이후 접근은 속성 조회다. `__slots__`를 쓰는 클래스에서는 동작하지 않는다.

`partial`은 인자 일부를 미리 묶는다. 콜백 등록과 멀티프로세싱 워커 함수 전달에서 자주 쓴다.

```python
import functools


def scale_and_clip(x: float, scale: float, lo: float, hi: float):
    return max(lo, min(hi, x * scale))


normalize_logit = functools.partial(scale_and_clip, scale=0.1, lo=-5.0, hi=5.0)
print([normalize_logit(v) for v in [-100.0, 0.0, 30.0, 100.0]])
```

람다 대신 `partial`을 쓰는 실무적 이유는 피클 가능성이다. `multiprocessing`이 워커에 함수를 보낼 때 람다는 직렬화되지 않지만 `partial`은 된다.

`singledispatch`는 첫 인자 타입에 따른 분기를 등록 방식으로 만든다.

```python
import functools
import numpy as np


@functools.singledispatch
def describe(obj):
    return f"unknown type {type(obj).__name__}"


@describe.register
def _(obj: np.ndarray):
    return f"ndarray shape={obj.shape} dtype={obj.dtype}"


@describe.register
def _(obj: list):
    return f"list len={len(obj)}"


@describe.register
def _(obj: dict):
    return f"dict keys={sorted(obj)}"


print(describe(np.zeros((2, 3))))
print(describe([1, 2, 3]))
print(describe({"b": 1, "a": 2}))
print(describe(3.14))
```

`isinstance` 사슬을 늘어놓는 대신 타입별 구현을 분산 등록할 수 있다. 텐서와 배열과 리스트를 모두 받는 전처리 유틸리티에 적합하다.

### 클래스 데코레이터와 메서드 데코레이터

데코레이터는 클래스에도 붙는다. 학습 컴포넌트 레지스트리가 전형적인 용도다.

```python
import functools

REGISTRY: dict[str, type] = {}


def register(name: str):
    def decorator(cls):
        if name in REGISTRY:
            raise ValueError(f"duplicate name: {name}")
        REGISTRY[name] = cls
        cls.registry_name = name
        return cls

    return decorator


@register("cosine")
class CosineSchedule:
    def __init__(self, base_lr: float, total_steps: int):
        self.base_lr = base_lr
        self.total_steps = total_steps

    def __call__(self, step: int):
        import math
        ratio = min(step / self.total_steps, 1.0)
        return 0.5 * self.base_lr * (1.0 + math.cos(math.pi * ratio))


@register("linear")
class LinearSchedule:
    def __init__(self, base_lr: float, total_steps: int):
        self.base_lr = base_lr
        self.total_steps = total_steps

    def __call__(self, step: int):
        return self.base_lr * max(0.0, 1.0 - step / self.total_steps)


def build_schedule(name: str, base_lr: float, total_steps: int):
    return REGISTRY[name](base_lr, total_steps)


sched = build_schedule("cosine", 3e-4, 1000)
print([round(sched(s), 8) for s in [0, 250, 500, 1000]])
print(sorted(REGISTRY))
```

설정 파일의 문자열 하나로 클래스를 고를 수 있게 된다. 중복 등록을 예외로 막은 것이 중요하다. 모듈이 두 번 임포트되어 조용히 덮어쓰이는 사고를 방지한다.

메서드에 붙는 데코레이터는 `self`가 첫 인자로 들어온다는 점만 다르다.

```python
import functools


def require_fitted(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "is_fitted", False):
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return method(self, *args, **kwargs)

    return wrapper


class Scaler:
    def __init__(self):
        self.is_fitted = False
        self.mean = 0.0
        self.std = 1.0

    def fit(self, values: list[float]):
        n = len(values)
        self.mean = sum(values) / n
        var = sum((v - self.mean) ** 2 for v in values) / n
        self.std = var ** 0.5 or 1.0
        self.is_fitted = True
        return self

    @require_fitted
    def transform(self, values: list[float]):
        return [(v - self.mean) / self.std for v in values]


scaler = Scaler()
try:
    scaler.transform([1.0, 2.0])
except RuntimeError as exc:
    print(f"caught: {exc}")

print([round(v, 4) for v in scaler.fit([1.0, 2.0, 3.0, 4.0]).transform([1.0, 4.0])])
```

### 데코레이터 순서

앞의 형식화에서 본 $D_1(D_2(f))$ 규칙이 실무 사고로 이어지는 지점이다.

```python
import functools

call_log: list[str] = []


def counting(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        call_log.append(fn.__qualname__)
        return fn(*args, **kwargs)

    return wrapper


@functools.lru_cache(maxsize=None)
@counting
def cache_outside(x: int):
    return x * 2


@counting
@functools.lru_cache(maxsize=None)
def cache_inside(x: int):
    return x * 3


call_log.clear()
cache_outside(1)
cache_outside(1)
print("cache outside:", len(call_log))

call_log.clear()
cache_inside(1)
cache_inside(1)
print("cache inside:", len(call_log))
```

`cache_outside`는 캐시가 바깥에 있으므로 두 번째 호출에서 `counting`까지 건너뛴다. `cache_inside`는 `counting`이 매번 실행된다. 재시도와 캐시를 조합할 때 이 순서를 잘못 잡으면 실패 결과가 캐싱되어 영구히 실패하는 상태가 된다. 캐시는 항상 재시도 안쪽에 둔다.

## 실무 관점

데코레이터를 쓸지 판단하는 기준은 관심사의 직교성이다. 측정, 재시도, 캐싱, 권한 확인처럼 원래 로직과 독립적인 관심사만 데코레이터로 뺀다. 도메인 로직의 일부를 데코레이터에 넣으면 함수를 읽어도 무엇을 하는지 알 수 없게 된다.

디버깅 비용이 늘어난다는 점을 감수해야 한다. 스택 트레이스에 `wrapper` 프레임이 끼어들고, 중첩이 깊으면 실제 함수까지 여러 단계를 지난다. `functools.wraps`의 `__wrapped__`를 따라가면 원본에 도달할 수 있다.

```python
def unwrap_all(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn
```

성능 오버헤드는 호출당 함수 프레임 하나 정도로, 마이크로초 단위다. 학습 스텝 함수처럼 초당 수십 번 호출되는 곳에서는 무시할 수 있지만, 텐서 원소 접근처럼 초당 수백만 번 호출되는 경로에는 붙이지 않는다.

`lru_cache`의 메모리 누수는 실무에서 반복적으로 발생한다. 세 가지 규칙으로 막는다. 첫째, `maxsize=None`은 인자 공간이 유한하다고 확신할 때만 쓴다. 둘째, 메서드에는 붙이지 않는다. 셋째, 캐시된 함수가 큰 객체를 반환하면 `maxsize`를 명시적으로 작게 잡는다.

스레드 안전성도 확인한다. 클로저 상태를 갱신하는 데코레이터는 여러 스레드에서 호출되면 경쟁 상태가 된다. `DataLoader` 워커가 스레드일 때 카운터가 어긋나는 문제가 여기서 나온다. 필요하면 `threading.Lock`으로 감싼다.

```python
import functools
import threading


def counted(fn):
    lock = threading.Lock()
    state = {"calls": 0}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with lock:
            state["calls"] += 1
        return fn(*args, **kwargs)

    wrapper.get_calls = lambda: state["calls"]
    return wrapper


@counted
def forward_step(x: float):
    return x * 2


for v in range(5):
    forward_step(float(v))
print(forward_step.get_calls())
```

타입 검사기와의 궁합도 고려한다. `*args, **kwargs`로 받는 래퍼는 원본 시그니처를 잃는다. `ParamSpec`을 쓰면 보존된다.

```python
import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def traced(fn: Callable[P, R]):
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        print(f"enter {fn.__qualname__}")
        result = fn(*args, **kwargs)
        print(f"exit {fn.__qualname__}")
        return result

    return wrapper


@traced
def add(a: int, b: int):
    return a + b


print(add(2, 3))
```

`ParamSpec` 덕분에 `add(2, "3")` 같은 호출이 검사기에서 잡힌다. `ParamSpec` 없이는 `*args`가 되어 아무 인자나 통과한다.

## 핵심 정리

`@D`가 붙은 정의는 `f = D(f)`와 정확히 같고, 여러 개가 쌓이면 아래쪽이 먼저 적용된다. 캐시는 재시도 안쪽에 둔다. 순서를 뒤집으면 실패 결과가 캐싱된다.

`functools.wraps`를 빠뜨리면 이름과 문서 문자열이 사라지고 `inspect.signature`가 `(*args, **kwargs)`를 반환한다. pytest 픽스처와 FastAPI 의존성 주입이 여기서 깨진다.

클로저 셀은 값이 아니라 참조를 담는다. 반복문 안에서 만든 함수들이 마지막 값을 공유하는 버그는 기본 인자로 값을 고정해 막는다.

인자를 받는 데코레이터는 3단 중첩이며, 인자 없이도 쓸 수 있게 하려면 첫 인자가 함수인지 판별하고 나머지를 키워드 전용으로 강제한다.

`lru_cache`는 메서드에 붙이지 않는다. `self`가 키에 들어가 인스턴스가 해제되지 않는다.

`ParamSpec`을 쓰면 래퍼가 원본 시그니처를 보존해 타입 검사가 유지된다.
