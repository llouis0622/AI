# 타입 힌트와 정적 검사, 제네릭, Protocol

## 한 줄 정의

타입 힌트는 실행에 영향을 주지 않는 메타데이터이지만, 정적 검사기가 이를 읽어 실행 전에 타입 불일치를 잡아내는 계약 선언이다.

## 문제 상황

타입 힌트가 없던 시절의 학습 코드베이스에서 반복되던 사고는 다음 형태였다.

```python
def compute_loss(logits, targets, weights=None):
    ...
```

이 함수의 `logits`가 `torch.Tensor`인지 `np.ndarray`인지, `targets`가 정수 인덱스인지 원-핫인지, `weights`가 클래스별 가중치인지 샘플별 가중치인지는 시그니처에서 알 수 없다. 본문을 읽거나, 호출부를 역추적하거나, 실행해서 예외를 봐야 한다. 학습 스크립트는 데이터 로딩과 초기화에만 몇 분이 걸리므로 "실행해서 확인"의 비용이 특히 크다. 30분짜리 학습을 돌린 뒤 마지막 검증 단계에서 형상 불일치로 죽는 상황이 흔했다.

더 심각한 것은 조용히 틀리는 경우다. `targets`가 `(B,)` 정수 배열이어야 하는데 `(B, 1)`이 들어오면 브로드캐스팅으로 `(B, B)` 손실이 만들어지고, 평균을 취하면 스칼라가 나오며, 학습은 계속 진행되지만 값이 틀린다. 예외가 나지 않으므로 며칠 뒤 성능이 이상하다는 것으로만 발견된다.

정적 검사만이 목적은 아니다. 타입 힌트는 IDE 자동완성의 근거이며, 무엇보다 함수 경계에서의 계약을 문서화한다. 여섯 달 뒤 자기 코드를 읽을 때 시그니처만으로 사용법을 복원할 수 있느냐가 갈린다.

## 직관적 이해

타입 힌트는 세관 신고서에 가깝다. Python 런타임은 신고서를 읽지 않고 물건을 그냥 통과시킨다. 그러나 검사관(mypy, pyright)은 신고서만 보고 "이 경로로는 이 물건이 갈 수 없다"를 미리 지적한다. 신고서가 실물과 달라도 런타임은 아무 말이 없으므로, 신고서의 가치는 전적으로 검사관을 돌리느냐에 달려 있다.

제네릭은 신고서에 "상자 안의 내용물 종류"를 적는 칸을 추가하는 것이다. `list`라고만 쓰면 무엇이 들었는지 모르지만 `list[int]`라고 쓰면 꺼냈을 때 무엇인지 안다.

`Protocol`은 검문 방식 자체를 바꾼다. "이 상자는 A사가 만든 것이어야 한다"가 상속 기반 검사(`ABC`)라면, "이 상자는 뚜껑을 열 수 있고 무게를 잴 수 있으면 된다"가 `Protocol`이다. 제조사를 묻지 않고 능력만 묻는다. Python이 원래 하던 덕 타이핑을 정적으로 검사 가능하게 만든 장치다.

## 형식화

타입 검사기가 하는 일은 부분 순서 관계 위에서의 포함 관계 판정이다. 타입 $T$의 값 집합을 $\llbracket T \rrbracket$이라 하면, $S$가 $T$의 서브타입이라는 것은 다음을 의미한다.

$$S <: T \iff \llbracket S \rrbracket \subseteq \llbracket T \rrbracket$$

여기서 $\llbracket \cdot \rrbracket$은 해당 타입이 가질 수 있는 런타임 값들의 집합이다. 함수 인자로 $T$를 요구하는 자리에 $S$를 넘길 수 있는 조건이 바로 $S <: T$다.

함수 타입 사이의 서브타입 관계는 인자에 대해 반공변, 반환에 대해 공변이다.

$$(T_1 \to R_1) <: (T_2 \to R_2) \iff T_2 <: T_1 \ \wedge \ R_1 <: R_2$$

인자 방향이 뒤집히는 이유는 대체 가능성의 정의에서 곧바로 나온다. $(T_2 \to R_2)$가 기대되는 자리에 $(T_1 \to R_1)$을 넣으려면, 호출자가 넘기는 모든 $T_2$ 값을 그 함수가 받아들일 수 있어야 하므로 $T_2 <: T_1$이 필요하다. 즉 "더 넓은 입력을 받는 함수"가 "더 좁은 입력만 받는 함수"를 대체한다. `float`를 받는 함수는 `int`만 받는 자리에 쓸 수 있지만 그 반대는 안 된다.

컨테이너의 가변성이 공변성을 막는 이유도 같은 논리다. $S <: T$일 때 `list[S]`를 `list[T]`로 취급할 수 있다고 하면 읽기는 안전하지만 쓰기가 깨진다. `list[int]`를 `list[float]`로 보고 `3.5`를 넣으면 원래 리스트의 계약이 무너지기 때문이다. 그래서 다음이 성립한다.

$$\text{Sequence}[S] <: \text{Sequence}[T] \iff S <: T$$

$$\text{list}[S] <: \text{list}[T] \iff S = T$$

`Sequence`는 읽기 전용이므로 공변, `list`는 가변이므로 불변(invariant)이다. 이 규칙이 실무에서 의미하는 바는 하나로 압축된다. 함수 인자는 가능한 한 `Sequence`, `Iterable`, `Mapping` 같은 읽기 전용 추상 타입으로 받고, 반환은 `list`, `dict` 같은 구체 타입으로 준다. 입력은 넓게, 출력은 좁게다.

## 구현

### 기본 힌트와 검사기 동작

앞의 문제 함수에 계약을 부여한다. 아래 코드에서 `np.ndarray | None`이 위 형식화의 유니온 타입 $\llbracket \text{ndarray} \rrbracket \cup \llbracket \text{None} \rrbracket$에 해당한다.

```python
import numpy as np


def compute_loss(logits: np.ndarray, targets: np.ndarray, class_weights: np.ndarray | None = None):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    picked = log_probs[np.arange(targets.shape[0]), targets]
    if class_weights is None:
        return -picked.mean()
    w = class_weights[targets]
    return -(picked * w).sum() / w.sum()


rng = np.random.default_rng(0)
logits = rng.normal(size=(8, 5))
targets = rng.integers(0, 5, size=(8,))
print(compute_loss(logits, targets))
print(compute_loss(logits, targets, class_weights=np.array([1.0, 2.0, 1.0, 1.0, 3.0])))
```

`np.ndarray | None`은 Python 3.10부터 쓸 수 있는 유니온 표기다. 그 이전 버전에서는 `Optional[np.ndarray]`를 쓴다. 기본값이 `None`인 인자에 유니온을 붙이지 않으면 검사기가 즉시 오류를 낸다.

인자명을 `weights`가 아니라 `class_weights`로 둔 이유는 타입이 같아도 이름이 계약의 일부이기 때문이다. `np.ndarray` 하나로는 길이가 클래스 수인지 배치 크기인지 구분되지 않는다.

검사기 설정은 `pyproject.toml`에 둔다.

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_unreachable = true
disallow_any_explicit = false

[[tool.mypy.overrides]]
module = ["sklearn.*", "matplotlib.*"]
ignore_missing_imports = true
```

`strict = true`는 여러 플래그의 묶음이다. 실무 효과가 큰 것은 `disallow_untyped_defs`(힌트 없는 함수 정의 금지), `no_implicit_optional`(기본값 `None`에 대한 암묵 Optional 금지), `warn_return_any`(Any 반환 경고)다. 신규 프로젝트는 처음부터 `strict`로 시작하고, 기존 프로젝트는 모듈 단위로 점진 적용한다.

### 제네릭

같은 로직을 여러 타입에 재사용하면서 타입 정보를 잃지 않으려면 타입 변수를 쓴다. 앞의 공변성 규칙에 따라 인자는 `Sequence[T]`로 받는다. `list[T]`로 받았다면 튜플을 넘길 수 없다.

```python
from collections.abc import Callable, Sequence


def batched[T](items: Sequence[T], size: int):
    out: list[list[T]] = []
    for start in range(0, len(items), size):
        out.append(list(items[start:start + size]))
    return out


def map_batches[T, R](items: Sequence[T], size: int, fn: Callable[[list[T]], R]):
    return [fn(chunk) for chunk in batched(items, size)]


sample_ids = ["a", "b", "c", "d", "e"]
print(batched(sample_ids, 2))
print(map_batches(sample_ids, 2, lambda chunk: "".join(chunk)))
```

`batched(sample_ids, 2)`의 반환 타입은 `list[list[str]]`로 추론된다. `map_batches`의 반환은 `list[str]`이다. 타입 변수 `R`이 람다의 반환에서 결정되므로 호출부마다 다른 타입이 나온다.

대괄호 문법은 Python 3.12부터다. 3.11 이하에서는 `TypeVar`를 명시적으로 선언한다.

```python
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def batched_legacy(items: Sequence[T], size: int):
    out: list[list[T]] = []
    for start in range(0, len(items), size):
        out.append(list(items[start:start + size]))
    return out


print(batched_legacy((1, 2, 3, 4, 5), 3))
```

### Protocol

학습 파이프라인에서 스케줄러, 로거, 체크포인트 저장기 같은 컴포넌트는 여러 구현이 교체된다. 공통 기반 클래스 상속을 강제하면 외부 라이브러리 객체를 그대로 끼울 수 없다. `Protocol`은 상속 없이 구조만으로 호환을 판정한다.

```python
from typing import Protocol


class MetricLogger(Protocol):
    def log_scalar(self, name: str, value: float, step: int): ...
    def close(self): ...


class StdoutLogger:
    def log_scalar(self, name: str, value: float, step: int):
        print(f"step={step} {name}={value:.6f}")

    def close(self):
        pass


class BufferedLogger:
    def __init__(self, path: str):
        self.path = path
        self.buffer: list[str] = []

    def log_scalar(self, name: str, value: float, step: int):
        self.buffer.append(f"{step}\t{name}\t{value:.6f}")

    def close(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.buffer))


def run_epoch(logger: MetricLogger, losses: list[float], epoch: int):
    for i, loss in enumerate(losses):
        logger.log_scalar("train/loss", loss, epoch * len(losses) + i)
    logger.log_scalar("train/loss_mean", sum(losses) / len(losses), epoch)


run_epoch(StdoutLogger(), [0.9, 0.7, 0.55], epoch=0)
```

`StdoutLogger`와 `BufferedLogger`는 아무것도 상속하지 않았지만 `MetricLogger` 자리에 들어간다. MLflow나 W&B 래퍼를 만들 때 이 구조가 결정적이다. 외부 객체를 우리 기반 클래스로 감쌀 필요가 없다.

`runtime_checkable`을 붙이면 `isinstance` 검사도 가능해지지만 메서드 존재 여부만 보고 시그니처는 보지 않는다. 정적 검사가 본 계약과 런타임 검사가 본 계약이 달라지므로 분기 조건으로 쓰지 않는 편이 낫다.

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class Closeable(Protocol):
    def close(self): ...


class Broken:
    def close(self, extra_arg):
        pass


print(isinstance(Broken(), Closeable))
```

시그니처가 다른데도 `True`가 나온다. 이것이 런타임 검사의 한계다.

`Protocol`과 `ABC`의 선택 기준은 소유권이다. 구현체를 우리가 모두 작성하고 공통 로직을 공유하고 싶으면 `ABC`, 구현체 중 일부가 외부 라이브러리이거나 경계에서 최소 계약만 요구하면 `Protocol`이다.

### TypedDict와 오버로드

설정 딕셔너리는 학습 코드에서 자주 쓰이지만 `dict[str, Any]`로 두면 타입 정보가 전부 사라진다.

```python
from typing import NotRequired, TypedDict


class OptimConfig(TypedDict):
    name: str
    lr: float
    weight_decay: NotRequired[float]
    betas: NotRequired[tuple[float, float]]


def build_optimizer_args(cfg: OptimConfig):
    args: dict[str, float | tuple[float, float]] = {"lr": cfg["lr"]}
    if "weight_decay" in cfg:
        args["weight_decay"] = cfg["weight_decay"]
    if "betas" in cfg:
        args["betas"] = cfg["betas"]
    return args


cfg: OptimConfig = {"name": "adamw", "lr": 3e-4, "weight_decay": 0.01}
print(build_optimizer_args(cfg))
```

`NotRequired`로 선택 키를 표시하면 검사기가 `cfg["betas"]` 직접 접근을 오류로 잡는다. 존재 확인 후 접근하도록 강제되는 셈이다.

인자 형태에 따라 반환 타입이 달라지는 함수에는 `overload`를 쓴다.

```python
from typing import overload
import numpy as np


@overload
def to_probability(scores: float): ...


@overload
def to_probability(scores: np.ndarray): ...


def to_probability(scores: float | np.ndarray):
    return 1.0 / (1.0 + np.exp(-scores))


print(to_probability(0.5))
print(to_probability(np.array([-1.0, 0.0, 1.0])))
```

`overload`로 선언된 시그니처는 검사기만 읽고 런타임에는 마지막 실제 정의만 남는다. 오버로드 본문에 `...` 외의 코드를 쓰면 안 된다.

### 형상까지 검사하기

타입 힌트는 `np.ndarray`까지만 말해주고 형상은 말해주지 않는다. 앞서 본 `(B, 1)` 사고가 여기서 발생한다. 함수 진입부의 런타임 단언이 가장 실용적인 대응이다.

```python
import numpy as np


def assert_shape(arr: np.ndarray, expected: tuple[int | None, ...], name: str):
    if arr.ndim != len(expected):
        raise ValueError(f"{name}: ndim {arr.ndim} != {len(expected)}")
    for axis, (got, want) in enumerate(zip(arr.shape, expected)):
        if want is not None and got != want:
            raise ValueError(f"{name}: axis {axis} size {got} != {want}")


def cross_entropy(logits: np.ndarray, targets: np.ndarray):
    assert_shape(logits, (None, None), "logits")
    assert_shape(targets, (logits.shape[0],), "targets")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -log_probs[np.arange(targets.shape[0]), targets].mean()


rng = np.random.default_rng(0)
print(cross_entropy(rng.normal(size=(4, 3)), rng.integers(0, 3, size=(4,))))

try:
    cross_entropy(rng.normal(size=(4, 3)), rng.integers(0, 3, size=(4, 1)))
except ValueError as exc:
    print(f"caught: {exc}")
```

`(4, 1)` 형상이 들어오면 즉시 `ValueError`가 난다. 조용한 브로드캐스팅 버그가 여기서 막힌다.

## 실무 관점

정적 검사 도입 순서를 정해야 한다. 전체 코드베이스에 한 번에 `strict`를 켜면 수천 개 오류가 쏟아지고 아무도 고치지 않는다. 새 파일에만 `strict`를 적용하고 기존 파일은 무시했다가 손댈 때마다 해제하는 방식이 실효성이 있다.

```toml
[[tool.mypy.overrides]]
module = ["legacy.*"]
ignore_errors = true
```

mypy와 pyright의 선택은 상황에 따라 다르다. pyright는 훨씬 빠르고 추론이 공격적이라 IDE 통합에 유리하다. mypy는 플러그인 생태계가 넓고 CI에서 안정적이다. 둘 다 돌려도 되지만 규칙이 미묘하게 달라 이중 관리 비용이 생기므로 CI 게이트는 하나만 둔다.

`Any`가 전염되는 문제를 경계한다. 함수 하나가 `Any`를 반환하면 그 값을 쓰는 모든 지점에서 검사가 무력화된다. `json.loads`, `yaml.safe_load`, `torch.load`가 모두 `Any`를 반환한다. 이 경계에서 즉시 구체 타입으로 좁히는 함수를 하나 두는 것이 표준 대응이다.

```python
import json
from typing import Any, TypedDict, cast


class RunMeta(TypedDict):
    run_id: str
    seed: int
    lr: float


def load_run_meta(path: str):
    with open(path, encoding="utf-8") as f:
        raw: Any = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("run meta must be an object")
    for key in ("run_id", "seed", "lr"):
        if key not in raw:
            raise ValueError(f"missing key: {key}")
    if not isinstance(raw["seed"], int):
        raise ValueError("seed must be int")
    return cast(RunMeta, raw)
```

`cast`는 런타임에 아무것도 하지 않는다. 검사기에게 "여기서는 내가 보증한다"고 말하는 것이므로 반드시 그 앞에 실제 검증 코드가 있어야 한다. 검증 없는 `cast`는 타입 시스템에 대한 거짓말이며 `Any`보다 위험하다. 최소한 `Any`는 검사기가 모른다는 사실을 알고 있지만 `cast`는 안다고 착각하게 만든다.

성능 비용은 사실상 없다. 힌트는 `__annotations__`에 저장될 뿐 호출마다 평가되지 않는다. 다만 모듈 임포트 시점에 어노테이션 표현식이 평가되므로, 타입 힌트 때문에 무거운 임포트를 하게 되는 경우가 있다. 이때 `TYPE_CHECKING` 가드를 쓴다.

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def move_to_cpu(tensor: torch.Tensor):
    return tensor.detach().cpu()
```

`from __future__ import annotations`가 어노테이션을 문자열로 지연 평가하게 만들어, 런타임에 `torch`가 임포트되지 않아도 문제가 없다. CLI 도구처럼 시작 시간이 중요한 경우 수백 밀리초가 절약된다.

흔한 실수를 정리한다.

가변 기본값에 힌트만 붙이고 문제를 방치하는 경우가 첫째다. `def f(items: list[int] = [])`는 힌트가 맞아도 버그다. `None`을 기본값으로 두고 본문에서 만든다.

`Union`의 모든 분기를 처리하지 않고 넘어가는 경우가 둘째다. `Literal`과 `assert_never`를 조합하면 분기 누락이 검사 시점에 잡힌다.

```python
from typing import Literal, assert_never
import numpy as np

Reduction = Literal["mean", "sum", "none"]


def reduce_loss(values: np.ndarray, mode: Reduction):
    if mode == "mean":
        return values.mean()
    if mode == "sum":
        return values.sum()
    if mode == "none":
        return values
    assert_never(mode)


print(reduce_loss(np.array([1.0, 2.0, 3.0]), "mean"))
```

`Reduction`에 새 값을 추가하면 `assert_never` 지점에서 검사기가 오류를 낸다. 열거형 확장 시 처리 누락을 자동으로 잡는 장치이며, 문자열 옵션을 받는 모든 함수에 적용할 가치가 있다.

셋째는 `Protocol`을 과도하게 잘게 쪼개는 것이다. 메서드 하나짜리 프로토콜을 수십 개 만들면 오히려 읽기 어려워진다. 실제 교체 가능성이 있는 컴포넌트 경계에만 둔다.

## 핵심 정리

타입 힌트는 런타임 동작을 바꾸지 않는다. 가치는 전적으로 검사기를 CI 게이트로 넣느냐에서 나온다.

함수 인자는 `Sequence`, `Iterable`, `Mapping` 같은 읽기 전용 공변 타입으로 넓게 받고 반환은 구체 타입으로 준다. 가변 컨테이너가 불변인 이유는 쓰기 안전성 때문이다.

`Protocol`은 상속 없이 구조만으로 호환을 판정하므로 외부 라이브러리 객체를 계약에 맞추는 데 적합하다. 구현체를 모두 소유하고 공통 로직을 공유할 때만 `ABC`를 쓴다.

`Any`는 전염된다. `json.load`, `yaml.safe_load`, `torch.load` 같은 경계에서 검증 후 `cast`로 즉시 좁힌다. 검증 없는 `cast`는 금지다.

타입 힌트는 형상을 검사하지 못한다. 배열을 다루는 함수는 진입부에 형상 단언을 두어 브로드캐스팅으로 인한 조용한 오류를 막는다.

`Literal` + `assert_never` 조합은 분기 누락을 정적으로 잡아준다.
