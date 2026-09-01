# 클래스 심화: 매직 메서드, dataclass, `__slots__`

## 한 줄 정의

Python의 클래스는 속성 조회와 연산자 동작을 매직 메서드로 위임하는 개방형 구조이며, `dataclass`는 그 보일러플레이트를 자동 생성하고 `__slots__`는 인스턴스 저장 방식을 딕셔너리에서 고정 배열로 바꾼다.

## 문제 상황

설정 객체와 데이터 레코드를 클래스로 만들면 같은 코드가 반복된다.

```python
class TrainConfig:
    def __init__(self, lr, batch_size, epochs, weight_decay=0.0):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay

    def __repr__(self):
        return (f"TrainConfig(lr={self.lr}, batch_size={self.batch_size}, "
                f"epochs={self.epochs}, weight_decay={self.weight_decay})")

    def __eq__(self, other):
        if not isinstance(other, TrainConfig):
            return NotImplemented
        return (self.lr, self.batch_size, self.epochs, self.weight_decay) == \
               (other.lr, other.batch_size, other.epochs, other.weight_decay)
```

필드를 하나 추가하면 네 곳을 고쳐야 하고, 한 곳을 빠뜨리면 `__eq__`가 조용히 틀린 결과를 낸다. 실험 설정 비교가 틀리면 중복 실행을 감지하지 못한다.

메모리 문제도 있다. 인스턴스마다 `__dict__`가 생기므로, 데이터셋 샘플을 100만 개 객체로 표현하면 딕셔너리 오버헤드만 수백 MB가 된다.

세 번째 결핍은 사용자 정의 타입이 언어의 기본 연산과 통합되지 않는다는 것이었다. 커스텀 텐서 래퍼를 만들었는데 `+`가 동작하지 않고, `len()`도 안 되고, `for`로 순회도 안 되면 표준 라이브러리와 조합할 수 없다.

## 직관적 이해

매직 메서드는 언어의 문법이 객체에게 보내는 신호다. `a + b`는 "덧셈을 해달라"는 요청이고 Python은 이를 `type(a).__add__(a, b)` 호출로 번역한다. 즉 문법과 구현 사이에 규약이 있고, 클래스는 그 규약에 응답함으로써 언어의 일급 시민이 된다.

`dataclass`는 서기관이다. 필드 목록을 보고 생성자, 표현, 비교 메서드를 대신 작성해 클래스에 붙여준다. 사람이 쓰는 것과 같은 코드를 생성하므로 마법이 아니라 자동 타이핑이다.

`__slots__`는 서랍장을 바꾸는 것이다. 기본 인스턴스는 이름표를 붙일 수 있는 무한 확장 서랍(딕셔너리)을 쓰지만, `__slots__`를 선언하면 미리 정해진 칸만 있는 고정 서랍이 된다. 확장성을 포기하고 공간과 접근 속도를 얻는다.

## 형식화

속성 조회 규칙이 클래스 동작의 핵심이다. `obj.name`을 평가할 때 Python은 다음 순서를 따른다.

$$\text{type}(obj).\_\_mro\_\_ \text{에서 } name \text{을 찾는다} \to \text{데이터 디스크립터면 즉시 그것을 사용}$$

$$\to obj.\_\_dict\_\_[name] \text{ 확인} \to \text{비데이터 디스크립터 또는 클래스 속성 사용} \to \_\_getattr\_\_ \text{호출}$$

여기서 데이터 디스크립터는 `__get__`과 `__set__`(또는 `__delete__`)을 모두 가진 객체이고, 비데이터 디스크립터는 `__get__`만 가진 객체다. 함수는 `__get__`만 가지므로 비데이터 디스크립터이며, 그래서 인스턴스 속성이 메서드를 가릴 수 있다. `property`는 데이터 디스크립터이므로 인스턴스 속성보다 우선한다.

메서드 조회에서 순서를 결정하는 MRO는 C3 선형화로 계산된다. 클래스 $C$가 기반 클래스 $B_1, \dots, B_n$을 가질 때

$$L(C) = C + \text{merge}(L(B_1), \dots, L(B_n), [B_1, \dots, B_n])$$

이며, merge는 각 리스트의 머리를 순서대로 검사해 다른 리스트의 꼬리에 나타나지 않는 첫 머리를 취한다. 이 규칙이 만족되지 않으면 클래스 정의 시점에 `TypeError`가 난다. 다중 상속에서 `super()`가 예측 가능한 순서를 갖는 근거다.

메모리 측면에서 인스턴스 크기를 비교한다. 필드 $k$개를 가진 일반 인스턴스는

$$M_{\text{dict}} = M_{\text{object header}} + M_{\text{dict ptr}} + M_{\text{dict}}(k)$$

이고, 딕셔너리 자체가 해시 테이블이므로 $k$가 작아도 최소 크기가 크다. `__slots__`를 쓰면

$$M_{\text{slots}} = M_{\text{object header}} + k \cdot M_{\text{ptr}}$$

로 줄어든다. CPython 64비트에서 대략 필드 5개짜리 객체가 152바이트에서 72바이트 수준으로 내려간다. 100만 인스턴스면 80MB 차이다.

접근 속도도 다르다. 딕셔너리 조회는 해시 계산과 충돌 탐색을 거치지만 슬롯은 고정 오프셋 배열 접근이므로 대략 20에서 30퍼센트 빠르다.

## 구현

### 매직 메서드로 언어에 통합하기

수치 배열 래퍼를 만들어 주요 프로토콜을 붙인다.

```python
import math


class Vector:
    def __init__(self, values):
        self.values = list(values)

    def __repr__(self):
        return f"Vector({self.values})"

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Vector(self.values[index])
        return self.values[index]

    def __iter__(self):
        return iter(self.values)

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return self.values == other.values

    def __hash__(self):
        return hash(tuple(self.values))

    def __add__(self, other):
        if isinstance(other, Vector):
            if len(other) != len(self):
                raise ValueError("length mismatch")
            return Vector(a + b for a, b in zip(self.values, other.values))
        if isinstance(other, (int, float)):
            return Vector(a + other for a in self.values)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector(a * scalar for a in self.values)

    __rmul__ = __mul__

    def __abs__(self):
        return math.sqrt(sum(a * a for a in self.values))

    def __bool__(self):
        return any(self.values)

    def __format__(self, spec):
        if spec.endswith("n"):
            return f"|{abs(self):{spec[:-1]}}|"
        return repr(self)


v = Vector([3.0, 4.0])
w = Vector([1.0, 2.0])
print(v + w, v + 1, 2 * v)
print(len(v), v[0], list(v[0:1]))
print(abs(v), bool(Vector([0, 0])))
print(f"{v:.3n}")
print(v == Vector([3.0, 4.0]), {v, Vector([3.0, 4.0])})
```

`NotImplemented`를 반환하는 부분이 중요하다. `None`이나 예외가 아니라 `NotImplemented`를 돌려주면 Python이 반사 연산(`__radd__`)을 시도하고, 그것도 실패하면 적절한 `TypeError`를 만든다. 직접 `TypeError`를 던지면 이 협상 과정이 끊긴다.

`__eq__`를 정의하면 `__hash__`가 `None`이 되어 해시 불가능해진다는 규칙도 확인된다. 위에서 `__hash__`를 명시했기 때문에 집합에 넣을 수 있다. 가변 객체에 `__hash__`를 붙이는 것은 위험하지만, 여기서는 값이 바뀌지 않는다는 전제로 두었다.

### property와 디스크립터

앞의 형식화에서 데이터 디스크립터가 인스턴스 속성보다 우선한다고 했다. 이를 검증 로직에 활용한다.

```python
class Schedule:
    def __init__(self, base_lr: float, total_steps: int):
        self._base_lr = 0.0
        self.base_lr = base_lr
        self.total_steps = total_steps

    @property
    def base_lr(self):
        return self._base_lr

    @base_lr.setter
    def base_lr(self, value: float):
        if value <= 0:
            raise ValueError(f"base_lr must be positive, got {value}")
        if value > 1.0:
            raise ValueError(f"base_lr suspiciously large: {value}")
        self._base_lr = float(value)


sched = Schedule(3e-4, 1000)
print(sched.base_lr)
try:
    sched.base_lr = -1.0
except ValueError as exc:
    print(f"caught: {exc}")
```

같은 검증을 여러 필드에 반복해야 하면 디스크립터를 직접 만든다.

```python
class Positive:
    def __set_name__(self, owner, name):
        self.private = f"_{name}"
        self.public = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private)

    def __set__(self, obj, value):
        if value <= 0:
            raise ValueError(f"{self.public} must be positive, got {value}")
        setattr(obj, self.private, value)


class OptimizerConfig:
    lr = Positive()
    weight_decay_scale = Positive()

    def __init__(self, lr: float, weight_decay_scale: float):
        self.lr = lr
        self.weight_decay_scale = weight_decay_scale


cfg = OptimizerConfig(1e-3, 0.01)
print(cfg.lr, cfg.weight_decay_scale)
try:
    OptimizerConfig(0.0, 0.01)
except ValueError as exc:
    print(f"caught: {exc}")
```

`__set_name__`이 클래스 정의 시점에 자동 호출되어 필드 이름을 알려준다. 디스크립터 인스턴스가 클래스 속성으로 공유되므로 값 자체는 인스턴스에 저장해야 한다는 점이 핵심이다. 디스크립터에 값을 저장하면 모든 인스턴스가 같은 값을 공유하는 버그가 된다.

PyTorch의 `nn.Parameter` 등록도 이와 같은 계열의 메커니즘이며, `__setattr__` 오버라이드로 구현되어 있다.

### dataclass

앞의 보일러플레이트가 어떻게 사라지는지 본다.

```python
from dataclasses import dataclass, field, asdict, replace


@dataclass(frozen=True, slots=True)
class TrainConfig:
    lr: float
    batch_size: int
    epochs: int
    weight_decay: float = 0.0
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


cfg = TrainConfig(lr=3e-4, batch_size=32, epochs=10, tags=("baseline",))
print(cfg)
print(cfg == TrainConfig(3e-4, 32, 10, 0.0, ("baseline",)))
print(asdict(cfg))
print(replace(cfg, lr=1e-4))
print(hash(cfg))
```

`frozen=True`가 불변성을 강제하므로 설정 객체가 실험 도중 수정되는 사고를 막는다. 불변이면 `__hash__`가 자동 생성되어 딕셔너리 키나 집합 원소로 쓸 수 있다. 실험 설정을 캐시 키로 쓰는 패턴이 여기서 나온다.

`slots=True`는 `__slots__`를 자동 생성한다. Python 3.10부터 지원한다.

가변 기본값은 `field(default_factory=...)`로 처리한다. 리스트를 기본값으로 직접 쓰면 `dataclass`가 오류를 낸다.

```python
from dataclasses import dataclass, field


@dataclass
class RunState:
    step: int = 0
    losses: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    run_id: str = field(default="", compare=False, repr=False)

    def record(self, loss: float):
        self.step += 1
        self.losses.append(loss)


a = RunState()
b = RunState()
a.record(0.5)
print(a.losses, b.losses)
print(a == RunState(step=1, losses=[0.5], run_id="other"))
```

`compare=False`로 표시한 필드는 `__eq__`에서 제외된다. 실행 ID처럼 내용과 무관한 메타데이터를 비교에서 빼는 용도다. `repr=False`는 로그를 짧게 유지한다.

정렬이 필요하면 `order=True`를 준다. 필드 선언 순서대로 튜플 비교가 생성된다.

```python
from dataclasses import dataclass, field


@dataclass(order=True)
class Candidate:
    score: float
    name: str = field(compare=False)


items = [Candidate(0.8, "b"), Candidate(0.95, "a"), Candidate(0.5, "c")]
print(sorted(items, reverse=True))
```

상속 시 기본값 있는 필드 뒤에 기본값 없는 필드를 둘 수 없다는 제약이 있다. 함수 인자 규칙과 같다. `kw_only=True`로 우회한다.

```python
from dataclasses import dataclass


@dataclass(kw_only=True)
class BaseConfig:
    seed: int = 42


@dataclass(kw_only=True)
class VisionConfig(BaseConfig):
    image_size: int
    channels: int = 3


print(VisionConfig(image_size=224))
```

### `__slots__`

메모리 차이를 실제로 측정한다.

```python
import sys


class WithDict:
    def __init__(self, uid: int, text: str, label: int, weight: float, split: str):
        self.uid = uid
        self.text = text
        self.label = label
        self.weight = weight
        self.split = split


class WithSlots:
    __slots__ = ("uid", "text", "label", "weight", "split")

    def __init__(self, uid: int, text: str, label: int, weight: float, split: str):
        self.uid = uid
        self.text = text
        self.label = label
        self.weight = weight
        self.split = split


a = WithDict(1, "x", 0, 1.0, "train")
b = WithSlots(1, "x", 0, 1.0, "train")

size_a = sys.getsizeof(a) + sys.getsizeof(a.__dict__)
size_b = sys.getsizeof(b)
print(f"dict-based: {size_a} bytes")
print(f"slots-based: {size_b} bytes")
print(f"per 1M instances: {(size_a - size_b) * 1_000_000 / 1024 / 1024:.1f} MiB saved")
```

접근 속도도 비교한다.

```python
import timeit


setup = """
class WithDict:
    def __init__(self):
        self.value = 1

class WithSlots:
    __slots__ = ("value",)
    def __init__(self):
        self.value = 1

a = WithDict()
b = WithSlots()
"""

t_dict = timeit.timeit("a.value", setup=setup, number=5_000_000)
t_slots = timeit.timeit("b.value", setup=setup, number=5_000_000)
print(f"dict:  {t_dict:.4f}s")
print(f"slots: {t_slots:.4f}s")
```

`__slots__`의 제약을 명확히 한다. 선언하지 않은 속성을 대입할 수 없고, `__dict__`가 없으므로 `functools.cached_property`가 동작하지 않으며, 약한 참조가 필요하면 `__weakref__`를 슬롯에 포함해야 한다. 다중 상속에서 두 부모가 모두 비어 있지 않은 `__slots__`를 가지면 레이아웃 충돌로 정의가 실패한다.

```python
class A:
    __slots__ = ("x",)


class B:
    __slots__ = ("y",)


try:
    class C(A, B):
        __slots__ = ()
except TypeError as exc:
    print(f"caught: {exc}")
```

상속 계층에서 자식이 `__slots__`를 선언하지 않으면 `__dict__`가 다시 생겨 이득이 사라진다는 점도 흔한 실수다.

```python
class Parent:
    __slots__ = ("a",)


class ChildWithoutSlots(Parent):
    pass


class ChildWithSlots(Parent):
    __slots__ = ("b",)


print(hasattr(ChildWithoutSlots(), "__dict__"))
print(hasattr(ChildWithSlots(), "__dict__"))
```

### `__getattr__`과 동적 속성

설정 객체에 점 표기 접근을 붙이는 패턴이다.

```python
class DotDict:
    def __init__(self, mapping: dict):
        object.__setattr__(self, "_mapping", dict(mapping))

    def __getattr__(self, name: str):
        try:
            value = self._mapping[name]
        except KeyError:
            raise AttributeError(name) from None
        return DotDict(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value):
        self._mapping[name] = value

    def __repr__(self):
        return f"DotDict({self._mapping})"


cfg = DotDict({"optim": {"name": "adamw", "lr": 3e-4}, "seed": 42})
print(cfg.optim.lr, cfg.seed)
try:
    cfg.missing
except AttributeError as exc:
    print(f"caught: {exc}")
```

`__getattr__`은 일반 조회가 실패했을 때만 호출된다. 반면 `__getattribute__`는 모든 조회에서 호출되므로 무한 재귀에 빠지기 쉽다. `_mapping` 접근에 `object.__setattr__`을 쓴 이유가 여기 있다. `self._mapping = ...`으로 쓰면 `__setattr__`이 호출되고 그 안에서 다시 `self._mapping`을 참조해 무한 재귀가 된다.

`AttributeError`를 던질 때 `from None`을 붙인 것도 의도적이다. 내부 `KeyError`가 트레이스백에 노출되면 사용자가 혼란스러워한다.

### `__init_subclass__`와 자동 등록

앞 문서의 데코레이터 레지스트리를 상속 기반으로 바꾼 형태다.

```python
class Backbone:
    registry: dict[str, type] = {}

    def __init_subclass__(cls, name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        key = name or cls.__name__.lower()
        if key in Backbone.registry:
            raise ValueError(f"duplicate backbone: {key}")
        Backbone.registry[key] = cls


class ResNet(Backbone, name="resnet"):
    def __init__(self, depth: int = 50):
        self.depth = depth


class ConvNeXt(Backbone, name="convnext"):
    def __init__(self, width: int = 96):
        self.width = width


print(sorted(Backbone.registry))
print(Backbone.registry["resnet"](depth=101).depth)
```

메타클래스를 쓰지 않고도 서브클래스 생성 시점에 개입할 수 있다. 메타클래스는 다중 상속 충돌을 일으키기 쉬우므로 `__init_subclass__`로 해결되는 경우 이쪽을 쓴다.

## 실무 관점

설정 객체는 `frozen=True`로 만든다. 학습 도중 설정이 변경되면 로그에 기록된 설정과 실제 실행이 달라져 재현이 불가능해진다. 변형이 필요하면 `replace`로 새 객체를 만든다.

데이터 레코드가 수십만 개 이상이면 `slots=True`를 켠다. 다만 그 정도 규모면 애초에 객체가 아니라 NumPy 배열이나 Arrow 테이블로 표현하는 것이 낫다. 객체당 72바이트도 100만 개면 72MB이고, 같은 데이터를 열 지향 배열로 두면 훨씬 작고 빠르다. `__slots__`는 중간 규모에서의 최적화 도구다.

`__eq__`를 정의할 때 `__hash__`를 함께 고려한다. 가변 객체를 해시 가능하게 만들면 딕셔너리에 넣은 뒤 값을 바꿨을 때 조회가 실패하는 버그가 난다. `dataclass`는 이 규칙을 강제해서, `eq=True`이고 `frozen=False`면 `__hash__`를 `None`으로 만든다.

`NotImplemented`와 `NotImplementedError`를 혼동하지 않는다. 전자는 이항 연산 협상용 싱글턴 값이고 후자는 추상 메서드용 예외다. `__add__`에서 후자를 던지면 반사 연산 시도가 막힌다.

`__repr__`은 디버깅 품질을 좌우한다. 원칙은 가능하면 `eval`로 되살릴 수 있는 형태, 불가능하면 각괄호로 감싼 설명형이다. 텐서처럼 큰 객체는 값 전체를 찍지 말고 형상과 dtype과 디바이스만 표시한다.

```python
class TensorLike:
    def __init__(self, shape: tuple[int, ...], dtype: str, device: str):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def __repr__(self):
        return f"<TensorLike shape={self.shape} dtype={self.dtype} device={self.device}>"


print(TensorLike((32, 3, 224, 224), "float16", "cuda:0"))
```

`__del__`은 쓰지 않는다. 호출 시점이 보장되지 않고, 순환 참조가 있으면 아예 호출되지 않을 수 있으며, 인터프리터 종료 시 전역이 이미 정리된 상태에서 호출되어 이상한 예외를 낸다. 자원 정리는 컨텍스트 매니저로 한다.

상속보다 합성을 기본으로 삼는다. 다중 상속 MRO 문제, `super()` 호출 누락, 부모 클래스 변경의 파급이 모두 상속 깊이에 비례해 커진다. PyTorch의 `nn.Module`처럼 프레임워크가 요구하는 경우가 아니면 합성과 `Protocol` 조합이 안전하다.

`dataclass`의 성능 특성도 알아둔다. 생성자는 소스 코드를 문자열로 만들어 `exec`로 컴파일하므로 런타임 오버헤드는 손으로 쓴 것과 같다. 다만 클래스 정의 시점 비용이 있어, 수백 개의 dataclass를 임포트하는 모듈은 시작이 느려질 수 있다.

`__post_init__`에서 무거운 계산을 하지 않는다. `dataclass`는 값 객체이며, 생성이 저렴하다는 기대가 있다. 검증 정도만 둔다.

## 핵심 정리

속성 조회는 데이터 디스크립터, 인스턴스 딕셔너리, 비데이터 디스크립터, `__getattr__` 순으로 진행된다. `property`가 인스턴스 속성을 이기는 이유가 이 순서다.

이항 연산에서 처리할 수 없는 타입을 만나면 `NotImplemented`를 반환한다. 예외를 던지면 반사 연산 협상이 끊긴다.

`dataclass(frozen=True)`는 설정 객체의 기본값으로 삼는다. 불변이면 해시 가능해져 캐시 키로 쓸 수 있고, 실행 도중 변경으로 인한 재현 불가 문제가 사라진다.

가변 기본값은 `field(default_factory=...)`로 준다. `compare=False`와 `repr=False`로 비교와 로그에서 필드를 제외할 수 있다.

`__slots__`는 인스턴스 크기를 절반 이하로 줄이고 접근을 20에서 30퍼센트 빠르게 한다. 대신 동적 속성, `cached_property`, 일부 다중 상속이 막힌다. 자식 클래스가 `__slots__`를 선언하지 않으면 이득이 사라진다.

`__getattr__`은 조회 실패 시에만 호출되고 `__getattribute__`는 항상 호출된다. 후자를 오버라이드하면 무한 재귀에 빠지기 쉽다.

`__del__`에 자원 정리를 의존하지 않는다. 컨텍스트 매니저를 쓴다.
