# 컨텍스트 매니저와 리소스 관리

## 한 줄 정의

컨텍스트 매니저는 블록 진입 시 자원을 확보하고, 블록을 어떻게 벗어나든 반드시 정리 코드를 실행하도록 언어 차원에서 보장하는 객체다.

## 문제 상황

자원을 쓰고 되돌려놓는 코드는 성공 경로와 실패 경로 양쪽에서 실행되어야 한다. `try`/`finally`만으로 이를 쓰면 중첩이 빠르게 감당하기 어려워진다.

```python
f = open("shard.jsonl", encoding="utf-8")
try:
    conn = open_db()
    try:
        prev_mode = model.training
        model.eval()
        try:
            run(f, conn, model)
        finally:
            model.train(prev_mode)
    finally:
        conn.close()
finally:
    f.close()
```

세 개의 자원에 세 겹의 `try`가 필요하고, 정리 순서가 역순이어야 한다는 규칙을 사람이 지켜야 한다. 하나라도 빠지면 학습 도중 예외가 났을 때 파일 핸들이 남거나, 모델이 `eval` 모드에 갇힌 채 다음 에폭으로 넘어간다.

딥러닝 코드에서 이 결핍이 특히 비싼 이유는 상태가 전역적이기 때문이다. `torch.no_grad()`를 켜고 예외가 나면 그래디언트가 꺼진 채 학습이 계속되고, 손실은 떨어지지 않는데 원인은 수십 줄 위에 있다. GPU 메모리 역시 참조가 남으면 회수되지 않아 다음 에폭에서 OOM으로 나타난다. 문제가 발생한 지점과 증상이 드러나는 지점이 멀어진다는 점이 핵심 어려움이다.

## 직관적 이해

컨텍스트 매니저는 출입 기록이 강제되는 방이다. 들어갈 때 이름을 적고, 나갈 때는 정문으로 나가든 창문으로 뛰어내리든(예외) 반드시 기록이 남는다. Python은 `with` 블록을 벗어나는 모든 경로에 정리 코드를 삽입한다.

`__exit__`이 예외 정보를 인자로 받는 것은 "어떻게 나갔는지"를 알아야 정리 방식을 바꿀 수 있기 때문이다. 정상 종료면 커밋하고 예외면 롤백하는 트랜잭션이 대표적이다.

## 형식화

`with` 문의 의미는 컴파일러가 수행하는 변환으로 정확히 정의된다. 다음 코드

```python
with expr as target:
    body
```

는 아래와 동등하다.

```python
manager = expr
exit_fn = type(manager).__exit__
value = type(manager).__enter__(manager)
ok = True
try:
    target = value
    body
except BaseException:
    ok = False
    if not exit_fn(manager, *sys.exc_info()):
        raise
finally:
    if ok:
        exit_fn(manager, None, None, None)
```

여기서 세 가지가 결정된다.

첫째, `__enter__`와 `__exit__`은 인스턴스가 아니라 타입에서 조회된다. 인스턴스 속성으로 덮어써도 무시된다는 뜻이다.

둘째, `__exit__`의 반환값이 참으로 평가되면 예외가 억제된다. 그렇지 않으면 원래 예외가 다시 올라간다. 실수로 `__exit__`에서 값을 반환하면 예외가 조용히 삼켜지므로, 억제 의도가 없으면 아무것도 반환하지 않는다.

셋째, `as target` 대입은 `__enter__` 반환값에 대해 일어난다. `__enter__`가 `self`를 반환할 의무는 없다.

여러 자원을 나열하면 좌에서 우로 진입하고 우에서 좌로 정리된다.

$$\texttt{with } A, B: \ \text{body} \quad \equiv \quad \texttt{with } A: \ \texttt{with } B: \ \text{body}$$

정리 순서가 역순인 이유는 나중에 얻은 자원이 먼저 얻은 자원에 의존할 수 있기 때문이다. 데이터베이스 연결 위에서 트랜잭션을 열었다면 트랜잭션을 먼저 닫아야 한다.

## 구현

### 클래스 기반 구현

앞의 문제 중 하나였던 모델 모드 전환을 컨텍스트 매니저로 만든다.

```python
class EvalMode:
    def __init__(self, module):
        self.module = module
        self.prev = None

    def __enter__(self):
        self.prev = self.module.training
        self.module.eval()
        return self.module

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.train(self.prev)


class FakeModule:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self, mode: bool = True):
        self.training = mode


module = FakeModule()
print(module.training)
try:
    with EvalMode(module):
        print(module.training)
        raise RuntimeError("validation crashed")
except RuntimeError:
    pass
print(module.training)
```

예외가 났는데도 `training`이 `True`로 복원된다. `__exit__`이 아무것도 반환하지 않으므로 예외는 그대로 전파된다. 앞의 형식화에서 본 세 번째 규칙이다.

`self.prev`를 저장했다가 복원하는 방식이 `model.train()`을 무조건 호출하는 것보다 낫다. 중첩 사용 시 바깥 컨텍스트의 상태가 보존되기 때문이다.

### contextlib.contextmanager

정리 로직이 짧으면 제너레이터 기반이 훨씬 읽기 좋다. `yield` 앞이 `__enter__`, 뒤가 `__exit__`이 된다.

```python
import contextlib
import time


@contextlib.contextmanager
def timed_block(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{name}: {time.perf_counter() - start:.4f}s")


with timed_block("tokenize"):
    total = sum(i * i for i in range(200000))

print(total)
```

`try`/`finally`가 필수다. 이것이 없으면 블록 안에서 예외가 났을 때 제너레이터가 `yield` 지점에서 멈춘 채 정리 코드가 실행되지 않는다.

예외 정보를 보고 분기하려면 `except`를 함께 쓴다.

```python
import contextlib


@contextlib.contextmanager
def transaction(store: dict):
    snapshot = dict(store)
    try:
        yield store
    except Exception:
        store.clear()
        store.update(snapshot)
        raise
    finally:
        pass


registry: dict[str, float] = {"lr": 3e-4}
try:
    with transaction(registry) as tx:
        tx["momentum"] = 0.9
        raise ValueError("bad config")
except ValueError:
    pass
print(registry)
```

`raise`를 다시 던지지 않으면 예외가 억제된다. 제너레이터 방식에서는 예외를 삼키려면 그냥 `pass`하면 되고, 전파하려면 `raise`를 명시한다.

`yield` 값을 `as`로 받는 것도 확인할 수 있다. `tx`가 `store`와 같은 객체다.

### 재진입과 재사용

제너레이터 기반 컨텍스트 매니저는 일회용이다. 같은 객체를 두 번 쓰면 실패한다.

```python
import contextlib


@contextlib.contextmanager
def once():
    print("enter")
    yield
    print("exit")


cm = once()
with cm:
    pass
try:
    with cm:
        pass
except RuntimeError as exc:
    print(f"caught: {exc}")
```

`generator didn't yield`라는 오류가 난다. 데코레이터로 재사용 가능하게 만들려면 `ContextDecorator`를 상속하거나 팩토리를 매번 호출한다. `contextlib.contextmanager`로 만든 객체는 실제로 `ContextDecorator`를 상속하므로 함수 데코레이터로 바로 쓸 수 있다.

```python
import contextlib
import time


@contextlib.contextmanager
def timed_block(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{name}: {time.perf_counter() - start:.4f}s")


@timed_block("epoch")
def run_epoch(n: int):
    return sum(range(n))


print(run_epoch(100000))
print(run_epoch(100000))
```

데코레이터로 쓸 때는 호출마다 새 제너레이터가 만들어지므로 재사용 문제가 없다.

### ExitStack

자원 수가 실행 시점에 결정되면 `with`를 정적으로 중첩할 수 없다. `ExitStack`이 이를 해결한다.

```python
import contextlib
import io


def merge_shards(paths: list[str]):
    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(io.StringIO(f"line from {p}\n")) for p in paths]
        return [h.getvalue().strip() for h in handles]


print(merge_shards(["a.jsonl", "b.jsonl", "c.jsonl"]))
```

`enter_context`로 등록한 자원은 스택이 닫힐 때 역순으로 정리된다. 개수가 0이어도 코드가 그대로 동작한다는 점이 중요하다.

`callback`으로 임의의 정리 함수를 등록할 수도 있고, `pop_all`로 정리 책임을 넘길 수도 있다. 후자는 팩토리 함수에서 자원을 만들어 반환할 때 쓴다.

```python
import contextlib


class Resource:
    def __init__(self, name: str):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True
        print(f"closed {self.name}")


def build_pipeline():
    with contextlib.ExitStack() as stack:
        first = Resource("reader")
        stack.callback(first.close)
        second = Resource("writer")
        stack.callback(second.close)
        if first.closed:
            raise RuntimeError("unexpected")
        stack.pop_all()
        return first, second, stack


reader, writer, owner = build_pipeline()
print(reader.closed, writer.closed)
owner.close()
```

`pop_all()` 이전에 예외가 나면 두 자원이 모두 정리되고, 성공하면 정리 책임이 반환된 스택으로 넘어간다. 부분 초기화 실패 시 누수를 막는 표준 패턴이다.

### 비동기 컨텍스트 매니저

`asyncio` 기반 코드에서는 `__aenter__`, `__aexit__`을 쓴다. 정리 과정 자체가 await를 필요로 할 때 필수다.

```python
import asyncio
import contextlib


@contextlib.asynccontextmanager
async def acquire_slot(sem: asyncio.Semaphore, name: str):
    await sem.acquire()
    try:
        yield name
    finally:
        sem.release()


async def worker(sem: asyncio.Semaphore, idx: int):
    async with acquire_slot(sem, f"w{idx}") as slot:
        await asyncio.sleep(0.01)
        return slot


async def main():
    sem = asyncio.Semaphore(2)
    results = await asyncio.gather(*(worker(sem, i) for i in range(5)))
    return results


print(asyncio.run(main()))
```

동시 실행 수가 2로 제한되면서도 예외 발생 시 세마포어가 반드시 반환된다.

### 실전 조합: 학습 스텝 컨텍스트

앞의 요소들을 모아 학습 루프에서 실제로 쓰는 형태를 만든다. 시드 고정, 자원 정리, 프로파일링 구간 표시를 하나의 진입점으로 묶는다.

```python
import contextlib
import os
import random
import time

import numpy as np


@contextlib.contextmanager
def deterministic(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    hash_seed = os.environ.get("PYTHONHASHSEED")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        yield seed
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = hash_seed


@contextlib.contextmanager
def stage(name: str, timings: dict[str, float]):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - start


timings: dict[str, float] = {}
with deterministic(42):
    with stage("sample", timings):
        first = np.random.normal(size=(4,))
with deterministic(42):
    with stage("sample", timings):
        second = np.random.normal(size=(4,))

print(np.allclose(first, second))
print({k: round(v, 6) for k, v in timings.items()})
```

`deterministic`이 이전 상태를 저장했다가 복원한다는 점이 중요하다. 시드를 고정한 채 빠져나오면 이후 코드의 무작위성이 전부 오염된다. 테스트에서 특히 문제가 된다.

`stage`가 누적 방식인 이유는 학습 루프에서 같은 구간이 반복 호출되기 때문이다. 마지막 값만 남기면 병목 판단에 쓸 수 없다.

### suppress와 redirect

간단한 도구 두 개를 확인한다.

```python
import contextlib
import io
import os

with contextlib.suppress(FileNotFoundError):
    os.remove("checkpoint_that_does_not_exist.pt")
print("continued")

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    print("this goes to the buffer")
print(f"captured: {buffer.getvalue().strip()}")
```

`suppress`는 `try`/`except`/`pass`보다 의도가 명확하다. 다만 억제할 예외 타입을 반드시 좁게 지정한다. `Exception`을 억제하면 버그를 숨긴다.

`redirect_stdout`은 출력을 직접 내보내는 외부 라이브러리를 로거로 흡수할 때 쓴다. 진행 표시줄을 찍는 라이브러리를 CI 로그에서 조용하게 만드는 용도가 흔하다.

## 실무 관점

컨텍스트 매니저를 만들 시점을 판단하는 기준은 "블록 진입 시 바꾼 상태를 나갈 때 되돌려야 하는가"다. 파일과 소켓은 명백하지만, 전역 설정 변경, 모델 모드, 그래디언트 활성화, 로깅 레벨, 작업 디렉터리, 환경 변수도 모두 대상이다. 이런 것들이 복원되지 않아 생기는 버그가 파일 핸들 누수보다 훨씬 자주 발생한다.

`__exit__`에서 예외를 억제할 때는 극도로 보수적으로 접근한다. 반환값이 참이면 예외가 사라지고 호출자는 성공했다고 믿는다. 학습 스크립트가 조용히 잘못된 체크포인트를 저장하는 경로가 여기서 생긴다. 억제하려면 예외 타입을 좁히고 로그를 남긴다.

`__exit__` 안에서 발생한 예외가 원래 예외를 가린다는 점도 주의한다. Python 3부터는 `__context__`로 원래 예외가 연결되어 트레이스백에 함께 표시되지만, 최상단 메시지는 정리 중 발생한 예외가 된다. 정리 코드는 자체적으로 방어한다.

```python
import contextlib


@contextlib.contextmanager
def safe_cleanup(resource):
    try:
        yield resource
    finally:
        try:
            resource.close()
        except Exception as exc:
            print(f"cleanup failed: {exc}")
```

GPU 메모리 관련해서는 컨텍스트 매니저만으로 해결되지 않는 부분이 있다. Python 객체 참조가 끊겨도 캐싱 할당자가 메모리를 붙들고 있기 때문이다. 참조를 끊는 것과 할당자에 반환하는 것은 다른 일이다.

```python
import contextlib
import gc


@contextlib.contextmanager
def released_after(*names):
    scope: dict = {}
    try:
        yield scope
    finally:
        for name in names:
            scope.pop(name, None)
        gc.collect()
```

실제 PyTorch 코드에서는 `finally` 안에서 `torch.cuda.empty_cache()`를 함께 호출한다. 다만 `empty_cache`는 할당자 성능을 떨어뜨리므로 학습 루프 안에서 매 스텝 호출하지 않는다. 검증 단계 전후처럼 큰 메모리 패턴이 바뀌는 경계에서만 쓴다.

중첩 깊이가 세 단계를 넘으면 `ExitStack`으로 평탄화한다. 가독성뿐 아니라 자원 수가 조건부일 때 코드 분기가 폭발하는 것을 막는다.

성능은 거의 무시할 수 있다. `with` 진입과 종료는 메서드 호출 두 번이며 마이크로초 이하다. 다만 제너레이터 기반은 클래스 기반보다 약간 느리므로, 텐서 연산마다 진입하는 초고빈도 경로에서는 클래스 기반이나 인라인 `try`/`finally`를 쓴다.

테스트에서의 활용도 중요하다. 픽스처가 곧 컨텍스트 매니저이며, pytest의 `yield` 픽스처는 앞에서 본 제너레이터 규약과 정확히 같은 구조다. 임시 디렉터리, 임시 환경 변수, 임시 설정 패치가 모두 이 형태로 작성된다.

```python
import contextlib
import os


@contextlib.contextmanager
def env_override(**overrides: str):
    previous = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


print(os.environ.get("CUDA_VISIBLE_DEVICES"))
with env_override(CUDA_VISIBLE_DEVICES="0,1"):
    print(os.environ["CUDA_VISIBLE_DEVICES"])
print(os.environ.get("CUDA_VISIBLE_DEVICES"))
```

원래 값이 없었던 경우와 빈 문자열이었던 경우를 구분하는 것이 핵심이다. `None` 체크 없이 복원하면 없던 환경 변수가 생겨난다.

## 핵심 정리

`with`는 진입 경로와 무관하게 정리 코드 실행을 언어 차원에서 보장한다. `__enter__`와 `__exit__`은 인스턴스가 아니라 타입에서 조회된다.

`__exit__`의 반환값이 참이면 예외가 억제된다. 억제 의도가 없으면 아무것도 반환하지 않는다. 실수로 값을 반환해 예외가 사라지는 사고가 흔하다.

`contextlib.contextmanager`에서 `try`/`finally`는 필수다. 없으면 예외 발생 시 정리 코드가 실행되지 않는다.

자원 수가 실행 시점에 결정되면 `ExitStack`을 쓴다. `pop_all()`로 정리 책임을 호출자에게 넘길 수 있어 부분 초기화 실패 시 누수를 막는다.

파일 핸들뿐 아니라 전역 상태(모델 모드, 시드, 환경 변수, 로깅 레벨)를 바꾸는 모든 코드가 컨텍스트 매니저 대상이다. 상태 복원 실패로 생기는 버그가 훨씬 찾기 어렵다.

시드를 고정하는 컨텍스트는 반드시 이전 상태를 저장했다가 복원한다. 그렇지 않으면 이후 모든 무작위성이 오염된다.
