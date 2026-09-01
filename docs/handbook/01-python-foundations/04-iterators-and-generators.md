# 이터레이터, 제너레이터, 지연 평가

## 한 줄 정의

이터레이터는 다음 값을 요청받을 때마다 하나씩 내주는 객체이고, 제너레이터는 함수의 실행 상태를 통째로 보존해 그 객체를 자동으로 만들어 주는 문법이다.

## 문제 상황

학습 데이터가 메모리보다 클 때 리스트 기반 코드는 그대로 실패한다.

```python
def load_all(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows
```

100GB짜리 코퍼스에 이 함수를 쓰면 프로세스가 죽는다. 전처리 파이프라인을 여러 단계로 연결하면 상황이 더 나빠진다. 토큰화 결과 리스트, 필터링 결과 리스트, 청킹 결과 리스트가 각각 만들어져 중간 산출물이 원본의 몇 배를 차지한다.

메모리가 충분한 경우에도 문제가 남는다. 첫 배치를 보려면 전체 전처리가 끝나야 한다. 100만 개 중 앞의 10개만 확인하고 싶어도 100만 개를 다 처리한다. 디버깅 반복 주기가 몇 분 단위로 늘어난다.

무한 스트림은 아예 표현할 수 없다. 학습률 스케줄, 무한 반복 데이터 로더, 온라인 학습 스트림은 끝이 없으므로 리스트로 만들 수 없다.

## 직관적 이해

리스트는 창고에 물건을 전부 쌓아놓는 것이고, 제너레이터는 주문이 올 때마다 하나씩 만들어 보내는 것이다. 창고 비용이 들지 않는 대신 이미 보낸 물건을 다시 요청하면 처음부터 다시 만들어야 한다.

제너레이터 함수의 특이한 점은 `return`이 아니라 `yield`에서 멈춘다는 것이다. 보통 함수는 반환하면 지역 변수와 실행 위치가 사라지지만, 제너레이터는 그 상태를 그대로 얼려둔다. 다음 요청이 오면 얼렸던 지점 바로 다음 줄부터 녹여서 이어간다. 함수 호출이 아니라 일시정지와 재개다.

파이프라인으로 연결하면 각 단계가 한 개씩만 처리하고 다음 단계로 넘긴다. 컨베이어 벨트에서 물건 하나가 모든 공정을 통과한 뒤 다음 물건이 출발하는 구조이며, 그래서 어느 시점에도 메모리에는 물건 몇 개만 있다.

## 형식화

이터레이션 프로토콜은 두 개의 메서드로 정의된다. 객체 $C$가 이터러블이라는 것은 `__iter__`를 구현해 이터레이터 $I$를 반환한다는 뜻이고, $I$가 이터레이터라는 것은 `__next__`를 구현하며 자기 자신을 반환하는 `__iter__`를 갖는다는 뜻이다.

`for` 문의 의미는 다음 변환으로 정의된다.

```python
for x in C:
    body
```

는 아래와 동등하다.

```python
it = iter(C)
while True:
    try:
        x = next(it)
    except StopIteration:
        break
    body
```

이터레이터가 자기 자신을 반환하는 `__iter__`를 갖는 이유가 여기서 나온다. `for`는 이터러블이든 이터레이터든 `iter()`를 부르므로, 이터레이터도 그 호출을 견뎌야 한다.

제너레이터 함수 $g$를 호출하면 본문은 실행되지 않고 제너레이터 객체가 생성된다. 상태는 다음 세 가지로 구성된다.

$$\text{state}(g) = (\text{frame}, \text{ip}, \text{status})$$

여기서 frame은 지역 변수 사전, ip는 다음에 실행할 바이트코드 위치, status는 `CREATED`, `SUSPENDED`, `RUNNING`, `CLOSED` 중 하나다. `next()`는 status를 `RUNNING`으로 바꾸고 ip부터 실행하다가 `yield`를 만나면 값을 내보내며 `SUSPENDED`로 돌아간다.

메모리 관점에서 리스트와 제너레이터의 차이는 명확하다. 원소 $n$개, 원소당 크기 $s$인 시퀀스에 대해

$$M_{\text{list}} = O(n \cdot s), \qquad M_{\text{gen}} = O(s + f)$$

이며 $f$는 프레임 크기(지역 변수 몇 개 수준)다. $k$단계 파이프라인이면 리스트 방식은 $O(k \cdot n \cdot s)$, 제너레이터 방식은 $O(k \cdot s + k \cdot f)$다. 단계 수에 대해서도 데이터 크기에 대해서도 상수 배만 든다.

시간 관점에서는 첫 원소까지의 지연이 다르다. 리스트는 $O(n)$을 소모한 뒤 첫 원소를 주지만 제너레이터는 $O(1)$이다. 전체를 소비하면 총 시간은 같으므로, 제너레이터의 이득은 메모리와 응답 지연에 있지 처리량에 있지 않다.

## 구현

### 프로토콜 직접 구현

먼저 이터레이터를 손으로 만들어 프로토콜을 확인한다.

```python
class Windowed:
    def __init__(self, values: list[int], size: int):
        self.values = values
        self.size = size

    def __iter__(self):
        return WindowedIterator(self.values, self.size)


class WindowedIterator:
    def __init__(self, values: list[int], size: int):
        self.values = values
        self.size = size
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos + self.size > len(self.values):
            raise StopIteration
        chunk = self.values[self.pos:self.pos + self.size]
        self.pos += 1
        return chunk


series = Windowed([1, 2, 3, 4, 5], 3)
print(list(series))
print(list(series))
```

이터러블과 이터레이터를 분리했기 때문에 두 번 순회해도 같은 결과가 나온다. 앞의 형식화에서 본 `iter(C)`가 매번 새 이터레이터를 만들기 때문이다.

만약 `Windowed`가 `__next__`를 직접 구현하고 `__iter__`에서 `self`를 반환했다면 두 번째 순회는 빈 리스트가 된다. 이것이 실무에서 "데이터 로더를 두 번째 에폭에서 돌렸더니 아무것도 안 나온다"의 원인이다.

### 제너레이터로 같은 것

같은 로직을 제너레이터로 쓰면 상태 관리 코드가 전부 사라진다.

```python
from collections.abc import Sequence


def windowed(values: Sequence[int], size: int):
    for pos in range(len(values) - size + 1):
        yield list(values[pos:pos + size])


print(list(windowed([1, 2, 3, 4, 5], 3)))
```

`self.pos`가 함수의 지역 변수 `pos`로 대체되고, `StopIteration`은 함수가 끝나면 자동으로 발생한다. 앞서 본 프레임 보존이 이 단순화의 실체다.

제너레이터 객체의 상태를 직접 확인할 수 있다.

```python
import inspect


def counter(limit: int):
    total = 0
    for i in range(limit):
        total += i
        yield total


gen = counter(3)
print(inspect.getgeneratorstate(gen))
print(next(gen))
print(inspect.getgeneratorstate(gen))
print(gen.gi_frame.f_locals)
print(list(gen))
print(inspect.getgeneratorstate(gen))
```

`gi_frame.f_locals`로 얼려둔 지역 변수를 들여다볼 수 있다. 형식화에서 말한 frame이 이것이다.

### 파이프라인 구성

핵심 이득이 드러나는 부분이다. 각 단계를 제너레이터로 만들고 연결한다.

```python
import json
from collections.abc import Iterable, Iterator


def read_lines(paths: Iterable[str]):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def parse_json(lines: Iterable[str]):
    for line in lines:
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def filter_by_length(records: Iterable[dict], min_chars: int, max_chars: int):
    for record in records:
        text = record.get("text", "")
        if min_chars <= len(text) <= max_chars:
            yield record


def to_chunks(records: Iterable[dict], chunk_chars: int, overlap: int):
    for record in records:
        text = record["text"]
        step = chunk_chars - overlap
        for start in range(0, max(len(text) - overlap, 1), step):
            piece = text[start:start + chunk_chars]
            if piece:
                yield {"source": record.get("id"), "text": piece}


def batched(items: Iterable[dict], size: int):
    buffer: list[dict] = []
    for item in items:
        buffer.append(item)
        if len(buffer) == size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer
```

동작을 확인하기 위해 임시 파일을 만든다.

```python
import json
import os
import tempfile

tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "corpus.jsonl")
with open(path, "w", encoding="utf-8") as f:
    for i in range(50):
        f.write(json.dumps({"id": i, "text": "가" * (20 + i * 3)}, ensure_ascii=False) + "\n")
    f.write("not json\n")

pipeline = batched(
    to_chunks(
        filter_by_length(parse_json(read_lines([path])), min_chars=30, max_chars=140),
        chunk_chars=50,
        overlap=10,
    ),
    size=8,
)

first_batch = next(pipeline)
print(len(first_batch))
print(first_batch[0])
```

`next(pipeline)`이 실행되는 순간 파일에서 읽히는 줄은 8개 청크를 만들기에 충분한 만큼뿐이다. 나머지 45줄은 아직 디스크에 있다. 파이프라인이 다섯 단계인데도 메모리에는 청크 몇 개만 존재한다.

`parse_json`에서 잘못된 줄을 건너뛴 것도 확인된다. 파싱 실패가 파이프라인 전체를 죽이지 않는다.

### 메모리 실측

이론적 주장을 숫자로 확인한다.

```python
import sys
import tracemalloc


def build_list(n: int):
    return [i * i for i in range(n)]


def build_gen(n: int):
    return (i * i for i in range(n))


n = 2_000_000

tracemalloc.start()
lst = build_list(n)
current_list, peak_list = tracemalloc.get_traced_memory()
del lst
tracemalloc.stop()

tracemalloc.start()
gen = build_gen(n)
first = next(gen)
current_gen, peak_gen = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"list peak: {peak_list / 1024 / 1024:.1f} MiB")
print(f"gen  peak: {peak_gen / 1024:.1f} KiB")
print(f"gen object size: {sys.getsizeof(gen)} bytes")
```

리스트는 수십 MiB, 제너레이터는 KiB 단위다. 형식화의 $O(n \cdot s)$ 대 $O(s + f)$가 그대로 나타난다.

전체를 소비할 때의 시간을 비교하면 제너레이터가 약간 느리다. `yield`마다 프레임 전환 비용이 들기 때문이다. 메모리와 지연을 시간과 맞바꾸는 거래다.

```python
import time


def timeit(fn, repeat: int = 3):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


n = 1_000_000
print(f"list sum: {timeit(lambda: sum([i * i for i in range(n)])):.4f}s")
print(f"gen  sum: {timeit(lambda: sum(i * i for i in range(n))):.4f}s")
```

### yield from과 위임

중첩 제너레이터를 평탄화할 때 수동 루프 대신 `yield from`을 쓴다.

```python
from collections.abc import Iterable


def flatten(nested):
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item


print(list(flatten([1, [2, 3, [4, [5]]], (6, 7)])))
```

`yield from`은 단순 루프의 축약이 아니다. `send`, `throw`, `close`를 하위 제너레이터에 그대로 전달하고 하위의 반환값을 표현식 값으로 준다.

```python
def inner():
    total = 0
    while True:
        received = yield
        if received is None:
            return total
        total += received


def outer():
    result = yield from inner()
    yield f"sum={result}"


gen = outer()
next(gen)
gen.send(10)
gen.send(20)
print(gen.send(None))
```

`inner`의 `return total`이 `yield from`의 값이 되어 `outer`로 전달된다.

### send와 코루틴형 제너레이터

`send`는 제너레이터에 값을 밀어 넣는다. 이동 평균처럼 상태를 유지하는 소비자를 만들 수 있다.

```python
import functools


def priming(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        gen = fn(*args, **kwargs)
        next(gen)
        return gen

    return wrapper


@priming
def running_mean():
    total = 0.0
    count = 0
    average = None
    while True:
        value = yield average
        total += value
        count += 1
        average = total / count


meter = running_mean()
for loss in [1.0, 0.8, 0.6, 0.4]:
    print(round(meter.send(loss), 4))
```

`priming` 데코레이터가 필요한 이유는 첫 `send` 전에 제너레이터가 첫 `yield`까지 진행되어 있어야 하기 때문이다. 이를 빠뜨리면 `TypeError: can't send non-None value to a just-started generator`가 난다.

현대 Python에서 이런 용도는 대부분 `async`/`await`로 대체되었지만, 프레임워크 내부와 레거시 코드에서 여전히 만난다.

### close와 정리

제너레이터 안에서 자원을 열면 정리 시점을 신경 써야 한다.

```python
def reading(path: str):
    print("open")
    handle = open(path, encoding="utf-8")
    try:
        for line in handle:
            yield line.rstrip("\n")
    finally:
        handle.close()
        print("close")


import os
import tempfile

tmp = os.path.join(tempfile.mkdtemp(), "x.txt")
with open(tmp, "w", encoding="utf-8") as f:
    f.write("a\nb\nc\n")

gen = reading(tmp)
print(next(gen))
gen.close()
```

`close()`는 `yield` 지점에 `GeneratorExit`를 던지고, `finally`가 실행된다. 부분 소비 후 버려지는 제너레이터가 자원을 잡고 있지 않게 하려면 `finally`가 필수다.

`GeneratorExit`를 잡아서 `yield`를 더 하면 `RuntimeError`가 난다. 정리 중에는 값을 내보낼 수 없다.

### itertools 조합

표준 라이브러리가 제공하는 지연 평가 도구를 학습 코드 맥락에서 정리한다.

```python
import itertools


def infinite_shards(shard_count: int):
    for epoch in itertools.count():
        order = list(range(shard_count))
        for shard in order:
            yield epoch, shard


stream = infinite_shards(3)
print(list(itertools.islice(stream, 7)))

lrs = itertools.chain(
    (1e-5 * (i + 1) for i in range(3)),
    itertools.repeat(3e-4, 4),
)
print([round(v, 6) for v in lrs])

records = [("a", 1), ("a", 2), ("b", 3), ("b", 4), ("c", 5)]
for key, group in itertools.groupby(records, key=lambda r: r[0]):
    print(key, [v for _, v in group])
```

`islice`가 무한 스트림에서 앞부분만 꺼내는 표준 방법이다. `groupby`는 정렬되어 있어야 의미가 있다는 점이 함정이다. 정렬되지 않은 입력에 쓰면 같은 키가 여러 그룹으로 쪼개진다.

`tee`는 하나의 이터레이터를 여러 개로 복제하지만, 소비 속도가 다르면 내부 버퍼가 커진다.

```python
import itertools

source = (i for i in range(10))
a, b = itertools.tee(source, 2)
print(list(itertools.islice(a, 3)))
print(list(itertools.islice(b, 3)))
```

앞선 쪽이 소비한 만큼이 뒤처진 쪽을 위해 버퍼에 남는다. 한쪽만 끝까지 소비하면 전체가 메모리에 올라가므로, `tee` 이후 두 갈래를 번갈아 소비하는 구조에서만 쓴다.

## 실무 관점

DataLoader의 `IterableDataset`이 제너레이터 기반이다. 이때 워커가 여러 개면 각 워커가 같은 스트림을 처음부터 읽어 데이터가 중복된다. 워커 정보를 읽어 샤딩해야 한다.

```python
from collections.abc import Iterator


def sharded_stream(paths: list[str], worker_id: int, num_workers: int):
    for idx, path in enumerate(paths):
        if idx % num_workers != worker_id:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")


paths = [f"shard_{i}.jsonl" for i in range(8)]
assigned = [[p for i, p in enumerate(paths) if i % 4 == w] for w in range(4)]
print(assigned)
```

파일 수가 워커 수로 나누어떨어지지 않으면 워커별 부하가 달라진다. 가장 느린 워커가 에폭 시간을 결정하므로, 파일 수를 워커 수의 배수로 맞추거나 파일 내부 오프셋 단위로 샤딩한다.

제너레이터가 소비되고 나면 비어 있다는 점이 실무 버그의 단골이다.

```python
scores = (i * 0.1 for i in range(5))
total = sum(scores)
count = sum(1 for _ in scores)
print(total, count)
```

`count`가 0이 된다. 두 번 순회해야 하면 리스트로 물질화하거나 이터러블을 반환하는 팩토리를 넘긴다.

`len()`이 동작하지 않는다는 점도 같은 성질이다. 진행 표시줄을 쓰려면 총 개수를 별도로 전달한다.

디버깅 방법을 정리한다. 파이프라인 중간을 들여다보려면 통과시키며 관찰하는 단계를 끼운다.

```python
from collections.abc import Iterable


def spy(items: Iterable, label: str, every: int = 1000):
    for idx, item in enumerate(items):
        if idx % every == 0:
            print(f"{label}[{idx}]: {str(item)[:80]}")
        yield item


sample = spy((i * i for i in range(5)), "squares", every=2)
print(list(sample))
```

예외 트레이스백이 읽기 어려워지는 문제도 있다. 파이프라인이 다섯 단계면 예외가 어느 단계에서 났는지 프레임을 따라가야 한다. 각 단계에서 예외를 잡아 맥락을 붙여 다시 던지면 진단이 쉬워진다.

```python
from collections.abc import Iterable


def annotated(items: Iterable, stage: str):
    idx = 0
    it = iter(items)
    while True:
        try:
            item = next(it)
        except StopIteration:
            return
        except Exception as exc:
            raise RuntimeError(f"stage={stage} index={idx}") from exc
        idx += 1
        yield item
```

`raise ... from exc`로 원인을 연결하는 것이 핵심이다. 원본 트레이스백이 유지된다.

성능 측면에서 주의할 점은 제너레이터가 CPU 병목을 해결하지 못한다는 것이다. 지연 평가는 메모리 문제를 풀지 처리량 문제를 풀지 않는다. 전처리가 GPU를 굶기고 있다면 제너레이터가 아니라 멀티프로세싱이나 벡터화가 답이다.

제너레이터 안에서 `StopIteration`을 직접 던지면 Python 3.7부터 `RuntimeError`로 변환된다. `PEP 479`의 변경으로, 내부 `next()` 호출이 던진 `StopIteration`이 조용히 제너레이터를 끝내던 버그를 막기 위한 것이다. 종료는 `return`으로 표현한다.

```python
def take_until_none(source):
    it = iter(source)
    while True:
        try:
            value = next(it)
        except StopIteration:
            return
        if value is None:
            return
        yield value


print(list(take_until_none([1, 2, None, 3])))
```

## 핵심 정리

이터러블은 `__iter__`로 새 이터레이터를 만들고, 이터레이터는 `__next__`와 자기 자신을 반환하는 `__iter__`를 갖는다. 두 역할을 한 객체에 합치면 두 번째 순회가 빈 결과를 낸다.

제너레이터는 프레임을 얼려 두었다가 재개한다. 메모리는 $O(n \cdot s)$에서 $O(s + f)$로 줄고 첫 원소까지의 지연은 $O(n)$에서 $O(1)$로 줄지만, 전체 처리량은 오히려 약간 느리다.

다단계 파이프라인에서 이득이 곱해진다. $k$단계 리스트 방식이 $O(k \cdot n \cdot s)$인 반면 제너레이터 방식은 단계 수에 대해서만 선형이다.

제너레이터는 한 번만 소비된다. 두 번 순회가 필요하면 물질화하거나 팩토리를 넘긴다. `len()`도 동작하지 않는다.

자원을 여는 제너레이터는 `try`/`finally`로 정리를 보장한다. `close()`가 `GeneratorExit`를 던지므로 부분 소비 후 버려져도 핸들이 닫힌다.

제너레이터 안에서 `StopIteration`을 던지면 `RuntimeError`가 된다. 종료는 `return`으로 표현한다.

지연 평가는 메모리 문제를 풀지 CPU 병목을 풀지 않는다. GPU가 굶고 있다면 답은 다른 곳에 있다.
