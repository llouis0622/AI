# 동시성: GIL, threading, multiprocessing, asyncio

## 한 줄 정의

Python의 동시성 선택은 작업이 GIL을 붙잡고 있는지 놓고 있는지에 따라 결정되며, CPU 작업은 프로세스로, 블로킹 I/O는 스레드로, 대량 네트워크 I/O는 코루틴으로 처리한다.

## 문제 상황

GPU 사용률이 30퍼센트에서 오르지 않는 상황을 생각한다. `nvidia-smi`를 보면 GPU가 주기적으로 놀고 있고, 프로파일러를 붙이면 학습 스텝 사이에 긴 공백이 있다. 원인은 대부분 데이터 파이프라인이다. 이미지 디코딩과 증강이 CPU에서 순차 실행되면서 GPU가 다음 배치를 기다린다.

단순히 스레드를 늘리는 것으로는 해결되지 않는다. 순수 Python 코드는 GIL 때문에 스레드를 늘려도 총 처리량이 늘지 않기 때문이다. 여기서 "왜 스레드가 안 되는가"를 정확히 이해하지 못하면 잘못된 방향으로 몇 시간을 쓴다.

반대 상황도 있다. LLM API를 1000번 호출해야 하는데 순차 실행하면 각 호출이 2초씩 걸려 33분이 든다. 여기에 프로세스를 쓰면 프로세스 1000개를 만드느라 메모리가 터진다. 이 경우 필요한 것은 코루틴이다.

세 번째 상황은 분산 학습이다. 8개 GPU에 프로세스를 하나씩 띄우는데, 각 프로세스가 다시 DataLoader 워커를 4개씩 만들면 32개 프로세스가 된다. fork 방식과 CUDA 초기화의 상호작용에서 데드락이 발생한다.

세 문제의 답이 각각 다르다는 것이 핵심이다.

## 직관적 이해

GIL은 인터프리터 내부 자료구조를 지키는 단 하나의 열쇠다. Python 바이트코드를 실행하려면 이 열쇠가 필요하고, 열쇠는 하나뿐이므로 한 순간에 한 스레드만 바이트코드를 실행한다. 다만 열쇠를 놓는 순간이 있다. 파일을 읽거나 네트워크를 기다리거나 NumPy가 C 레벨 계산을 할 때는 열쇠를 반납하고 작업한다.

따라서 판단 기준은 하나다. 그 작업이 열쇠를 붙잡고 있는가. 붙잡고 있으면(순수 Python 계산) 스레드를 늘려도 소용없고 프로세스를 써야 한다. 놓고 있으면(I/O, NumPy 연산, PyTorch 커널 실행) 스레드가 실제로 병렬로 진행된다.

코루틴은 다른 종류의 해법이다. 스레드는 운영체제가 강제로 전환하지만 코루틴은 스스로 양보한다. 양보 지점이 `await`로 명시되어 있어 전환 비용이 훨씬 싸고, 수만 개를 동시에 띄울 수 있다. 대신 양보하지 않는 코드가 하나 있으면 전체가 멈춘다.

## 형식화

작업 하나의 실행 시간을 GIL 보유 구간과 해제 구간으로 나눈다.

$$T = T_{\text{gil}} + T_{\text{free}}$$

$n$개 작업을 $k$개 스레드로 처리할 때, GIL 구간은 직렬화되고 해제 구간만 겹칠 수 있으므로 총 시간의 하한은 다음과 같다.

$$T_{\text{thread}}(n, k) \geq n \cdot T_{\text{gil}} + \frac{n \cdot T_{\text{free}}}{\min(k, n)}$$

여기서 $T_{\text{free}} = 0$(순수 Python 계산)이면 $T_{\text{thread}} = n \cdot T_{\text{gil}}$로 스레드 수와 무관하다. 반대로 $T_{\text{gil}} \approx 0$(대부분 I/O 대기)이면 $T_{\text{thread}} \approx n \cdot T_{\text{free}} / k$로 완전한 선형 가속이 나온다.

프로세스를 쓰면 GIL 제약이 사라지지만 직렬화 비용이 붙는다. 작업당 입력 크기 $s_{in}$, 출력 크기 $s_{out}$, 직렬화 대역폭 $\beta$라 하면

$$T_{\text{process}}(n, k) \approx C_{\text{spawn}} \cdot k + \frac{n \cdot (T_{\text{gil}} + T_{\text{free}})}{k} + \frac{n \cdot (s_{in} + s_{out})}{\beta}$$

이 식에서 프로세스가 손해를 보는 조건이 읽힌다. 작업 하나가 짧고($T$가 작고) 데이터가 크면($s$가 크면) 직렬화 항이 지배해 순차 실행보다 느려진다. 실무 기준으로 작업당 최소 수십 밀리초 이상은 되어야 프로세스 풀이 이득이다.

코루틴은 컨텍스트 전환 비용이 함수 호출 수준이므로

$$T_{\text{async}}(n, k) \approx \max\left(n \cdot C_{\text{switch}}, \frac{n \cdot T_{\text{wait}}}{k}\right)$$

이고 $C_{\text{switch}}$가 마이크로초 단위라 $k$를 수천으로 키워도 오버헤드가 작다. 스레드는 스택 때문에 하나당 최소 수십 KB를 쓰지만 코루틴은 수백 바이트다.

## 구현

### GIL 효과 측정

먼저 주장을 숫자로 확인한다.

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def cpu_bound(n: int):
    total = 0
    for i in range(n):
        total += i * i
    return total


def io_bound(seconds: float):
    time.sleep(seconds)
    return seconds


def measure(fn, *args, repeat: int = 1):
    start = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    return time.perf_counter() - start


N = 3_000_000
WORK = 4

serial = measure(lambda: [cpu_bound(N) for _ in range(WORK)])
with ThreadPoolExecutor(max_workers=4) as pool:
    threaded = measure(lambda: list(pool.map(cpu_bound, [N] * WORK)))

print(f"cpu serial:   {serial:.3f}s")
print(f"cpu threaded: {threaded:.3f}s  speedup={serial / threaded:.2f}x")
```

스레드 4개를 써도 가속이 1배 근처이거나 오히려 느리다. 형식화의 $T_{\text{free}} = 0$ 경우다. 컨텍스트 전환 비용 때문에 약간 손해를 보기까지 한다.

I/O 작업은 다르다.

```python
import time
from concurrent.futures import ThreadPoolExecutor


def io_bound(seconds: float):
    time.sleep(seconds)
    return seconds


start = time.perf_counter()
for _ in range(8):
    io_bound(0.1)
serial_io = time.perf_counter() - start

with ThreadPoolExecutor(max_workers=8) as pool:
    start = time.perf_counter()
    list(pool.map(io_bound, [0.1] * 8))
    threaded_io = time.perf_counter() - start

print(f"io serial:   {serial_io:.3f}s")
print(f"io threaded: {threaded_io:.3f}s  speedup={serial_io / threaded_io:.2f}x")
```

거의 8배 가속이 나온다. `time.sleep`이 GIL을 놓기 때문이다.

NumPy 연산도 GIL을 놓는다는 것을 확인한다. 이것이 실무에서 중요한 이유는, 전처리가 NumPy 기반이면 스레드로도 병렬화가 된다는 뜻이기 때문이다.

```python
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def matmul_work(size: int):
    a = np.random.default_rng(0).normal(size=(size, size))
    return float((a @ a).sum())


SIZE = 900
start = time.perf_counter()
for _ in range(4):
    matmul_work(SIZE)
serial_np = time.perf_counter() - start

with ThreadPoolExecutor(max_workers=4) as pool:
    start = time.perf_counter()
    list(pool.map(matmul_work, [SIZE] * 4))
    threaded_np = time.perf_counter() - start

print(f"numpy serial:   {serial_np:.3f}s")
print(f"numpy threaded: {threaded_np:.3f}s  speedup={serial_np / threaded_np:.2f}x")
```

BLAS가 이미 내부적으로 스레드를 쓰기 때문에 결과 해석에 주의가 필요하다. `OMP_NUM_THREADS=1`로 설정하고 측정하면 순수한 GIL 해제 효과를 볼 수 있다.

### 프로세스 풀

CPU 작업의 정답이다.

```python
import time
from concurrent.futures import ProcessPoolExecutor


def cpu_bound(n: int):
    total = 0
    for i in range(n):
        total += i * i
    return total


def main():
    N = 3_000_000
    WORK = 4

    start = time.perf_counter()
    for _ in range(WORK):
        cpu_bound(N)
    serial = time.perf_counter() - start

    with ProcessPoolExecutor(max_workers=4) as pool:
        start = time.perf_counter()
        list(pool.map(cpu_bound, [N] * WORK))
        parallel = time.perf_counter() - start

    print(f"serial:   {serial:.3f}s")
    print(f"parallel: {parallel:.3f}s  speedup={serial / parallel:.2f}x")


if __name__ == "__main__":
    main()
```

`if __name__ == "__main__"` 가드가 필수다. spawn 방식에서 자식 프로세스가 모듈을 다시 임포트하므로, 가드가 없으면 프로세스가 무한 생성된다. macOS와 Windows는 spawn이 기본이므로 이 문제가 즉시 드러나고, Linux는 fork가 기본이라 늦게 발견된다.

직렬화 비용이 지배하는 경우를 확인한다.

```python
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def trivial(arr: np.ndarray):
    return float(arr.sum())


def main():
    chunks = [np.random.default_rng(i).normal(size=(500, 500)) for i in range(16)]

    start = time.perf_counter()
    serial = [trivial(c) for c in chunks]
    t_serial = time.perf_counter() - start

    with ProcessPoolExecutor(max_workers=4) as pool:
        start = time.perf_counter()
        parallel = list(pool.map(trivial, chunks))
        t_parallel = time.perf_counter() - start

    print(f"serial:   {t_serial:.4f}s")
    print(f"parallel: {t_parallel:.4f}s")
    print(f"same result: {np.allclose(serial, parallel)}")


if __name__ == "__main__":
    main()
```

각 배열이 2MB인데 작업은 마이크로초 수준이라 프로세스 풀이 훨씬 느리다. 형식화의 직렬화 항 $n(s_{in} + s_{out})/\beta$가 지배하는 경우다.

시작 방식을 명시적으로 지정한다. CUDA를 초기화한 뒤 fork하면 자식 프로세스의 CUDA 컨텍스트가 깨진다.

```python
import multiprocessing as mp


def show_methods():
    print(f"default: {mp.get_start_method()}")
    print(f"available: {mp.get_all_start_methods()}")


if __name__ == "__main__":
    show_methods()
```

PyTorch DataLoader에서 `num_workers > 0`이고 워커 안에서 CUDA를 쓰려면 `spawn`이 필요하다. 다만 spawn은 시작이 느리고 워커가 모듈을 다시 임포트하므로, 무거운 전역 초기화가 있으면 워커 시작 시간이 크게 늘어난다.

### 공유 메모리

프로세스 간 큰 배열을 복사 없이 전달한다.

```python
import numpy as np
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor


def sum_slice(args):
    name, shape, dtype_str, start, stop = args
    shm = shared_memory.SharedMemory(name=name)
    try:
        arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        return float(arr[start:stop].sum())
    finally:
        shm.close()


def main():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(4_000_000,)).astype(np.float32)
    shm = shared_memory.SharedMemory(create=True, size=source.nbytes)
    try:
        shared = np.ndarray(source.shape, dtype=source.dtype, buffer=shm.buf)
        shared[:] = source
        bounds = [(i * 1_000_000, (i + 1) * 1_000_000) for i in range(4)]
        payloads = [(shm.name, source.shape, source.dtype.str, a, b) for a, b in bounds]
        with ProcessPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(sum_slice, payloads))
        print(f"shared sum: {sum(parts):.4f}")
        print(f"direct sum: {float(source.sum()):.4f}")
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
```

배열 자체가 아니라 이름과 메타데이터만 전달된다. 직렬화 비용이 사라지므로 앞의 형식화에서 $s_{in} \approx 0$이 된다.

`unlink`를 반드시 호출한다. 호출하지 않으면 프로세스가 끝나도 `/dev/shm`에 세그먼트가 남는다. 학습 작업이 반복 실행되면 공유 메모리가 차서 다음 실행이 실패한다.

### asyncio

대량 네트워크 호출의 정답이다.

```python
import asyncio
import time


async def fake_request(idx: int, latency: float = 0.2):
    await asyncio.sleep(latency)
    return f"result-{idx}"


async def bounded_gather(count: int, concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(idx: int):
        async with semaphore:
            return await fake_request(idx)

    return await asyncio.gather(*(guarded(i) for i in range(count)))


async def main():
    start = time.perf_counter()
    results = await bounded_gather(200, concurrency=32)
    elapsed = time.perf_counter() - start
    print(f"{len(results)} requests in {elapsed:.2f}s")
    print(f"serial would take {200 * 0.2:.1f}s")


asyncio.run(main())
```

200개 요청이 순차로는 40초 걸릴 것을 1.5초 내에 끝낸다. 세마포어로 동시성을 제한한 이유는 상대 서버의 속도 제한 때문이다. 무제한으로 띄우면 429 응답을 받거나 연결이 거부된다.

재시도와 타임아웃을 붙인 실전 형태다.

```python
import asyncio
import random


class RateLimitError(Exception):
    pass


async def flaky_call(idx: int, rng: random.Random):
    await asyncio.sleep(0.05)
    roll = rng.random()
    if roll < 0.2:
        raise RateLimitError(f"429 on {idx}")
    if roll < 0.25:
        raise TimeoutError(f"timeout on {idx}")
    return idx * 2


async def call_with_retry(idx: int, rng: random.Random, attempts: int = 4):
    delay = 0.05
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            async with asyncio.timeout(1.0):
                return await flaky_call(idx, rng)
        except (RateLimitError, TimeoutError) as exc:
            last = exc
            if attempt < attempts - 1:
                jitter = rng.uniform(0, delay * 0.3)
                await asyncio.sleep(delay + jitter)
                delay *= 2
    raise RuntimeError(f"exhausted retries for {idx}") from last


async def main():
    rng = random.Random(0)
    semaphore = asyncio.Semaphore(16)

    async def guarded(idx: int):
        async with semaphore:
            try:
                return await call_with_retry(idx, rng)
            except RuntimeError:
                return None

    results = await asyncio.gather(*(guarded(i) for i in range(60)))
    ok = [r for r in results if r is not None]
    print(f"succeeded {len(ok)} of {len(results)}")


asyncio.run(main())
```

`asyncio.timeout`이 Python 3.11부터 제공되는 컨텍스트 매니저다. 그 이전에는 `asyncio.wait_for`를 쓴다. 지수 백오프에 지터를 더한 이유는 앞의 데코레이터 문서에서 설명한 것과 같다.

### 이벤트 루프를 막지 않기

asyncio의 가장 흔한 실패다. 코루틴 안에서 동기 블로킹 호출을 하면 전체가 멈춘다.

```python
import asyncio
import time


def blocking_cpu(n: int):
    total = 0
    for i in range(n):
        total += i * i
    return total


async def wrong():
    start = time.perf_counter()
    await asyncio.gather(*(asyncio.to_thread(time.sleep, 0.1) for _ in range(5)))
    return time.perf_counter() - start


async def blocked():
    start = time.perf_counter()
    for _ in range(3):
        blocking_cpu(2_000_000)
        await asyncio.sleep(0)
    return time.perf_counter() - start


async def offloaded():
    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=3) as pool:
        await asyncio.gather(*(loop.run_in_executor(pool, blocking_cpu, 2_000_000) for _ in range(3)))
    return time.perf_counter() - start


async def main():
    print(f"io via to_thread: {await wrong():.3f}s")
    print(f"cpu inline:       {await blocked():.3f}s")
    print(f"cpu offloaded:    {await offloaded():.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
```

`asyncio.to_thread`는 블로킹 I/O를 스레드로 밀어낸다. CPU 작업은 스레드로 밀어도 GIL 때문에 소용없으므로 `run_in_executor`에 프로세스 풀을 준다.

### 선택 기준을 코드로

앞의 형식화를 판단 함수로 정리한다.

```python
def choose_concurrency(task_kind: str, task_seconds: float, payload_mb: float, count: int):
    if task_kind == "network":
        return "asyncio" if count > 50 else "threads"
    if task_kind == "blocking_io":
        return "threads"
    if task_kind == "numpy_or_torch":
        return "threads"
    if task_kind == "pure_python_cpu":
        if task_seconds < 0.01:
            return "serial (overhead dominates)"
        if payload_mb > 50:
            return "processes with shared_memory"
        return "processes"
    return "serial"


cases = [
    ("network", 2.0, 0.01, 1000),
    ("blocking_io", 0.05, 1.0, 20),
    ("pure_python_cpu", 0.001, 0.1, 10000),
    ("pure_python_cpu", 0.5, 200.0, 64),
    ("numpy_or_torch", 0.2, 8.0, 16),
]
for case in cases:
    print(f"{case} -> {choose_concurrency(*case)}")
```

## 실무 관점

DataLoader의 `num_workers` 선택은 이 문서의 내용이 직접 적용되는 지점이다. 워커는 프로세스이므로 GIL 제약이 없고, 이미지 디코딩 같은 CPU 작업이 실제로 병렬화된다. 값은 CPU 코어 수를 넘지 않게 하되, 각 워커가 메모리를 복사한다는 점을 고려한다. 데이터셋 객체가 큰 리스트를 들고 있으면 워커 수만큼 복사되어 메모리가 터진다. 이를 피하려면 인덱스만 들고 있고 실제 데이터는 파일에서 읽거나 공유 메모리에 둔다.

`persistent_workers=True`를 켜면 에폭마다 워커를 재생성하지 않는다. spawn 방식에서 워커 시작이 수 초 걸릴 수 있으므로 에폭이 짧으면 이 설정이 큰 차이를 만든다.

`pin_memory=True`는 호스트 메모리를 페이지 고정해 GPU 전송을 비동기로 만든다. 별도의 스레드가 관여하며, 전송이 계산과 겹칠 수 있어 실질적 이득이 크다.

fork와 스레드의 조합은 위험하다. fork 시점에 다른 스레드가 락을 잡고 있었다면 자식 프로세스에서 그 락이 영원히 잠긴 채로 남는다. 자식이 그 락을 요구하면 데드락이다. OpenMP, CUDA, gRPC 라이브러리가 내부 스레드를 쓰므로 이 상황이 실제로 발생한다. 대응은 두 가지다. 워커 시작 전에 무거운 라이브러리를 초기화하지 않거나, `spawn`을 쓴다.

```python
import multiprocessing as mp


def configure_start_method():
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    return mp.get_start_method()


if __name__ == "__main__":
    print(configure_start_method())
```

`set_start_method`는 프로세스당 한 번만 호출할 수 있으므로 `try`로 감싼다.

BLAS 스레드 수를 통제한다. NumPy와 PyTorch가 각각 내부 스레드 풀을 쓰는데, DataLoader 워커 8개가 각각 8개 BLAS 스레드를 띄우면 64개 스레드가 코어를 두고 경쟁해 오히려 느려진다.

```python
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
```

이 설정은 NumPy 임포트 전에 해야 효과가 있다. 스크립트 최상단에 둔다.

좀비 프로세스와 자원 누수를 확인한다. 프로세스 풀을 `with` 없이 쓰거나 예외로 빠져나가면 자식이 남는다. `concurrent.futures`의 컨텍스트 매니저는 `shutdown(wait=True)`를 보장한다.

Python 3.13부터 GIL을 비활성화한 빌드가 실험적으로 제공된다. 사용 가능하면 순수 Python CPU 작업의 스레드 병렬화가 가능해지지만, C 확장이 스레드 안전하게 재작성되어야 하므로 생태계 전환에 시간이 걸린다. 당분간은 이 문서의 선택 기준이 유효하다.

디버깅 방법도 정리한다. 데드락이 의심되면 `faulthandler`로 모든 스레드의 스택을 덤프한다.

```python
import faulthandler
import signal

faulthandler.register(signal.SIGUSR1)
print("send SIGUSR1 to dump stacks of all threads")
```

프로세스가 멈췄을 때 `kill -USR1 <pid>`로 어디서 막혔는지 즉시 확인할 수 있다. 학습 스크립트에 기본으로 넣어둘 가치가 있다.

## 핵심 정리

판단 기준은 하나다. 그 작업이 GIL을 붙잡고 있는가. 순수 Python 계산은 붙잡고, I/O와 NumPy와 PyTorch 커널 실행은 놓는다.

순수 Python CPU 작업에 스레드를 늘리면 가속이 없거나 오히려 느려진다. 프로세스를 쓴다.

프로세스는 직렬화 비용이 있다. 작업당 시간이 수십 밀리초 미만이거나 페이로드가 크면 순차 실행보다 느리다. 큰 배열은 `shared_memory`로 전달하고 `unlink`를 반드시 호출한다.

대량 네트워크 호출은 asyncio다. 코루틴 하나가 수백 바이트라 수천 개를 띄울 수 있다. 반드시 세마포어로 동시성을 제한한다.

이벤트 루프를 막으면 전체가 멈춘다. 블로킹 I/O는 `asyncio.to_thread`, CPU 작업은 `run_in_executor`에 프로세스 풀로 밀어낸다.

`if __name__ == "__main__"` 가드는 spawn 방식에서 필수다. 없으면 프로세스가 무한 생성된다.

fork와 스레드 조합은 데드락을 만든다. CUDA나 OpenMP를 초기화한 뒤 fork하지 않는다. DataLoader 워커에서 CUDA를 쓰려면 spawn을 지정한다.

DataLoader 워커를 늘릴 때 `OMP_NUM_THREADS=1`을 함께 설정한다. 그렇지 않으면 스레드가 코어를 두고 경쟁한다.
