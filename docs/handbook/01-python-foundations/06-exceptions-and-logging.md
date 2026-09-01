# 예외 설계와 로깅 전략

## 한 줄 정의

예외는 호출자가 처리 방법을 결정할 수 있도록 실패의 종류를 타입으로 표현하는 제어 흐름이고, 로깅은 실행 이력을 사후에 재구성할 수 있게 남기는 구조화된 기록이다.

## 문제 상황

장시간 학습 작업에서 실패 처리가 잘못되면 비용이 직접적으로 발생한다. 8장짜리 GPU 노드에서 12시간 학습을 돌렸는데 마지막 검증 단계에서 예외가 나고, 로그에는 다음 한 줄만 남아 있는 상황이 전형적이다.

```
RuntimeError: CUDA error: device-side assert triggered
```

어느 배치에서, 어떤 샘플로, 어떤 형상으로 터졌는지 알 수 없다. 재현하려면 12시간을 다시 써야 한다.

반대 방향의 실패도 흔하다. 예외를 지나치게 넓게 잡아 문제를 숨기는 경우다.

```python
try:
    batch = next(loader_iter)
    loss = train_step(batch)
except Exception:
    continue
```

데이터 로딩 오류를 건너뛰려는 의도였지만 `train_step` 안의 모든 오류도 함께 삼켜진다. 학습은 계속 돌고 손실은 기록되지 않으며, 며칠 뒤 모델이 학습되지 않았다는 것만 발견한다.

로깅 쪽 결핍은 다르다. `print`로 남긴 출력은 레벨이 없어 전부 나오거나 전부 안 나온다. 분산 학습에서 8개 프로세스가 같은 스트림에 쓰면 줄이 섞이고 어느 랭크의 출력인지 알 수 없다. 시간 정보가 없으면 어느 단계가 느린지 판단할 수 없고, 기계 판독이 불가능하면 수천 줄에서 패턴을 찾을 수 없다.

## 직관적 이해

예외 타입은 실패의 주소다. `ValueError` 하나로 뭉뚱그리면 호출자는 "뭔가 잘못됐다"만 알고 대응을 고를 수 없다. 데이터가 손상된 것인지, 설정이 틀린 것인지, 외부 서비스가 죽은 것인지에 따라 재시도할지 건너뛸지 즉시 중단할지가 달라진다. 타입 계층이 곧 대응 전략의 분류다.

예외 연쇄(`raise ... from`)는 진술의 층위를 나누는 것이다. 아래층은 "파일을 열 수 없다"고 말하고 위층은 "샤드 7을 로드할 수 없다"고 말한다. 둘 다 필요하다. 하나만 남기면 원인이나 맥락 중 하나를 잃는다.

로깅 레벨은 독자를 나누는 장치다. DEBUG는 개발 중인 나, INFO는 운영 중 상태를 확인하는 나, WARNING은 이상 징후를 찾는 나, ERROR는 장애 대응 중인 나를 위한 것이다. 같은 사건도 독자에 따라 필요한 상세도가 다르다.

## 형식화

예외 처리의 의미는 스택 되감기로 정의된다. 프레임 스택 $F_1, \dots, F_n$에서 $F_n$이 예외 $e$를 발생시키면, 인터프리터는 $i = n, n-1, \dots, 1$ 순으로 각 프레임의 활성 `except` 절을 검사한다. 절의 타입 $T$에 대해 다음이 성립하면 그 절이 선택된다.

$$\text{type}(e) <: T$$

앞 문서의 서브타입 관계와 정확히 같은 판정이다. 따라서 `except Exception`은 거의 모든 예외 타입의 상위이므로 모든 것을 잡고, `except (OSError, ValueError)`는 두 집합의 합집합을 잡는다.

`except` 절은 위에서 아래로 검사되므로 좁은 타입을 먼저 써야 한다. 넓은 타입이 위에 있으면 아래 절은 도달 불가능하다.

예외 객체는 세 개의 연결 필드를 갖는다.

$$e.\_\_context\_\_, \quad e.\_\_cause\_\_, \quad e.\_\_traceback\_\_$$

`__context__`는 처리 중 다른 예외가 발생했을 때 자동으로 설정되고, `__cause__`는 `raise ... from ...`으로 명시적으로 설정된다. 트레이스백 출력에서 전자는 "During handling of the above exception, another exception occurred", 후자는 "The above exception was the direct cause of the following exception"으로 표시된다. 의도한 연쇄인지 사고인지가 이 문구로 구분된다.

로깅의 비용 구조도 형식화해둘 가치가 있다. 레벨 $\ell$의 메시지 하나를 남기는 비용은

$$C(\ell) = C_{\text{check}} + \mathbb{1}[\ell \geq \ell_{\text{threshold}}] \cdot (C_{\text{format}} + C_{\text{emit}})$$

이며 $C_{\text{check}}$는 정수 비교 하나다. 그러나 아래처럼 쓰면 $C_{\text{format}}$이 임계값과 무관하게 항상 발생한다.

```python
logger.debug(f"tensor stats: {tensor.mean()} {tensor.std()}")
```

f-string은 호출 전에 평가되므로 DEBUG가 꺼져 있어도 `mean()`과 `std()`가 실행된다. GPU 동기화까지 유발하면 학습 속도가 눈에 띄게 떨어진다. 지연 포매팅으로 이를 막는다.

## 구현

### 예외 계층 설계

도메인별 최상위 예외를 하나 두고 그 아래로 분류한다.

```python
class PipelineError(Exception):
    pass


class DataError(PipelineError):
    def __init__(self, message: str, shard: str | None = None, index: int | None = None):
        super().__init__(message)
        self.shard = shard
        self.index = index

    def __str__(self):
        base = super().__str__()
        parts = [base]
        if self.shard is not None:
            parts.append(f"shard={self.shard}")
        if self.index is not None:
            parts.append(f"index={self.index}")
        return " ".join(parts)


class CorruptRecordError(DataError):
    pass


class SchemaMismatchError(DataError):
    pass


class ConfigError(PipelineError):
    pass


class TransientError(PipelineError):
    pass


def classify(exc: Exception):
    if isinstance(exc, TransientError):
        return "retry"
    if isinstance(exc, CorruptRecordError):
        return "skip"
    if isinstance(exc, (ConfigError, SchemaMismatchError)):
        return "abort"
    return "abort"


for exc in [TransientError("s3 timeout"), CorruptRecordError("bad utf-8", shard="s7", index=42), ConfigError("lr missing")]:
    print(f"{type(exc).__name__}: {exc} -> {classify(exc)}")
```

계층의 목적은 `classify` 같은 함수가 성립하게 하는 것이다. 예외 타입만 보고 재시도, 건너뛰기, 중단 중 무엇을 할지 결정할 수 있으면 호출부의 분기 로직이 단순해진다.

예외에 구조화된 필드를 붙인 것도 의도적이다. 메시지 문자열을 파싱해서 샤드 번호를 뽑아내는 코드는 깨지기 쉽다.

### 맥락을 붙여 다시 던지기

앞의 문제 상황에서 본 "어느 배치인지 모른다"를 해결한다.

```python
import json


def parse_record(line: str, shard: str, index: int):
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise CorruptRecordError("invalid json", shard=shard, index=index) from exc
    if "text" not in record:
        raise SchemaMismatchError("missing field 'text'", shard=shard, index=index)
    return record


try:
    parse_record("{broken", shard="shard_007", index=1234)
except DataError as exc:
    print(f"{type(exc).__name__}: {exc}")
    print(f"cause: {type(exc.__cause__).__name__}: {exc.__cause__}")
```

`from exc`가 원인을 연결한다. 트레이스백에는 JSON 파서의 실제 실패 위치와 우리 계층의 맥락이 모두 남는다.

`from None`은 반대로 연쇄를 끊는다. 내부 구현 세부를 노출하고 싶지 않을 때 쓴다. 앞 문서의 `DotDict.__getattr__`에서 `KeyError`를 감춘 경우가 그 예다.

Python 3.11부터는 예외를 다시 만들지 않고 메모만 붙일 수 있다.

```python
def load_batch(records: list[str], shard: str):
    parsed = []
    for index, line in enumerate(records):
        try:
            parsed.append(parse_record(line, shard, index))
        except DataError as exc:
            exc.add_note(f"batch position {len(parsed)} of {len(records)}")
            raise
    return parsed


try:
    load_batch(['{"text": "ok"}', "{broken"], shard="shard_007")
except DataError as exc:
    print(exc.__notes__)
```

`add_note`는 원본 예외 타입과 트레이스백을 그대로 유지하면서 정보만 추가한다. 재래핑보다 가볍다.

### 예외 그룹

병렬 작업에서 여러 실패가 동시에 발생하면 하나만 남기고 버리는 것이 손실이다. Python 3.11의 `ExceptionGroup`이 이를 다룬다.

```python
def process_shards(shards: list[str]):
    results = []
    errors = []
    for shard in shards:
        try:
            if shard.endswith("3"):
                raise CorruptRecordError("bad utf-8", shard=shard)
            if shard.endswith("5"):
                raise TransientError(f"timeout on {shard}")
            results.append(shard)
        except PipelineError as exc:
            errors.append(exc)
    if errors:
        raise ExceptionGroup("shard processing failed", errors)
    return results


try:
    process_shards([f"shard_{i}" for i in range(7)])
except* CorruptRecordError as group:
    print(f"corrupt: {[str(e) for e in group.exceptions]}")
except* TransientError as group:
    print(f"transient: {[str(e) for e in group.exceptions]}")
```

`except*`는 그룹에서 해당 타입만 골라내고 나머지는 다시 던진다. 여러 종류의 실패에 각각 다른 대응을 하면서 어느 것도 잃지 않는다.

### 로깅 기본 구성

`print`를 로거로 교체하고 구조를 잡는다.

```python
import logging
import sys


def build_logger(name: str, level: int = logging.INFO, rank: int = 0):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stderr)
    fmt = f"%(asctime)s | %(levelname)-7s | rank{rank} | %(name)s:%(lineno)d | %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    if rank != 0:
        handler.setLevel(logging.WARNING)
    logger.addHandler(handler)
    return logger


logger = build_logger("train", level=logging.DEBUG, rank=0)
logger.debug("loading shard %s", "shard_000")
logger.info("epoch %d started, lr=%.2e", 1, 3e-4)
logger.warning("grad norm %.2f exceeds threshold %.2f", 12.4, 10.0)
logger.error("checkpoint save failed")
```

세 가지가 핵심이다.

첫째, `%` 스타일 지연 포매팅을 쓴다. 앞의 비용 형식화에서 본 대로 f-string은 레벨과 무관하게 평가된다. `logger.debug("stats %s", expensive())`도 인자 평가는 일어나므로, 정말 무거운 계산은 `logger.isEnabledFor`로 감싼다.

```python
import logging

logger = logging.getLogger("train")
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("param norm %.4f", sum(p * p for p in range(100000)) ** 0.5)
```

둘째, 랭크 0이 아닌 프로세스는 WARNING 이상만 남긴다. 8프로세스가 모두 INFO를 쓰면 로그가 8배가 되고 읽을 수 없다. 다만 오류는 모든 랭크에서 남겨야 한다. 특정 랭크에서만 터지는 문제가 실제로 많다.

셋째, `propagate = False`로 루트 로거 중복 출력을 막는다. 라이브러리가 루트에 핸들러를 붙이면 같은 메시지가 두 번 나오는 문제가 생긴다.

### 구조화 로깅

기계 판독이 필요하면 JSON 라인으로 남긴다.

```python
import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "line": f"{record.filename}:{record.lineno}",
        }
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def log_event(logger: logging.Logger, level: int, message: str, **fields):
    logger.log(level, message, extra={"extra_fields": fields})


json_logger = logging.getLogger("train.json")
json_logger.setLevel(logging.INFO)
json_logger.propagate = False
if not json_logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    json_logger.addHandler(h)

log_event(json_logger, logging.INFO, "step_end", step=100, loss=0.4213, lr=2.8e-4, grad_norm=1.72)
log_event(json_logger, logging.WARNING, "grad_spike", step=101, grad_norm=87.3, threshold=10.0)
```

이 형식이면 `jq`로 필터링하거나 로그 수집기가 필드 단위로 색인할 수 있다. "grad_norm이 10을 넘은 스텝을 전부 찾아라" 같은 질의가 가능해진다.

### 예외 로깅

예외를 로그에 남길 때 트레이스백을 잃지 않는 것이 중요하다.

```python
import logging
import sys

logger = logging.getLogger("train.exc")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(h)


def failing():
    raise CorruptRecordError("bad utf-8", shard="shard_003", index=88)


try:
    failing()
except DataError:
    logger.exception("failed to parse record")

try:
    failing()
except DataError as exc:
    logger.error("failed to parse record: %s", exc)
```

첫 번째는 트레이스백 전체를 남기고 두 번째는 메시지만 남긴다. `logger.exception`은 `except` 블록 안에서만 쓸 수 있으며 레벨이 ERROR로 고정된다. 다른 레벨이 필요하면 `logger.warning(..., exc_info=True)`를 쓴다.

### 학습 루프 통합

앞의 요소들을 실제 루프 형태로 합친다. 예외 분류에 따른 대응과 로깅이 함께 동작한다.

```python
import logging
import random
import time


def train_loop(steps: int, logger: logging.Logger, max_skips: int = 5):
    rng = random.Random(0)
    skipped = 0
    loss = 1.0
    start = time.perf_counter()

    for step in range(steps):
        try:
            roll = rng.random()
            if roll < 0.08:
                raise CorruptRecordError("bad sample", shard="s0", index=step)
            if roll < 0.12:
                raise TransientError("nccl timeout")
            loss = max(0.05, loss * 0.97 + rng.gauss(0, 0.01))
            if step % 20 == 0:
                logger.info("step=%d loss=%.4f elapsed=%.2fs", step, loss, time.perf_counter() - start)
        except CorruptRecordError as exc:
            skipped += 1
            logger.warning("skipping step %d: %s (total skipped=%d)", step, exc, skipped)
            if skipped > max_skips:
                raise PipelineError(f"too many corrupt records: {skipped}") from exc
        except TransientError as exc:
            logger.warning("transient failure at step %d, retrying once: %s", step, exc)
            try:
                loss = max(0.05, loss * 0.97)
            except Exception:
                logger.exception("retry failed at step %d", step)
                raise
    return loss, skipped


run_logger = build_logger("train.loop", level=logging.INFO)
try:
    final_loss, skips = train_loop(120, run_logger)
    run_logger.info("finished loss=%.4f skipped=%d", final_loss, skips)
except PipelineError:
    run_logger.exception("training aborted")
```

건너뛴 횟수에 상한을 둔 것이 중요하다. 무한히 건너뛰면 데이터가 전부 손상되었는데도 학습이 계속 도는 상태가 된다. 임계값을 넘으면 상승시켜 중단한다.

### 경고와 로깅의 구분

`warnings` 모듈은 코드 사용 방식에 대한 지적이고 로깅은 실행 이력이다. 라이브러리 코드에서는 deprecation 같은 것을 `warnings`로 알린다.

```python
import warnings


def old_api(x: float):
    warnings.warn(
        "old_api is deprecated, use new_api instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return x * 2


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    old_api(1.0)
    print(caught[0].category.__name__, caught[0].message)
```

`stacklevel=2`가 중요하다. 이것이 없으면 경고가 라이브러리 내부 줄을 가리켜서 사용자가 어디를 고쳐야 하는지 알 수 없다.

`logging.captureWarnings(True)`를 호출하면 경고가 로거로 흘러들어 한 곳에서 관리된다.

## 실무 관점

`except Exception`을 쓰는 경우는 최상위 루프 하나뿐이어야 한다. 그 위치에서는 로그를 남기고 체크포인트를 저장한 뒤 종료 코드를 설정한다. 중간 계층에서 넓게 잡으면 문제가 숨는다.

`BaseException`은 잡지 않는다. `KeyboardInterrupt`와 `SystemExit`가 여기 속하므로, 잡아버리면 Ctrl+C로 학습을 멈출 수 없다. `finally`와 컨텍스트 매니저는 `BaseException`에서도 실행되므로 정리는 그쪽에 맡긴다.

빈 `except: pass`는 금지다. 정말 무시해도 되는 좁은 예외라면 `contextlib.suppress`로 의도를 명시한다.

예외 메시지에는 값을 넣는다. "invalid shape"보다 "expected (B, C, H, W), got (32, 224, 224)"가 압도적으로 유용하다. 다만 개인정보나 자격증명이 메시지에 들어가지 않도록 주의한다. 로그는 수집되어 여러 사람이 본다.

로그 파일 회전을 설정한다. 장시간 학습이 수 GB 로그를 만들어 디스크를 채우는 사고가 실제로 발생한다.

```python
import logging
from logging.handlers import RotatingFileHandler


def add_file_handler(logger: logging.Logger, path: str, max_bytes: int = 50 * 1024 * 1024, backups: int = 3):
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return handler
```

멀티프로세싱에서 여러 프로세스가 같은 파일에 쓰면 줄이 섞인다. `QueueHandler`로 한 프로세스에 모으거나, 프로세스별 파일에 쓰고 나중에 병합한다.

```python
import logging
import logging.handlers
import multiprocessing as mp


def worker_setup(queue: mp.Queue, rank: int):
    logger = logging.getLogger(f"worker{rank}")
    logger.handlers.clear()
    logger.addHandler(logging.handlers.QueueHandler(queue))
    logger.setLevel(logging.INFO)
    return logger
```

CUDA 관련 예외는 특별히 다룬다. 비동기 실행 때문에 예외가 실제 원인 지점보다 훨씬 뒤에서 보고된다. 디버깅 시에는 `CUDA_LAUNCH_BLOCKING=1`을 설정해 동기 실행으로 바꾼다. 속도가 크게 느려지므로 진단할 때만 켠다. `device-side assert triggered`는 대부분 인덱스 범위 초과이며, 임베딩 레이어에 어휘 크기를 넘는 토큰 ID가 들어간 경우가 가장 흔하다.

로깅 레벨의 실무 기준을 정리한다. DEBUG는 텐서 형상, 중간 통계처럼 정상 운영에서 필요 없는 것이다. INFO는 에폭 시작과 종료, 체크포인트 저장, 주요 지표처럼 사후에 진행을 재구성할 정보다. WARNING은 그래디언트 급증, 샘플 건너뛰기, 자동 재시도처럼 지금은 진행되지만 확인이 필요한 사건이다. ERROR는 작업 일부가 실패했으나 프로세스는 계속되는 경우, CRITICAL은 프로세스를 종료해야 하는 경우다.

스텝마다 로그를 남기지 않는다. 초당 수십 스텝이면 로그가 수백만 줄이 된다. 주기적으로 남기되, 처음 몇 스텝은 매번 남겨 초기 이상을 놓치지 않는다.

```python
def should_log(step: int, every: int = 50, warmup: int = 10):
    return step < warmup or step % every == 0


print([s for s in range(200) if should_log(s)][:15])
```

## 핵심 정리

예외 타입 계층은 대응 전략의 분류다. 타입만 보고 재시도, 건너뛰기, 중단을 결정할 수 있게 설계한다.

`raise ... from exc`로 원인을 연결하고 맥락을 덧붙인다. Python 3.11의 `add_note`는 타입과 트레이스백을 유지하며 정보만 추가한다.

`except Exception`은 최상위 루프 하나에서만 쓴다. `BaseException`은 잡지 않는다. Ctrl+C가 막힌다.

빈 `except: pass`는 금지다. 좁은 예외를 정말 무시한다면 `contextlib.suppress`로 의도를 명시한다.

로깅은 `%` 지연 포매팅을 쓴다. f-string은 레벨과 무관하게 평가되어 GPU 동기화까지 유발할 수 있다. 무거운 계산은 `isEnabledFor`로 감싼다.

분산 학습에서 랭크 0만 INFO를 남기고 나머지는 WARNING 이상만 남긴다. 다만 오류는 모든 랭크에서 남긴다.

구조화 로깅(JSON 라인)은 필드 단위 질의를 가능하게 한다. 수천 스텝의 로그에서 패턴을 찾으려면 필수다.

CUDA 예외는 비동기 실행 때문에 원인 지점과 보고 지점이 다르다. 진단 시 `CUDA_LAUNCH_BLOCKING=1`을 켠다.
