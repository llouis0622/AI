# 가상환경, 의존성 관리, 패키징

## 한 줄 정의

가상환경은 프로젝트별로 독립된 인터프리터와 패키지 트리를 만드는 격리 장치이고, 의존성 관리는 그 트리를 시점과 무관하게 동일하게 복원할 수 있게 만드는 작업이다.

## 문제 상황

학습 실험의 재현성이 깨지는 가장 흔한 경로는 코드가 아니라 환경이다. 3개월 전 실험을 다시 돌렸더니 성능이 2퍼센트 떨어지는데 커밋은 동일한 상황을 생각해보자. 원인 후보는 다음과 같다.

`requirements.txt`에 `torch>=2.0`이라고만 적혀 있어 그사이 2.1이 설치되었다. 2.1에서 `scaled_dot_product_attention`의 기본 백엔드가 바뀌어 수치가 미세하게 달라졌다. 또는 `transformers`가 업데이트되면서 토크나이저의 기본 동작이 바뀌었다. 또는 CUDA 빌드가 다른 PyTorch 휠이 설치되어 커널 구현이 달라졌다.

버전 범위만 기록하는 방식은 재현을 보장하지 못한다. 직접 의존성을 고정해도 전이 의존성이 움직이면 같은 문제가 생긴다.

두 번째 문제는 격리 실패다. 시스템 Python에 `pip install`을 하면 프로젝트 간 버전 충돌이 발생한다. 프로젝트 A가 `numpy<2`를 요구하고 B가 `numpy>=2`를 요구하면 한쪽은 반드시 깨진다.

세 번째는 배포다. 학습 코드를 다른 사람이 쓰려면 `sys.path` 조작이나 상대 경로 임포트에 의존하지 않고 설치 가능한 패키지여야 한다. `from src.models.resnet import ResNet`이 실행 디렉터리에 따라 실패하는 코드는 공유할 수 없다.

## 직관적 이해

가상환경은 프로젝트마다 별도의 작업대를 주는 것이다. 같은 공구(패키지)를 여러 작업대에 서로 다른 버전으로 둘 수 있고, 한 작업대를 통째로 버려도 다른 곳에 영향이 없다.

의존성 명세와 잠금 파일의 관계는 요구사항과 계약서의 관계다. `pyproject.toml`에는 "PyTorch 2.x 이상이면 된다"는 요구를 적고, 잠금 파일에는 "실제로 설치된 것은 2.4.1이고 해시는 이것이다"라는 사실을 적는다. 사람이 관리하는 것은 앞이고 기계가 생성하는 것은 뒤다.

패키징은 코드에 주소를 부여하는 일이다. 설치되지 않은 코드는 실행 위치에 따라 임포트가 달라지지만, 설치된 패키지는 어디서 실행하든 같은 이름으로 찾힌다.

## 형식화

의존성 해석은 제약 만족 문제다. 패키지 집합 $P$, 각 패키지 $p \in P$의 사용 가능한 버전 집합 $V_p$가 있고, 선택 $\sigma : P \to V$가 모든 제약을 만족해야 한다.

$$\forall (p, q, C) \in \mathcal{D} : \sigma(q) \in C$$

여기서 $\mathcal{D}$는 "패키지 $p$가 $q$의 버전 범위 $C$를 요구한다"는 의존 관계의 집합이다. 이 문제는 일반적으로 NP-난해다. 실제 해석기는 백트래킹 탐색을 쓰며, 그래서 충돌이 있을 때 후보를 하나씩 내려가며 수십 개의 버전을 다운로드하다가 느려진다.

재현성은 별개의 조건이다. 시점 $t_1$과 $t_2$에서의 해석 결과가 같으려면

$$\sigma_{t_1} = \sigma_{t_2}$$

가 보장되어야 하는데, $V_p$가 시간에 따라 커지므로(새 버전이 올라오므로) 제약만으로는 보장되지 않는다. 잠금 파일은 $\sigma$ 자체를 기록해 해석을 건너뛰게 만든다. 여기에 해시를 함께 기록하면 같은 버전 번호로 다른 내용이 배포되는 경우까지 막는다.

버전 번호 자체의 순서는 PEP 440이 정의한다. 정규 형태는 다음과 같다.

$$[N!]N(.N)^*[\{a|b|rc\}N][.postN][.devN][+local]$$

에폭 $N!$이 앞에 오는 이유는 버전 체계를 통째로 바꿔야 할 때 이전 것보다 큰 값을 만들기 위해서다. 로컬 버전 `+local`은 CUDA 빌드 구분에 쓰인다. `2.4.1+cu121`과 `2.4.1+cpu`가 같은 공개 버전이지만 다른 아티팩트다.

## 구현

### 가상환경 만들기

표준 도구부터 확인한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -c "import sys; print(sys.prefix); print(sys.base_prefix)"
```

`sys.prefix`가 가상환경을, `sys.base_prefix`가 원본 인터프리터를 가리킨다. 두 값이 다르면 가상환경 안이다. 스크립트에서 이를 확인하는 코드는 다음과 같다.

```python
import sys


def in_virtualenv():
    return sys.prefix != sys.base_prefix


def describe_env():
    return {
        "python": sys.version.split()[0],
        "prefix": sys.prefix,
        "isolated": in_virtualenv(),
        "executable": sys.executable,
    }


for key, value in describe_env().items():
    print(f"{key}: {value}")
```

학습 스크립트 시작 시 이 정보를 로그에 남기면, 나중에 "어떤 환경에서 돌렸는지"를 재구성할 수 있다.

### 프로젝트 명세

`pyproject.toml` 하나로 메타데이터, 의존성, 도구 설정을 모두 기술한다.

```toml
[project]
name = "ai-handbook-training"
version = "0.1.0"
description = "학습 파이프라인"
requires-python = ">=3.11,<3.13"
dependencies = [
    "numpy>=1.26,<3",
    "torch>=2.3",
    "pyyaml>=6.0",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "mypy>=1.10",
    "ruff>=0.5",
]
track = [
    "mlflow>=2.14",
]

[project.scripts]
train = "handbook.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/handbook"]

[tool.ruff]
line-length = 110
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = ["slow: 오래 걸리는 테스트", "gpu: GPU가 필요한 테스트"]
```

몇 가지가 의도적이다.

`requires-python`에 상한을 둔 이유는 새 Python 버전이 나왔을 때 C 확장 휠이 아직 없어 소스 빌드로 넘어가 실패하는 것을 막기 위해서다.

`numpy>=1.26,<3`처럼 주 버전 상한을 두는 것은 NumPy 2.0의 ABI 변경 같은 사건에 대비한다. 상한이 없으면 어느 날 갑자기 깨진다.

`[project.scripts]`가 진입점을 만든다. 설치 후 `train` 명령이 생기므로 `python -m` 이나 경로 지정 없이 실행된다.

`src` 레이아웃을 쓴 것도 중요하다. 패키지를 `src/handbook`에 두면 프로젝트 루트에서 `import handbook`이 실패하므로, 테스트가 반드시 설치된 패키지를 임포트하게 된다. 설치 누락으로 인한 오류를 개발 중에 발견한다.

### 디렉터리 구조

```
project/
├── pyproject.toml
├── uv.lock
├── src/
│   └── handbook/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py
│       └── models/
│           ├── __init__.py
│           └── resnet.py
├── tests/
│   ├── conftest.py
│   └── test_config.py
└── configs/
    └── baseline.yaml
```

### 편집 가능 설치

개발 중에는 소스를 고칠 때마다 재설치하지 않도록 편집 가능 모드로 설치한다.

```bash
pip install -e ".[dev]"
python -c "import handbook; print(handbook.__file__)"
```

`handbook.__file__`이 `src/handbook/__init__.py`를 가리킨다. 소스를 고치면 즉시 반영된다.

### 잠금 파일

`pip freeze`는 잠금 파일로 부적절하다. 직접 의존성과 전이 의존성이 구분되지 않고, 해시가 없으며, 플랫폼별 차이를 표현하지 못한다. `uv` 또는 `pip-tools`를 쓴다.

```bash
uv lock
uv sync --frozen
```

`--frozen`이 핵심이다. 잠금 파일과 명세가 불일치하면 해석을 다시 하지 않고 실패한다. CI에서 이 옵션 없이 설치하면 잠금 파일이 조용히 갱신되어 재현성이 깨진다.

`pip-tools`를 쓰는 경우는 다음과 같다.

```bash
pip-compile --generate-hashes --output-file requirements.lock pyproject.toml
pip-sync requirements.lock
```

`--generate-hashes`가 앞의 형식화에서 말한 내용 검증을 추가한다. 같은 버전 번호로 다른 내용이 배포되는 공격을 막는다.

`pip install --require-hashes -r requirements.lock`으로 설치 시 해시 검증을 강제한다.

### CUDA 휠 처리

PyTorch는 CUDA 버전별로 다른 인덱스에서 배포된다. 로컬 버전 표기가 여기서 쓰인다.

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```

macOS 개발 환경과 Linux 학습 서버가 다른 휠을 쓰게 된다. 마커로 분기하지 않으면 한쪽이 반드시 실패한다.

설치된 빌드를 검증하는 코드를 학습 스크립트에 둔다.

```python
import importlib.metadata as md
import platform
import sys


def collect_env_report(packages: list[str]):
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for name in packages:
        try:
            report[name] = md.version(name)
        except md.PackageNotFoundError:
            report[name] = "not installed"
    return report


for key, value in collect_env_report(["numpy", "pyyaml"]).items():
    print(f"{key}: {value}")
```

`importlib.metadata`는 실제 설치된 배포판 버전을 읽는다. `module.__version__`은 패키지마다 있을 수도 없을 수도 있으므로 이쪽이 신뢰할 만하다.

이 보고서를 체크포인트와 함께 저장하면 나중에 환경 차이를 즉시 확인할 수 있다.

```python
import hashlib
import json


def env_fingerprint(report: dict):
    payload = json.dumps(report, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


report = collect_env_report(["numpy", "pyyaml"])
print(env_fingerprint(report))
```

두 실행의 지문이 다르면 환경이 달라진 것이다. 성능 차이의 원인 후보를 즉시 좁힌다.

### 설정 로딩

`pyproject.toml`이 프로젝트 메타데이터라면 실험 설정은 별도 파일이다. 앞 문서의 dataclass와 결합한다.

```python
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DataConfig:
    root: str
    batch_size: int = 32
    num_workers: int = 4


@dataclass(frozen=True, slots=True)
class OptimConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    seed: int = 42
    epochs: int = 10
    dataset: DataConfig = field(default_factory=lambda: DataConfig(root="./data"))
    optim: OptimConfig = field(default_factory=OptimConfig)


def load_config(path: str | Path):
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    dataset = DataConfig(**raw.pop("dataset", {}))
    optim = OptimConfig(**raw.pop("optim", {}))
    return ExperimentConfig(dataset=dataset, optim=optim, **raw)


import tempfile

tmp = Path(tempfile.mkdtemp()) / "exp.toml"
tmp.write_text(
    """
name = "baseline"
seed = 1234
epochs = 20

[dataset]
root = "/mnt/data/imagenet"
batch_size = 128

[optim]
lr = 1e-3
""",
    encoding="utf-8",
)

cfg = load_config(tmp)
print(cfg)
print(asdict(cfg))
```

`tomllib`은 Python 3.11 표준 라이브러리다. 별도 의존성이 필요 없다. 알 수 없는 키가 들어오면 dataclass 생성자가 `TypeError`를 내므로 오타가 즉시 잡힌다. YAML은 이 검증이 없어 오타가 조용히 무시된다.

### CLI 진입점

`[project.scripts]`가 가리키는 함수를 만든다.

```python
import argparse
import sys
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(prog="train", description="학습 실행")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_overrides(pairs: list[str]):
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"invalid override: {pair}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    overrides = parse_overrides(args.override)
    print(f"config={args.config} overrides={overrides} dry_run={args.dry_run}")
    return 0


print(main(["--config", "configs/baseline.toml", "--override", "optim.lr=1e-4", "--dry-run"]))
```

`main`이 `argv`를 인자로 받는 형태여야 테스트가 쉽다. `sys.argv`를 직접 읽으면 테스트에서 패치해야 한다.

### 빌드와 배포

휠을 만들어 내용을 확인한다.

```bash
python -m build
unzip -l dist/*.whl | head -30
```

휠 안에 `src` 디렉터리가 아니라 `handbook`이 최상위로 들어가야 한다. `[tool.hatch.build.targets.wheel]`의 `packages` 설정이 이를 담당한다.

설치 검증은 새 환경에서 한다.

```bash
python -m venv /tmp/verify
/tmp/verify/bin/pip install dist/*.whl
/tmp/verify/bin/python -c "import handbook; print(handbook.__version__)"
```

개발 환경에서만 동작하고 설치 후 깨지는 문제(데이터 파일 누락, 상대 임포트 오류)가 여기서 잡힌다.

## 실무 관점

도구 선택 기준을 정리한다. `uv`는 Rust로 작성되어 해석과 설치가 pip보다 한 자릿수 빠르고 잠금 파일을 기본 제공한다. 신규 프로젝트의 기본 선택으로 삼을 만하다. `poetry`는 성숙하고 기능이 많지만 느리고 `pyproject.toml` 표준에서 일부 벗어난 부분이 있었다. `conda`는 Python 외 바이너리 의존성(CUDA 툴킷, MKL, 시스템 라이브러리)이 필요할 때 유일한 선택지인 경우가 있다. `pip` + `venv`는 어디서나 동작한다는 장점이 있어 CI 이미지가 제한적일 때 쓴다.

실무에서 흔한 조합은 개발과 CI는 `uv`, 배포는 Docker다. Docker에서는 가상환경이 필요 없다고 생각하기 쉽지만, 시스템 Python을 오염시키면 OS 패키지가 깨지므로 컨테이너 안에서도 가상환경을 쓰는 편이 안전하다.

레이어 캐시를 활용하는 Dockerfile 순서가 중요하다.

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

COPY src/ src/
RUN uv pip install --no-deps -e .

COPY configs/ configs/

ENTRYPOINT ["train"]
```

의존성 설치가 소스 복사보다 앞에 있어야 코드 변경 시 의존성 레이어가 재사용된다. 순서를 뒤집으면 한 줄만 고쳐도 전체 설치가 다시 돈다.

`PYTHONUNBUFFERED=1`은 로그가 즉시 나오게 한다. 이것이 없으면 컨테이너가 죽었을 때 마지막 로그가 버퍼에 남아 사라진다.

`OMP_NUM_THREADS=1`을 이미지 수준에서 설정한 것은 앞 문서의 스레드 경쟁 문제 때문이다.

버전 고정 정책도 정한다. 라이브러리를 만든다면 의존성 범위를 넓게 두고 잠금 파일을 배포하지 않는다. 사용자의 다른 의존성과 충돌하면 안 되기 때문이다. 애플리케이션이나 학습 파이프라인이라면 잠금 파일을 저장소에 커밋한다. 재현성이 유연성보다 중요하다.

전이 의존성 문제를 진단하는 방법이다.

```bash
uv pip tree
pip install pipdeptree && pipdeptree --warn fail
```

충돌이 있으면 어느 패키지가 어떤 범위를 요구하는지 트리로 보인다. 해석기가 뱉는 오류 메시지만으로는 원인을 찾기 어려운 경우가 많다.

보안 관점에서 이름 유사 공격(typosquatting)을 경계한다. `numpy` 대신 `nunpy`를 설치하면 임의 코드가 실행된다. 해시 고정과 내부 미러 사용이 방어책이다. 사설 인덱스를 쓸 때 `--index-url`이 아니라 `--extra-index-url`을 쓰면 공개 PyPI에 같은 이름의 더 높은 버전이 올라왔을 때 그쪽이 설치되는 의존성 혼동 공격이 가능하다. 사설 패키지는 `--index-url`로 지정하고 공개 인덱스를 명시적으로 분리한다.

임포트 성능도 고려한다. 학습 스크립트가 시작까지 20초 걸리는 경우가 있고, 대부분 무거운 임포트 때문이다.

```bash
python -X importtime -c "import numpy" 2>&1 | tail -12
```

누적 시간이 큰 모듈이 드러난다. 자주 쓰지 않는 무거운 의존성은 함수 안에서 지연 임포트한다.

```python
def build_tracker(kind: str):
    if kind == "mlflow":
        import mlflow
        return mlflow
    if kind == "none":
        return None
    raise ValueError(f"unknown tracker: {kind}")
```

`__init__.py`에서 하위 모듈을 전부 임포트하는 습관이 시작 시간을 늘리는 주범이다. 공개 API가 필요하면 `__getattr__`로 지연 노출한다.

```python
_LAZY = {"ResNet": "handbook.models.resnet"}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(name)
```

## 핵심 정리

가상환경 여부는 `sys.prefix != sys.base_prefix`로 확인한다. 학습 시작 시 환경 정보를 로그에 남긴다.

`pyproject.toml`에는 사람이 관리하는 버전 범위를, 잠금 파일에는 기계가 생성한 정확한 버전과 해시를 둔다. `pip freeze`는 잠금 파일이 아니다.

CI에서는 `uv sync --frozen` 또는 `pip install --require-hashes`로 잠금 파일을 강제한다. 이 옵션이 없으면 재현성이 조용히 깨진다.

`src` 레이아웃을 쓰면 테스트가 반드시 설치된 패키지를 임포트하므로 설치 누락을 개발 중에 발견한다.

PyTorch 같은 CUDA 빌드는 로컬 버전(`2.4.1+cu121`)으로 구분된다. 플랫폼 마커로 인덱스를 분기하지 않으면 개발 환경과 학습 서버 중 하나가 깨진다.

설정은 TOML + frozen dataclass 조합이 안전하다. 알 수 없는 키가 생성자에서 `TypeError`로 즉시 잡힌다.

Dockerfile은 의존성 설치를 소스 복사보다 앞에 둔다. 순서를 뒤집으면 레이어 캐시가 무력화된다.

사설 인덱스는 `--extra-index-url`이 아니라 `--index-url`로 지정한다. 전자는 의존성 혼동 공격에 노출된다.

시작이 느리면 `python -X importtime`으로 원인을 찾고 무거운 의존성은 지연 임포트한다.
