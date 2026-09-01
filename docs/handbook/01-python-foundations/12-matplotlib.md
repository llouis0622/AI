# Matplotlib 체계 정리

## 한 줄 정의

Matplotlib은 Figure라는 캔버스 위에 Axes라는 좌표계를 배치하고, 각 Axes에 Artist(선, 점, 텍스트)를 그리는 계층 구조이며, 이 구조를 명시적으로 다루면 나머지 API가 전부 정리된다.

## 문제 상황

Matplotlib을 상태 기반 `pyplot` 인터페이스로만 쓰면 그림이 조금만 복잡해져도 통제가 어려워진다.

```python
plt.plot(x1, y1)
plt.title("loss")
plt.figure()
plt.plot(x2, y2)
plt.legend(["train"])
plt.savefig("out.png")
```

`plt.legend`가 어느 그림에 붙는지, `plt.title`이 어느 축에 붙는지가 "현재 활성 상태"에 의존한다. 함수로 분리하는 순간 어떤 함수가 어떤 그림을 건드리는지 알 수 없게 되고, 반복문 안에서 그리면 이전 그림에 덧그려진다.

두 번째 문제는 학습 코드에서의 메모리 누수다. 에폭마다 그림을 그리고 저장하는데 닫지 않으면 Figure 객체가 계속 쌓인다. 100에폭이면 100개의 Figure가 메모리에 남고, 경고가 나올 즈음에는 이미 수 GB를 쓰고 있다.

세 번째는 헤드리스 환경이다. 학습 서버에 디스플레이가 없는데 기본 백엔드가 GUI를 시도해 예외가 나거나, SSH 세션이 끊기면 프로세스가 죽는다.

## 직관적 이해

세 계층으로 이해한다. Figure는 종이 한 장이다. Axes는 그 종이 위에 그려진 좌표 평면 하나이며, 종이 한 장에 여러 개를 놓을 수 있다. Artist는 좌표 평면 위에 놓이는 모든 것(선, 마커, 텍스트, 범례)이다.

`pyplot`은 "지금 보고 있는 종이"와 "지금 쓰고 있는 평면"을 전역 변수로 들고 있다가 함수 호출을 그쪽으로 전달하는 편의 계층이다. 대화형 탐색에서는 편하지만 프로그램에서는 그 전역 상태가 곧 버그다.

따라서 규칙은 하나다. `fig, ax = plt.subplots()`로 객체를 손에 쥐고, 이후 모든 조작을 `ax.` 또는 `fig.`로 한다.

## 형식화

좌표 변환 파이프라인을 명시하면 축 설정과 주석 배치가 이해된다. Matplotlib은 네 개의 좌표계를 갖는다.

데이터 좌표 $(x, y)$는 사용자 데이터의 단위다. 축 좌표는 Axes 영역을 $[0,1]^2$로 정규화한 것이고, 그림 좌표는 Figure 전체를 $[0,1]^2$로 정규화한 것이며, 디스플레이 좌표는 픽셀 단위다.

데이터에서 디스플레이까지의 변환은 합성으로 정의된다.

$$T_{\text{display}} = T_{\text{fig}} \circ T_{\text{axes}} \circ T_{\text{scale}}$$

여기서 $T_{\text{scale}}$이 선형이면 일반 축, $\log$면 로그 축이다. 로그 축에서 데이터 좌표 $x$는 $\log_{10} x$로 변환된 뒤 선형 매핑된다. 0이나 음수를 로그 축에 그리면 사라지는 이유가 이것이다.

출력 크기 계산도 명확히 해둔다. `figsize`가 인치 단위 $(w, h)$이고 `dpi`가 인치당 픽셀이면 저장되는 이미지의 픽셀 크기는

$$(W_{\text{px}}, H_{\text{px}}) = (w \cdot \text{dpi}, \ h \cdot \text{dpi})$$

이다. 논문 그림에서 폰트 크기가 이상하게 보이는 문제는 대부분 여기서 나온다. 폰트 크기는 포인트(1/72인치) 단위로 지정되므로 dpi와 무관하게 물리적 크기가 고정된다. 그림을 크게 만들면(figsize를 키우면) 상대적으로 글자가 작아지고, dpi만 키우면 픽셀 수만 늘고 비율은 그대로다. 논문 제출 시 그림을 축소해 넣으면 글자가 작아지는 것도 같은 이유다.

## 구현

### 백엔드와 기본 설정

학습 스크립트 최상단에서 백엔드를 고정한다.

```python
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

print(matplotlib.get_backend())
```

`Agg`는 파일 출력 전용 래스터 백엔드다. 디스플레이가 없어도 동작한다. `matplotlib.use`는 `pyplot` 임포트 전에 호출해야 한다.

전역 스타일을 한 곳에서 정한다.

```python
import matplotlib.pyplot as plt


def apply_style():
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "figure.autolayout": False,
    })


apply_style()
print(plt.rcParams["figure.figsize"], plt.rcParams["savefig.dpi"])
```

`savefig.bbox="tight"`가 잘리는 레이블 문제를 대부분 해결한다. `figure.dpi`와 `savefig.dpi`를 분리한 이유는 화면 확인은 가볍게, 저장은 고해상도로 하기 위해서다.

한글 폰트가 필요하면 명시적으로 지정한다. 지정하지 않으면 네모 상자가 나온다.

```python
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager


def set_korean_font():
    candidates = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


print(set_korean_font())
```

`axes.unicode_minus = False`가 필요한 이유는 유니코드 마이너스 기호가 대부분의 한글 폰트에 없어 음수 눈금이 깨지기 때문이다.

### Figure와 Axes 구조

명시적 방식의 기본형이다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng(0)
steps = np.arange(200)
train_loss = 2.0 * np.exp(-steps / 60) + 0.15 + rng.normal(scale=0.03, size=200)
val_loss = 2.0 * np.exp(-steps / 55) + 0.22 + rng.normal(scale=0.05, size=200)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, train_loss, label="train")
ax.plot(steps, val_loss, label="validation")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_title("learning curve")
ax.legend(loc="upper right")
fig.savefig("curve.png")
plt.close(fig)
print("saved")
```

`plt.close(fig)`가 핵심이다. 이것이 없으면 앞서 말한 누수가 발생한다.

여러 축을 배치한다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng(0)
steps = np.arange(300)
loss = 2.0 * np.exp(-steps / 80) + 0.2 + rng.normal(scale=0.04, size=300)
lr = 3e-4 * 0.5 * (1 + np.cos(np.pi * steps / 300))
grad_norm = np.abs(rng.normal(loc=1.5, scale=0.8, size=300))
accuracy = 1 - np.exp(-steps / 100) * 0.6 + rng.normal(scale=0.01, size=300)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

axes[0, 0].plot(steps, loss, color="tab:blue")
axes[0, 0].set_ylabel("loss")
axes[0, 0].set_title("training loss")

axes[0, 1].plot(steps, lr, color="tab:orange")
axes[0, 1].set_ylabel("learning rate")
axes[0, 1].set_title("cosine schedule")
axes[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

axes[1, 0].plot(steps, grad_norm, color="tab:green", alpha=0.7)
axes[1, 0].axhline(5.0, color="tab:red", linestyle="--", label="clip threshold")
axes[1, 0].set_xlabel("step")
axes[1, 0].set_ylabel("grad norm")
axes[1, 0].legend()

axes[1, 1].plot(steps, accuracy, color="tab:purple")
axes[1, 1].set_xlabel("step")
axes[1, 1].set_ylabel("accuracy")
axes[1, 1].set_ylim(0, 1)

fig.suptitle("training diagnostics", fontsize=13)
fig.tight_layout()
fig.savefig("diagnostics.png")
plt.close(fig)
print("saved")
```

`sharex=True`가 x축을 동기화해 확대 시 함께 움직인다. `tight_layout`이 레이블 겹침을 해소한다.

불규칙한 배치는 `subplot_mosaic`이 가장 읽기 쉽다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng(0)
fig, axd = plt.subplot_mosaic(
    """
    AAB
    AAC
    """,
    figsize=(10, 5),
)

axd["A"].plot(np.cumsum(rng.normal(size=200)))
axd["A"].set_title("main curve")
axd["B"].hist(rng.normal(size=500), bins=25, color="tab:orange")
axd["B"].set_title("distribution")
axd["C"].scatter(rng.normal(size=100), rng.normal(size=100), s=8, alpha=0.6)
axd["C"].set_title("scatter")

fig.tight_layout()
fig.savefig("mosaic.png")
plt.close(fig)
print("saved")
```

### 플롯 종류별 선택

데이터 성격에 따른 선택 기준을 코드로 정리한다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 3, figsize=(13, 7))

x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), label="sin")
axes[0, 0].plot(x, np.cos(x), label="cos", linestyle="--")
axes[0, 0].set_title("line: 연속 추세")
axes[0, 0].legend(fontsize=8)

points = rng.normal(size=(300, 2))
colors = points[:, 0] + points[:, 1]
scatter = axes[0, 1].scatter(points[:, 0], points[:, 1], c=colors, s=12, cmap="viridis", alpha=0.7)
axes[0, 1].set_title("scatter: 두 변수 관계")
fig.colorbar(scatter, ax=axes[0, 1], label="sum")

samples = rng.normal(loc=0, scale=1, size=2000)
axes[0, 2].hist(samples, bins=40, density=True, alpha=0.7, color="tab:green")
axes[0, 2].set_title("hist: 분포 형태")

groups = [rng.normal(loc=m, scale=s, size=200) for m, s in [(0, 1), (1, 1.5), (-0.5, 0.7)]]
axes[1, 0].boxplot(groups, tick_labels=["a", "b", "c"])
axes[1, 0].set_title("box: 그룹 간 분포 비교")

categories = ["ResNet", "ViT", "ConvNeXt", "Swin"]
scores = [76.1, 79.8, 82.1, 83.0]
bars = axes[1, 1].bar(categories, scores, color="tab:blue", alpha=0.8)
axes[1, 1].set_ylim(70, 85)
axes[1, 1].set_title("bar: 범주별 크기")
for bar, score in zip(bars, scores):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, score + 0.2, f"{score}", ha="center", fontsize=8)

matrix = rng.random((8, 8))
image = axes[1, 2].imshow(matrix, cmap="magma", aspect="auto")
axes[1, 2].set_title("imshow: 2차원 배열")
fig.colorbar(image, ax=axes[1, 2])

fig.tight_layout()
fig.savefig("plot_types.png")
plt.close(fig)
print("saved")
```

막대그래프의 `set_ylim(70, 85)`은 의도적이면서 위험한 선택이다. 차이를 강조하지만 절대 크기 인식을 왜곡한다. 0에서 시작하지 않은 축은 명시적으로 표시하는 것이 정직하다.

### 축과 눈금

축 제어가 필요한 상황들이다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


steps = np.arange(1, 10001)
loss = 5.0 / np.sqrt(steps) + 0.1

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].plot(steps, loss)
axes[0].set_title("linear")

axes[1].plot(steps, loss)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_title("log-log")

axes[2].plot(steps, loss)
axes[2].set_yscale("log")
axes[2].xaxis.set_major_locator(ticker.MultipleLocator(2500))
axes[2].xaxis.set_minor_locator(ticker.MultipleLocator(500))
axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
axes[2].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
axes[2].grid(which="minor", alpha=0.15)
axes[2].set_title("custom ticks")

fig.tight_layout()
fig.savefig("axes.png")
plt.close(fig)
print("saved")
```

로그-로그 축에서 직선이 되면 멱법칙 관계다. 스케일링 법칙 분석에서 이 축 설정이 기본이다.

이중 축은 신중하게 쓴다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


steps = np.arange(500)
loss = 2.0 * np.exp(-steps / 120) + 0.2
lr = 3e-4 * 0.5 * (1 + np.cos(np.pi * steps / 500))

fig, ax_left = plt.subplots(figsize=(7, 4))
line_loss, = ax_left.plot(steps, loss, color="tab:blue", label="loss")
ax_left.set_xlabel("step")
ax_left.set_ylabel("loss", color="tab:blue")
ax_left.tick_params(axis="y", labelcolor="tab:blue")

ax_right = ax_left.twinx()
line_lr, = ax_right.plot(steps, lr, color="tab:red", linestyle="--", label="lr")
ax_right.set_ylabel("learning rate", color="tab:red")
ax_right.tick_params(axis="y", labelcolor="tab:red")
ax_right.grid(False)

ax_left.legend(handles=[line_loss, line_lr], loc="upper right")
fig.tight_layout()
fig.savefig("twin.png")
plt.close(fig)
print("saved")
```

두 축의 격자가 겹치면 읽기 어려우므로 한쪽을 끈다. 범례는 두 축의 핸들을 모아 한 번에 만든다. 이중 축은 두 계열의 스케일이 근본적으로 다를 때만 쓰고, 그렇지 않으면 별도 서브플롯이 낫다.

### 범례와 주석

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng(0)
steps = np.arange(400)
runs = {
    "baseline": 2.0 * np.exp(-steps / 90) + 0.30,
    "with warmup": 2.0 * np.exp(-steps / 70) + 0.24,
    "with EMA": 2.0 * np.exp(-steps / 65) + 0.21,
}

fig, ax = plt.subplots(figsize=(8, 4.5))
for label, values in runs.items():
    noisy = values + rng.normal(scale=0.015, size=values.shape)
    ax.plot(steps, noisy, label=label, alpha=0.85)

best_step = int(np.argmin(runs["with EMA"]))
best_value = float(runs["with EMA"][best_step])
ax.scatter([best_step], [best_value], color="black", zorder=5, s=30)
ax.annotate(
    f"best {best_value:.3f}",
    xy=(best_step, best_value),
    xytext=(best_step - 130, best_value + 0.25),
    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.0},
    fontsize=9,
)

ax.axvspan(0, 40, color="tab:gray", alpha=0.15)
ax.text(20, ax.get_ylim()[1] * 0.95, "warmup", ha="center", va="top", fontsize=9)

ax.set_xlabel("step")
ax.set_ylabel("validation loss")
ax.legend(loc="upper right", ncols=1)
fig.tight_layout()
fig.savefig("annotated.png")
plt.close(fig)
print("saved")
```

`zorder`로 그리기 순서를 제어한다. 값이 클수록 위에 그려진다. 마커가 선에 가려지는 문제를 이것으로 해결한다.

### 색상과 접근성

색상 선택에도 규칙이 있다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

x = np.linspace(0, 10, 200)
for i in range(6):
    axes[0].plot(x, np.sin(x + i * 0.5) + i * 0.3, label=f"run {i}")
axes[0].set_title("qualitative: 구분 가능한 색")
axes[0].legend(fontsize=7, ncols=2)

gradient = np.linspace(0, 1, 256).reshape(1, -1)
axes[1].imshow(gradient, aspect="auto", cmap="viridis")
axes[1].set_title("sequential: viridis")
axes[1].set_yticks([])

diverging = np.linspace(-1, 1, 256).reshape(1, -1)
axes[2].imshow(diverging, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
axes[2].set_title("diverging: 0 기준 대칭")
axes[2].set_yticks([])

fig.tight_layout()
fig.savefig("colors.png")
plt.close(fig)
print("saved")
```

세 종류를 구분한다. 범주형에는 구분 가능한 색(기본 `tab10`), 단조 증가하는 값에는 순차형(`viridis`, `magma`), 0을 기준으로 양방향인 값에는 발산형(`RdBu_r`)을 쓴다. `jet`은 지각적으로 균등하지 않아 존재하지 않는 경계를 만들어내므로 쓰지 않는다.

발산형에서는 `vmin`과 `vmax`를 대칭으로 잡아야 0이 중앙색에 온다. 이를 빠뜨리면 색이 의미를 잃는다.

### 저장과 형식

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100) ** 2)
ax.set_xlabel("x")
ax.set_ylabel("y")

tmpdir = tempfile.mkdtemp()
for name, kwargs in [
    ("plot.png", {"dpi": 200}),
    ("plot_transparent.png", {"dpi": 200, "transparent": True}),
    ("plot.pdf", {}),
    ("plot.svg", {}),
]:
    path = os.path.join(tmpdir, name)
    fig.savefig(path, bbox_inches="tight", **kwargs)
    print(f"{name}: {os.path.getsize(path) / 1024:.1f} KiB")

plt.close(fig)
```

논문과 문서에는 PDF나 SVG를 쓴다. 벡터 형식이라 확대해도 깨지지 않고 폰트가 텍스트로 남는다. 산점도의 점이 수만 개면 벡터 파일이 거대해지므로, 이 경우 `rasterized=True`로 해당 Artist만 래스터화한다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile


rng = np.random.default_rng(0)
points = rng.normal(size=(50_000, 2))

tmpdir = tempfile.mkdtemp()

fig, ax = plt.subplots()
ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.3)
path_vector = os.path.join(tmpdir, "dense_vector.pdf")
fig.savefig(path_vector)
plt.close(fig)

fig, ax = plt.subplots()
ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.3, rasterized=True)
ax.set_xlabel("x")
path_mixed = os.path.join(tmpdir, "dense_mixed.pdf")
fig.savefig(path_mixed, dpi=200)
plt.close(fig)

print(f"full vector: {os.path.getsize(path_vector) / 1024**2:.2f} MiB")
print(f"rasterized points: {os.path.getsize(path_mixed) / 1024**2:.2f} MiB")
```

축과 레이블은 벡터로, 점은 래스터로 남아 파일 크기가 크게 줄면서 텍스트 품질은 유지된다.

### 학습 코드 통합

앞의 요소를 재사용 가능한 함수로 정리한다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def figure(path: str | Path, figsize: tuple[float, float] = (7.0, 4.5), dpi: int = 200):
    fig, ax = plt.subplots(figsize=figsize)
    try:
        yield fig, ax
    finally:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_curves(path: str | Path, curves: dict[str, np.ndarray], xlabel: str = "step", ylabel: str = "value", logy: bool = False):
    with figure(path) as (fig, ax):
        for label, values in curves.items():
            ax.plot(np.arange(len(values)), values, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if len(curves) > 1:
            ax.legend()


def plot_confusion(path: str | Path, matrix: np.ndarray, labels: list[str]):
    with figure(path, figsize=(5.5, 5.0)) as (fig, ax):
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color = "white" if normalized[i, j] > 0.5 else "black"
                ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046)


import tempfile
tmpdir = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)
steps = np.arange(200)
plot_curves(tmpdir / "loss.png", {
    "train": 2.0 * np.exp(-steps / 60) + 0.15,
    "val": 2.0 * np.exp(-steps / 55) + 0.22,
}, ylabel="loss")

confusion = rng.integers(0, 40, size=(4, 4)) + np.diag([120, 130, 110, 140])
plot_confusion(tmpdir / "confusion.png", confusion, ["cat", "dog", "bird", "fish"])
print(sorted(p.name for p in tmpdir.iterdir()))
```

컨텍스트 매니저가 `plt.close`를 보장한다. 앞의 컨텍스트 매니저 문서에서 다룬 패턴의 직접 적용이다.

혼동행렬에서 텍스트 색을 배경 밝기에 따라 바꾼 것이 가독성의 핵심이다. 어두운 배경에 검은 글씨는 읽을 수 없다.

## 실무 관점

학습 스크립트에서 그림을 그릴 때 세 가지를 지킨다. 백엔드를 `Agg`로 고정하고, Figure를 반드시 닫으며, 그림 생성이 학습 루프를 막지 않도록 주기를 조절한다. 매 스텝 그리면 그림 생성 비용이 학습보다 커진다.

메모리 누수를 확인하는 방법이다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


for i in range(5):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, i])
    plt.close(fig)
print(f"open figures after close: {len(plt.get_fignums())}")

for i in range(5):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, i])
print(f"open figures without close: {len(plt.get_fignums())}")
plt.close("all")
print(f"after close all: {len(plt.get_fignums())}")
```

`plt.get_fignums()`가 열린 Figure 수를 준다. 학습 후 이 값이 0이 아니면 누수가 있다.

실험 추적 도구와의 통합에서는 Figure 객체를 그대로 넘긴다. 파일로 저장했다가 다시 읽지 않는다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_figure(values: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(values)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    return fig


def log_figure(tracker, name: str, fig, step: int):
    try:
        tracker.log_figure(fig, name, step)
    finally:
        plt.close(fig)


class DummyTracker:
    def log_figure(self, fig, name: str, step: int):
        print(f"logged {name} at step {step}, axes={len(fig.axes)}")


log_figure(DummyTracker(), "train/loss", make_figure(np.linspace(1, 0.2, 100)), step=100)
print(f"open figures: {len(plt.get_fignums())}")
```

`finally`에서 닫는 것이 중요하다. 로깅이 실패해도 Figure는 정리된다.

대용량 산점도는 그대로 그리지 않는다. 10만 점을 그리면 렌더링이 느리고 겹쳐서 아무것도 보이지 않는다. 2차원 히스토그램이나 육각 비닝을 쓴다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng(0)
n = 200_000
x = rng.normal(size=n)
y = x * 0.6 + rng.normal(scale=0.8, size=n)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].scatter(x[:5000], y[:5000], s=2, alpha=0.2)
axes[0].set_title("scatter (subsampled 5k)")

hist = axes[1].hist2d(x, y, bins=80, cmap="viridis")
axes[1].set_title("hist2d (200k)")
fig.colorbar(hist[3], ax=axes[1])

hexes = axes[2].hexbin(x, y, gridsize=60, cmap="magma", mincnt=1)
axes[2].set_title("hexbin (200k)")
fig.colorbar(hexes, ax=axes[2])

fig.tight_layout()
fig.savefig("density.png")
plt.close(fig)
print("saved")
```

밀도 표현이 겹친 점보다 정보를 훨씬 많이 전달한다.

폰트 크기 문제는 앞의 형식화에서 유도한 관계로 해결한다. 논문에 넣을 그림은 최종 게재 크기와 같은 `figsize`로 만들고 축소하지 않는다. 두 칼럼 논문에서 칼럼 폭이 3.3인치면 `figsize=(3.3, 2.5)`로 만든다. 크게 만들어 축소하면 글자가 작아진다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def paper_figure(width_inches: float = 3.3, aspect: float = 0.7):
    fig, ax = plt.subplots(figsize=(width_inches, width_inches * aspect))
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    return fig, ax


fig, ax = paper_figure()
steps = np.arange(100)
ax.plot(steps, 1 / (1 + steps / 20), label="proposed")
ax.plot(steps, 1 / (1 + steps / 12), label="baseline", linestyle="--")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.legend(fontsize=7)
fig.savefig("paper.pdf", bbox_inches="tight")
plt.close(fig)
print("saved")
```

색맹 접근성도 고려한다. 색만으로 계열을 구분하면 8퍼센트 정도의 남성 독자가 구별하지 못한다. 선 스타일이나 마커를 함께 바꾼다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


styles = [
    {"color": "tab:blue", "linestyle": "-", "marker": "o"},
    {"color": "tab:orange", "linestyle": "--", "marker": "s"},
    {"color": "tab:green", "linestyle": "-.", "marker": "^"},
]

fig, ax = plt.subplots(figsize=(6, 3.5))
x = np.linspace(0, 10, 20)
for i, style in enumerate(styles):
    ax.plot(x, np.sin(x + i), label=f"run {i}", markevery=3, markersize=4, **style)
ax.legend()
fig.tight_layout()
fig.savefig("accessible.png")
plt.close(fig)
print("saved")
```

`markevery`로 마커를 솎아내야 조밀한 곡선에서 마커가 뭉치지 않는다.

## 핵심 정리

`fig, ax = plt.subplots()`로 객체를 손에 쥐고 이후 모든 조작을 `ax.` 또는 `fig.`로 한다. 전역 상태에 의존하는 `plt.` 함수는 대화형 탐색에만 쓴다.

학습 스크립트는 `matplotlib.use("Agg")`를 `pyplot` 임포트 전에 호출한다. 디스플레이 없는 서버에서 필수다.

Figure는 반드시 `plt.close(fig)`로 닫는다. 컨텍스트 매니저로 감싸면 예외 시에도 보장된다. `plt.get_fignums()`로 누수를 확인한다.

저장 픽셀 크기는 `figsize × dpi`이고 폰트는 물리 크기가 고정된 포인트 단위다. 논문 그림은 최종 게재 크기와 같은 `figsize`로 만들고 축소하지 않는다.

색상은 범주형, 순차형, 발산형을 구분해 쓴다. `jet`은 지각적으로 불균등하므로 쓰지 않는다. 발산형은 `vmin`과 `vmax`를 대칭으로 잡는다.

색만으로 계열을 구분하지 않고 선 스타일과 마커를 함께 바꾼다.

수만 점 이상의 산점도는 `hist2d`나 `hexbin`으로 대체한다. 겹친 점보다 밀도가 정보를 많이 준다.

벡터 형식(PDF, SVG)에 조밀한 점을 그릴 때는 해당 Artist만 `rasterized=True`로 지정한다.
