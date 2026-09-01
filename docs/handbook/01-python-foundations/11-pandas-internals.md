# Pandas: 인덱싱, groupby, 메모리

## 한 줄 정의

DataFrame은 열마다 dtype이 다를 수 있는 NumPy 배열들의 묶음에 레이블 축(인덱스)을 붙인 구조이며, 성능과 버그의 대부분이 이 레이블 정렬과 블록 저장 방식에서 나온다.

## 문제 상황

데이터가 수백만 행을 넘어가면 Pandas 코드가 갑자기 느려지거나 메모리를 터뜨린다. 원인이 명확하지 않다는 것이 문제다. 같은 연산인데 어떤 날은 3초, 어떤 날은 3분이 걸린다.

전형적인 사례들이다. `for idx, row in df.iterrows()`로 100만 행을 돌면서 각 행을 처리하면 30분이 걸린다. 같은 작업을 벡터 연산으로 쓰면 2초다. `df[df.col == x] = y` 형태의 대입이 원본을 바꾸지 않고 `SettingWithCopyWarning`만 남긴다. 문자열 열이 `object` dtype으로 저장되어 100만 행에 8GB를 쓴다. `groupby().apply()`가 그룹마다 파이썬 함수를 호출해 수십 배 느리다. 두 DataFrame을 더했더니 인덱스가 정렬되면서 행이 사라지거나 NaN이 생긴다.

이 문제들의 공통 원인은 Pandas가 NumPy 위에 레이블 정렬 계층을 얹었다는 사실이다. 그 계층이 편의를 주는 대신 비용과 함정을 함께 만든다.

## 직관적 이해

DataFrame을 스프레드시트로 생각하면 오해가 생긴다. 실제 구조는 열 지향 저장소에 가깝다. 같은 dtype의 열들이 하나의 2차원 블록으로 뭉쳐 저장되고, dtype이 다르면 별도 블록이 된다. 그래서 열 단위 연산은 빠르고 행 단위 접근은 느리다. 한 행을 꺼내려면 여러 블록에서 값을 모아 새 객체를 만들어야 하기 때문이다.

인덱스는 행에 붙은 이름표다. 두 Series를 더할 때 Pandas는 위치가 아니라 이름표를 맞춘다. 이름표가 다르면 합집합을 만들고 없는 자리는 NaN이 된다. NumPy에서 위치 기반으로 사고하던 습관이 여기서 깨진다.

`groupby`는 정렬 또는 해시로 그룹을 나눈 뒤 각 그룹에 함수를 적용한다. 그 함수가 C로 구현된 내장 집계면 빠르고, 파이썬 함수면 그룹 수만큼 인터프리터를 왕복한다. 성능 차이가 여기서 갈린다.

## 형식화

Series의 이항 연산은 인덱스 정렬을 포함한다. 인덱스 집합 $I_A$, $I_B$를 가진 두 Series의 연산 결과는

$$(A \oplus B)_k = \begin{cases} A_k \oplus B_k & k \in I_A \cap I_B \\ \text{NaN} & k \in (I_A \cup I_B) \setminus (I_A \cap I_B) \end{cases}$$

이고 결과 인덱스는 $I_A \cup I_B$다. 이 정렬 비용은 인덱스가 정렬되어 있으면 $O(n)$, 아니면 해시 테이블 구축으로 $O(n)$이지만 상수가 크다. 반복 호출하면 누적된다.

`groupby` 연산의 비용을 분해한다. 행 $n$개, 그룹 $g$개일 때

$$T_{\text{groupby}} = T_{\text{factorize}}(n) + T_{\text{aggregate}}$$

이고 factorize는 그룹 키를 정수 코드로 바꾸는 단계로 $O(n)$이다. 집계 단계는 구현에 따라 크게 갈린다.

$$T_{\text{aggregate}} = \begin{cases} O(n) & \text{내장 집계 (Cython 경로)} \\ g \cdot (C_{\text{python}} + O(n/g)) & \text{apply / lambda} \end{cases}$$

$C_{\text{python}}$은 파이썬 함수 호출과 Series 객체 생성 비용으로 수 마이크로초 수준이다. 그룹이 10만 개면 이 항만으로 수 초가 든다. 그룹이 적고 그룹당 데이터가 크면 `apply`의 상대 비용이 작아지므로, 그룹 수가 판단 기준이 된다.

메모리는 dtype이 지배한다. 행 $n$개인 열 하나의 크기는

$$M_{\text{numeric}} = n \cdot b, \qquad M_{\text{object}} = n \cdot 8 + \sum_{i=1}^{n} (49 + \text{len}(s_i))$$

이다. object dtype은 포인터 배열에 더해 각 문자열 객체의 헤더(CPython에서 약 49바이트)와 내용을 따로 갖는다. 카디널리티가 $c$인 범주형으로 바꾸면

$$M_{\text{category}} = n \cdot b_{\text{code}} + \sum_{j=1}^{c} (49 + \text{len}(u_j))$$

로 줄어든다. $c \ll n$이면 절감이 극적이다. 100만 행에 고유값 50개인 문자열 열이 60MB에서 1MB로 줄어드는 일이 흔하다.

## 구현

### 저장 구조 확인

블록 구조를 직접 들여다본다.

```python
import numpy as np
import pandas as pd


rng = np.random.default_rng(0)
frame = pd.DataFrame({
    "user_id": rng.integers(0, 1000, size=6),
    "score": rng.normal(size=6),
    "weight": rng.normal(size=6).astype(np.float32),
    "split": rng.choice(["train", "val"], size=6),
})

print(frame.dtypes)
print(frame.info(memory_usage="deep"))
print(frame.memory_usage(deep=True))
```

`memory_usage(deep=True)`가 object 열의 실제 문자열 크기까지 계산한다. `deep=False`면 포인터 배열 크기만 세므로 실제보다 훨씬 작게 나온다.

### 인덱싱 규칙

`loc`, `iloc`, `[]`의 차이를 정리한다.

```python
import numpy as np
import pandas as pd


frame = pd.DataFrame(
    {"score": [0.1, 0.2, 0.3, 0.4], "label": [0, 1, 0, 1]},
    index=["a", "b", "c", "d"],
)

print(frame.loc["b"])
print(frame.iloc[1])
print(frame.loc["b":"c"])
print(frame.iloc[1:3])
print(frame["score"])
print(frame[["score", "label"]])
```

`loc` 슬라이스는 끝 포함, `iloc` 슬라이스는 끝 제외다. 이 차이가 오프바이원 버그의 단골이다.

정수 인덱스에서 특히 혼란스럽다.

```python
import pandas as pd


series = pd.Series([10, 20, 30], index=[2, 0, 1])
print(f"loc[0] (label): {series.loc[0]}")
print(f"iloc[0] (position): {series.iloc[0]}")
```

레이블 0이 위치 1에 있으므로 결과가 다르다. 정수 인덱스를 쓸 때는 `[]` 대신 항상 `loc` 또는 `iloc`을 명시한다.

불리언 마스킹과 조건 조합도 정리한다.

```python
import numpy as np
import pandas as pd


rng = np.random.default_rng(0)
frame = pd.DataFrame({
    "score": rng.normal(size=1000),
    "group": rng.choice(["a", "b", "c"], size=1000),
    "count": rng.integers(0, 100, size=1000),
})

mask = (frame["score"] > 0) & (frame["count"] < 50)
print(f"combined mask: {mask.sum()} rows")

print(frame.loc[mask, ["score", "group"]].head())
print(frame.query("score > 0 and count < 50").shape)
print(frame[frame["group"].isin(["a", "b"])].shape)
```

`&`와 `|`를 쓰고 괄호로 감싸야 한다. `and`, `or`는 Series에 대해 `ValueError`를 낸다. 진리값이 모호하다는 메시지가 나오면 이 실수다.

### SettingWithCopyWarning

가장 악명 높은 함정이다.

```python
import numpy as np
import pandas as pd


rng = np.random.default_rng(0)
source = pd.DataFrame({
    "score": rng.normal(size=10),
    "group": rng.choice(["a", "b"], size=10),
})

subset = source[source["group"] == "a"]
subset["score"] = 0.0
print(f"original changed: {(source['score'] == 0.0).any()}")

correct = source.copy()
correct.loc[correct["group"] == "a", "score"] = 0.0
print(f"correct assignment worked: {(correct['score'] == 0.0).any()}")
```

첫 번째 방식은 `subset`이 뷰인지 복사본인지 확정되지 않아 원본 반영 여부가 상황에 따라 달라진다. 규칙은 단순하다. 대입은 항상 `loc` 한 번의 호출로 한다. 연쇄 인덱싱(`df[...][...] = `)을 쓰지 않는다.

Pandas 3.0부터는 Copy-on-Write가 기본이 되어 이 모호함이 사라진다. 그 전 버전에서도 명시적으로 켤 수 있다.

```python
import pandas as pd


pd.options.mode.copy_on_write = True
print(pd.options.mode.copy_on_write)
```

CoW에서는 모든 인덱싱 결과가 논리적으로 복사본처럼 동작하고, 실제 복사는 쓰기가 일어날 때만 발생한다. 원본 오염이 구조적으로 불가능해진다.

### 반복 대신 벡터화

성능 차이를 측정한다.

```python
import time

import numpy as np
import pandas as pd


def timeit(fn, repeat: int = 3):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


rng = np.random.default_rng(0)
n = 200_000
frame = pd.DataFrame({
    "a": rng.normal(size=n),
    "b": rng.normal(size=n),
})


def with_iterrows():
    out = []
    for _, row in frame.iterrows():
        out.append(row["a"] * 2 + row["b"])
    return out


def with_itertuples():
    return [r.a * 2 + r.b for r in frame.itertuples(index=False)]


def with_apply():
    return frame.apply(lambda r: r["a"] * 2 + r["b"], axis=1)


def with_vector():
    return frame["a"] * 2 + frame["b"]


def with_numpy():
    return frame["a"].to_numpy() * 2 + frame["b"].to_numpy()


for name, fn in [
    ("iterrows", with_iterrows),
    ("itertuples", with_itertuples),
    ("apply axis=1", with_apply),
    ("vectorized", with_vector),
    ("to_numpy", with_numpy),
]:
    print(f"{name:14s} {timeit(fn):.4f}s")
```

`iterrows`가 압도적으로 느리다. 행마다 Series 객체를 새로 만들기 때문이다. `itertuples`는 namedtuple이라 훨씬 가볍지만 여전히 벡터 연산보다 두 자릿수 느리다. `apply(axis=1)`은 사실상 `iterrows`와 같다.

`iterrows`의 또 다른 문제는 dtype 손실이다.

```python
import pandas as pd


mixed = pd.DataFrame({"count": [1, 2, 3], "score": [0.1, 0.2, 0.3]})
first = next(mixed.iterrows())[1]
print(f"row dtype: {first.dtype}")
print(f"count value type: {type(first['count'])}")
print(f"original count dtype: {mixed['count'].dtype}")
```

행 하나에 int와 float가 섞이면 공통 dtype인 float64로 승격되어 정수가 실수가 된다. ID를 이렇게 꺼내면 큰 값에서 정밀도가 손실된다.

### groupby

내장 집계와 apply의 차이를 측정한다.

```python
import time

import numpy as np
import pandas as pd


def timeit(fn, repeat: int = 3):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


rng = np.random.default_rng(0)
n = 500_000
frame = pd.DataFrame({
    "key": rng.integers(0, 20_000, size=n),
    "value": rng.normal(size=n),
})


def builtin_agg():
    return frame.groupby("key")["value"].mean()


def apply_lambda():
    return frame.groupby("key")["value"].apply(lambda s: s.mean())


def agg_multiple():
    return frame.groupby("key")["value"].agg(["mean", "std", "count"])


def transform_normalize():
    grouped = frame.groupby("key")["value"]
    return (frame["value"] - grouped.transform("mean")) / grouped.transform("std")


print(f"builtin mean:     {timeit(builtin_agg):.4f}s")
print(f"apply lambda:     {timeit(apply_lambda):.4f}s")
print(f"agg multiple:     {timeit(agg_multiple):.4f}s")
print(f"transform:        {timeit(transform_normalize):.4f}s")
```

형식화에서 예측한 대로 그룹 수가 2만 개이므로 `apply`의 파이썬 호출 비용이 지배한다.

`transform`은 집계 결과를 원본 형상으로 되돌려준다. 그룹별 정규화 같은 작업의 표준 도구이며, `merge`로 되돌리는 것보다 빠르고 짧다.

여러 집계를 한 번에 하는 방법이다.

```python
import numpy as np
import pandas as pd


rng = np.random.default_rng(0)
frame = pd.DataFrame({
    "user": rng.choice(["u1", "u2", "u3"], size=100),
    "item": rng.choice(["i1", "i2"], size=100),
    "score": rng.normal(size=100),
    "duration": rng.integers(1, 100, size=100),
})

summary = frame.groupby(["user", "item"]).agg(
    score_mean=("score", "mean"),
    score_std=("score", "std"),
    total_duration=("duration", "sum"),
    events=("score", "size"),
).reset_index()

print(summary)
```

명명된 집계(`named aggregation`)를 쓰면 결과 열 이름이 명확해진다. 다단계 열 이름이 생기지 않아 이후 처리가 단순하다.

`observed=True`를 범주형 그룹 키에 지정하는 것이 중요하다.

```python
import pandas as pd


frame = pd.DataFrame({
    "cat": pd.Categorical(["a", "b"], categories=["a", "b", "c", "d", "e"]),
    "value": [1.0, 2.0],
})

print(frame.groupby("cat", observed=False)["value"].mean())
print(frame.groupby("cat", observed=True)["value"].mean())
```

`observed=False`면 데이터에 없는 범주까지 결과에 포함되어 NaN 행이 생긴다. 범주 조합이 많으면 결과가 폭발한다.

### 결측 처리

Pandas의 결측 표현은 dtype마다 다르다.

```python
import numpy as np
import pandas as pd


float_series = pd.Series([1.0, np.nan, 3.0])
print(f"float64: {float_series.dtype}, isna={float_series.isna().tolist()}")

int_series = pd.Series([1, None, 3])
print(f"int with None: {int_series.dtype}")

nullable_int = pd.Series([1, None, 3], dtype="Int64")
print(f"nullable Int64: {nullable_int.dtype}, isna={nullable_int.isna().tolist()}")
print(f"sum: {nullable_int.sum()}")

string_series = pd.Series(["a", None, "c"], dtype="string")
print(f"string dtype: {string_series.dtype}, isna={string_series.isna().tolist()}")
```

기본 int64 열에 결측이 들어가면 float64로 승격된다. ID 열이 이렇게 바뀌면 큰 정수에서 정밀도를 잃는다. nullable 확장 dtype(`Int64`, `Float64`, `string`, `boolean`)을 쓰면 dtype을 유지한 채 결측을 표현할 수 있다.

결측 처리 전략을 코드로 정리한다.

```python
import numpy as np
import pandas as pd


rng = np.random.default_rng(0)
frame = pd.DataFrame({
    "numeric": rng.normal(size=20),
    "category": rng.choice(["a", "b", None], size=20),
    "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
})
frame.loc[rng.choice(20, size=5, replace=False), "numeric"] = np.nan


def missing_report(df: pd.DataFrame):
    counts = df.isna().sum()
    ratios = counts / len(df)
    return pd.DataFrame({"missing": counts, "ratio": ratios.round(3)})


print(missing_report(frame))

filled = frame.copy()
filled["numeric"] = filled["numeric"].fillna(filled["numeric"].median())
filled["category"] = filled["category"].fillna("unknown")
print(missing_report(filled))
```

중앙값 대치를 쓴 것은 이상치에 강건하기 때문이다. 다만 결측 자체가 정보인 경우가 많으므로, 결측 지시 열을 따로 만드는 편이 낫다.

```python
import numpy as np
import pandas as pd


def add_missing_indicators(df: pd.DataFrame, columns: list[str]):
    out = df.copy()
    for col in columns:
        out[f"{col}_was_missing"] = out[col].isna().astype(np.int8)
    return out


rng = np.random.default_rng(0)
sample = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
print(add_missing_indicators(sample, ["x"]))
```

시계열에서는 대치 방향이 중요하다. 미래 값으로 과거를 채우면 데이터 누수다.

```python
import numpy as np
import pandas as pd


series = pd.Series([1.0, np.nan, np.nan, 4.0], index=pd.date_range("2024-01-01", periods=4))
print(f"ffill (safe):  {series.ffill().tolist()}")
print(f"bfill (leak):  {series.bfill().tolist()}")
print(f"interpolate:   {series.interpolate().tolist()}")
```

`bfill`과 `interpolate`는 미래 정보를 쓰므로 예측 파이프라인에서 금지한다.

### 메모리 최적화

형식화의 dtype별 크기 공식을 실측한다.

```python
import numpy as np
import pandas as pd


def optimize_dtypes(df: pd.DataFrame, category_threshold: float = 0.5):
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_integer_dtype(series):
            out[col] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            out[col] = pd.to_numeric(series, downcast="float")
        elif pd.api.types.is_object_dtype(series):
            uniques = series.nunique(dropna=False)
            if uniques / max(len(series), 1) < category_threshold:
                out[col] = series.astype("category")
    return out


rng = np.random.default_rng(0)
n = 200_000
frame = pd.DataFrame({
    "user_id": rng.integers(0, 30_000, size=n),
    "score": rng.normal(size=n),
    "country": rng.choice(["KR", "US", "JP", "DE", "FR"], size=n),
    "device": rng.choice(["ios", "android", "web"], size=n),
    "flag": rng.integers(0, 2, size=n),
})

before = frame.memory_usage(deep=True).sum()
optimized = optimize_dtypes(frame)
after = optimized.memory_usage(deep=True).sum()

print(f"before: {before / 1024**2:.2f} MiB")
print(f"after:  {after / 1024**2:.2f} MiB")
print(f"ratio:  {after / before:.3f}")
print(optimized.dtypes)
```

문자열 열의 범주형 변환이 절감의 대부분을 차지한다. 형식화에서 본 $c \ll n$ 조건이 성립하기 때문이다.

`downcast`는 값 범위를 보고 최소 정수 타입을 고른다. 다만 이후 값이 커질 수 있으면 위험하다. 오프라인 분석에는 좋고 계속 갱신되는 파이프라인에는 주의가 필요하다.

범주형의 함정도 확인한다.

```python
import pandas as pd


cat = pd.Series(["a", "b", "a"], dtype="category")
print(cat.cat.categories.tolist())

try:
    cat.iloc[0] = "z"
except (TypeError, ValueError) as exc:
    print(f"caught: {type(exc).__name__}")

extended = cat.cat.add_categories(["z"])
extended.iloc[0] = "z"
print(extended.tolist())
```

범주에 없는 값을 넣을 수 없다. 학습 데이터로 범주를 만들고 추론 데이터에 새 값이 나타나는 상황이 실무에서 흔하다. 범주 목록을 명시적으로 관리해야 한다.

### 조인

`merge`의 동작과 함정이다.

```python
import numpy as np
import pandas as pd


left = pd.DataFrame({"key": [1, 2, 2, 3], "value_left": ["a", "b", "c", "d"]})
right = pd.DataFrame({"key": [2, 2, 3, 4], "value_right": ["x", "y", "z", "w"]})

inner = left.merge(right, on="key", how="inner")
print(f"inner rows: {len(inner)}")
print(inner)

outer = left.merge(right, on="key", how="outer", indicator=True)
print(outer["_merge"].value_counts().to_dict())
```

키가 양쪽에서 중복되면 곱집합이 만들어져 행이 늘어난다. 왼쪽 2개와 오른쪽 2개가 4행이 된다. 대규모 데이터에서 이것이 메모리 폭발의 원인이다.

`validate`로 관계를 강제한다.

```python
import pandas as pd


left = pd.DataFrame({"key": [1, 2, 3], "v": ["a", "b", "c"]})
right_dup = pd.DataFrame({"key": [1, 1, 2], "w": ["x", "y", "z"]})

try:
    left.merge(right_dup, on="key", validate="one_to_one")
except Exception as exc:
    print(f"caught: {type(exc).__name__}: {exc}")

ok = left.merge(right_dup, on="key", validate="one_to_many")
print(f"one_to_many ok, rows={len(ok)}")
```

`validate`는 조인 전에 관계를 확인해 의도하지 않은 팽창을 막는다. 모든 merge에 붙일 가치가 있다.

`indicator=True`로 매칭 결과를 확인하는 것도 습관으로 삼는다. 조인 후 행 수만 보면 매칭 실패를 놓친다.

### 대용량 처리

메모리보다 큰 파일을 다룬다.

```python
import os
import tempfile

import numpy as np
import pandas as pd


tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "events.csv")

rng = np.random.default_rng(0)
pd.DataFrame({
    "user": rng.integers(0, 5000, size=200_000),
    "value": rng.normal(size=200_000),
    "country": rng.choice(["KR", "US", "JP"], size=200_000),
}).to_csv(path, index=False)


def streaming_group_mean(csv_path: str, chunksize: int = 50_000):
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=["country", "value"]):
        grouped = chunk.groupby("country")["value"].agg(["sum", "count"])
        for country, row in grouped.iterrows():
            sums[country] = sums.get(country, 0.0) + float(row["sum"])
            counts[country] = counts.get(country, 0) + int(row["count"])
    return {k: sums[k] / counts[k] for k in sums}


streamed = streaming_group_mean(path)
full = pd.read_csv(path).groupby("country")["value"].mean().to_dict()
print({k: round(v, 6) for k, v in sorted(streamed.items())})
print({k: round(v, 6) for k, v in sorted(full.items())})
```

청크 단위로 합과 개수를 누적하면 평균을 정확히 계산할 수 있다. 분산이나 분위수는 더 복잡한 병합이 필요하다.

파일 형식 선택도 성능에 직결된다.

```python
import os
import tempfile
import time

import numpy as np
import pandas as pd


rng = np.random.default_rng(0)
frame = pd.DataFrame({
    "a": rng.normal(size=200_000),
    "b": rng.integers(0, 1000, size=200_000),
    "c": rng.choice(["x", "y", "z"], size=200_000),
})

tmpdir = tempfile.mkdtemp()
csv_path = os.path.join(tmpdir, "d.csv")
parquet_path = os.path.join(tmpdir, "d.parquet")

start = time.perf_counter()
frame.to_csv(csv_path, index=False)
csv_write = time.perf_counter() - start

try:
    start = time.perf_counter()
    frame.to_parquet(parquet_path, index=False)
    parquet_write = time.perf_counter() - start

    start = time.perf_counter()
    pd.read_csv(csv_path)
    csv_read = time.perf_counter() - start

    start = time.perf_counter()
    pd.read_parquet(parquet_path)
    parquet_read = time.perf_counter() - start

    print(f"csv     write={csv_write:.3f}s read={csv_read:.3f}s size={os.path.getsize(csv_path)/1024**2:.2f} MiB")
    print(f"parquet write={parquet_write:.3f}s read={parquet_read:.3f}s size={os.path.getsize(parquet_path)/1024**2:.2f} MiB")
except ImportError:
    print("pyarrow not installed, skipping parquet comparison")
```

Parquet은 열 지향이고 압축되며 dtype을 보존한다. CSV는 모든 값을 문자열로 저장했다가 읽을 때 다시 추론하므로 느리고 크고 dtype이 깨진다. 중간 산출물은 Parquet으로 저장한다.

## 실무 관점

Pandas를 쓸지 다른 도구를 쓸지 판단하는 기준을 정한다. 수백만 행 이하의 탐색적 분석이면 Pandas가 편의성에서 앞선다. 수천만 행 이상이거나 반복 실행되는 파이프라인이면 Polars나 DuckDB가 훨씬 빠르다. 학습 데이터 자체는 Pandas를 거치지 않고 Arrow나 NumPy 메모리맵으로 직접 다루는 것이 최선인 경우가 많다.

메서드 체이닝을 쓰면 중간 변수와 그로 인한 원본 오염 위험이 사라진다.

```python
import numpy as np
import pandas as pd


rng = np.random.default_rng(0)
raw = pd.DataFrame({
    "user": rng.integers(0, 100, size=1000),
    "amount": rng.normal(loc=50, scale=20, size=1000),
    "country": rng.choice(["KR", "US", "JP"], size=1000),
})

result = (
    raw
    .loc[lambda d: d["amount"] > 0]
    .assign(
        log_amount=lambda d: np.log1p(d["amount"]),
        country=lambda d: d["country"].astype("category"),
    )
    .groupby("country", observed=True)
    .agg(mean_log=("log_amount", "mean"), users=("user", "nunique"))
    .reset_index()
    .sort_values("mean_log", ascending=False)
)
print(result)
```

`lambda`를 쓰면 체인 중간 상태를 참조할 수 있다. `raw`를 다시 언급하지 않으므로 어느 단계의 결과인지 헷갈리지 않는다.

인덱스를 남용하지 않는다. 인덱스는 조회 성능과 정렬 연산에서 이득을 주지만, 실수로 정렬이 발생하면 결과가 조용히 바뀐다. 파이프라인 중간에는 `reset_index(drop=True)`로 정수 인덱스를 유지하고, 조회가 반복되는 지점에서만 인덱스를 설정한다.

```python
import numpy as np
import pandas as pd


rng = np.random.default_rng(0)
lookup = pd.DataFrame({
    "user": rng.permutation(50_000),
    "embedding_id": np.arange(50_000),
})

no_index = lookup
with_index = lookup.set_index("user")

import time
targets = rng.choice(50_000, size=2000)

start = time.perf_counter()
for t in targets:
    _ = no_index.loc[no_index["user"] == t, "embedding_id"].iloc[0]
scan_time = time.perf_counter() - start

start = time.perf_counter()
for t in targets:
    _ = with_index.at[t, "embedding_id"]
index_time = time.perf_counter() - start

print(f"boolean scan: {scan_time:.3f}s")
print(f"index lookup: {index_time:.3f}s")
```

인덱스 조회가 두 자릿수 빠르다. 다만 이 정도 반복 조회가 필요하면 애초에 딕셔너리나 NumPy 배열이 더 적합하다.

문자열 처리는 `.str` 접근자를 쓴다. 파이썬 루프보다 빠르지만 여전히 object dtype이면 느리다. `string[pyarrow]` dtype이 크게 개선된다.

```python
import numpy as np
import pandas as pd


rng = np.random.default_rng(0)
texts = pd.Series([f"user_{i}_event_{i % 7}" for i in range(100_000)])

print(texts.str.split("_").str[1].head().tolist())
print(texts.str.contains("event_3").sum())
print(texts.str.len().mean())

try:
    arrow_texts = texts.astype("string[pyarrow]")
    print(f"arrow dtype ok: {arrow_texts.dtype}")
except ImportError:
    print("pyarrow not available")
```

시간대 처리는 명시적으로 한다. 나이브 datetime과 aware datetime을 섞으면 비교가 실패하고, UTC로 통일하지 않으면 서머타임에서 어긋난다.

```python
import pandas as pd


naive = pd.date_range("2024-03-10 01:00", periods=3, freq="h")
aware = naive.tz_localize("UTC")
converted = aware.tz_convert("Asia/Seoul")
print(naive[0], aware[0], converted[0])
```

내부 저장은 UTC로, 표시할 때만 지역 시간으로 변환하는 것이 원칙이다.

`inplace=True`를 쓰지 않는다. 성능 이득이 거의 없고(대부분 내부적으로 복사한다), 체이닝을 깨뜨리며, CoW 도입 후 동작이 달라진다. 반환값을 받아 쓰는 방식이 안전하다.

`copy=False`나 `sort=False` 같은 옵션은 실제 효과를 측정하고 쓴다. 문서에 있는 옵션이 항상 의도대로 동작하지 않는 경우가 있다.

## 핵심 정리

DataFrame은 dtype별 블록으로 열 지향 저장된다. 열 단위 연산은 빠르고 행 단위 접근은 느리다.

Series 이항 연산은 인덱스를 정렬한다. 인덱스가 다르면 합집합에 NaN이 생긴다. 위치 기반 사고를 유지하려면 `to_numpy()`로 내려간다.

`loc` 슬라이스는 끝 포함, `iloc`은 끝 제외다. 정수 인덱스에서는 `[]`를 쓰지 않고 항상 둘 중 하나를 명시한다.

대입은 항상 `loc` 한 번의 호출로 한다. 연쇄 인덱싱은 원본 반영 여부가 불확정이다. Copy-on-Write를 켜면 구조적으로 해결된다.

`iterrows`는 행마다 Series를 만들고 dtype을 승격시킨다. 벡터 연산 대비 두 자릿수 느리며 정수 ID의 정밀도를 잃는다.

`groupby`의 내장 집계는 Cython 경로로 $O(n)$이지만 `apply(lambda)`는 그룹 수만큼 파이썬을 왕복한다. 그룹이 많으면 이 항이 지배한다.

object dtype 문자열 열을 범주형으로 바꾸면 고유값이 적을 때 메모리가 수십 배 줄어든다. 단, 학습 시 범주 목록을 고정해 추론 시 새 값 문제를 관리한다.

`merge`에 `validate`와 `indicator`를 붙인다. 양쪽 중복 키가 곱집합을 만들어 행이 폭발하는 사고를 사전에 막는다.

중간 산출물은 CSV가 아니라 Parquet으로 저장한다. dtype이 보존되고 읽기가 훨씬 빠르다.

시계열 결측에 `bfill`과 `interpolate`를 쓰면 미래 정보가 새어 들어간다.
