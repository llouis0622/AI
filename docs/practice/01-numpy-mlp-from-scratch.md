# 01. NumPy로 신경망 밑바닥 구현

**만드는 것**: 프레임워크 없이 NumPy만으로 2층 MLP를 구현해 MNIST 손글씨를 97%+ 정확도로 분류한다.

**왜 이걸 먼저 하는가**: `loss.backward()` 한 줄 뒤에서 일어나는 일을 한 번이라도 직접 구현해 본 사람과 아닌 사람은, 학습이 안 될 때 디버깅하는 깊이가 다르다. 이 코드랩이 끝나면 역전파는 더 이상 마법이 아니다.

**선행 지식**: [역전파 유도](/handbook/02-mathematics/10-backpropagation-derivation), [신경망의 원리](/curriculum/ch04/lecture08)

## 전체 구조

순전파는 다음 계산을 한다.

$$
z_1 = XW_1 + b_1,\quad h = \mathrm{ReLU}(z_1),\quad z_2 = hW_2 + b_2,\quad \hat{y} = \mathrm{softmax}(z_2)
$$

손실은 교차엔트로피, 역전파는 이 순서를 거꾸로 밟는다. 소프트맥스+교차엔트로피의 그래디언트가 $\hat{y} - y$로 깔끔하게 떨어지는 것이 핵심 유도다([유도 과정](/handbook/04-deep-learning/03-loss-functions)).

## 전체 코드

`mlp_numpy.py` 하나로 완결된다. MNIST는 `sklearn`의 fetch_openml 또는 아래처럼 직접 내려받는다.

```python
"""NumPy만으로 구현하는 2층 MLP — MNIST 분류.

실행: python mlp_numpy.py
의존성: numpy (데이터 로딩에 torchvision 사용, 학습에는 불사용)
"""
import numpy as np


# ---------- 데이터 ----------
def load_mnist():
    # torchvision은 다운로드 용도로만 쓴다. 학습은 순수 NumPy.
    from torchvision import datasets

    tr = datasets.MNIST("./data", train=True, download=True)
    te = datasets.MNIST("./data", train=False, download=True)
    x_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    x_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_tr = tr.targets.numpy()
    y_te = te.targets.numpy()
    return x_tr, y_tr, x_te, y_te


def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


# ---------- 모델 ----------
class TwoLayerMLP:
    def __init__(self, d_in=784, d_hidden=256, d_out=10, seed=42):
        rng = np.random.default_rng(seed)
        # He 초기화: ReLU가 절반을 죽이므로 분산을 2/fan_in으로.
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, d_hidden)).astype(np.float32)
        self.b1 = np.zeros(d_hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / d_hidden), (d_hidden, d_out)).astype(np.float32)
        self.b2 = np.zeros(d_out, dtype=np.float32)

    def forward(self, X):
        # 역전파에 필요한 중간값을 캐시에 저장한다 — autograd가 하는 일이 바로 이것.
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h = np.maximum(0, self.z1)          # ReLU
        self.z2 = self.h @ self.W2 + self.b2
        # 수치 안정 소프트맥스: 최댓값을 빼도 결과는 동일하다.
        z = self.z2 - self.z2.max(axis=1, keepdims=True)
        e = np.exp(z)
        self.probs = e / e.sum(axis=1, keepdims=True)
        return self.probs

    def loss(self, Y):
        # 교차엔트로피. 1e-12는 log(0) 방지.
        n = Y.shape[0]
        return -np.sum(Y * np.log(self.probs + 1e-12)) / n

    def backward(self, Y):
        n = Y.shape[0]
        # softmax + CE의 그래디언트: (probs - Y) / n  ← 이 한 줄이 유도의 결정체
        dz2 = (self.probs - Y) / n                        # (n, 10)
        dW2 = self.h.T @ dz2                              # (256, 10)
        db2 = dz2.sum(axis=0)
        dh = dz2 @ self.W2.T                              # (n, 256)
        dz1 = dh * (self.z1 > 0)                          # ReLU의 미분: 통과 or 0
        dW1 = self.X.T @ dz1                              # (784, 256)
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads, lr):
        for name, g in grads.items():
            setattr(self, name, getattr(self, name) - lr * g)


# ---------- 학습 ----------
def train():
    x_tr, y_tr, x_te, y_te = load_mnist()
    Y_tr = one_hot(y_tr)
    model = TwoLayerMLP()
    rng = np.random.default_rng(0)

    batch_size, lr, epochs = 128, 0.1, 10
    n = x_tr.shape[0]

    for epoch in range(epochs):
        idx = rng.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            model.forward(x_tr[b])
            total_loss += model.loss(Y_tr[b]) * len(b)
            grads = model.backward(Y_tr[b])
            model.step(grads, lr)

        pred = model.forward(x_te).argmax(axis=1)
        acc = (pred == y_te).mean()
        print(f"epoch {epoch+1:2d}  loss {total_loss/n:.4f}  test acc {acc:.4f}")


if __name__ == "__main__":
    train()
```

10 에포크에 테스트 정확도 ~97.5%가 나온다. 코드 전체에서 `torch`는 데이터 다운로드에만 쓰였다 — 학습의 모든 수학은 위에 다 있다.

## 반드시 이해할 세 지점

**1. 캐시 = autograd의 계산 그래프.** `forward`가 저장한 `X, z1, h, probs`가 곧 PyTorch가 그래프에 저장하는 중간 텐서다. `torch.no_grad()`가 메모리를 아끼는 이유는 이 저장을 생략하기 때문이다.

**2. `(probs - Y) / n` 한 줄.** 소프트맥스와 교차엔트로피를 각각 미분해 곱하면 복잡하지만, 합성하면 이렇게 단순해진다. 프레임워크들이 `CrossEntropyLoss`에 소프트맥스를 내장하는(로짓을 받는) 이유다 — 수치 안정성과 효율을 함께 얻는다.

**3. 그래디언트 검증.** 역전파 구현이 맞는지 확인하는 표준 기법은 수치 미분과의 비교다.

```python
def grad_check(model, X, Y, eps=1e-5):
    model.forward(X)
    grads = model.backward(Y)
    # W2의 임의 원소 하나로 검증
    i, j = 3, 7
    analytic = grads["W2"][i, j]
    model.W2[i, j] += eps
    model.forward(X); lp = model.loss(Y)
    model.W2[i, j] -= 2 * eps
    model.forward(X); lm = model.loss(Y)
    model.W2[i, j] += eps
    numeric = (lp - lm) / (2 * eps)
    print(f"analytic {analytic:.8f}  numeric {numeric:.8f}")  # 소수 5~6자리 일치해야 정상

grad_check(TwoLayerMLP(), *[np.random.rand(8, 784).astype(np.float32), one_hot(np.arange(8) % 10)])
```

## 확장 과제

1. **층 추가** — 3층으로 확장하라. `backward`에서 무엇이 반복 패턴이 되는지 관찰하면, 임의 깊이 네트워크의 역전파가 왜 "층별 국소 미분의 곱"인지 보인다.
2. **모멘텀 구현** — `step`에 속도 항 $v \leftarrow \beta v - \eta g$를 추가하고 수렴 속도를 비교하라.
3. **일부러 망가뜨리기** — He 초기화를 `rng.normal(0, 1, ...)`로 바꾸면 학습이 어떻게 되는가? 왜 그런가? ([초기화 이론](/handbook/04-deep-learning/05-weight-initialization))
4. **미니배치 크기 실험** — batch_size를 1, 128, 60000으로 바꿔 손실 곡선의 노이즈와 수렴을 비교하라.

## 다음

같은 문제를 PyTorch로 다시 풀며 프레임워크가 무엇을 대신해 주는지 확인한다 → [02. PyTorch 학습 파이프라인](/practice/02-pytorch-training-pipeline)
