# 20. 모델 서빙

**만드는 것**: 학습한 모델을 FastAPI로 API 서버화하고, 동적 배칭으로 처리량을 끌어올리고, Docker로 포장해 어디서든 배포 가능하게 만든다. "모델이 있다"와 "서비스가 있다" 사이의 간극을 메우는 마지막 코드랩이다.

**선행 지식**: [FastAPI 기반 최소 서빙](/book/27-to-production), [서빙 인프라](/book/27-to-production)

## 1. 최소 서빙 — 그러나 올바르게

[02에서 학습한 CIFAR 모델](/practice/02-pytorch-training-pipeline)을 서빙한다. 작아 보여도 실무 서빙의 필수 요소가 다 있다: 시작 시 1회 로드, 전처리 일치, 검증, 헬스체크.

```python
"""모델 서빙 API. 실행: uvicorn serve:app --host 0.0.0.0 --port 8000
의존성: pip install fastapi uvicorn python-multipart pillow torch torchvision
"""
import io
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
device = "cuda" if torch.cuda.is_available() else "cpu"
state = {}

# 학습 때와 '동일한' 전처리 — 다르면 조용히 성능이 깎인다 (학습-서빙 왜곡)
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델은 서버 시작 시 '한 번만' 로드한다. 요청마다 로드하는 것은 최악의 안티패턴.
    from train_cifar import build_model                # 02 코드랩의 모델 정의 재사용
    model = build_model()
    ckpt = torch.load("ckpt_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    state["model"] = model
    yield
    state.clear()


app = FastAPI(title="CIFAR classifier", lifespan=lifespan)


class Prediction(BaseModel):
    label: str
    confidence: float
    top3: dict[str, float]


@app.get("/health")
def health():
    """로드밸런서·k8s가 살아있는지 확인하는 엔드포인트 — 반드시 있어야 한다."""
    return {"status": "ok", "device": device}


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, f"지원하지 않는 형식: {file.content_type}")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "이미지를 해석할 수 없습니다")

    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = state["model"](x).softmax(dim=1)[0]
    top = probs.topk(3)
    return Prediction(
        label=CLASSES[top.indices[0]],
        confidence=round(top.values[0].item(), 4),
        top3={CLASSES[i]: round(v.item(), 4)
              for v, i in zip(top.values, top.indices)},
    )
```

```bash
curl -s -X POST http://localhost:8000/predict -F "file=@cat.jpg" | python -m json.tool
```

## 2. 동적 배칭 — GPU 서빙의 핵심 기법

요청을 1건씩 GPU에 넣으면 GPU가 거의 논다. 짧은 시간(예: 10ms) 동안 도착한 요청을 **모아서 한 배치로** 추론하면 처리량이 몇 배로 뛴다 — Triton·vLLM이 내부에서 하는 일의 원형이다.

```python
"""동적 배칭: asyncio.Queue로 요청을 모아 배치 추론."""
import asyncio
import torch

MAX_BATCH, MAX_WAIT_MS = 32, 10


class DynamicBatcher:
    def __init__(self, model):
        self.model = model
        self.queue: asyncio.Queue = asyncio.Queue()

    async def start(self):
        asyncio.create_task(self._loop())

    async def infer(self, x: torch.Tensor) -> torch.Tensor:
        fut = asyncio.get_event_loop().create_future()
        await self.queue.put((x, fut))         # 요청을 큐에 넣고
        return await fut                        # 배치 처리 결과를 기다린다

    async def _loop(self):
        while True:
            x0, fut0 = await self.queue.get()  # 첫 요청이 올 때까지 대기
            batch, futs = [x0], [fut0]
            deadline = asyncio.get_event_loop().time() + MAX_WAIT_MS / 1000
            while len(batch) < MAX_BATCH:      # 마감까지 추가 요청을 모은다
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                try:
                    x, fut = await asyncio.wait_for(self.queue.get(), timeout)
                    batch.append(x); futs.append(fut)
                except asyncio.TimeoutError:
                    break
            with torch.no_grad():              # 한 번의 순전파로 전부 처리
                out = self.model(torch.cat(batch).to(device)).softmax(dim=1).cpu()
            for i, fut in enumerate(futs):
                fut.set_result(out[i])
```

lifespan에서 `batcher = DynamicBatcher(model); await batcher.start()`로 띄우고, 핸들러에서 `probs = await batcher.infer(x)`로 바꾸면 끝이다. 부하 테스트(`hey`, `locust`)로 배칭 전후 처리량을 비교해 보라 — GPU에서는 극적인 차이가 난다.

## 3. Docker로 포장

```dockerfile
FROM python:3.12-slim

WORKDIR /app
# 의존성 레이어를 코드와 분리 — 코드만 바뀌면 캐시로 빌드가 수 초에 끝난다
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py train_cifar.py ckpt_best.pt ./

EXPOSE 8000
# 워커 수: CPU 추론은 코어 수만큼, GPU 추론은 1(모델 복제 방지) + 배칭이 정석
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

```bash
docker build -t cifar-serve .
docker run --rm -p 8000:8000 --gpus all cifar-serve      # GPU 사용 시
```

## 실무 체크리스트 — 이 다음에 필요한 것들

| 항목 | 최소한의 실천 |
| --- | --- |
| 로깅 | 요청 ID, 지연시간, 예측 분포를 구조화 로그로 — [드리프트 감지](/book/27-to-production)의 원재료 |
| 지연시간 | p50이 아니라 **p95/p99**를 본다. 배칭의 MAX_WAIT는 지연-처리량 트레이드오프 다이얼이다 |
| 버전 관리 | 응답에 모델 버전을 넣고, [레지스트리 승격 절차](/book/27-to-production)로 배포한다 |
| 성능 | 병목이면 ONNX Runtime/TensorRT 변환, [양자화](/book/27-to-production) 순으로 |
| LLM 서빙 | 자기회귀 생성은 이 방식으론 부족 — vLLM 등 [전용 엔진](/book/27-to-production)을 쓴다 |
| 대규모 | 모델별 서버 관리가 힘들어지면 Triton/TorchServe, k8s [오토스케일링](/book/27-to-production)으로 |

## 확장 과제

1. **부하 테스트** — `locust`로 동시 사용자 100명을 흉내 내 배칭 유/무의 p95 지연과 처리량(RPS)을 표로 만들어라.
2. **ONNX 내보내기** — 모델을 ONNX로 변환해 `onnxruntime`으로 서빙하고 CPU 추론 속도를 비교하라. ([익스포트](/practice/02-pytorch-training-pipeline))
3. **LLM 서빙 체험** — `vllm serve Qwen/Qwen2.5-1.5B-Instruct`로 [11에서 만든 병합 모델](/practice/11-lora-finetuning)을 띄우고, OpenAI 호환 API로 호출해 보라. 연속 배칭이 켜진 처리량을 관찰하라.
4. **모니터링 대시보드** — 예측 클래스 분포를 시간별로 집계해, 입력 분포가 변하면 알 수 있는 최소 드리프트 모니터를 만들어라.

## 마치며

여기까지 왔다면 — 역전파를 손으로 짰고, CNN·Transformer·확산 모델·RL 에이전트를 밑바닥부터 만들었고, LLM을 튜닝하고 RAG와 에이전트를 붙였고, 그 결과물을 API로 배포했다. 이제 어떤 AI 논문이나 코드베이스를 만나도 "처음 보는 마법"은 없을 것이다. [책의 여는 글](/book/)로 돌아가 빈 곳을 채우고, [복습 퀴즈](/review/)로 기억을 다져라.
