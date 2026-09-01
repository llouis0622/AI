# 18. CLIP과 멀티모달

**만드는 것**: CLIP으로 (1) 학습 없이 임의 클래스를 분류하는 제로샷 분류기, (2) 텍스트로 이미지를 찾는 시맨틱 이미지 검색을 구현하고, 대조 학습 손실을 미니 구현으로 확인한다. 이미지와 텍스트가 **같은 임베딩 공간**에 산다는 것이 멀티모달 AI의 출발점이다.

**선행 지식**: [자기지도 비전 학습](/handbook/07-computer-vision/11-self-supervised-vision), [문장 임베딩](/handbook/08-sequence-nlp/13-t5-and-sentence-embeddings)

## CLIP의 원리 한 문단

이미지 인코더와 텍스트 인코더를 (이미지, 캡션) 쌍 4억 개로 함께 학습한다. 배치 안에서 **짝인 쌍의 유사도는 높이고 나머지 조합은 낮추는** 대조(contrastive) 손실 — 그 결과 "강아지 사진"의 이미지 벡터와 "a photo of a dog"의 텍스트 벡터가 가까워진다. 분류 레이블이라는 고정 어휘에서 해방되어, 텍스트로 표현 가능한 무엇이든 시각 개념과 연결된다.

## 1. 제로샷 분류

```python
"""CLIP 제로샷 분류. 의존성: pip install open_clip_torch pillow torch"""
import torch
import open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

# 내가 정의한 임의의 클래스 — 학습 데이터에 이 조합이 있었을 필요가 없다
classes = ["골든리트리버", "고양이", "자전거", "라면 한 그릇", "야경 사진"]
# 프롬프트 템플릿: 단어만 넣는 것보다 문장이 텍스트 인코더 분포에 가깝다
prompts = [f"a photo of {c}" for c in classes]

image = preprocess(Image.open("test.jpg")).unsqueeze(0).to(device)
text = tokenizer(prompts).to(device)

with torch.no_grad(), torch.autocast(device_type=device):
    img_feat = model.encode_image(image)
    txt_feat = model.encode_text(text)
    # 정규화 후 내적 = 코사인 유사도. 100은 학습된 온도의 근사.
    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)

for c, p in sorted(zip(classes, probs[0].tolist()), key=lambda x: -x[1]):
    print(f"{c:12s} {p:.3f}")
```

## 2. 텍스트로 이미지 검색

이미지 폴더를 한 번 임베딩해 두면, 어떤 문장으로도 검색된다 — [RAG](/practice/12-rag-system)의 이미지판이다.

```python
"""시맨틱 이미지 검색: python search.py "노을 지는 바다" """
import sys
from pathlib import Path
import torch
import open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()


@torch.no_grad()
def build_index(photo_dir="./photos"):
    paths = [p for p in Path(photo_dir).rglob("*")
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    feats = []
    for i in range(0, len(paths), 64):                       # 배치 임베딩
        batch = torch.stack([preprocess(Image.open(p).convert("RGB"))
                             for p in paths[i:i+64]]).to(device)
        f = model.encode_image(batch)
        feats.append(f / f.norm(dim=-1, keepdim=True))
    return paths, torch.cat(feats)


@torch.no_grad()
def search(query, paths, feats, k=5):
    t = model.encode_text(tokenizer([query]).to(device))
    t /= t.norm(dim=-1, keepdim=True)
    sims = (feats @ t.T).squeeze(1)
    for score, idx in zip(*sims.topk(k)):
        print(f"{score.item():.3f}  {paths[idx]}")


if __name__ == "__main__":
    paths, feats = build_index()
    search(sys.argv[1] if len(sys.argv) > 1 else "a dog running on grass", paths, feats)
```

## 3. 대조 손실 미니 구현 — CLIP 학습의 핵심 20줄

CLIP 학습 자체를 재현할 수는 없지만(4억 쌍), 손실 함수는 완전히 이해할 수 있다.

```python
import torch
import torch.nn.functional as F

def clip_loss(img_feat, txt_feat, temperature=0.07):
    """배치 (B, D) 두 개. 대각선이 정답 쌍이다."""
    img = F.normalize(img_feat, dim=-1)
    txt = F.normalize(txt_feat, dim=-1)
    logits = img @ txt.T / temperature          # (B, B) 유사도 행렬
    labels = torch.arange(len(img))             # i번째 이미지의 짝은 i번째 텍스트
    # 두 방향의 교차엔트로피: 이미지→텍스트 검색과 텍스트→이미지 검색을 동시에
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2
```

핵심 통찰: 배치 크기 B에서 정답 1개 대 오답 B−1개의 분류 문제다. **배치가 클수록 오답(negative)이 많아져 학습 신호가 좋아진다** — CLIP이 수만 단위 배치로 학습된 이유이고, 대조 학습 전반의 공통 원리다([SimCLR](/handbook/07-computer-vision/11-self-supervised-vision)과 같은 구조).

## 멀티모달의 다음 단계 — VLM

CLIP은 이해(정렬)까지다. 이미지에 대해 "대화"하는 VLM(GPT-4V, Claude, LLaVA)은 대체로: 비전 인코더(CLIP 계열) → 프로젝션 층 → LLM의 토큰 공간으로 이미지 특징을 주입하는 구조다. LLaVA의 레시피가 대표적이다 — CLIP 인코더는 동결하고, 프로젝션과 LLM을 (이미지, 지시, 응답) 데이터로 파인튜닝한다. [11. LoRA](/practice/11-lora-finetuning)의 기법이 그대로 쓰인다.

## 확장 과제

1. **프롬프트 앙상블** — "a photo of X", "a blurry photo of X" 등 템플릿 7개의 텍스트 임베딩을 평균해 제로샷 정확도가 오르는지 CIFAR-10으로 측정하라(CLIP 논문의 실제 기법).
2. **CIFAR-10 제로샷 벤치마크** — 학습 없이 CLIP 제로샷으로 CIFAR-10 정확도를 재고, [02에서 학습한 ResNet](/practice/02-pytorch-training-pipeline)과 비교하라.
3. **선형 프로브** — CLIP 이미지 특징 위에 로지스틱 회귀만 학습해(=[03의 선형 프로브](/practice/03-transfer-learning)) 제로샷과 비교하라. 몇 장의 레이블이면 제로샷을 넘는가?
4. **미니 CLIP 학습** — MNIST 이미지와 "a photo of the digit three" 텍스트로 위 `clip_loss`를 써서 장난감 CLIP을 학습시켜 보라. 인코더는 [06의 ViT](/practice/06-vit-from-scratch)와 작은 텍스트 Transformer면 된다.

## 다음

이미지 다음은 소리 — 오디오 모달리티 → [19. 음성 인식과 오디오](/practice/19-whisper-audio)
