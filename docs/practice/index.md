# 실전 코드랩

이론을 아는 것과 코드를 짜는 것 사이의 간극을 메우는 트랙이다. 각 문서는 **처음부터 끝까지 실행 가능한 완전한 코드**를 중심으로, 밑바닥 구현(라이브러리 없이 원리 확인)과 실전 파이프라인(현업 도구 사용)을 모두 다룬다. [커리큘럼](/curriculum/ch01/lecture01)에서 개념을, [핸드북](/handbook/01-python-foundations/)에서 수식을 확인한 뒤 여기서 손을 움직인다.

## 환경 준비

모든 코드랩은 Python 3.10+ 기준이다. 가상환경을 만들고 공통 의존성을 설치한다.

```bash
# uv 사용 (권장 — 빠르다)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv && source .venv/bin/activate
uv pip install torch torchvision numpy matplotlib

# 또는 pip
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy matplotlib
```

GPU가 있다면 [pytorch.org](https://pytorch.org/get-started/locally/)에서 CUDA 버전에 맞는 설치 명령을 확인한다. GPU가 없어도 모든 코드랩은 CPU로 실행된다(작은 데이터셋 기준으로 설계했다).

재현성을 위한 시드 고정은 모든 코드랩에서 이 패턴을 쓴다.

```python
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
```

## 코드랩 목록

### A. 기초 — 훈련 루프를 몸에 익힌다

| 코드랩 | 만드는 것 | 핵심 기술 |
| --- | --- | --- |
| [01. NumPy로 신경망 밑바닥 구현](/practice/01-numpy-mlp-from-scratch) | MNIST 분류 MLP | 순전파·역전파 수동 구현 |
| [02. PyTorch 학습 파이프라인](/practice/02-pytorch-training-pipeline) | CIFAR-10 ResNet | Dataset, AMP, 스케줄러, 체크포인트 |
| [03. 전이학습과 파인튜닝](/practice/03-transfer-learning) | 커스텀 이미지 분류기 | 사전학습 모델, 단계적 해동, 차등 학습률 |

### B. 컴퓨터 비전

| 코드랩 | 만드는 것 | 핵심 기술 |
| --- | --- | --- |
| [04. U-Net 세그멘테이션](/practice/04-unet-segmentation) | 픽셀 단위 분할 모델 | 인코더-디코더, 스킵 연결, Dice 손실 |
| [05. 객체 탐지](/practice/05-object-detection) | 탐지 파이프라인 | IoU/NMS 구현, Faster R-CNN 파인튜닝, YOLO |
| [06. ViT 밑바닥 구현](/practice/06-vit-from-scratch) | Vision Transformer | 패치 임베딩, 어텐션 재사용 |

### C. NLP와 LLM

| 코드랩 | 만드는 것 | 핵심 기술 |
| --- | --- | --- |
| [07. BPE 토크나이저 구현](/practice/07-bpe-tokenizer) | 서브워드 토크나이저 | BPE 학습·인코딩·디코딩 |
| [08. Transformer 밑바닥 구현](/practice/08-transformer-from-scratch) | 완전한 Transformer | 어텐션, 마스킹, 인코더-디코더 |
| [09. GPT 밑바닥 구현과 학습](/practice/09-gpt-from-scratch) | 문자 단위 언어 모델 | 인과 어텐션, 학습, 샘플링, KV 캐시 |
| [10. Hugging Face로 BERT 파인튜닝](/practice/10-bert-finetuning-hf) | 텍스트 분류기 | datasets, Trainer, 평가 |
| [11. LoRA로 LLM 파인튜닝](/practice/11-lora-finetuning) | 지시 따르는 LLM | PEFT, QLoRA, SFT |
| [12. RAG 시스템 구축](/practice/12-rag-system) | 문서 질의응답 시스템 | 청킹, 임베딩, FAISS, 생성, 평가 |
| [13. LLM 에이전트 밑바닥 구현](/practice/13-agent-from-scratch) | 도구 쓰는 에이전트 | 도구 루프, 함수 호출, 안전장치 |

### D. 생성 모델

| 코드랩 | 만드는 것 | 핵심 기술 |
| --- | --- | --- |
| [14. VAE와 GAN 구현](/practice/14-vae-gan) | 이미지 생성 모델 | ELBO, 재파라미터화, 적대적 학습 |
| [15. Diffusion(DDPM) 밑바닥 구현](/practice/15-ddpm-from-scratch) | 확산 생성 모델 | 노이즈 스케줄, 학습, 샘플링 |

### E. 강화학습

| 코드랩 | 만드는 것 | 핵심 기술 |
| --- | --- | --- |
| [16. DQN으로 CartPole 정복](/practice/16-dqn-cartpole) | 가치 기반 RL 에이전트 | 리플레이 버퍼, 타깃 네트워크 |
| [17. PPO 구현](/practice/17-ppo-from-scratch) | 정책 기반 RL 에이전트 | GAE, 클리핑, 액터-크리틱 |

### F. 멀티모달과 운영

| 코드랩 | 만드는 것 | 핵심 기술 |
| --- | --- | --- |
| [18. CLIP과 멀티모달](/practice/18-clip-multimodal) | 제로샷 분류·이미지 검색 | 대조 학습, 임베딩 공간 |
| [19. 음성 인식과 오디오](/practice/19-whisper-audio) | 음성→텍스트 파이프라인 | 멜 스펙트로그램, Whisper |
| [20. 모델 서빙](/practice/20-serving-fastapi) | 모델 API 서버 | FastAPI, 배칭, Docker |

## 학습 방법

1. **베껴 쓰지 말고 따라 쓴다** — 코드를 읽고 이해한 뒤, 보지 않고 다시 짜 본다. 막히는 지점이 이해가 빈 지점이다.
2. **의도적으로 망가뜨린다** — 학습률을 100배 키우면? 정규화를 빼면? 시드를 바꾸면? 예상과 결과를 비교하는 것이 최고의 디버깅 훈련이다.
3. **확장 과제를 푼다** — 각 코드랩 끝의 확장 과제는 "이해했는가"를 검증하는 문제다. 최소 하나는 풀고 넘어간다.
