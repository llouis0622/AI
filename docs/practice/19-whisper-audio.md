# 19. 음성 인식과 오디오

**만드는 것**: 오디오가 신경망 입력이 되는 과정(파형 → 멜 스펙트로그램)을 직접 계산해 보고, Whisper로 음성 인식(STT) 파이프라인을 구축하고, 간단한 오디오 분류기를 학습한다.

**핵심 통찰**: 스펙트로그램으로 바꾸는 순간 오디오는 "이미지"가 된다 — CV에서 배운 CNN/Transformer가 거의 그대로 적용된다. 모달리티가 달라도 도구 상자는 같다.

**선행 지식**: [15장. 듣는다는 것](/book/15-audio)

## 1. 오디오 → 멜 스펙트로그램

소리는 1차원 파형(초당 16,000 샘플)이다. 그대로는 너무 길고 구조가 안 보이므로, 짧은 구간마다 푸리에 변환해 "시간 × 주파수" 2차원 표현으로 바꾼다. 멜 스케일은 사람 귀의 민감도(저주파에 민감)를 반영한 주파수 축이다.

```python
"""파형 → 멜 스펙트로그램. 의존성: pip install torchaudio matplotlib"""
import torch
import torchaudio
import matplotlib.pyplot as plt

wav, sr = torchaudio.load("speech.wav")          # (채널, 샘플 수)
wav = wav.mean(dim=0, keepdim=True)              # 모노로
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=400,          # 25ms 창 — 이 안에서는 소리가 '정상적'이라고 가정
    hop_length=160,     # 10ms 스트라이드 — 초당 100 프레임
    n_mels=80,          # 멜 주파수 빈 수 (Whisper도 80/128을 쓴다)
)(wav)
log_mel = torch.log(mel + 1e-9)                  # 로그: 사람의 소리 크기 지각도 로그다

print(log_mel.shape)                             # (1, 80, 시간 프레임)
plt.imshow(log_mel[0], origin="lower", aspect="auto")
plt.xlabel("time frames"); plt.ylabel("mel bins")
plt.savefig("mel.png", dpi=120)
```

이 `(80, T)` 행렬이 음성 인식·화자 인식·오디오 분류 모두의 표준 입력이다.

## 2. Whisper로 음성 인식

Whisper는 (로그 멜 → Transformer 인코더) + (텍스트 디코더) 구조의 인코더-디코더다 — [08 코드랩](/practice/08-transformer-from-scratch)의 구조에서 소스 임베딩이 오디오 특징일 뿐이다. 68만 시간의 다국어 데이터로 학습되어 한국어도 바로 된다.

```python
"""Whisper STT 파이프라인. 의존성: pip install faster-whisper"""
from faster_whisper import WhisperModel

# faster-whisper: CTranslate2 기반 재구현 — 원본 대비 4배 빠르고 메모리 절반
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
# GPU가 없으면: WhisperModel("small", device="cpu", compute_type="int8")

segments, info = model.transcribe(
    "meeting.mp3",
    language="ko",                # 지정하면 언어 감지 오류 방지
    vad_filter=True,              # 무음 구간 제거 — 환각(무음에서 문장 생성) 억제
    beam_size=5,
)
print(f"감지 언어: {info.language} (p={info.language_probability:.2f})")

for seg in segments:              # 제너레이터 — 순회할 때 실제 디코딩된다
    print(f"[{seg.start:7.2f} → {seg.end:7.2f}] {seg.text}")
```

실무 팁 세 가지:

- **환각 주의** — 무음·음악 구간에서 그럴듯한 문장을 지어내는 것이 Whisper의 대표 실패 모드다. `vad_filter`가 1차 방어선이고, 자막 등 정밀 용도는 세그먼트별 `no_speech_prob`를 확인한다.
- **긴 파일은 30초 창** — Whisper는 내부적으로 30초 단위로 처리한다. faster-whisper가 슬라이딩을 대신해 주지만, 경계에서 문장이 잘리는 경우 타임스탬프 후처리가 필요하다.
- **화자 분리는 별도** — "누가 말했는가"(diarization)는 Whisper 밖의 문제다. `pyannote.audio`로 화자 구간을 얻어 전사와 시간축으로 병합하는 것이 표준 조합이다.

## 3. 오디오 분류기 학습 — 스펙트로그램은 이미지다

명령어 인식(SpeechCommands: "yes", "no", "stop" 등 35 단어)을 2D CNN으로 푼다.

```python
"""키워드 분류 — 멜 스펙트로그램 + CNN. 실행: python audio_clf.py"""
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
mel_tf = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, hop_length=160, n_mels=64)


class SubsetSC(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, subset):
        super().__init__("./data", download=True, subset=subset)

def collate(batch):
    xs, ys = [], []
    for wav, sr, label, *_ in batch:
        if label not in LABELS:
            continue
        wav = torch.nn.functional.pad(wav, (0, 16000 - wav.shape[1]))  # 1초로 패딩
        xs.append(torch.log(mel_tf(wav) + 1e-9))
        ys.append(LABELS.index(label))
    return torch.stack(xs), torch.tensor(ys)


model = nn.Sequential(                                  # 입력 (B, 1, 64, 101)
    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, len(LABELS)),
).to(device)

train_dl = DataLoader(SubsetSC("training"), batch_size=128, shuffle=True,
                      num_workers=4, collate_fn=collate)
test_dl = DataLoader(SubsetSC("testing"), batch_size=256, num_workers=4, collate_fn=collate)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(8):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = sum((model(x.to(device)).argmax(1).cpu() == y).float().mean().item()
                  for x, y in test_dl) / len(test_dl)
    print(f"epoch {epoch+1}  acc {acc:.4f}")
```

8 에포크에 ~95%. [02 코드랩](/practice/02-pytorch-training-pipeline)의 CIFAR CNN과 코드가 거의 같다는 점이 요지다 — 표현만 바꾸면 도구는 이식된다.

## 오디오 생태계 지도

| 과제 | 대표 모델/도구 |
| --- | --- |
| 음성 인식(STT) | Whisper, faster-whisper |
| 음성 합성(TTS) | VITS 계열, Bark, 상용 API |
| 화자 분리 | pyannote.audio |
| 오디오 임베딩/분류 | CLAP(오디오판 CLIP), AST |
| 음악 생성 | MusicGen |

## 확장 과제

1. **증강 실험** — SpecAugment(시간·주파수 마스킹, `torchaudio.transforms.TimeMasking/FrequencyMasking`)를 분류기에 넣어 정확도 변화를 측정하라. 스펙트로그램판 Cutout이다.
2. **회의록 파이프라인** — Whisper 전사 + [13의 에이전트](/practice/13-agent-from-scratch) 요약을 이어붙여, 오디오 파일 → 요약 회의록 자동화 스크립트를 완성하라.
3. **스트리밍 STT** — 마이크 입력을 3초 청크로 잘라 Whisper small로 준실시간 자막을 만들어 보라. 지연과 정확도의 트레이드오프를 어디서 잡을 것인가?

## 다음

만든 모델을 세상에 내보내는 마지막 단계 → [20. 모델 서빙](/practice/20-serving-fastapi)
