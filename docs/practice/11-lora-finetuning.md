# 11. LoRA로 LLM 파인튜닝

**만드는 것**: 오픈소스 LLM을 LoRA/QLoRA로 지시 데이터에 파인튜닝(SFT)한다. 소비자 GPU 한 장으로 수B 파라미터 모델을 조정하는, 현재 가장 보편적인 LLM 커스터마이징 레시피다.

**선행 지식**: [LoRA 수식 유도](/handbook/10-llm-engineering/10-lora), [QLoRA](/handbook/10-llm-engineering/11-peft-variants-and-qlora), [SFT 데이터 설계](/handbook/10-llm-engineering/07-sft)

## 원리 요약

가중치 $W$를 동결하고 갱신량만 저랭크로 학습한다: $W' = W + \frac{\alpha}{r}BA$. 학습 파라미터가 전체의 ~0.1–1%로 줄고, QLoRA는 여기에 베이스 모델 4비트 양자화를 더해 메모리를 다시 1/4로 줄인다. 어댑터는 수십 MB짜리 파일이라 과제별로 여러 개를 갈아끼울 수 있다.

## 1. LoRA를 손으로 먼저 — 30줄 구현

라이브러리를 쓰기 전에, LoRA가 얼마나 단순한지 직접 확인한다.

```python
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """기존 Linear를 감싸 W를 동결하고 BA만 학습한다."""
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                      # W, b 동결
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # B=0 → 초기엔 ΔW=0
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def merge(self):
        """추론용: ΔW를 W에 합쳐 지연 없는 일반 Linear로 되돌린다."""
        self.base.weight.data += self.scale * (self.B @ self.A)
        return self.base
```

B를 0으로 초기화하는 이유: 학습 시작 시점에 모델이 원본과 정확히 같아야, 사전학습 성능에서 출발해 점진적으로 이동한다.

## 2. 실전: TRL + PEFT로 SFT

```python
"""QLoRA SFT — 단일 GPU에서 LLM 지시 튜닝.
의존성: pip install transformers peft trl datasets bitsandbytes accelerate
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # 예시. GPU 여유에 따라 더 큰 모델로 교체


def main():
    # 1) 4비트 양자화 로드 (QLoRA). LoRA만 쓰려면 quantization_config를 빼면 된다.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            # 정규분포 가중치에 최적화된 4비트 형식
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto")
    tok = AutoTokenizer.from_pretrained(BASE)

    # 2) LoRA 설정 — 어텐션과 MLP의 선형층을 타깃으로
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # 3) 데이터 — messages(chat) 형식이면 SFTTrainer가 모델의 챗 템플릿을 자동 적용
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")  # 사람 작성 지시 데이터

    # 4) 학습
    cfg = SFTConfig(
        output_dir="./sft-out",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,        # 유효 배치 16
        learning_rate=2e-4,                   # LoRA는 풀 파인튜닝보다 큰 lr을 쓴다
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=1024,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        gradient_checkpointing=True,          # 활성값 메모리 절약 (재계산과 교환)
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model("./sft-out/final")     # 어댑터만 저장된다 (수십 MB)


if __name__ == "__main__":
    main()
```

## 3. 추론과 병합

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                            device_map="auto")
model = PeftModel.from_pretrained(base, "./sft-out/final")
tok = AutoTokenizer.from_pretrained(BASE)

messages = [{"role": "user", "content": "사내 회의록 요약 규칙을 3가지로 정리해줘"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))

# 배포 시: 어댑터를 병합해 단일 모델로 저장 → vLLM 등에서 바로 서빙
merged = model.merge_and_unload()
merged.save_pretrained("./sft-merged")
tok.save_pretrained("./sft-merged")
```

## 실무 감각

**언제 파인튜닝하고 언제 안 하는가.** 형식·문체·도메인 어휘를 일관되게 따르게 하는 데는 SFT가 강하다. 반면 **새로운 지식 주입**에는 비효율적이다 — 최신 정보·사내 문서 참조는 [RAG](/practice/12-rag-system)가 먼저다. 프롬프트 엔지니어링 → RAG → 파인튜닝 순으로 시도하는 것이 비용 순서다.

**데이터 품질 > 수량.** 잘 만든 수천 건이 긁어모은 수십만 건을 이긴다. 응답 형식이 들쭉날쭉한 데이터로 SFT하면 모델도 들쭉날쭉해진다. ([SFT 데이터 설계](/handbook/10-llm-engineering/07-sft))

**손실은 응답에만.** 챗 SFT에서는 프롬프트 토큰을 손실에서 제외(마스킹)하는 것이 표준이다 — TRL이 챗 템플릿 기반으로 처리해 준다. 직접 루프를 짤 때 놓치기 쉬운 지점이다.

**하이퍼파라미터 출발점.** r=16, alpha=2r, lr=1e-4~3e-4, 1~3 에포크. 과적합 신호(응답이 데이터 말투를 앵무새처럼 복제)가 보이면 에포크·r을 줄인다.

## 확장 과제

1. **수동 LoRA 검증** — 1절의 `LoRALinear`를 [09의 GPT](/practice/09-gpt-from-scratch)의 `qkv`에 끼워 파인튜닝하고, 전체 파인튜닝과 손실 곡선을 비교하라.
2. **r 스윕** — r=4/16/64로 학습해 성능과 어댑터 크기를 비교하라. 어디서 포화되는가?
3. **평가 자동화** — 파인튜닝 전후 모델에 같은 프롬프트 50개를 넣고, 더 강한 LLM에게 어느 쪽 응답이 나은지 채점시켜(LLM-as-judge) 승률을 계산하라. ([LLM 평가](/handbook/10-llm-engineering/22-llm-evaluation))
4. **DPO 이어가기** — SFT 모델에 선호 쌍 데이터(`trl`의 `DPOTrainer`)로 DPO를 이어 붙여라. ([DPO 유도](/handbook/10-llm-engineering/09-dpo))

## 다음

모델을 바꾸지 않고 지식을 주입하는 다른 축 → [12. RAG 시스템 구축](/practice/12-rag-system)
