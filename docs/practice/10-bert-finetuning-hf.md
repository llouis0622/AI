# 10. Hugging Face로 BERT 파인튜닝

**만드는 것**: Hugging Face 생태계(`transformers`, `datasets`, `evaluate`)로 한국어 감성 분류기를 파인튜닝한다. NLP 실무의 표준 워크플로우 — 데이터 로드부터 학습, 평가, 추론 파이프라인, 모델 저장까지.

**선행 지식**: [BERT 계열](/book/18-transformer), [03. 전이학습](/practice/03-transfer-learning)의 개념이 텍스트로 그대로 이어진다.

## 전체 코드

NSMC(네이버 영화 리뷰, 긍/부정 20만 건)를 klue/bert-base로 분류한다.

```python
"""BERT 감성 분류 파인튜닝. 실행: python bert_nsmc.py
의존성: pip install transformers datasets evaluate accelerate
"""
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

MODEL = "klue/bert-base"


def main():
    # 1) 데이터 — datasets는 캐싱·스트리밍·병렬 map을 제공한다
    ds = load_dataset("e9t/nsmc", trust_remote_code=True)     # train 15만 / test 5만
    ds = ds.filter(lambda x: x["document"] is not None and len(x["document"]) > 0)

    # 2) 토크나이즈 — 모델과 짝인 토크나이저를 반드시 사용
    tok = AutoTokenizer.from_pretrained(MODEL)

    def tokenize(batch):
        # 여기서는 패딩하지 않는다 — 배치 시점에 collator가 배치 내 최장 길이로 패딩(동적 패딩)
        return tok(batch["document"], truncation=True, max_length=128)

    ds = ds.map(tokenize, batched=True, remove_columns=["id", "document"])
    ds = ds.rename_column("label", "labels")

    # 3) 모델 — 분류 헤드가 새로 초기화된다는 경고가 뜨는 것이 정상
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, id2label={0: "부정", 1: "긍정"})

    # 4) 평가 지표
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {**metric_acc.compute(predictions=preds, references=labels),
                **metric_f1.compute(predictions=preds, references=labels)}

    # 5) 학습 설정
    args = TrainingArguments(
        output_dir="./bert-nsmc",
        num_train_epochs=2,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=256,
        learning_rate=2e-5,                # BERT 파인튜닝의 표준 범위: 1e-5 ~ 5e-5
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,                         # Ampere+ GPU. 아니면 fp16=True 또는 둘 다 False
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,       # 학습 종료 시 최고 성능 체크포인트 복원
        metric_for_best_model="accuracy",
        logging_steps=100,
        report_to="none",                  # W&B 등 연동 시 "wandb"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=DataCollatorWithPadding(tok),   # 동적 패딩
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())

    # 6) 저장 — 토크나이저를 함께 저장해야 완결된 아티팩트다
    trainer.save_model("./bert-nsmc-final")
    tok.save_pretrained("./bert-nsmc-final")


if __name__ == "__main__":
    main()
```

2 에포크에 정확도 ~90%가 나온다.

## 추론 파이프라인

학습된 모델은 세 가지 방식으로 쓴다. 프로토타입은 pipeline, 서비스는 직접 배치 추론.

```python
from transformers import pipeline
import torch

# (a) 가장 간단: pipeline
clf = pipeline("text-classification", model="./bert-nsmc-final", device=0)
print(clf(["연기가 미쳤다 최고의 영화", "시간이 아깝다 정말"]))
# [{'label': '긍정', 'score': 0.99}, {'label': '부정', 'score': 0.99}]

# (b) 직접 추론: 배치 제어·후처리가 필요할 때
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("./bert-nsmc-final")
model = AutoModelForSequenceClassification.from_pretrained("./bert-nsmc-final").cuda().eval()

texts = ["스토리는 약한데 배우가 다 살렸다"]
with torch.no_grad():
    inputs = tok(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
    probs = model(**inputs).logits.softmax(-1)
print(probs)
```

## 워크플로우의 결정들

**동적 패딩.** 데이터셋 전체를 max_length로 패딩하면 짧은 문장 배치에서 계산 낭비가 크다. `DataCollatorWithPadding`은 배치마다 그 배치의 최장 길이로만 패딩한다 — 학습이 눈에 띄게 빨라진다. 길이가 비슷한 샘플끼리 배치를 묶으면(`group_by_length=True`) 더 빨라진다.

**learning rate 2e-5.** 사전학습 지식을 보존하면서 적응시키는 범위다. 1e-3처럼 크게 잡으면 파국적 망각으로 성능이 무너진다 — [03. 전이학습](/practice/03-transfer-learning)의 "백본은 작게"와 같은 원리다.

**`load_best_model_at_end`.** 마지막 스텝이 최고 성능이 아닐 수 있다. 검증 지표 기준 최고 체크포인트를 자동 복원한다.

**한국어 모델 선택.** klue/bert-base 외에 klue/roberta-base(대체로 더 강함), 문장 임베딩이 목적이면 BGE-m3·KoSimCSE 계열을 본다. 후보는 [Hugging Face Hub](https://huggingface.co/models?language=ko)에서 다운로드 수와 최신성을 함께 본다.

## 확장 과제

1. **Trainer 없이 직접 루프** — 같은 학습을 순수 PyTorch 루프(+`get_linear_schedule_with_warmup`)로 재현하라. Trainer가 해 주던 일(그래디언트 누적, 혼합 정밀도, 체크포인트)이 무엇인지 명확해진다.
2. **토큰 분류로 확장** — `AutoModelForTokenClassification`으로 KLUE-NER(개체명 인식)을 파인튜닝하라. 레이블 정렬(서브워드 ↔ 단어)이 핵심 난관이다.
3. **오분류 분석** — 검증 셋에서 confidence가 높은데 틀린 샘플 30개를 뽑아 보라. 레이블 노이즈인가, 진짜 어려운 문장인가? ([평가의 기본기](/book/08-evaluation))
4. **경량화** — distilbert 계열로 바꿔 정확도 하락 대비 속도 향상을 측정하라.

## 다음

인코더 분류를 넘어, 생성형 LLM을 내 데이터로 조정한다 → [11. LoRA로 LLM 파인튜닝](/practice/11-lora-finetuning)
