# Part 10 · LLM Engineering

## 이 파트의 목표

이 아카이브에서 가장 큰 파트다. LLM 엔지니어링은 단일 주제가 아니라 아키텍처, 학습, 압축, 추론, 응용 다섯 층이 겹쳐 있는 영역이고, 각 층에서 내리는 결정이 다른 층의 예산을 잡아먹는다. GQA를 쓰면 KV 캐시가 줄어 배치 크기를 키울 수 있고, 배치 크기가 커지면 처리량이 오르는 대신 첫 토큰 지연이 늘어난다. 이런 연쇄를 숫자로 계산할 수 있게 만드는 것이 목표다.

Part 08의 Transformer 구현을 전제로 시작한다. 어텐션의 형상 변화가 손에 익지 않았다면 먼저 그쪽을 마친다.

## 다섯 층

```mermaid
flowchart TD
    SCALE["스케일링 법칙<br/>파라미터 · 토큰 · 컴퓨트 배분"] --> ARCH
    subgraph ARCH["아키텍처"]
        A1["MQA · GQA · KV 캐시"]
        A2["MoE 라우팅"]
        A3["SwiGLU · RMSNorm"]
        A4["긴 컨텍스트"]
    end
    ARCH --> TRAIN
    subgraph TRAIN["학습"]
        B1["사전학습 데이터"]
        B2["SFT"]
        B3["RLHF"]
        B4["DPO"]
    end
    TRAIN --> ADAPT
    subgraph ADAPT["적응과 압축"]
        C1["LoRA"]
        C2["QLoRA · Adapter · IA3"]
        C3["양자화 PTQ/QAT"]
    end
    ADAPT --> INFER
    subgraph INFER["추론"]
        D1["FlashAttention"]
        D2["PagedAttention · 연속 배칭"]
        D3["Speculative Decoding"]
        D4["디코딩 전략"]
    end
    INFER --> APP
    subgraph APP["응용"]
        E1["프롬프트 엔지니어링"]
        E2["RAG · 벡터 인덱스"]
        E3["에이전트"]
        E4["평가"]
    end

    classDef l1 fill:#dbeafe,stroke:#3b7dd8,color:#000000
    classDef l2 fill:#fef3c7,stroke:#d97706,color:#000000
    classDef l3 fill:#fce7f3,stroke:#db2777,color:#000000
    classDef l4 fill:#e9d5ff,stroke:#7c3aed,color:#000000
    classDef l5 fill:#d1fae5,stroke:#10b981,color:#000000
    class SCALE,A1,A2,A3,A4 l1
    class B1,B2,B3,B4 l2
    class C1,C2,C3 l3
    class D1,D2,D3,D4 l4
    class E1,E2,E3,E4 l5
```

## 역할별 진입 경로

전부 순서대로 읽는 것이 정석이지만, 당장 필요한 작업이 있다면 아래 경로로 들어간다.

파인튜닝을 맡았다면 07(SFT) → 10(LoRA) → 11(QLoRA) → 12(양자화) → 22(평가) 순으로 읽는다.

추론 서빙을 맡았다면 02(KV 캐시) → 13(FlashAttention) → 14(PagedAttention과 배칭) → 15(Speculative) → 16(디코딩) 순이다. Part 13의 메모리 계산 문서를 함께 본다.

RAG 시스템을 맡았다면 17(프롬프트) → 18(RAG 파이프라인) → 19(벡터 인덱스) → 20(평가) 순이다.

정렬 학습을 맡았다면 07 → 08(RLHF) → 09(DPO) 순으로 읽는다.

## 자주 필요한 계산

| 계산 | 공식 |
| --- | --- |
| KV 캐시 크기 | $2 \times L_{seq} \times n_{layer} \times n_{kv} \times d_h \times \text{bytes}$ |
| 학습 메모리(AdamW, FP32 상태) | 파라미터 $4P$ + 그래디언트 $4P$ + 옵티마이저 $8P$ = $16P$ 바이트 |
| 추론 가중치 메모리 | $P \times \text{bytes per weight}$ |
| 학습 FLOPs 근사 | $6 \times P \times D$ ($D$는 학습 토큰 수) |
| 추론 FLOPs 근사 | 토큰당 $2P$ |
| Chinchilla 최적 비율 | $D \approx 20P$ |

각 공식의 유도와 실제 모델 크기 대입 예시는 해당 문서에서 다룬다.
