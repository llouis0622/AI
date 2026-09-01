# 퀴즈 08. LLM 엔지니어링 실무

범위: [핸드북 Part 10 · LLM Engineering](/handbook/10-llm-engineering/).

<QuizProgress prefix="le" :total="12" />

<Quiz id="le-01" q="KV 캐시란 무엇이며, 왜 LLM 추론 메모리의 지배 요인이 되는가?">

**답:** 자기회귀 생성에서 이전 토큰들의 Key/Value 텐서를 매 스텝 재계산하지 않도록 저장해 두는 캐시다. 크기가 (레이어 수 × 헤드 × 시퀀스 길이 × 배치)에 비례해, 긴 컨텍스트·큰 배치에서 가중치보다 커질 수 있다. MQA/GQA는 KV 헤드 수를 줄여 이 메모리를 절감하는 구조적 선택이다. → [Part 10 · 02편](/handbook/10-llm-engineering/02-attention-variants-and-kv-cache)

</Quiz>

<Quiz id="le-02" q="LoRA의 수식적 아이디어와 하이퍼파라미터 r, alpha의 의미를 설명하라.">

**답:** 사전학습 가중치 $W$를 동결하고 갱신량을 저랭크 분해 $\Delta W = \frac{\alpha}{r} BA$ ($B \in \mathbb{R}^{d\times r}, A \in \mathbb{R}^{r\times k}$, $r \ll d$)로 학습한다. $r$은 랭크(표현력과 파라미터 수), $\alpha$는 갱신 스케일이다. 학습 파라미터가 수백분의 일로 줄고, 추론 시 $W + \Delta W$로 병합하면 지연이 없다. → [Part 10 · 10편](/handbook/10-llm-engineering/10-lora)

</Quiz>

<Quiz id="le-03" q="QLoRA는 무엇을 조합한 기법이며 어떤 문제를 푸는가?">

**답:** 베이스 모델을 4비트(NF4)로 양자화해 동결하고, 그 위에 LoRA 어댑터만 bf16으로 학습한다(+ 이중 양자화, 페이지드 옵티마이저). 풀 파인튜닝은커녕 LoRA조차 못 올리는 GPU 메모리 제약에서 대형 모델 미세조정을 가능하게 한다 — 예컨대 수십B 모델을 단일 GPU에서 튜닝한다. → [Part 10 · 11편](/handbook/10-llm-engineering/11-peft-variants-and-qlora)

</Quiz>

<Quiz id="le-04" q="PTQ와 QAT의 차이, 그리고 GPTQ/AWQ가 해결하는 문제를 말하라.">

**답:** PTQ는 학습 후 가중치를 저비트로 변환(빠르고 싸지만 정확도 손실 위험), QAT는 양자화를 흉내내며 재학습(정확도 좋지만 비쌈)한다. GPTQ는 층별로 양자화 오차를 보정하는 근사 2차 방법, AWQ는 활성값 크기 기준으로 중요한 채널을 보호하는 방법으로 — 둘 다 재학습 없이 4비트에서 정확도를 유지하는 PTQ 계열이다. → [Part 10 · 12편](/handbook/10-llm-engineering/12-quantization)

</Quiz>

<Quiz id="le-05" q="FlashAttention이 빠른 이유는 무엇을 줄였기 때문인가? (FLOPs가 아니다)">

**답:** HBM 메모리 접근을 줄였다. 표준 어텐션은 $L \times L$ 어텐션 행렬을 HBM에 쓰고 다시 읽지만, FlashAttention은 온라인 소프트맥스로 타일 단위 계산을 SRAM 안에서 끝내 중간 행렬을 구체화하지 않는다. 어텐션은 메모리 바운드 연산이므로 I/O 감소가 곧 속도이며, 메모리도 $O(L^2)$에서 $O(L)$로 준다. → [Part 10 · 13편](/handbook/10-llm-engineering/13-flashattention)

</Quiz>

<Quiz id="le-06" q="연속 배칭(continuous batching)과 PagedAttention이 서빙 처리량을 올리는 원리를 설명하라.">

**답:** 연속 배칭은 배치 전체가 끝나길 기다리지 않고, 시퀀스가 끝나는 즉시 그 자리에 대기 중인 요청을 끼워 넣어 GPU를 놀리지 않는다. PagedAttention은 KV 캐시를 OS 페이징처럼 고정 크기 블록으로 관리해, 길이를 몰라 과할당하던 메모리 단편화를 없애 동시 배치 크기를 키운다 — vLLM의 두 축이다. → [Part 10 · 14편](/handbook/10-llm-engineering/14-inference-serving-and-batching)

</Quiz>

<Quiz id="le-07" q="Speculative decoding의 동작 원리와 출력 분포가 보존되는 이유를 말하라.">

**답:** 작은 드래프트 모델이 토큰 여러 개를 앞서 제안하면, 큰 모델이 한 번의 병렬 순전파로 검증해 맞는 접두사까지 수용하고 틀린 지점부터 다시 뽑는다. 수용/기각 확률을 두 모델의 확률비로 설계(기각 시 보정 분포에서 재샘플링)해, 최종 출력 분포가 큰 모델 단독 샘플링과 수학적으로 동일하게 유지된다. → [Part 10 · 15편](/handbook/10-llm-engineering/15-speculative-decoding)

</Quiz>

<Quiz id="le-08" q="top-k, top-p, temperature 샘플링의 차이를 설명하라.">

**답:** temperature는 로짓을 나눠 분포의 뾰족함을 조절한다. top-k는 확률 상위 $k$개만 남기고 재정규화 — 후보 수가 고정된다. top-p(nucleus)는 누적 확률이 $p$가 될 때까지의 후보만 남긴다 — 분포가 뾰족하면 후보가 적어지고 평평하면 많아져 적응적이다. 실무에서는 temperature + top-p 조합이 흔하다. → [Part 10 · 16편](/handbook/10-llm-engineering/16-decoding-strategies)

</Quiz>

<Quiz id="le-09" q="RAG 파이프라인의 단계를 나열하고, 청킹이 검색 품질에 미치는 영향을 설명하라.">

**답:** 문서 수집 → 청킹 → 임베딩 → 벡터 인덱스 저장 → (질의 시) 질의 임베딩 → 유사도 검색(+재순위화) → 컨텍스트로 증강해 생성. 청크가 너무 크면 임베딩이 여러 주제를 뭉개 검색 정밀도가 떨어지고, 너무 작으면 답에 필요한 맥락이 잘린다 — 의미 단위 분할과 오버랩, 제목 메타데이터 부착이 기본기다. → [Part 10 · 18편](/handbook/10-llm-engineering/18-rag-pipeline)

</Quiz>

<Quiz id="le-10" q="벡터 인덱스에서 Flat, IVF, HNSW의 트레이드오프를 비교하라.">

**답:** Flat은 전수 비교 — 정확하지만 $O(n)$이라 큰 코퍼스에서 느리다. IVF는 클러스터로 나눠 일부만 탐색 — 빠르지만 nprobe에 따라 재현율이 떨어질 수 있다. HNSW는 계층 그래프 탐색 — 높은 재현율과 빠른 질의를 주지만 메모리를 많이 쓰고 구축이 느리다. PQ 압축은 어디에나 결합해 메모리를 줄인다(정확도 손실). → [Part 10 · 19편](/handbook/10-llm-engineering/19-vector-indexes)

</Quiz>

<Quiz id="le-11" q="RoPE의 아이디어와, 긴 컨텍스트 확장에서 위치 인코딩이 왜 문제가 되는지 설명하라.">

**답:** RoPE는 Query/Key를 위치에 비례한 각도로 회전시켜, 내적이 상대 위치에만 의존하게 만드는 위치 인코딩이다. 학습 길이보다 긴 위치는 본 적 없는 회전 각도가 되어 성능이 무너지므로, 위치 인덱스를 압축(Position Interpolation)하거나 주파수를 재조정(NTK, YaRN)하는 보정으로 컨텍스트를 확장한다. → [Part 10 · 05편](/handbook/10-llm-engineering/05-long-context)

</Quiz>

<Quiz id="le-12" q="LLM 평가에서 벤치마크 점수만 믿으면 안 되는 이유를 두 가지 이상 들어라.">

**답:** (1) 오염 — 벤치마크가 사전학습 데이터에 섞여 있으면 암기를 측정하게 된다. (2) 과제 대표성 — MMLU 같은 객관식이 실제 사용 분포(생성, 대화, 도구 사용)를 대표하지 않는다. (3) 프롬프트·채점 방식에 민감해 순위가 뒤집힌다. 실무에서는 자기 과제 기반 평가셋과 LLM/사람 채점을 병행해야 한다. → [Part 10 · 22편](/handbook/10-llm-engineering/22-llm-evaluation)

</Quiz>
