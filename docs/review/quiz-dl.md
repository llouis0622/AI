# 퀴즈 03. 딥러닝

범위: [Lecture 08~10](/curriculum/ch04/lecture08), [핸드북 Part 04 · Deep Learning](/handbook/04-deep-learning/), [Part 05 · PyTorch](/handbook/05-pytorch/).

<QuizProgress prefix="dl" :total="12" />

<Quiz id="dl-01" q="신경망에 비선형 활성함수가 없다면 어떤 일이 생기는가?">

**답:** 선형 계층의 합성은 다시 선형이므로, 아무리 층을 쌓아도 전체가 하나의 선형변환과 같아진다. 비선형 활성함수가 있어야 층을 쌓는 것이 표현력 증가로 이어지고, 보편 근사 정리가 말하는 임의 연속함수 근사가 가능해진다. → [핸드북 Part 04 · 01편](/handbook/04-deep-learning/01-mlp-and-nonlinearity)

</Quiz>

<Quiz id="dl-02" q="역전파는 무엇을 계산하는 알고리즘이며, 왜 효율적인가?">

**답:** 손실에 대한 모든 파라미터의 그래디언트를 연쇄법칙으로 계산하는 알고리즘이다. 출력에서 입력 방향으로 중간 결과(각 노드의 그래디언트)를 재사용하며 한 번의 역방향 통과로 모든 파라미터의 미분을 얻으므로, 순전파 비용의 상수배로 끝난다 — 파라미터마다 따로 수치미분하는 것보다 압도적으로 싸다. → [핸드북 Part 04 · 04편](/handbook/04-deep-learning/04-backpropagation-in-matrix-form)

</Quiz>

<Quiz id="dl-03" q="기울기 소실은 왜 일어나며, 이를 완화하는 구조적 장치 두 가지를 들어라.">

**답:** 깊은 층을 거치며 연쇄법칙의 야코비안이 반복 곱해질 때, 크기가 1보다 작은 항이 누적되면 그래디언트가 지수적으로 사라진다(시그모이드 포화가 대표 원인). 완화 장치: 잔차 연결(그래디언트의 항등 경로 확보), 정규화 계층(BatchNorm/LayerNorm으로 활성 분포 안정화), 그리고 ReLU 계열 활성함수와 적절한 초기화. → [핸드북 Part 04 · 08편](/handbook/04-deep-learning/08-residual-connections-and-gradient-flow)

</Quiz>

<Quiz id="dl-04" q="가중치 초기화가 왜 중요한가? Xavier와 He 초기화는 무엇을 맞추려 하는가?">

**답:** 초기 분산이 부적절하면 순전파의 활성값과 역전파의 그래디언트가 층을 거치며 폭발하거나 소실된다. Xavier(tanh용)와 He(ReLU용) 초기화는 각 층 입출력의 분산이 대략 보존되도록 가중치 분산을 팬인/팬아웃에 맞춰 설정한다 — ReLU는 절반을 죽이므로 분산을 2배 키우는 것이 He 초기화다. → [핸드북 Part 04 · 05편](/handbook/04-deep-learning/05-weight-initialization)

</Quiz>

<Quiz id="dl-05" q="BatchNorm과 LayerNorm의 정규화 축 차이와 각각의 주 사용처를 말하라.">

**답:** BatchNorm은 배치 축으로 채널별 평균·분산을 계산한다 — 배치 통계에 의존하므로 큰 배치의 CNN에 적합하고, 추론 시 이동평균을 쓴다. LayerNorm은 샘플 하나 안에서 특징 축으로 정규화한다 — 배치 크기와 무관하고 시퀀스 길이가 변해도 안정적이라 Transformer의 표준이다. → [핸드북 Part 04 · 06편](/handbook/04-deep-learning/06-normalization-layers)

</Quiz>

<Quiz id="dl-06" q="드롭아웃의 동작과 훈련/추론 시 차이를 설명하라.">

**답:** 훈련 시 각 뉴런을 확률 $p$로 무작위로 꺼서 특정 뉴런 간 공적응을 막고 암묵적 앙상블 효과를 낸다. 추론 시에는 모든 뉴런을 켜되 기대값이 맞도록 스케일을 보정한다(inverted dropout은 훈련 시 $1/(1-p)$로 미리 나눔). → [핸드북 Part 04 · 07편](/handbook/04-deep-learning/07-dropout)

</Quiz>

<Quiz id="dl-07" q="CNN이 이미지에 잘 맞는 이유를 귀납 편향(inductive bias) 관점에서 설명하라.">

**답:** 합성곱은 지역 수용영역(가까운 픽셀끼리 상관이 높다)과 가중치 공유(같은 패턴은 위치와 무관하게 유용하다 — 평행이동 등변성)라는 이미지의 통계적 성질을 구조에 새겨 넣었다. 덕분에 완전연결 대비 파라미터가 극적으로 줄고 적은 데이터로도 일반화한다. → [Lecture 10](/curriculum/ch04/lecture10), [핸드북 Part 07 · 01편](/handbook/07-computer-vision/01-convolution-arithmetic)

</Quiz>

<Quiz id="dl-08" q="표현 학습(representation learning)이란 무엇이며, 딥러닝이 전통 ML과 구별되는 지점은?">

**답:** 원시 데이터에서 과제에 유용한 특징(표현)을 손수 설계하는 대신 모델이 스스로 학습하게 하는 것이다. 전통 ML은 사람이 특징 공학을 하고 모델은 그 위에서 학습하지만, 딥러닝은 층을 거치며 저수준→고수준 표현을 점진적으로 구성해 특징 추출과 예측을 끝까지(end-to-end) 함께 최적화한다. → [Lecture 09](/curriculum/ch04/lecture09)

</Quiz>

<Quiz id="dl-09" q="학습 손실이 발산할 때 가장 먼저 점검할 것 세 가지를 순서대로 들어라.">

**답:** (1) 학습률이 너무 큰지 — 가장 흔한 원인, 10배 낮춰 확인. (2) 손실/레이블 정의 오류 — 로짓에 소프트맥스를 중복 적용했는지, 레이블 인덱스 범위, NaN 입력. (3) 그래디언트 폭발 — 그래디언트 노름을 로깅하고 클리핑 적용. 이후 초기화, 정규화 계층, 데이터 파이프라인을 본다. → [핸드북 Part 04 · 10편](/handbook/04-deep-learning/10-training-failure-diagnosis)

</Quiz>

<Quiz id="dl-10" q="PyTorch에서 model.eval()과 torch.no_grad()는 각각 무엇을 하며 왜 둘 다 필요한가?">

**답:** `model.eval()`은 Dropout/BatchNorm 등 훈련·추론 동작이 다른 모듈을 추론 모드로 바꾼다. `torch.no_grad()`는 autograd 그래프 생성을 꺼서 메모리와 연산을 아낀다. 서로 다른 기능이므로 평가 시 둘 다 써야 한다 — eval만 하면 그래프가 쌓이고, no_grad만 하면 드롭아웃이 계속 적용된다. → [핸드북 Part 05](/handbook/05-pytorch/05-training-loop-template)

</Quiz>

<Quiz id="dl-11" q="optimizer.zero_grad()를 빼먹으면 무슨 일이 일어나는가?">

**답:** PyTorch는 `.backward()`가 그래디언트를 덮어쓰지 않고 누적하므로, 매 스텝 초기화하지 않으면 이전 배치들의 그래디언트가 계속 더해져 사실상 잘못된(점점 커지는) 방향으로 갱신된다. 학습이 불안정해지거나 발산하는 단골 버그다. → [핸드북 Part 05 · 12편](/handbook/05-pytorch/12-common-bug-patterns)

</Quiz>

<Quiz id="dl-12" q="DDP(분산 데이터 병렬)의 기본 동작 원리를 한 문단으로 설명하라.">

**답:** 각 GPU(프로세스)가 모델 복제본을 갖고 서로 다른 데이터 샤드로 순전파·역전파를 수행한 뒤, all-reduce로 그래디언트를 평균해 모든 복제본이 동일한 갱신을 적용한다. 통신은 역전파와 겹쳐(overlap) 진행되어 효율적이며, 모델이 한 GPU에 안 들어갈 때는 FSDP/ZeRO처럼 파라미터·옵티마이저 상태 자체를 샤딩하는 방식으로 넘어간다. → [핸드북 Part 05 · 09~10편](/handbook/05-pytorch/09-ddp)

</Quiz>
