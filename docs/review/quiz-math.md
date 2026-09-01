# 퀴즈 06. AI 수학

범위: [Lecture 21~23](/curriculum/ch11/lecture21), [핸드북 Part 02 · Mathematics](/handbook/02-mathematics/).

<QuizProgress prefix="mt" :total="10" />

<Quiz id="mt-01" q="행렬의 랭크(rank)는 무엇을 의미하며, 랭크가 낮다는 것은 데이터 관점에서 어떤 뜻인가?">

**답:** 랭크는 행렬의 선형독립인 열(또는 행)의 최대 개수 — 그 선형변환이 보존하는 차원이다. 데이터 행렬의 랭크가 낮다는 것은 특징들이 소수의 방향(요인)의 선형결합으로 설명된다는 뜻으로, 저랭크 근사·압축·LoRA 같은 기법의 근거가 된다. → [핸드북 Part 02 · 01편](/handbook/02-mathematics/01-vector-spaces-and-rank)

</Quiz>

<Quiz id="mt-02" q="SVD의 형태와 각 행렬의 의미를 쓰고, 저랭크 근사와의 관계를 말하라.">

**답:** $A = U\Sigma V^\top$ — $U, V$는 직교(회전), $\Sigma$는 특이값의 대각(축별 스케일)이다. 상위 $k$개 특이값만 남긴 $A_k$는 프로베니우스 노름 기준 최적의 랭크-$k$ 근사다(Eckart–Young). PCA, 압축, LoRA의 저랭크 분해가 모두 이 사실 위에 서 있다. → [핸드북 Part 02 · 05편](/handbook/02-mathematics/05-svd-and-low-rank-approximation)

</Quiz>

<Quiz id="mt-03" q="그래디언트는 기하학적으로 무엇을 가리키며, 경사하강법은 왜 그 반대 방향으로 가는가?">

**답:** 그래디언트 $\nabla f$는 함수가 가장 가파르게 증가하는 방향이며 크기는 그 증가율이다. 손실을 줄이려면 가장 가파르게 감소하는 방향, 즉 $-\nabla f$로 이동한다: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$. 학습률 $\eta$가 이동 크기를 조절한다. → [Lecture 23](/curriculum/ch11/lecture23)

</Quiz>

<Quiz id="mt-04" q="연쇄법칙이 역전파의 기반이 되는 이유를 계산 그래프 관점에서 설명하라.">

**답:** 신경망은 단순 연산들의 합성함수이고, 연쇄법칙은 합성함수의 미분을 각 구간 미분의 곱으로 분해한다. 계산 그래프에서 출력→입력 방향으로 각 노드의 국소 미분을 곱하고 분기에서는 더하며 전파하면, 중간 결과를 재사용해 모든 파라미터의 그래디언트를 한 번의 역방향 통과로 얻는다. → [핸드북 Part 02 · 09~10편](/handbook/02-mathematics/09-chain-rule-and-computational-graphs)

</Quiz>

<Quiz id="mt-05" q="MLE와 MAP의 차이를 수식으로 말하고, MAP의 사전분포가 딥러닝의 무엇과 대응되는가?">

**답:** MLE는 $\arg\max_\theta p(D|\theta)$ — 데이터 가능도만 최대화. MAP는 $\arg\max_\theta p(D|\theta)p(\theta)$ — 사전분포를 곱해 최대화한다. 가우시안 사전은 L2 정규화(가중치 감쇠), 라플라스 사전은 L1 정규화와 정확히 대응된다 — 정규화는 사전 지식의 베이즈적 표현이다. → [핸드북 Part 02 · 13편](/handbook/02-mathematics/13-bayes-mle-map)

</Quiz>

<Quiz id="mt-06" q="교차엔트로피와 KL 발산의 관계를 쓰고, 분류 손실로 교차엔트로피를 쓰는 것이 왜 KL 최소화와 같은가?">

**답:** $H(p, q) = H(p) + D_{KL}(p\,\|\,q)$. 진짜 분포 $p$(원핫 레이블)는 고정이라 $H(p)$가 상수이므로, 교차엔트로피 최소화는 모델 분포 $q$를 $p$에 가깝게 하는 KL 발산 최소화와 동치다. 이는 곧 가능도 최대화(MLE)이기도 하다. → [핸드북 Part 02 · 14편](/handbook/02-mathematics/14-entropy-and-divergences)

</Quiz>

<Quiz id="mt-07" q="볼록 함수에서 경사하강법이 특별히 좋은 이유는 무엇인가? 딥러닝 손실은 볼록인가?">

**답:** 볼록 함수는 국소 최소가 곧 전역 최소이므로, 적절한 학습률의 경사하강법이 전역 최적으로 수렴함이 보장된다. 딥러닝 손실은 비볼록이라 이런 보장이 없지만, 고차원에서는 나쁜 국소 최소보다 안장점이 지배적이고 SGD의 노이즈가 탈출을 돕는다는 것이 경험적 이해다. → [핸드북 Part 02 · 16편](/handbook/02-mathematics/16-convexity-and-gradient-descent)

</Quiz>

<Quiz id="mt-08" q="모멘텀과 Adam이 순수 SGD에 무엇을 더하는지 각각 설명하라.">

**답:** 모멘텀은 과거 그래디언트의 지수이동평균으로 진동을 상쇄하고 일관된 방향을 가속한다. Adam은 여기에 그래디언트 제곱의 이동평균으로 좌표별 적응 학습률을 더해, 파라미터마다 갱신 크기를 자동 조절한다(+ 초기 편향 보정). AdamW는 가중치 감쇠를 그래디언트에서 분리해 올바르게 적용한 변형이다. → [핸드북 Part 02 · 17편](/handbook/02-mathematics/17-optimizers-and-schedules)

</Quiz>

<Quiz id="mt-09" q="소프트맥스 함수의 정의와, 온도(temperature)를 높이면 분포가 어떻게 변하는지 말하라.">

**답:** $\text{softmax}(z)_i = e^{z_i/T} / \sum_j e^{z_j/T}$ — 로짓을 확률 분포로 변환한다. 온도 $T$를 높이면 로짓 차이가 상대적으로 줄어 분포가 균등에 가까워지고(다양한 샘플), 낮추면 최대값에 집중되어 결정적이 된다($T \to 0$이면 argmax). LLM 디코딩의 temperature가 바로 이것이다. → [Lecture 22](/curriculum/ch11/lecture22)

</Quiz>

<Quiz id="mt-10" q="기대값과 분산의 정의를 쓰고, 몬테카를로 추정이 무엇인지 한 문장으로 설명하라.">

**답:** $\mathbb{E}[X] = \sum x\,p(x)$(연속이면 적분), $\mathrm{Var}[X] = \mathbb{E}[(X-\mathbb{E}[X])^2]$. 몬테카를로 추정은 계산 불가능한 기대값을 분포에서 뽑은 표본들의 평균으로 근사하는 방법으로, 표본 수가 늘수록 큰 수의 법칙에 의해 참값에 수렴한다 — 미니배치 그래디언트, VAE의 ELBO, 정책 경사가 모두 이 원리다. → [Lecture 22](/curriculum/ch11/lecture22)

</Quiz>
