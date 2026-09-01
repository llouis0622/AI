# 퀴즈 05. 강화학습과 에이전트

범위: [Lecture 17~20](/curriculum/ch08/lecture17), [핸드북 Part 11](/handbook/11-other-domains/) 05~06편, [Part 10 · 21편](/handbook/10-llm-engineering/21-agents).

<QuizProgress prefix="rl" :total="10" />

<Quiz id="rl-01" q="MDP를 구성하는 다섯 요소를 나열하고 각각의 의미를 말하라.">

**답:** 상태 집합 $S$, 행동 집합 $A$, 전이 확률 $P(s'|s,a)$, 보상 함수 $R(s,a)$, 할인율 $\gamma$. 에이전트는 상태를 관찰해 행동을 고르고, 환경은 다음 상태와 보상을 돌려준다. 마르코프 성질 — 다음 상태가 현재 상태와 행동에만 의존 — 이 이론의 기반이다. → [Lecture 17](/curriculum/ch08/lecture17)

</Quiz>

<Quiz id="rl-02" q="가치 함수 V(s)와 행동가치 함수 Q(s,a)의 차이는 무엇인가?">

**답:** $V^\pi(s)$는 상태 $s$에서 정책 $\pi$를 따를 때의 기대 누적 할인 보상이고, $Q^\pi(s,a)$는 상태 $s$에서 먼저 행동 $a$를 취한 뒤 $\pi$를 따를 때의 기대 누적 보상이다. Q가 있으면 각 상태에서 $\arg\max_a Q(s,a)$로 정책을 바로 얻을 수 있어 모델 프리 제어에 유용하다. → [Lecture 17](/curriculum/ch08/lecture17)

</Quiz>

<Quiz id="rl-03" q="탐색(exploration)과 활용(exploitation)의 딜레마를 설명하고 대표적 해법 하나를 들어라.">

**답:** 현재 알기에 가장 좋은 행동만 반복하면(활용) 더 나은 행동을 발견할 기회를 잃고, 새로운 행동만 시도하면(탐색) 보상을 놓친다. 대표 해법은 $\epsilon$-greedy — 확률 $\epsilon$로 무작위 행동, 나머지는 최선 행동을 택하고 $\epsilon$을 점차 줄인다. → [Lecture 17](/curriculum/ch08/lecture17)

</Quiz>

<Quiz id="rl-04" q="Q-learning의 갱신식을 쓰고, 오프폴리시(off-policy)라 불리는 이유를 설명하라.">

**답:** $Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,]$. 행동은 탐색 정책($\epsilon$-greedy)으로 고르지만 갱신 타깃은 $\max$ — 즉 탐욕 정책의 가치 — 를 쓰므로, 행동을 만든 정책과 평가하는 정책이 달라 오프폴리시다. → [핸드북 Part 11 · 05편](/handbook/11-other-domains/05-rl-foundations)

</Quiz>

<Quiz id="rl-05" q="가치 기반 방법과 정책 기반 방법의 차이, 그리고 정책 경사가 필요한 상황을 말하라.">

**답:** 가치 기반(DQN 등)은 Q를 학습하고 그로부터 정책을 유도한다. 정책 기반(REINFORCE, PPO)은 정책 $\pi_\theta(a|s)$를 직접 파라미터화해 기대 보상의 그래디언트로 최적화한다. 연속 행동 공간이거나 확률적 정책이 필요할 때는 $\max_a$가 불가능하거나 부적절하므로 정책 경사가 필요하다. → [Lecture 18](/curriculum/ch08/lecture18)

</Quiz>

<Quiz id="rl-06" q="PPO의 클리핑 목적함수는 어떤 문제를 막기 위한 것인가?">

**답:** 정책 경사에서 한 번의 갱신이 너무 커서 정책이 급변하면 성능이 붕괴하고 회복이 어렵다. PPO는 새 정책과 이전 정책의 확률비를 $[1-\epsilon, 1+\epsilon]$로 클리핑한 대리 목적을 최대화해, 이점(advantage)이 큰 방향으로도 정책이 한 번에 너무 멀리 이동하지 못하게 막는다 — TRPO의 신뢰영역을 단순한 클리핑으로 근사한 것이다. → [핸드북 Part 11 · 06편](/handbook/11-other-domains/06-policy-gradient-and-ppo)

</Quiz>

<Quiz id="rl-07" q="RLHF에서 강화학습이 쓰이는 지점을 MDP 요소로 대응시켜 설명하라.">

**답:** 상태는 프롬프트와 지금까지 생성된 토큰, 행동은 다음 토큰 선택, 정책은 LLM 자체, 보상은 응답 완성 시 보상 모델이 주는 점수(+ 참조 모델과의 KL 페널티로 과도한 이탈 방지)다. PPO로 이 보상을 최대화하도록 LLM을 갱신한다. → [Lecture 15](/curriculum/ch06/lecture15), [핸드북 Part 10 · 08편](/handbook/10-llm-engineering/08-rlhf)

</Quiz>

<Quiz id="rl-08" q="LLM 에이전트가 단순 챗봇과 구별되는 구성 요소 세 가지를 들어라.">

**답:** (1) 도구 사용 — 검색, 코드 실행, API 호출 등 외부 행동으로 세계와 상호작용, (2) 루프 구조 — 관찰→추론→행동을 반복하며 중간 결과에 따라 계획을 수정, (3) 메모리/상태 — 컨텍스트를 넘어 작업 상태를 유지·참조. 목표를 받아 여러 단계에 걸쳐 자율적으로 수행한다는 점이 본질이다. → [Lecture 20](/curriculum/ch10/lecture20), [핸드북 Part 10 · 21편](/handbook/10-llm-engineering/21-agents)

</Quiz>

<Quiz id="rl-09" q="ReAct 패턴은 무엇이며 왜 효과적인가?">

**답:** 추론(Thought)과 행동(Action)을 교대로 생성하게 하는 프롬프트/설계 패턴이다. 행동 전에 명시적 추론을 생성하면 도구 선택이 정확해지고, 도구의 관찰 결과(Observation)를 다음 추론에 반영해 오류를 중간에 교정할 수 있다 — 추론만(CoT) 하면 사실 접근이 안 되고, 행동만 하면 계획이 없는 문제를 함께 해결한다. → [핸드북 Part 10 · 21편](/handbook/10-llm-engineering/21-agents)

</Quiz>

<Quiz id="rl-10" q="에이전트 시스템에서 실행 제어(가드레일)가 필요한 이유와 대표 장치를 들어라.">

**답:** 에이전트는 외부에 실제 영향을 주는 행동(파일 수정, API 호출, 결제 등)을 하므로 잘못된 추론이 실제 피해로 이어진다. 대표 장치: 도구 권한 제한과 허용 목록, 되돌리기 어려운 행동에 대한 사람 승인 게이트, 최대 반복/비용 한도, 행동 로그와 샌드박스 실행. → [Lecture 20](/curriculum/ch10/lecture20)

</Quiz>
