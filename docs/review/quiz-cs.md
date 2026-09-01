# 퀴즈 07. CS 기초

범위: [Lecture 24~28](/curriculum/ch12/lecture24) — 자료구조, 컴퓨터 구조, 운영체제, 네트워크, 데이터베이스.

<QuizProgress prefix="cs" :total="10" />

<Quiz id="cs-01" q="해시 테이블의 평균/최악 탐색 복잡도와, 최악이 발생하는 조건을 말하라.">

**답:** 평균 $O(1)$, 최악 $O(n)$. 최악은 해시 충돌이 한 버킷에 몰릴 때(나쁜 해시 함수, 적대적 입력) 발생한다. 체이닝이나 오픈 어드레싱으로 충돌을 처리하고, 적재율이 임계치를 넘으면 리사이징한다. → [Lecture 24](/curriculum/ch12/lecture24)

</Quiz>

<Quiz id="cs-02" q="배열과 연결 리스트의 접근/삽입 복잡도를 비교하고, 캐시 관점에서 배열이 유리한 이유를 말하라.">

**답:** 배열: 인덱스 접근 $O(1)$, 중간 삽입 $O(n)$. 연결 리스트: 접근 $O(n)$, (위치를 알 때) 삽입 $O(1)$. 배열은 메모리가 연속적이라 공간 지역성이 좋아 캐시 라인 적중률이 높다 — 실제 성능에서는 빅오가 같아도 배열 계열(vector)이 훨씬 빠른 경우가 많다. NumPy 텐서가 연속 메모리인 것도 같은 이유다. → [Lecture 24](/curriculum/ch12/lecture24)

</Quiz>

<Quiz id="cs-03" q="CPU 캐시 계층 구조가 존재하는 이유와, 이것이 GPU 딥러닝 성능과 어떻게 연결되는지 말하라.">

**답:** 연산 속도와 메모리 속도의 격차(메모리 벽)를 메우기 위해 작고 빠른 저장소(레지스터→L1→L2→L3→DRAM)를 계층화한다. GPU도 같다 — HBM 접근이 연산보다 훨씬 느려서, FlashAttention처럼 온칩 SRAM에서 타일 단위로 계산해 HBM 왕복을 줄이는 것이 핵심 최적화가 된다. → [Lecture 25](/curriculum/ch12/lecture25), [핸드북 Part 13](/handbook/13-systems-hardware/)

</Quiz>

<Quiz id="cs-04" q="프로세스와 스레드의 차이, 그리고 Python GIL이 멀티스레딩에 주는 제약을 설명하라.">

**답:** 프로세스는 독립된 주소 공간을 갖는 실행 단위, 스레드는 프로세스 안에서 메모리를 공유하는 실행 흐름이다. CPython의 GIL은 한 시점에 한 스레드만 바이트코드를 실행하게 하므로 CPU 바운드 작업은 멀티스레딩으로 빨라지지 않는다 — 그래서 DataLoader는 멀티프로세싱을 쓰고, I/O 바운드에는 스레드/asyncio가 유효하다. → [Lecture 26](/curriculum/ch12/lecture26), [핸드북 Part 01 · 07편](/handbook/01-python-foundations/07-concurrency)

</Quiz>

<Quiz id="cs-05" q="가상 메모리와 페이징의 개념을 말하고, GPU OOM과 CPU OOM의 대응 차이를 설명하라.">

**답:** 가상 메모리는 각 프로세스에 연속된 가상 주소 공간을 주고 페이지 단위로 물리 메모리에 매핑하는 기법으로, 부족하면 디스크로 스왑한다. CPU는 스왑으로 버티지만(느려질 뿐), GPU 메모리는 일반적으로 스왑이 없어 초과 즉시 OOM 에러가 난다 — 배치 축소, 그래디언트 누적, 체크포인팅, 혼합 정밀도로 대응한다. → [Lecture 26](/curriculum/ch12/lecture26)

</Quiz>

<Quiz id="cs-06" q="TCP와 UDP의 차이와, 각각이 쓰이는 ML 시스템 사례를 들어라.">

**답:** TCP는 연결 지향으로 순서·신뢰성을 보장하고(재전송, 흐름 제어), UDP는 비연결로 빠르지만 손실을 허용한다. 모델 API 서빙(HTTP/gRPC)은 TCP 위에서 돌고, 실시간 스트리밍·일부 분산 통신처럼 지연이 더 중요한 곳에 UDP 계열이 쓰인다. NCCL 같은 GPU 집합 통신은 별도로 NVLink/IB 위에서 최적화된다. → [Lecture 27](/curriculum/ch12/lecture27)

</Quiz>

<Quiz id="cs-07" q="HTTP 요청-응답 흐름에서 REST API와 gRPC의 차이를 말하라.">

**답:** REST는 HTTP + JSON 텍스트 기반으로 범용적이고 디버깅이 쉽다. gRPC는 HTTP/2 + Protocol Buffers 이진 직렬화로 페이로드가 작고 빠르며, 스트리밍과 강타입 계약을 지원한다 — 내부 마이크로서비스 간 고성능 모델 서빙에는 gRPC, 외부 공개 API에는 REST가 흔한 선택이다. → [Lecture 27](/curriculum/ch12/lecture27)

</Quiz>

<Quiz id="cs-08" q="데이터베이스 인덱스는 왜 조회를 빠르게 하며, 그 대가는 무엇인가?">

**답:** B-tree 등의 정렬된 보조 구조를 유지해 전체 스캔 $O(n)$ 대신 $O(\log n)$ 탐색을 가능하게 한다. 대가는 쓰기 시 인덱스 갱신 비용과 저장 공간 — 쓰기가 많은 테이블에 인덱스를 남발하면 삽입·갱신이 느려진다. 벡터 검색의 HNSW/IVF 인덱스도 같은 트레이드오프(정확도·메모리 vs 속도) 구조다. → [Lecture 28](/curriculum/ch12/lecture28), [핸드북 Part 10 · 19편](/handbook/10-llm-engineering/19-vector-indexes)

</Quiz>

<Quiz id="cs-09" q="트랜잭션의 ACID 네 가지 성질을 나열하고 한 줄씩 설명하라.">

**답:** 원자성(Atomicity) — 전부 성공하거나 전부 취소. 일관성(Consistency) — 트랜잭션 전후에 제약 조건이 유지. 격리성(Isolation) — 동시 트랜잭션이 서로의 중간 상태를 보지 못함. 지속성(Durability) — 커밋된 결과는 장애에도 보존. → [Lecture 28](/curriculum/ch12/lecture28)

</Quiz>

<Quiz id="cs-10" q="시간 복잡도 O(n log n)이 하한이 되는 대표적 문제와, 이를 우회하는 정렬의 예를 들어라.">

**답:** 비교 기반 정렬은 결정 트리 논증에 의해 $\Omega(n \log n)$이 하한이다. 비교를 쓰지 않는 계수 정렬·기수 정렬은 값의 범위가 제한될 때 $O(n + k)$로 우회한다 — "가정을 추가하면 하한을 깰 수 있다"는 알고리즘 설계의 전형적 패턴이다. → [Lecture 24](/curriculum/ch12/lecture24)

</Quiz>
