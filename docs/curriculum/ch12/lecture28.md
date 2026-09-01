# Lecture 28. 데이터베이스

## 개요

**핵심 질문**

- 관계형 데이터베이스는 어떻게 데이터를 구조화하고 보장하는가?
- 트랜잭션과 ACID는 무엇이며 왜 필요한가?
- NoSQL은 관계형 DB와 어떤 철학적 차이를 갖는가?
- 벡터 DB는 왜 필요하며 어떻게 작동하는가?

**학습 목표**

- 관계형 DB의 핵심 개념(테이블·키·정규화·SQL)을 설명할 수 있다.
- 트랜잭션의 ACID 속성과 격리 수준을 이해한다.
- NoSQL 유형별 특징과 적합한 사용 사례를 구분할 수 있다.
- 벡터 DB의 설계 철학과 ANN 검색 원리를 설명할 수 있다.

---

## 핵심 개념

### 1. 데이터베이스 개요

**데이터베이스 (Database)**

구조화된 데이터의 집합. 여러 사용자와 프로그램이 공유하고, 중복을 최소화하며, 효율적으로 접근·관리하기 위해 조직된 데이터 모음.

**DBMS (Database Management System)**

데이터베이스를 생성·관리하는 소프트웨어. 데이터 정의·조작·제어 기능 제공.

**데이터베이스의 주요 분류**

| 유형 | 특징 | 대표 시스템 |
|---|---|---|
| 관계형 (RDBMS) | 테이블·SQL·ACID | PostgreSQL, MySQL, Oracle |
| Key-Value | 단순 매핑, 초고속 | Redis, DynamoDB |
| Document | JSON 유사 문서 | MongoDB, CouchDB |
| Column-Family | 열 단위 저장 | Cassandra, HBase |
| Graph | 노드·간선 구조 | Neo4j, Amazon Neptune |
| 벡터 DB | 고차원 벡터 검색 | Pinecone, Weaviate, Qdrant, pgvector |
| 시계열 | 시간 순서 데이터 | InfluxDB, TimescaleDB |

---

### 2. 관계형 데이터베이스

**핵심 개념**

- **테이블 (Relation)**: 행(Row, Tuple)과 열(Column, Attribute)로 구성된 2차원 구조
- **스키마**: 테이블의 구조 정의 — 열 이름·데이터 타입·제약 조건
- **도메인**: 각 열에서 허용되는 값의 범위

**키의 종류**

| 키 | 설명 |
|---|---|
| 기본 키 (Primary Key) | 각 행을 유일하게 식별하는 열(들). NULL 불가, 중복 불가 |
| 외래 키 (Foreign Key) | 다른 테이블의 기본 키를 참조. 참조 무결성 보장 |
| 후보 키 (Candidate Key) | 기본 키가 될 수 있는 모든 키 |
| 슈퍼 키 (Super Key) | 행을 유일하게 식별할 수 있는 열의 집합 |
| 대리 키 (Surrogate Key) | 자연 속성 없이 인공적으로 생성된 식별자 (AUTO_INCREMENT) |

**정규화 (Normalization)**

중복 제거와 이상(Anomaly) 방지를 위해 테이블을 분해하는 과정.

| 정규형 | 조건 |
|---|---|
| 1NF | 각 열의 값이 원자값(Atomic) — 반복 그룹 없음 |
| 2NF | 1NF + 부분 함수 종속 제거 (복합 기본 키에서 일부에만 종속되는 열 분리) |
| 3NF | 2NF + 이행 함수 종속 제거 (기본 키가 아닌 열이 다른 비키 열에 종속 제거) |
| BCNF | 3NF + 모든 결정자가 후보 키 |

역정규화 (Denormalization): 읽기 성능 향상을 위해 의도적으로 중복을 허용하는 최적화.

---

### 3. SQL

**DDL (Data Definition Language)**

스키마 정의·변경.

```sql
CREATE TABLE users (
    id      BIGINT      PRIMARY KEY AUTO_INCREMENT,
    name    VARCHAR(50) NOT NULL,
    email   VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE users ADD COLUMN age INT;
DROP TABLE users;
```

**DML (Data Manipulation Language)**

데이터 조회·삽입·수정·삭제.

```sql
-- 조회
SELECT u.name, o.amount
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.amount > 10000
ORDER BY o.amount DESC
LIMIT 10;

-- 삽입
INSERT INTO users (name, email) VALUES ('루이', 'louis@example.com');

-- 수정
UPDATE users SET name = '오신의' WHERE id = 1;

-- 삭제
DELETE FROM users WHERE id = 1;
```

**집계 함수와 그룹화**

```sql
SELECT user_id,
       COUNT(*) AS order_count,
       SUM(amount) AS total_amount,
       AVG(amount) AS avg_amount
FROM orders
GROUP BY user_id
HAVING COUNT(*) >= 3
ORDER BY total_amount DESC;
```

**서브쿼리와 윈도우 함수**

```sql
-- 서브쿼리
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE amount > 50000);

-- 윈도우 함수: 각 사용자의 주문 누적 합계
SELECT user_id, amount,
       SUM(amount) OVER (PARTITION BY user_id ORDER BY created_at) AS running_total
FROM orders;
```

**인덱스 (Index)**

특정 열의 빠른 탐색을 위한 자료구조.

```sql
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_user_amount ON orders(user_id, amount);
```

- **B-Tree 인덱스**: 기본 인덱스 구조. 범위 검색·정렬에 유리. 탐색 $O(\log n)$.
- **Hash 인덱스**: 정확 일치(Equality) 검색에 최적. 범위 검색 불가.
- **Full-Text 인덱스**: 텍스트 전문 검색용.

인덱스 설계 원칙:
- 카디널리티(Cardinality)가 높은 열에 생성 (성별×, 이메일○)
- 쿼리의 WHERE·JOIN·ORDER BY 절 기준으로 생성
- 쓰기 성능과 트레이드오프 — 불필요한 인덱스는 오히려 성능 저하

---

### 4. 트랜잭션과 ACID

**트랜잭션 (Transaction)**

하나의 논리적 작업 단위로 묶인 SQL 연산의 집합. 전부 성공하거나, 전부 실패해야 한다.

```sql
BEGIN;
UPDATE accounts SET balance = balance - 10000 WHERE id = 1;
UPDATE accounts SET balance = balance + 10000 WHERE id = 2;
COMMIT;  -- 모두 성공 시
ROLLBACK; -- 하나라도 실패 시
```

**ACID 속성**

| 속성 | 영문 | 의미 |
|---|---|---|
| 원자성 | Atomicity | 트랜잭션 내 모든 연산은 전부 반영되거나 전부 취소 |
| 일관성 | Consistency | 트랜잭션 전후로 DB의 무결성 제약 조건이 유지 |
| 격리성 | Isolation | 동시 실행 트랜잭션이 서로 간섭하지 않음 |
| 지속성 | Durability | 커밋된 트랜잭션 결과는 장애가 발생해도 영구 보존 |

**트랜잭션 격리 수준 (Isolation Level)**

격리 수준이 높을수록 동시성 문제가 줄지만 성능이 낮아진다.

| 격리 수준 | Dirty Read | Non-Repeatable Read | Phantom Read |
|---|---|---|---|
| READ UNCOMMITTED | 발생 | 발생 | 발생 |
| READ COMMITTED | 방지 | 발생 | 발생 |
| REPEATABLE READ | 방지 | 방지 | 발생 |
| SERIALIZABLE | 방지 | 방지 | 방지 |

- **Dirty Read**: 커밋되지 않은 데이터를 다른 트랜잭션이 읽음
- **Non-Repeatable Read**: 같은 조회를 두 번 했을 때 결과가 다름
- **Phantom Read**: 같은 조건으로 조회했을 때 행 수가 달라짐

**동시성 제어**

- **낙관적 잠금 (Optimistic Lock)**: 충돌이 드물다 가정, 버전 번호로 충돌 감지
- **비관적 잠금 (Pessimistic Lock)**: 충돌이 잦다 가정, 접근 시 즉시 잠금 (`SELECT FOR UPDATE`)
- **MVCC (Multi-Version Concurrency Control)**: 각 트랜잭션마다 데이터의 스냅샷을 제공 → 읽기-쓰기 충돌 최소화. PostgreSQL·MySQL(InnoDB)이 사용.

---

### 5. NoSQL 데이터베이스

**NoSQL의 등장 배경**

관계형 DB는 수평 확장(Scale-out)과 유연한 스키마 변경이 어렵다. 웹 서비스의 폭발적 데이터 증가로 특정 사용 사례에 최적화된 NoSQL이 등장.

**CAP 정리 (CAP Theorem)**

분산 시스템에서 세 가지 속성을 동시에 완전히 만족할 수 없다.

$$
\text{일관성 (Consistency)} + \text{가용성 (Availability)} + \text{분할 허용성 (Partition Tolerance)}
$$

- **CP 시스템**: 일관성·분할 허용성 — 네트워크 분할 시 가용성 포기 (HBase, Zookeeper)
- **AP 시스템**: 가용성·분할 허용성 — 네트워크 분할 시 일관성 포기 (Cassandra, DynamoDB)

**BASE 원칙**

ACID의 대안. 대부분의 NoSQL이 따른다.

- **Basically Available**: 항상 응답은 하되 최신 데이터가 아닐 수 있음
- **Soft State**: 시스템 상태가 시간에 따라 변할 수 있음
- **Eventually Consistent**: 일정 시간 후에는 모든 노드가 일관된 상태가 됨

**NoSQL 유형별 특징**

**Key-Value 스토어**

가장 단순한 구조. 키로 값을 $O(1)$에 조회.

```
SET user:1001 '{"name": "루이", "score": 95}'
GET user:1001
```

Redis 활용: 세션 저장, 캐시 레이어, 순위표 (Sorted Set), Pub/Sub 메시징.

**Document DB**

JSON/BSON 형태의 문서를 컬렉션에 저장. 스키마 유연.

```json
{
  "_id": "abc123",
  "name": "루이",
  "tags": ["AI", "ML"],
  "address": { "city": "부산" }
}
```

MongoDB 활용: 컨텐츠 관리, 사용자 프로필, 이벤트 로그.

**Column-Family**

행 키 기준으로 열을 동적으로 묶어 저장. 쓰기 성능 탁월.

Cassandra 활용: 시계열 로그, IoT 센서 데이터, 대규모 분산 쓰기.

**Graph DB**

노드(엔티티)와 간선(관계)으로 데이터 표현. 관계 탐색에 최적화.

```cypher
MATCH (u:User)-[:FOLLOWS]->(f:User)
WHERE u.name = '루이'
RETURN f.name
```

Neo4j 활용: 소셜 네트워크, 추천 시스템, 지식 그래프.

---

### 6. 인덱싱 구조

**B-Tree 인덱스**

균형 다진 탐색 트리. 모든 리프 노드의 깊이가 동일하여 최악의 경우에도 $O(\log n)$ 보장.

$$
\text{탐색 복잡도} = O(\log_M n) \quad (M: \text{노드당 최대 자식 수})
$$

B+Tree: 실제 데이터를 모두 리프 노드에 저장, 리프 노드를 연결 리스트로 연결 → 범위 검색 효율적. 대부분의 RDBMS 인덱스가 B+Tree.

**LSM-Tree (Log-Structured Merge Tree)**

쓰기에 최적화된 자료구조. 메모리의 MemTable에 먼저 쓴 뒤 일정 크기가 되면 디스크의 SSTable로 순차 플러시.

- 쓰기: $O(1)$ (메모리 추가)
- 읽기: 여러 레벨 탐색 필요 → Bloom Filter로 최적화

Cassandra, RocksDB(Meta AI, TiKV), LevelDB가 LSM-Tree 기반.

---

### 7. 벡터 데이터베이스

**왜 관계형 DB로 벡터 검색이 안 되는가**

관계형 DB의 B-Tree 인덱스는 정확한 값 또는 범위 비교에 최적화되어 있다. 수백~수천 차원의 부동소수점 벡터 간 유사도 비교에는 구조적으로 부적합하다.

| 비교 | 관계형 DB | 벡터 DB |
|---|---|---|
| 검색 패러다임 | 정확 일치·범위 검색 | 의미적 유사도 검색 |
| 인덱스 구조 | B-Tree, Hash | HNSW, IVF, PQ |
| 쿼리 | SQL `WHERE` | k-NN, ANN |
| 스케일링 | 수직 확장 중심 | 수평 확장 설계 |

**벡터 유사도 측정**

| 방법 | 수식 | 특징 |
|---|---|---|
| 코사인 유사도 | $\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$ | 방향만 비교, 임베딩 검색 표준 |
| 유클리드 거리 | $\sqrt{\sum_i (a_i - b_i)^2}$ | 크기·방향 모두 반영 |
| 내적 (Dot Product) | $\mathbf{a} \cdot \mathbf{b}$ | 벡터 정규화 시 코사인과 동치 |

**ANN 검색 (Approximate Nearest Neighbor)**

정확한 최근접 이웃(Exact k-NN)은 $O(nd)$ ($n$: 벡터 수, $d$: 차원)으로 대규모 데이터에서 너무 느리다. ANN은 정확도를 약간 희생하고 속도를 크게 향상시킨다.

$$
\text{Recall@k} = \frac{|\text{ANN 결과} \cap \text{정확 k-NN 결과}|}{k}
$$

Recall과 검색 속도는 트레이드오프 관계.

**HNSW (Hierarchical Navigable Small World)**

계층적 그래프 구조 기반 ANN 알고리즘.

- 상위 레이어: 성긴 그래프 (빠른 탐색)
- 하위 레이어: 조밀한 그래프 (정밀한 탐색)

```
레이어 2: [A] --- [E]
레이어 1: [A]-[B] [D]-[E]
레이어 0: [A]-[B]-[C]-[D]-[E] (모든 벡터)
```

탐색 과정:
1. 최상위 레이어에서 시작 → 쿼리에 가까운 방향으로 이동
2. 아래 레이어로 이동 → 더 정밀하게 탐색
3. 최하위 레이어에서 Top-K 결과 반환

| 특성 | 값 |
|---|---|
| 탐색 복잡도 | $O(\log n)$ |
| 삽입 복잡도 | $O(\log n)$ |
| 메모리 사용 | 높음 (그래프 구조) |
| Recall | 0.95+ 달성 가능 |

**IVF (Inverted File Index)**

벡터 공간을 Voronoi 셀(클러스터)로 분할하여 쿼리에 인접한 클러스터만 탐색.

1. K-Means로 $k$개 클러스터 중심(센트로이드) 생성
2. 각 벡터를 가장 가까운 센트로이드에 할당 (Inverted List)
3. 쿼리 시: 쿼리에 가까운 상위 `nprobe`개 클러스터만 탐색

$$
\text{탐색 범위} \approx \frac{\text{nprobe}}{k} \times n
$$

`nprobe` 증가 → Recall 향상, 속도 감소.

**PQ (Product Quantization)**

고차원 벡터를 여러 서브벡터로 분할하고 각각을 양자화(코드북)하여 메모리 압축.

$$
\mathbf{x} \in \mathbb{R}^d \to \mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_m], \quad \mathbf{x}_j \in \mathbb{R}^{d/m}
$$

각 서브벡터를 코드북의 인덱스로 대체 → 저장 공간을 수십 배 압축.

**IVF + PQ 조합**: FAISS(Facebook AI Similarity Search)의 핵심 인덱스 구조. 대규모 벡터 DB의 표준 기법.

**RAG 파이프라인에서 벡터 DB의 역할**

```
[문서] → [청킹] → [임베딩 모델] → [벡터 DB 적재]
                                        ↓
[사용자 질문] → [임베딩] → [ANN 검색 Top-K] → [LLM 컨텍스트] → [응답]
```

벡터 DB 선택 기준:

| DB | 특징 | 적합한 경우 |
|---|---|---|
| Pinecone | 완전 관리형 클라우드 서비스 | 빠른 프로토타이핑 |
| Weaviate | 그래프·벡터 하이브리드, 자체 호스팅 가능 | 멀티모달·복잡 필터링 |
| Qdrant | Rust 기반, 고성능 필터링 | 프로덕션 자체 호스팅 |
| ChromaDB | 경량, 로컬 개발 최적화 | 연구·개발 환경 |
| pgvector | PostgreSQL 확장 | 기존 RDBMS에 벡터 기능 추가 |
| FAISS | 라이브러리 형태, 최고 성능 | 대규모 배치 검색, 커스텀 파이프라인 |

**하이브리드 검색 (Hybrid Search)**

벡터 검색(의미 유사도)과 BM25(키워드 매칭)를 결합하여 더 정확한 검색 결과 제공.

$$
\text{score} = \alpha \cdot \text{vector\_score} + (1-\alpha) \cdot \text{keyword\_score}
$$

$\alpha$: 벡터와 키워드 비중 조절 하이퍼파라미터.

---

### 8. 데이터베이스 설계 원칙

**ER 다이어그램 (Entity-Relationship Diagram)**

개체(Entity)·속성(Attribute)·관계(Relationship)를 시각화. 테이블 설계의 출발점.

관계의 종류:
- **1:1**: 각 행이 다른 테이블의 정확히 1개 행과 대응
- **1:N**: 하나의 행이 여러 행과 대응 (사용자 : 주문)
- **M:N**: 여러 행이 여러 행과 대응 → 중간 테이블 필요

**데이터베이스 파티셔닝**

대용량 테이블을 물리적으로 분할하여 성능 향상.

- **수평 파티셔닝 (Sharding)**: 행 단위로 분할하여 여러 DB 서버에 분산
- **수직 파티셔닝**: 열 단위로 분할 (자주 사용하는 열과 그렇지 않은 열 분리)

**AI 개발자 심화 — 데이터 파이프라인과 Feature Store**

- **ETL (Extract-Transform-Load)**: 원천 데이터를 추출·변환·적재하는 데이터 파이프라인
- **Feature Store**: 머신러닝 피처를 중앙 관리하는 시스템 (Feast, Tecton)
  - 오프라인 스토어: 과거 피처 저장 (Parquet, Hive) — 학습용
  - 온라인 스토어: 실시간 피처 서빙 (Redis, DynamoDB) — 추론용
- **데이터 레이크 vs 데이터 웨어하우스**
  - 데이터 레이크: 원시 데이터를 스키마 없이 저장 (S3, HDFS)
  - 데이터 웨어하우스: 정제된 구조화 데이터, 분석 최적화 (Snowflake, BigQuery)
- **벡터 DB와 캐싱**: 동일 쿼리의 임베딩 결과를 Redis에 캐싱 → ANN 검색 중복 방지

---

## 수식 정리

**코사인 유사도**

$$
\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}
$$

**유클리드 거리**

$$
d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^d (a_i - b_i)^2}
$$

**ANN Recall@k**

$$
\text{Recall@k} = \frac{|\text{ANN 결과} \cap \text{정확 k-NN 결과}|}{k}
$$

**하이브리드 검색 스코어**

$$
\text{score} = \alpha \cdot s_{\text{vector}} + (1-\alpha) \cdot s_{\text{keyword}}
$$

**B-Tree 탐색 복잡도**

$$
O(\log_M n), \quad M: \text{노드당 최대 자식 수}
$$

**IVF 탐색 범위 근사**

$$
\text{탐색 벡터 수} \approx \frac{\text{nprobe}}{k} \times n
$$

**PQ 메모리 압축률**

$$
\text{압축률} = \frac{m \times \log_2 k'}{d \times 32} \quad (\text{FP32 대비})
$$

- $m$: 서브벡터 수, $k'$: 코드북 크기, $d$: 원본 차원
