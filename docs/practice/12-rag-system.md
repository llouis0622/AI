# 12. RAG 시스템 구축

**만드는 것**: 내 문서를 근거로 답하는 질의응답 시스템 — 청킹 → 임베딩 → 벡터 인덱스(FAISS) → 검색 → LLM 생성 → 출처 표기까지 전 파이프라인을 한 파일로 구현하고, 검색 품질 평가까지 붙인다.

**왜 RAG인가**: LLM은 학습 시점 이후의 정보와 비공개 문서를 모른다. 파인튜닝으로 지식을 넣는 것은 비싸고 불안정하다. 질문마다 관련 문서를 **검색해 컨텍스트로 제공**하는 것이 RAG이고, 환각을 줄이고 출처를 제시할 수 있는 실무 표준이다.

**선행 지식**: [RAG 파이프라인과 청킹](/handbook/10-llm-engineering/18-rag-pipeline), [벡터 인덱스](/handbook/10-llm-engineering/19-vector-indexes), [RAG 평가](/handbook/10-llm-engineering/20-rag-evaluation)

## 전체 코드

```python
"""미니 RAG — 문서 폴더를 근거로 답하는 QA 시스템.
의존성: pip install sentence-transformers faiss-cpu anthropic
환경변수: ANTHROPIC_API_KEY (콘솔에서 발급)
실행: python rag.py "질문"
"""
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

EMBED_MODEL = "BAAI/bge-m3"          # 다국어(한국어 포함) 임베딩. 가벼운 대안: snowflake-arctic-embed-s
CHUNK_SIZE, CHUNK_OVERLAP = 800, 150  # 문자 기준
TOP_K = 5


# ---------- 1) 청킹: 문단 경계를 존중하며 자른다 ----------
def chunk_text(text: str, source: str) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) > CHUNK_SIZE and buf:
            chunks.append(buf)
            buf = buf[-CHUNK_OVERLAP:] + "\n" + p     # 오버랩: 경계에 걸친 문맥 보존
        else:
            buf = (buf + "\n\n" + p) if buf else p
    if buf:
        chunks.append(buf)
    return [{"text": c, "source": source, "idx": i} for i, c in enumerate(chunks)]


# ---------- 2) 인덱스 구축 ----------
class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chunks: list[dict] = []
        self.index = None

    def build(self, docs_dir: str):
        for path in sorted(Path(docs_dir).rglob("*")):
            if path.suffix in {".md", ".txt"}:
                self.chunks += chunk_text(path.read_text(encoding="utf-8"), str(path))
        print(f"{len(self.chunks)}개 청크 임베딩 중...")
        vecs = self.embedder.encode([c["text"] for c in self.chunks],
                                    normalize_embeddings=True,   # 정규화 → 내적 = 코사인
                                    batch_size=64, show_progress_bar=True)
        self.index = faiss.IndexFlatIP(vecs.shape[1])            # 소규모는 전수(Flat)로 충분
        self.index.add(np.asarray(vecs, dtype=np.float32))

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        qv = self.embedder.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(qv, dtype=np.float32), k)
        return [{**self.chunks[i], "score": float(s)}
                for s, i in zip(scores[0], ids[0]) if i != -1]


# ---------- 3) 생성: 검색 결과를 근거로 답변 ----------
SYSTEM = """당신은 제공된 문서 발췌만을 근거로 답하는 어시스턴트입니다.
- 발췌에 없는 내용은 지어내지 말고 "문서에서 찾을 수 없다"고 답합니다.
- 답변 끝에 사용한 발췌 번호를 [1][3] 형식으로 표기합니다."""


def answer(store: VectorStore, question: str) -> str:
    hits = store.search(question)
    context = "\n\n".join(
        f"[{i+1}] (출처: {h['source']}, 유사도 {h['score']:.3f})\n{h['text']}"
        for i, h in enumerate(hits))

    client = Anthropic()
    resp = client.messages.create(
        model="claude-opus-5",
        max_tokens=2048,
        system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"문서 발췌:\n{context}\n\n질문: {question}"}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


if __name__ == "__main__":
    store = VectorStore()
    store.build("./docs")                     # 아무 md/txt 문서 폴더
    q = sys.argv[1] if len(sys.argv) > 1 else "이 프로젝트의 배포 절차는?"
    print(answer(store, q))
```

## 검색 품질 평가 — RAG 디버깅의 핵심

RAG가 틀리는 원인의 다수는 생성이 아니라 **검색 실패**다. 생성을 보기 전에 검색부터 측정한다. (질문, 정답이 담긴 청크) 쌍 수십 개를 만들어 Recall@k와 MRR을 계산한다.

```python
def evaluate_retrieval(store: VectorStore, eval_set: list[dict], k: int = 5):
    """eval_set: [{"question": ..., "relevant_source": ..., "relevant_idx": ...}, ...]"""
    hit, rr = 0, 0.0
    for ex in eval_set:
        results = store.search(ex["question"], k)
        for rank, r in enumerate(results, start=1):
            if r["source"] == ex["relevant_source"] and r["idx"] == ex["relevant_idx"]:
                hit += 1
                rr += 1.0 / rank
                break
    n = len(eval_set)
    print(f"Recall@{k}: {hit/n:.3f}   MRR: {rr/n:.3f}")
```

Recall@5가 0.7이라면 — 다섯 개나 보여줘도 30%의 질문은 정답 근거 자체가 컨텍스트에 없다는 뜻이다. 아무리 좋은 LLM도 그 30%는 맞출 수 없다. 이때 손댈 곳은 프롬프트가 아니라 청킹 전략, 임베딩 모델, 질의 확장이다.

## 품질을 올리는 다음 단계들 (우선순위 순)

1. **청킹 개선** — 마크다운 제목 계층을 메타데이터로 붙이고, 표·코드 블록은 자르지 않는다. 청크에 문서 제목을 프리픽스로 넣으면 "이 조각이 무엇에 관한 글인지"가 임베딩에 반영된다.
2. **하이브리드 검색** — 벡터 검색은 고유명사·코드 식별자에 약하다. BM25(키워드)와 벡터 점수를 결합(RRF)하면 두 약점이 상쇄된다.
3. **리랭커** — 1차 검색으로 상위 20개를 뽑고, cross-encoder 리랭커(`BAAI/bge-reranker-v2-m3`)로 재정렬해 상위 5개만 쓴다. Recall이 정밀도로 바뀌는 가장 확실한 투자다.
4. **질의 재작성** — 대화 중의 후속 질문("그건 왜 그래?")은 그대로 검색하면 실패한다. LLM으로 독립형 질의로 재작성한 뒤 검색한다.
5. **인덱스 교체** — 청크가 수십만 개를 넘으면 Flat 대신 HNSW로. ([트레이드오프](/handbook/10-llm-engineering/19-vector-indexes))

## 확장 과제

1. **답변 충실도 평가** — 생성 답변이 컨텍스트에 근거하는지(faithfulness)를 LLM 채점으로 자동화하라. "컨텍스트에서 답의 각 주장을 뒷받침하는 문장을 찾을 수 있는가?"를 채점 기준으로.
2. **하이브리드 검색 구현** — `rank_bm25` 패키지로 BM25를 붙이고 RRF(Reciprocal Rank Fusion)로 결합해 Recall@5 변화를 측정하라.
3. **스트리밍 응답** — `client.messages.stream(...)`으로 답변을 토큰 단위 출력하도록 바꿔라(체감 지연이 크게 준다).
4. **PDF 지원** — `pypdf`로 PDF 텍스트 추출을 추가하라. 표가 깨지는 문제를 어떻게 다룰 것인가?

## 다음

검색을 넘어, LLM이 스스로 도구를 골라 쓰게 한다 → [13. LLM 에이전트 밑바닥 구현](/practice/13-agent-from-scratch)
