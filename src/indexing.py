# 회차 색인·중복 방지와 similarity/MMR 벡터 검색

import hashlib
import json
import math
import os
import time

from src.settings import resolve_path


def query_collection(collection, **kwargs):
    from chromadb.errors import InternalError

    for attempt in range(3):
        try:
            return collection.query(**kwargs)
        except InternalError as exc:
            if (
                "Error creating hnsw segment reader: Nothing found on disk" not in str(exc)
                or attempt == 2
            ):
                raise
            time.sleep(0.1 * (2**attempt))


def make_embeddings(settings):
    from langchain_openai import OpenAIEmbeddings

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("색인하려면 OPENAI_API_KEY 환경변수를 설정하세요.")
    return OpenAIEmbeddings(
        model=settings["embedding"]["model"],
        api_key=key,
        base_url=settings["llm"]["base_url"],
        check_embedding_ctx_length=False,
        request_timeout=30,
        max_retries=2,
    )


def open_collection(settings, kind="round"):
    import chromadb

    client = chromadb.PersistentClient(path=str(resolve_path(settings["vector_store"]["path"])))
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model": settings["embedding"]["model"],
                "base_url": settings["llm"]["base_url"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    collection = client.get_or_create_collection(
        settings["vector_store"][f"{kind}_collection"],
        embedding_function=None,
        metadata={"embedding_fingerprint": fingerprint},
    )
    if (collection.metadata or {}).get("embedding_fingerprint") != fingerprint:
        raise ValueError("임베딩 설정이 기존 색인과 다릅니다. 새 컬렉션 이름을 설정하세요.")
    return collection


def index_rounds(draws, collection, embeddings, collected_at, batch_size=64):
    if batch_size < 1:
        raise ValueError("배치 크기는 양수여야 합니다.")
    expected = {f"lotto_{draw.round}" for draw in draws}
    if len(expected) != len(draws):
        raise ValueError("색인 입력에 중복 회차가 있습니다.")
    changed = 0
    for start in range(0, len(draws), batch_size):
        batch = draws[start : start + batch_size]
        ids = [f"lotto_{draw.round}" for draw in batch]
        stored = collection.get(ids=ids, include=["documents", "metadatas"])
        previous = {
            key: (text, meta)
            for key, text, meta in zip(stored["ids"], stored["documents"], stored["metadatas"])
        }
        pending = []
        for draw in batch:
            pair = (draw.search_text(), draw.metadata(collected_at))
            if previous.get(f"lotto_{draw.round}") != pair:
                pending.append(draw)
        if pending:
            documents = [draw.search_text() for draw in pending]
            vectors = embeddings.embed_documents(documents)
            if len(vectors) != len(pending):
                raise ValueError("임베딩 응답 건수가 입력 건수와 다릅니다.")
            collection.upsert(
                ids=[f"lotto_{draw.round}" for draw in pending],
                documents=documents,
                embeddings=vectors,
                metadatas=[draw.metadata(collected_at) for draw in pending],
            )
            changed += len(pending)
    # 모든 새 배치가 성공한 뒤에만 회차 전용 컬렉션의 이전 항목을 정리한다.
    existing = set(collection.get(include=[])["ids"])
    stale = existing - expected
    if stale:
        collection.delete(ids=sorted(stale))
    actual = set(collection.get(include=[])["ids"])
    if actual != expected or collection.count() != len(draws):
        raise ValueError("색인 ID 또는 건수가 원본 회차와 일치하지 않습니다.")
    return {
        "input_count": len(draws),
        "collection_count": collection.count(),
        "changed_count": changed,
        "removed_count": len(stale),
        "ids_verified": True,
    }


def search_rounds(
    question,
    collection,
    embeddings,
    top_k=3,
    round_number=None,
    search_type="similarity",
    fetch_k=20,
    lambda_mult=0.5,
):
    if top_k < 1:
        raise ValueError("Top-k는 양수여야 합니다.")
    if not collection.count():
        return []
    rows, _ = vector_search(
        collection,
        embeddings.embed_query(question),
        top_k,
        where={"round": round_number} if round_number else None,
        search_type=search_type,
        fetch_k=max(fetch_k, top_k),
        lambda_mult=lambda_mult,
    )
    return rows


def cosine(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("임베딩 차원이 일치해야 합니다.")
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0.0


def mmr_indices(query, vectors, k, lambda_mult=0.5):
    if k < 1 or not 0 <= lambda_mult <= 1:
        raise ValueError("Top-k와 lambda_mult 범위를 확인하세요.")
    if not vectors:
        return []
    relevance = [cosine(query, vector) for vector in vectors]
    selected = [max(range(len(vectors)), key=lambda index: (relevance[index], -index))]
    while len(selected) < min(k, len(vectors)):
        remaining = [index for index in range(len(vectors)) if index not in selected]
        selected.append(
            max(
                remaining,
                key=lambda index: (
                    lambda_mult * relevance[index]
                    - (1 - lambda_mult)
                    * max(cosine(vectors[index], vectors[other]) for other in selected),
                    -index,
                ),
            )
        )
    return selected


def diversity(vectors):
    pairs = [1 - cosine(a, b) for i, a in enumerate(vectors) for b in vectors[i + 1 :]]
    return sum(pairs) / len(pairs) if pairs else 0.0


def source_kind(metadata):
    return metadata.get("source_type") or {
        "law_article": "statute",
        "government_guidance": "government_guidance",
        "structured_summary": "structured_summary",
    }.get(metadata.get("document_type"), "round_csv")


def vector_search(
    collection,
    query_vector,
    top_k,
    where=None,
    search_type="similarity",
    fetch_k=20,
    lambda_mult=0.5,
    eligible_ids=None,
):
    if (
        search_type not in {"similarity", "mmr"}
        or top_k < 1
        or fetch_k < top_k
        or not 0 <= lambda_mult <= 1
    ):
        raise ValueError("검색 방식·Top-k·fetch_k·lambda_mult 설정을 확인하세요.")
    if not collection.count() or eligible_ids == []:
        return [], {"search_type": search_type, "candidate_ids": [], "diversity": 0.0}
    # 두 검색 방식이 같은 질문과 후보 풀을 사용하도록 동일한 조건으로 후보를 조회한다.
    result = query_collection(
        collection,
        query_embeddings=[query_vector],
        ids=eligible_ids,
        n_results=min(
            fetch_k, len(eligible_ids) if eligible_ids is not None else collection.count()
        ),
        where=where,
        include=["documents", "metadatas", "embeddings", "distances"],
    )
    vectors = result["embeddings"][0]
    vectors = [list(vector) for vector in vectors]
    similarity = sorted(
        range(len(vectors)), key=lambda index: (-cosine(query_vector, vectors[index]), index)
    )
    chosen = (
        similarity[:top_k]
        if search_type == "similarity"
        else mmr_indices(query_vector, vectors, top_k, lambda_mult)
    )
    rows = [
        {
            "chunk_id": result["ids"][0][i],
            "text": result["documents"][0][i],
            "metadata": result["metadatas"][0][i],
            "rank": rank,
            "query_similarity": cosine(query_vector, vectors[i]),
        }
        for rank, i in enumerate(chosen, 1)
    ]
    audit = {
        "search_type": search_type,
        "fetch_k": fetch_k,
        "lambda_mult": lambda_mult,
        "candidate_ids": result["ids"][0],
        "selected_ids": [row["chunk_id"] for row in rows],
        "diversity": diversity([vectors[i] for i in chosen]),
        "source_type_count": len({source_kind(row["metadata"]) for row in rows}),
    }
    return rows, audit
