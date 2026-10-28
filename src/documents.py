# 조문 단위 청킹, 문서 버전 동기화, 수집 결과 캐시

import hashlib
import json
import threading
import unicodedata
from datetime import datetime, timezone

from src.contracts import PublicError, Selection
from src.data import atomic_write
from src.settings import resolve_path

_LOCK = threading.RLock()


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def chunks(article, size=500, overlap=80):
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("청크 크기 > 중복 길이 >= 0이어야 합니다.")
    # 조문별로 처리해 중복 구간이 다른 조문으로 넘어가지 않도록 한다.
    text, start, index = article.text, 0, 0
    result = []
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + max(overlap + 1, size // 2), end)
            if boundary > start:
                end = boundary + 1
        content = text[start:end]
        metadata = {
            "document_id": article.document_id,
            "chunk_index": index,
            "document_title": article.document_title,
            "document_type": article.document_type,
            "source_type": article.source_type,
            "scope": "global",
            "round_id": "",
            "source_name": article.source_name,
            "source_url": article.source_url,
            "law_title": article.law_title,
            "article_no": article.article_no,
            "effective_date": article.effective_date.isoformat() if article.effective_date else "",
            "verified_on": article.verified_on.isoformat(),
            "collected_at": article.collected_at,
            "content_hash": digest(content),
            "article_hash": digest(text),
            "char_start": start,
            "char_end": end,
            "chunk_size": size,
            "chunk_overlap": overlap,
        }
        result.append(
            {"id": f"{article.document_id}_{index}", "text": content, "metadata": metadata}
        )
        if end == len(text):
            break
        start, index = end - overlap, index + 1
    return result


def index_articles(articles, collection, embeddings, config):
    all_ids = []
    for article in articles:
        batch = chunks(article, config["document_chunk_size"], config["document_chunk_overlap"])
        old = collection.get(where={"document_id": article.document_id}, include=["metadatas"])
        current = {item["id"] for item in batch}
        unchanged = set(old["ids"]) == current and all(
            metadata.get("article_hash") == digest(article.text)
            and metadata.get("chunk_size") == config["document_chunk_size"]
            and metadata.get("chunk_overlap") == config["document_chunk_overlap"]
            and metadata.get("source_url") == article.source_url
            and metadata.get("source_type") == article.source_type
            and metadata.get("verified_on") == article.verified_on.isoformat()
            for metadata in old["metadatas"]
        )
        if not unchanged:
            for offset in range(0, len(batch), 64):
                part = batch[offset : offset + 64]
                vectors = embeddings.embed_documents([item["text"] for item in part])
                if len(vectors) != len(part):
                    raise PublicError("index")
                collection.upsert(
                    ids=[item["id"] for item in part],
                    documents=[item["text"] for item in part],
                    metadatas=[item["metadata"] for item in part],
                    embeddings=vectors,
                )
            stale = set(old["ids"]) - current
            if stale:
                collection.delete(ids=sorted(stale))
        actual = collection.get(where={"document_id": article.document_id}, include=[])["ids"]
        if set(actual) != current:
            raise PublicError("index")
        all_ids.extend(sorted(current))
    return all_ids


class DocumentCollector:
    def __init__(self, settings, client, selector, collection, embeddings):
        self.settings, self.client, self.selector = settings, client, selector
        self.collection, self.embeddings = collection, embeddings
        self.cache_path = resolve_path(settings["law"]["cache_path"])
        self.registry_path = resolve_path(settings["data"]["source_registry"])

    def _read(self, path):
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (ValueError, OSError):
            return {}  # 손상된 캐시는 근거로 사용하지 않고 캐시가 없는 것으로 처리한다.

    def collect(self, question, plan, as_of, progress=lambda _: None):
        ids, audit = [], []
        for query in plan.search_queries:
            normalized = " ".join(unicodedata.normalize("NFKC", query).split()).casefold()
            signature = json.dumps(
                {
                    "query": normalized,
                    "source": "law.go.kr",
                    "as_of": str(as_of),
                    "intent": plan.intent,
                    "version": self.settings["law"]["index_version"],
                    "chunking": self.settings["chunking"],
                    "embedding": self.settings["embedding"],
                    "endpoint": self.settings["llm"]["base_url"],
                },
                sort_keys=True,
            )
            key = digest(signature)
            with _LOCK:
                cache = self._read(self.cache_path)
                entry = cache.get(key, {})
            now = datetime.now(timezone.utc).timestamp()
            cached_ids = entry.get("chunk_ids", [])
            if entry.get("status") == "indexed" and now < entry.get("expires_at", 0) and cached_ids:
                stored = self.collection.get(ids=cached_ids, include=["metadatas"])
                actual = set(stored["ids"])
                if actual == set(cached_ids) and all(
                    meta.get("verified_on") == str(as_of) and meta.get("source_type") == "statute"
                    for meta in stored["metadatas"]
                ):
                    ids.extend(cached_ids)
                    audit.append({"query": normalized, "cache": "hit", "chunk_ids": cached_ids})
                    continue
            if entry.get("status") in {"empty", "failed"} and now < entry.get("retry_after", 0):
                audit.append({"query": normalized, "cache": entry["status"]})
                continue
            progress("법령 후보 검색")
            record = {
                "query_normalized": normalized,
                "source_id": "law.go.kr",
                "chunk_ids": [],
                "last_checked_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                candidates = self.client.search(query, as_of)
                progress("법령 후보 선별")
                selection = (
                    Selection.model_validate(self.selector(question, plan, candidates))
                    if candidates
                    else Selection(decisions=[])
                )
                decisions = {decision.candidate_id: decision for decision in selection.decisions}
                known = {candidate.candidate_id: candidate for candidate in candidates}
                if len(decisions) != len(selection.decisions) or set(decisions) != set(known):
                    raise PublicError("selection")
                articles = []
                allowed = set(self.settings["law"]["allowed_titles"])
                for decision in sorted(selection.decisions, key=lambda item: item.priority):
                    candidate = known[decision.candidate_id]
                    if decision.decision != "index":
                        continue
                    if candidate.law_title not in allowed or candidate.effective_date > as_of:
                        continue
                    progress("법령 본문 수집 및 청킹·색인")
                    wanted = self.settings["law"]["article_numbers"].get(candidate.law_title, [])
                    articles.extend(
                        article
                        for article in self.client.articles(candidate, as_of)
                        if not wanted or article.article_no in wanted
                    )
                record.update(
                    {
                        "candidates": [
                            candidate.model_dump(mode="json") for candidate in candidates
                        ],
                        "selection": selection.model_dump(mode="json"),
                    }
                )
                record["chunk_ids"] = index_articles(
                    articles, self.collection, self.embeddings, self.settings["chunking"]
                )
                with _LOCK:
                    registry = self._read(self.registry_path)
                    for article in articles:
                        registry[article.document_id] = {
                            **article.model_dump(mode="json", exclude={"text"}),
                            "source_name": "국가법령정보센터",
                            "license": "법령 원문 (저작권법 제7조 대상)",
                            "usage_scope": "법령 조문 검색·교육용 색인",
                            "content_hash": digest(article.text),
                        }
                    atomic_write(
                        self.registry_path, json.dumps(registry, ensure_ascii=False, indent=2)
                    )
                record["status"] = "indexed" if record["chunk_ids"] else "empty"
                record["expires_at"] = now + self.settings["law"]["cache_ttl_seconds"]
                record["retry_after"] = now + self.settings["law"]["empty_retry_seconds"]
            except Exception as exc:
                record.update(
                    status="failed",
                    error_code=exc.code if isinstance(exc, PublicError) else "collection_failed",
                    retry_after=now + self.settings["law"]["failed_retry_seconds"],
                )
                # 공개 가능한 실패 상태만 저장하고 관련 없는 기존 문서는 유지한다.
                with _LOCK:
                    cache = self._read(self.cache_path)
                    cache[key] = record
                    atomic_write(self.cache_path, json.dumps(cache, ensure_ascii=False, indent=2))
                if isinstance(exc, PublicError):
                    raise
                raise PublicError("law_response") from None
            with _LOCK:
                cache = self._read(self.cache_path)
                cache[key] = record
                atomic_write(self.cache_path, json.dumps(cache, ensure_ascii=False, indent=2))
            ids.extend(record["chunk_ids"])
            audit.append({"query": normalized, "cache": "miss", **record})
        return {"chunk_ids": sorted(set(ids)), "queries": audit}
