# 두 컬렉션 검색·검색 방식 비교·RRF 병합·중복 제거

import time
from collections import defaultdict

from src.documents import digest
from src.indexing import search_rounds, vector_search


def applicable(metadata, as_of, round_number=None):
    if metadata.get("scope") == "round" and metadata.get("round_id") != f"lotto_{round_number}":
        return False
    if metadata.get("document_type") in {"law_article", "government_guidance"}:
        if metadata.get("document_type") == "government_guidance":
            return metadata.get("verified_on") == str(as_of)
        return bool(
            metadata.get("effective_date")
            and metadata["effective_date"] <= str(as_of)
            and metadata.get("verified_on") == str(as_of)
        )
    return True


def document_search(
    question,
    collection,
    embeddings,
    top_k,
    as_of,
    round_number=None,
    search_type="similarity",
    fetch_k=20,
    lambda_mult=0.5,
):
    stored = collection.get(include=["metadatas"])
    ids = [
        key
        for key, metadata in zip(stored["ids"], stored["metadatas"])
        if applicable(metadata, as_of, round_number)
    ]
    if not ids:
        return []
    eligible = set(ids)
    rows, _ = vector_search(
        collection,
        embeddings.embed_query(question),
        top_k,
        where={
            "document_id": {
                "$in": sorted(
                    {
                        m["document_id"]
                        for key, m in zip(stored["ids"], stored["metadatas"])
                        if key in eligible
                    }
                )
            }
        },
        search_type=search_type,
        fetch_k=max(fetch_k, top_k),
        lambda_mult=lambda_mult,
        eligible_ids=ids,
    )
    return [
        {**row, "collection": "lotto_documents"}
        for row in rows
        if applicable(row["metadata"], as_of, round_number)
    ]


def complete_article(collection, law_title, article_no, as_of):
    stored = collection.get(
        where={
            "$and": [
                {"law_title": law_title},
                {"article_no": article_no},
                {"verified_on": str(as_of)},
            ]
        },
        include=["documents", "metadatas"],
    )
    groups = defaultdict(list)
    for key, text, meta in zip(stored["ids"], stored["documents"], stored["metadatas"]):
        if applicable(meta, as_of):
            groups[meta["document_id"]].append((key, text, meta))
    results = []
    for document_id, items in groups.items():
        items.sort(key=lambda item: item[2]["char_start"])
        end, content = 0, ""
        valid = True
        for _, text, meta in items:
            start = meta["char_start"]
            if start > end or (start < end and content[start:end] != text[: end - start]):
                valid = False
                break
            content += text[max(0, end - start) :]
            end = meta["char_end"]
        if valid and digest(content) == items[0][2]["article_hash"]:
            results.append(
                {
                    "chunk_id": document_id,
                    "text": content,
                    "metadata": items[0][2],
                    "collection": "lotto_documents",
                    "rank": 1,
                    "original_chunk_ids": [item[0] for item in items],
                }
            )
    return results


def merge_evidence(lists, rrf_k=60):
    if rrf_k < 1:
        raise ValueError("RRF 상수는 양수여야 합니다.")
    merged = {}
    for items in lists:
        seen_in_list = set()
        for rank, item in enumerate(items, 1):
            meta = item["metadata"]
            # 같은 사실을 담더라도 서로 다른 출처의 근거는 따로 보존한다.
            identity = (
                meta.get("document_id", item["chunk_id"]),
                digest(item["text"]),
                meta.get("source_url", ""),
            )
            if identity in seen_in_list:
                continue
            seen_in_list.add(identity)
            if identity not in merged:
                merged[identity] = {
                    **item,
                    "rrf_score": 0.0,
                    "original_chunk_ids": item.get("original_chunk_ids", [item["chunk_id"]]),
                }
            merged[identity]["rrf_score"] += 1 / (rrf_k + rank)
            merged[identity]["exact"] = merged[identity].get("exact", False) or item.get(
                "exact", False
            )
    ranked = sorted(
        merged.values(),
        key=lambda item: (not item.get("exact", False), -item["rrf_score"], item["chunk_id"]),
    )
    compact = []
    for item in ranked:
        # 완전한 조문을 재구성했으면 그 안에 포함되는 부분 청크는 제거한다.
        if any(
            other["metadata"].get("document_id") == item["metadata"].get("document_id")
            and item["text"] in other["text"]
            and len(other["text"]) > len(item["text"])
            for other in ranked
        ):
            continue
        compact.append(item)
    # 검색 순위와 관계없이 같은 원문의 인접 중복 구간을 병합한다.
    changed = True
    while changed:
        changed = False
        for i, left in enumerate(compact):
            for j, right in enumerate(compact):
                a, b = left["metadata"], right["metadata"]
                if (
                    i == j
                    or not a.get("document_id")
                    or a.get("document_id") != b.get("document_id")
                    or a.get("source_url") != b.get("source_url")
                    or "char_start" not in a
                    or "char_start" not in b
                    or not a["char_start"] < b["char_start"] < a["char_end"] < b["char_end"]
                ):
                    continue
                overlap = a["char_end"] - b["char_start"]
                if left["text"][-overlap:] == right["text"][:overlap]:
                    left["text"] += right["text"][overlap:]
                    left["metadata"] = {**a, "char_end": b["char_end"]}
                    left["original_chunk_ids"].extend(right["original_chunk_ids"])
                    left["rrf_score"] = max(left["rrf_score"], right["rrf_score"])
                    compact.pop(j)
                    changed = True
                    break
            if changed:
                break
    compact.sort(
        key=lambda item: (not item.get("exact", False), -item["rrf_score"], item["chunk_id"])
    )
    for i, item in enumerate(compact, 1):
        item["evidence_id"] = f"E{i}"
    return compact


def retrieve(
    question,
    plan,
    draws,
    round_collection,
    document_collection,
    embeddings,
    config,
    as_of,
    collected_at="",
):
    options = {
        "search_type": config.get("search_type", "similarity"),
        "fetch_k": config.get("mmr_fetch_k", 20),
        "lambda_mult": config.get("mmr_lambda_mult", 0.5),
    }
    round_results = search_rounds(
        question, round_collection, embeddings, config["round_top_k"], plan.filters.round, **options
    )
    round_results = [{**item, "collection": "lotto_rounds"} for item in round_results]
    exact = [draw for draw in draws if draw.round == plan.filters.round]
    # 수치의 기준은 CSV이며 벡터 메타데이터가 이를 덮어쓰지 못하게 한다.
    for draw in exact:
        round_results = [
            item for item in round_results if item["metadata"].get("round") != draw.round
        ]
        round_results.insert(
            0,
            {
                "chunk_id": f"lotto_{draw.round}",
                "text": draw.search_text(),
                "metadata": draw.metadata(collected_at),
                "collection": "lotto_rounds",
                "exact": True,
            },
        )
    documents = document_search(
        question,
        document_collection,
        embeddings,
        config["document_top_k"],
        as_of,
        plan.filters.round,
        **options,
    )
    required = []
    if plan.intent == "tax":
        for title, number in [("소득세법", "84"), ("소득세법", "129"), ("지방세법", "103의13")]:
            required.extend(complete_article(document_collection, title, number, as_of))
        required.extend(complete_article(document_collection, "", "", as_of))
    if plan.intent == "claim_guide":
        required.extend(complete_article(document_collection, "복권 및 복권기금법", "9", as_of))
    documents = required + documents
    return merge_evidence([round_results, documents], config["rrf_k"]), {
        "lotto_rounds": [item["chunk_id"] for item in round_results],
        "lotto_documents": [item["chunk_id"] for item in documents],
        "settings": options,
    }


def compare_search(question, engine, as_of, top_k=5, fetch_k=20, lambda_mult=0.5):
    vector = engine.embeddings.embed_query(question)
    output = {}
    for name, collection in (
        ("lotto_rounds", engine.round_collection),
        ("lotto_documents", engine.document_collection),
    ):
        where = None
        eligible_ids = None
        if name == "lotto_documents":
            stored = collection.get(include=["metadatas"])
            ids = sorted(
                {meta["document_id"] for meta in stored["metadatas"] if applicable(meta, as_of)}
            )
            eligible_ids = [
                key
                for key, meta in zip(stored["ids"], stored["metadatas"])
                if applicable(meta, as_of)
            ]
            if not ids:
                output[name] = {
                    method: {"results": [], "diversity": 0.0, "candidate_ids": []}
                    for method in ("similarity", "mmr")
                }
                continue
            where = {"document_id": {"$in": ids}}
        output[name] = {}
        for method in ("similarity", "mmr"):
            started = time.perf_counter()
            rows, audit = vector_search(
                collection,
                vector,
                top_k,
                where=where,
                search_type=method,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                eligible_ids=eligible_ids,
            )
            output[name][method] = {
                **audit,
                "results": rows,
                "elapsed_seconds": time.perf_counter() - started,
            }
        if output[name]["similarity"]["candidate_ids"] != output[name]["mmr"]["candidate_ids"]:
            raise ValueError("비교 중 후보군이 변경되었습니다. 다시 비교하세요.")
    return {
        "question": question,
        "as_of": str(as_of),
        "top_k": top_k,
        "fetch_k": fetch_k,
        "lambda_mult": lambda_mult,
        "collections": output,
        "note": "두 방식은 같은 질문 임베딩과 후보군을 사용합니다. 다양성은 선택한 벡터 쌍의 평균 cosine 거리입니다.",
    }
