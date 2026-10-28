from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_settings(path=None):
    with Path(path or ROOT / "config.yaml").open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if settings["data"]["target_rounds"] < settings["data"]["minimum_rounds"]:
        raise ValueError("target_rounds는 minimum_rounds 이상이어야 합니다.")
    if settings["embedding"]["batch_size"] < 1 or settings["retrieval"]["round_top_k"] < 1:
        raise ValueError("배치 크기와 Top-k는 양수여야 합니다.")
    if (
        not 0
        <= settings["chunking"]["document_chunk_overlap"]
        < settings["chunking"]["document_chunk_size"]
    ):
        raise ValueError("청크 크기 > 중복 길이 >= 0이어야 합니다.")
    if settings["retrieval"]["document_top_k"] < 1 or settings["retrieval"]["rrf_k"] < 1:
        raise ValueError("문서 Top-k와 RRF 상수는 양수여야 합니다.")
    retrieval = settings["retrieval"]
    if (
        retrieval.get("search_type", "similarity") not in {"similarity", "mmr"}
        or not 0 <= retrieval.get("mmr_lambda_mult", 0.5) <= 1
        or retrieval.get("mmr_fetch_k", 20)
        < max(retrieval["round_top_k"], retrieval["document_top_k"])
    ):
        raise ValueError("MMR 검색 방식·fetch_k·lambda_mult 설정을 확인하세요.")
    law = settings["law"]
    if (
        not 1 <= law["candidate_limit"] <= 100
        or law["max_retries"] < 0
        or law["request_interval"] < 0
    ):
        raise ValueError("법령 호출 설정을 확인하세요.")
    return settings


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
