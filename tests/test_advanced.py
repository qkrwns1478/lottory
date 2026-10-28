from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import chromadb
import pytest
from chromadb.config import Settings
from pydantic import ValidationError

from src.contracts import Article, PublicError
from src.evaluation import evaluate, evaluation_cases, score
from src.frontend import handle_event
from src.guidance import NTS_URL, GuidanceCollector, extract_guidance
from src.indexing import cosine, diversity, mmr_indices, vector_search
from src.settings import load_settings


def test_mmr_trades_relevance_for_diversity():
    vectors = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    assert mmr_indices([1, 0], vectors, 2, 1) == [0, 1]
    assert mmr_indices([1, 0], vectors, 2, 0) == [0, 2]
    assert diversity([vectors[0], vectors[2]]) > diversity(vectors[:2])
    assert cosine([0, 0], [1, 1]) == 0
    assert mmr_indices([1, 0], [], 5) == []


@pytest.mark.parametrize("k,value", [(0, 0.5), (2, -1), (2, 1.1)])
def test_mmr_rejects_invalid_settings(k, value):
    with pytest.raises(ValueError):
        mmr_indices([1, 0], [[1, 0]], k, value)


def test_real_chroma_algorithms_share_pool_and_respect_filter(tmp_path):
    client = chromadb.PersistentClient(
        path=str(tmp_path / "vectors"), settings=Settings(anonymized_telemetry=False)
    )
    try:
        collection = client.get_or_create_collection("mmr_test", embedding_function=None)
        collection.upsert(
            ids=["a", "b", "c", "excluded"],
            embeddings=[[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [1.0, 0.0]],
            documents=["relevant", "duplicate", "diverse", "wrong date"],
            metadatas=[
                {"eligible": True},
                {"eligible": True},
                {"eligible": True},
                {"eligible": False},
            ],
        )
        similarity, sim = vector_search(collection, [1, 0], 2, where={"eligible": True}, fetch_k=4)
        mmr, audit = vector_search(
            collection,
            [1, 0],
            2,
            where={"eligible": True},
            search_type="mmr",
            fetch_k=4,
            lambda_mult=0,
        )
        assert sim["candidate_ids"] == audit["candidate_ids"]
        assert "excluded" not in sim["candidate_ids"]
        assert [row["chunk_id"] for row in similarity] == ["a", "b"]
        assert [row["chunk_id"] for row in mmr] == ["a", "c"]
        assert audit["diversity"] > sim["diversity"]
        filtered, _ = vector_search(collection, [1, 0], 2, fetch_k=4, eligible_ids=["c"])
        assert [row["chunk_id"] for row in filtered] == ["c"]
    finally:
        client._system.stop()


HTML = """<html><nav>irrelevant menu</nav><h2>원천징수세율</h2>
<p>기타소득 과세 안내</p><table><tr><td>복권 당첨금의 소득금액이 3억원을 초과하는 경우</td><td>30%</td></tr>
<tr><td>일반적인 기타소득</td><td>20%</td></tr></table>
<p>연금에 원천징수세율을 적용하여 계산함</p><div>콘텐츠 만족도 조사</div><script>bad()</script></html>"""


def test_guidance_extracts_heading_not_last_mention():
    text = extract_guidance(HTML)
    assert "복권" in text and "30%" in text
    assert "irrelevant" not in text and "만족도" not in text and "bad()" not in text
    with pytest.raises(PublicError):
        extract_guidance("<h2>로그인 필요</h2>")


def test_guidance_collects_and_second_request_is_cached(tmp_path):
    settings = load_settings()
    settings["guidance"]["cache_path"] = str(tmp_path / "guidance.json")
    settings["data"]["source_registry"] = str(tmp_path / "sources.json")

    class Session:
        calls = 0

        def get(self, url, **kwargs):
            assert url == NTS_URL and not kwargs["allow_redirects"]
            self.calls += 1
            return SimpleNamespace(status_code=200, content=HTML.encode(), text=HTML)

    class Embeddings:
        def embed_documents(self, texts):
            return [[1.0, 2.0, 3.0] for _ in texts]

    client = chromadb.PersistentClient(
        path=str(tmp_path / "db"), settings=Settings(anonymized_telemetry=False)
    )
    try:
        collection = client.get_or_create_collection("guidance", embedding_function=None)
        session = Session()
        collector = GuidanceCollector(settings, collection, Embeddings(), session)
        today = datetime.now(ZoneInfo("Asia/Seoul")).date()
        first, second = collector.collect(today), collector.collect(today)
        assert first["status"] == "indexed" and second["cache"] == "hit"
        assert session.calls == 1 and second["http_calls"] == 0
        metadata = collection.get(include=["metadatas"])["metadatas"][0]
        assert metadata["source_type"] == "government_guidance"
        assert metadata["effective_date"] == ""  # 수집일을 법령 시행일로 추정하지 않는다.
    finally:
        client._system.stop()


def test_guidance_cannot_impersonate_statutes():
    with pytest.raises(ValidationError):
        Article(
            document_id="bad",
            law_title="fake",
            article_no="1",
            document_title="fake",
            text="fake",
            effective_date="2026-01-01",
            verified_on="2026-10-22",
            collected_at="today",
            source_url=NTS_URL,
        )


def test_evaluation_is_paired_and_includes_twenty_cases(draws):
    cases = evaluation_cases(draws)
    assert len(cases) == 20 and len({case["id"] for case in cases}) == 20
    seen = []

    def baseline(question, selected, data):
        seen.append(("baseline", question))
        return {"status": "insufficient_evidence", "answer": "모릅니다", "sources": []}

    def rag(question, selected, data):
        from src.service import answer_question

        seen.append(("rag", question))
        return answer_question(question, selected, data)

    report = evaluate(draws, baseline, rag, cases=cases[:2] + [cases[14]])
    assert [question for name, question in seen if name == "baseline"] == [
        question for name, question in seen if name == "rag"
    ]
    assert report["summary"]["rag"]["accuracy"] == 1
    assert report["summary"]["baseline"]["accuracy"] == pytest.approx(1 / 3)
    assert report["summary"]["rag"]["abstention_accuracy"] == 1


def test_evaluation_counts_failures_in_denominator(draws):
    def broken(*_):
        raise RuntimeError("private-test")

    report = evaluate(draws, broken, broken, cases=evaluation_cases(draws)[:2])
    assert report["summary"]["rag"]["errors"] == 2
    assert report["summary"]["rag"]["accuracy"] == 0
    assert "private-test" not in str(report)


def test_concept_correctness_is_not_confused_with_abstention_label():
    case = {"kind": "probability", "expected": {}}
    result = score(
        case,
        {
            "status": "insufficient_evidence",
            "answer": "과거 빈도가 높아도 다음 회차에 유리하지 않습니다.",
            "sources": [],
        },
    )
    assert result["correct"]


def test_frontend_filters_and_rejects_bad_events(draws):
    def no_api(settings):
        raise AssertionError("Local filter must not call APIs")

    result = handle_event(
        {"id": "filter1", "action": "filter", "filters": {"numbers": [22]}},
        load_settings(),
        draws,
        no_api,
    )
    assert len(result["rows"]) == 60
    assert all(22 in row["numbers"] for row in result["rows"])
    for request in (
        {"id": "x", "action": "eval_code"},
        {"id": "x", "action": "filter", "filters": {"numbers": [99]}},
    ):
        with pytest.raises(ValueError):
            handle_event(request, load_settings(), draws, no_api)


def test_frontend_local_question_is_connected_to_real_service(draws):
    result = handle_event(
        {"id": "ask1", "action": "ask", "question": "1100회 당첨금은?", "selected_round": 1100},
        load_settings(),
        draws,
        lambda _: None,
    )
    assert result["response"]["status"] == "answered"
    assert result["response"]["rows"][0]["round"] == 1100


def test_frontend_evaluation_connects_runners_and_saves_provenance(draws, monkeypatch):
    from types import SimpleNamespace

    import src.frontend as frontend

    baseline, rag = object(), object()
    saved = []
    monkeypatch.setattr(frontend, "make_baseline", lambda settings: baseline)

    def run(data, before, after, progress):
        assert data is draws and before is baseline and after is rag
        return {"case_count": 20}

    monkeypatch.setattr(frontend, "evaluate", run)
    monkeypatch.setattr(frontend, "save_evaluation", saved.append)
    result = handle_event(
        {"id": "eval1", "action": "evaluate", "search_type": "mmr"},
        load_settings(),
        draws,
        lambda _: SimpleNamespace(answer=rag),
    )
    assert result["kind"] == "evaluation" and saved[0] is result["evaluation"]
    assert saved[0]["retrieval"]["search_type"] == "mmr"
    assert saved[0]["data_range"] == {
        "from_round": min(d.round for d in draws),
        "to_round": max(d.round for d in draws),
    }
    assert saved[0]["evaluated_at"].endswith("+09:00")
