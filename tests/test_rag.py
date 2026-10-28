# 합성 법령과 실제 ChromaDB·LangGraph를 사용한 오프라인 통합 테스트

from datetime import date
from types import SimpleNamespace

import chromadb
import pytest
from chromadb.config import Settings

from src.contracts import Article, Candidate, GeneratedAnswer, PublicError, Selection
from src.documents import DocumentCollector, chunks, index_articles
from src.generation import conflicts, verify_answer
from src.query import Filters, SearchPlan, local_plan
from src.retrieval import complete_article, merge_evidence
from src.settings import load_settings
from src.tax import calculate_tax
from src.workflow import LottoWorkflow

DAY = date(2026, 10, 22)


def test_prize_extremes_survive_rag_generation_failure(environment, draws):
    def generator(*args):
        raise ValueError("offline generation failure")

    engine = LottoWorkflow(
        environment.settings,
        planner=lambda question, selected, _: local_plan(question, selected),
        collector=None,
        round_collection=environment.rounds,
        document_collection=environment.documents,
        embeddings=environment.embeddings,
        generator=generator,
        today=lambda: DAY,
    )
    response = engine.answer("최근 100회 중 당첨금의 최대 금액과 최소 금액은?", 1100, draws)
    assert response["status"] == "answered"
    assert "최대 금액: 1,119,000,000원" in response["answer"]
    assert "최소 금액: 1,020,000,000원" in response["answer"]
    assert response["calculation"]["row_count"] == 100
    assert any(s["document_type"] == "verified_calculation" for s in response["sources"])


class OfflineEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text) % 100), 1.0, 0.5] for text in texts]

    def embed_query(self, text):
        return [float(len(text) % 100), 1.0, 0.5]


@pytest.fixture
def articles():
    # 테스트용으로 짧게 작성한 합성 법령이며 실제 API로 수집한 기록이 아니다.
    specs = [
        (
            "1",
            "소득세법",
            "84",
            "복권 당첨금은 건별로 200만원 이하인 경우 과세하지 아니한다.",
        ),
        (
            "1",
            "소득세법",
            "129",
            "소득금액이 3억원을 초과하는 경우 그 초과하는 분은 100분의 30.\n그 밖의 기타소득은 100분의 20.",
        ),
        (
            "2",
            "지방세법",
            "103의13",
            "원천징수하는 소득세의 100분의 10을 특별징수한다.",
        ),
        (
            "3",
            "복권 및 복권기금법",
            "9",
            "온라인복권 당첨금을 받을 권리는 지급개시일부터 1년간 행사하지 않으면 소멸한다.",
        ),
    ]
    return [
        Article(
            document_id=f"law_{serial}_{number}",
            law_title=title,
            article_no=number,
            document_title=f"{title} 제{number}조",
            text=text,
            effective_date=date(2026, 1, 1),
            source_url=f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={serial}",
            collected_at="2026-10-22T00:00:00+00:00",
            verified_on=DAY,
        )
        for serial, title, number, text in specs
    ]


@pytest.fixture
def environment(tmp_path):
    settings = load_settings()
    settings["data"]["source_registry"] = str(tmp_path / "registry.json")
    settings["law"]["cache_path"] = str(tmp_path / "cache.json")
    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False)
    )
    rounds = client.get_or_create_collection("rounds", embedding_function=None)
    documents = client.get_or_create_collection("documents", embedding_function=None)
    yield SimpleNamespace(
        settings=settings,
        rounds=rounds,
        documents=documents,
        embeddings=OfflineEmbeddings(),
    )
    # 다음 통합 테스트 전에 임시 네이티브 클라이언트가 사용한 자원을 반환한다.
    client._system.stop()


def selector(question, plan, candidates):
    return Selection(
        decisions=[
            {
                "candidate_id": candidate.candidate_id,
                "decision": "index",
                "reason": "test relevance",
                "priority": 1,
            }
            for candidate in candidates
        ]
    )


def make_collector(env, articles):
    class Client:
        def __init__(self):
            self.call_count = 0

        def search(self, query, as_of):
            self.call_count += 1
            return [
                Candidate(
                    candidate_id=str(i),
                    law_title=title,
                    effective_date=date(2026, 1, 1),
                    source_url=f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={i}",
                )
                for i, title in [
                    (1, "소득세법"),
                    (2, "지방세법"),
                    (3, "복권 및 복권기금법"),
                ]
                if title == query
            ]

        def articles(self, candidate, as_of):
            self.call_count += 1
            return [article for article in articles if article.law_title == candidate.law_title]

    client = Client()
    return DocumentCollector(env.settings, client, selector, env.documents, env.embeddings)


def test_chunks_and_overlap_reconstruct_article(environment, articles):
    article = articles[1].model_copy(update={"text": "가나다라" * 90 + "\n" + "마바사아" * 90})
    config = {"document_chunk_size": 100, "document_chunk_overlap": 20}
    pieces = chunks(article, 100, 20)
    assert all(len(piece["text"]) <= 100 for piece in pieces)
    assert pieces[0]["text"][-20:] == pieces[1]["text"][:20]
    ids = index_articles([article], environment.documents, environment.embeddings, config)
    restored = complete_article(environment.documents, "소득세법", "129", DAY)
    assert restored[0]["text"] == article.text
    assert set(restored[0]["original_chunk_ids"]) == set(ids)
    index_articles(
        [article],
        environment.documents,
        environment.embeddings,
        {"document_chunk_size": 200, "document_chunk_overlap": 0},
    )
    assert environment.documents.count() < len(ids)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_configuration(articles, size, overlap):
    with pytest.raises(ValueError):
        chunks(articles[0], size, overlap)


def test_cache_second_request_has_zero_collection_calls(environment, articles):
    collector = make_collector(environment, articles)
    plan = SearchPlan(
        intent="tax", need_external_docs=True, search_queries=["소득세법", "지방세법"]
    )
    first = collector.collect("세금", plan, DAY)
    assert len(first["chunk_ids"]) == 3
    assert collector.client.call_count == 4
    second = collector.collect("세금", plan, DAY)
    assert collector.client.call_count == 4
    assert all(item["cache"] == "hit" for item in second["queries"])
    # 성공 캐시가 있어도 컬렉션에 문서가 없으면 다시 색인해야 한다.
    environment.documents.delete(ids=[first["chunk_ids"][0]])
    collector.collect("세금", plan, DAY)
    assert collector.client.call_count > 4


def test_selection_rejects_invented_or_missing_candidate_ids(environment, articles):
    collector = make_collector(environment, articles)
    collector.selector = lambda *_: Selection(
        decisions=[
            {
                "candidate_id": "invented",
                "decision": "index",
                "reason": "test",
                "priority": 1,
            }
        ]
    )
    with pytest.raises(PublicError, match="후보 선별"):
        collector.collect("세금", SearchPlan(intent="tax", search_queries=["소득세법"]), DAY)
    assert environment.documents.count() == 0


def test_review_usage_never_downloads_or_indexes(environment, articles):
    collector = make_collector(environment, articles)
    collector.selector = lambda q, p, candidates: Selection(
        decisions=[
            {
                "candidate_id": candidate.candidate_id,
                "decision": "review_usage",
                "reason": "조건 불명확",
                "priority": 1,
            }
            for candidate in candidates
        ]
    )
    result = collector.collect("세금", SearchPlan(intent="tax", search_queries=["소득세법"]), DAY)
    assert collector.client.call_count == 1
    assert not result["chunk_ids"]
    collector.collect("세금", SearchPlan(intent="tax", search_queries=["소득세법"]), DAY)
    assert collector.client.call_count == 1  # 빈 결과의 재시도 대기 시간에는 다시 호출하지 않는다.


def evidence_for(articles):
    return merge_evidence(
        [
            [
                {
                    "chunk_id": article.document_id,
                    "text": article.text,
                    "metadata": chunks(article)[0]["metadata"],
                }
                for article in articles
            ]
        ]
    )


@pytest.mark.parametrize(
    "amount,income,local,net",
    [
        (0, 0, 0, 0),
        (2_000_000, 0, 0, 2_000_000),
        (2_000_001, 400_000, 40_000, 1_560_001),
        (300_000_000, 60_000_000, 6_000_000, 234_000_000),
        (1_000_000_000, 270_000_000, 27_000_000, 703_000_000),
    ],
)
def test_tax_boundaries_with_legal_evidence(articles, amount, income, local, net):
    result = calculate_tax(amount, evidence_for(articles), DAY)
    assert result["income_tax"] == income
    assert result["local_income_tax"] == local
    assert result["net_amount"] == net
    assert len(result["source_evidence_ids"]) == 3


def test_tax_is_withheld_without_verified_rules(articles):
    assert calculate_tax(1_000_000_000, evidence_for(articles[:2]), DAY) is None
    amended = [
        article.model_copy(update={"text": article.text.replace("100분의 30", "100분의 35")})
        for article in articles
    ]
    assert calculate_tax(1_000_000_000, evidence_for(amended), DAY) is None
    assert calculate_tax(1_000_000_000, evidence_for(articles), date(2025, 1, 1)) is None


def test_generation_ids_quotes_and_calculated_numbers_are_checked(articles):
    evidence = evidence_for(articles)
    item = evidence[0]
    answer = GeneratedAnswer(
        status="answered",
        answer=item["text"],
        evidence_ids=[item["evidence_id"]],
        citations=[{"evidence_id": item["evidence_id"], "quote": item["text"]}],
    )
    assert verify_answer(answer, evidence).status == "answered"
    for change in [
        {"evidence_ids": ["FAKE"]},
        {"answer": "세금은 999,999,999원입니다."},
        {"citations": [{"evidence_id": item["evidence_id"], "quote": "근거에 없는 문장"}]},
    ]:
        with pytest.raises(PublicError, match="답변을 보류"):
            verify_answer({**answer.model_dump(), **change}, evidence)


def test_conflicting_versions_withhold_answers(articles):
    conflicting = articles[0].model_copy(
        update={"document_id": "law_other_84", "text": "과세 조건이 다른 본문"}
    )
    assert conflicts(evidence_for([articles[0], conflicting]))


def grounded_generator(question, plan, evidence, calculation, canonical):
    item = evidence[0]
    return GeneratedAnswer(
        status="answered",
        answer=canonical or item["text"],
        evidence_ids=[item["evidence_id"]],
        citations=[{"evidence_id": item["evidence_id"], "quote": item["text"]}],
    )


def workflow(env, articles, plan, generator=grounded_generator):
    return LottoWorkflow(
        env.settings,
        lambda *_: plan,
        make_collector(env, articles),
        env.rounds,
        env.documents,
        env.embeddings,
        generator,
        today=lambda: DAY,
    )


def test_end_to_end_graph_combines_csv_law_calculation_and_sources(environment, articles, draws):
    plan = SearchPlan(
        intent="tax",
        filters=Filters(round=1100),
        search_queries=["소득세법", "지방세법"],
    )
    engine = workflow(environment, articles, plan)
    assert set(engine.graph.get_graph().nodes) - {"__start__", "__end__"} == {
        "retrieve",
        "generate",
    }
    progress = []
    response = engine.answer("1100회 1등 당첨금과 세금", 1120, draws, progress.append)
    assert response["status"] == "answered"
    assert response["calculation"]["gross_amount"] == next(
        draw.first_prize_amount for draw in draws if draw.round == 1100
    )
    assert response["calculation"]["net_amount"] > 0
    assert "예상 실수령액" in response["answer"]
    assert response["audit"]["search"]["lotto_rounds"]
    assert response["audit"]["search"]["lotto_documents"]
    assert any(source.get("law_title") == "지방세법" for source in response["sources"])
    assert any(source.get("round") == 1100 for source in response["sources"])
    assert "법령 후보 선별" in progress and "인용·수치 검증" in progress
    calls = engine.collector.client.call_count
    engine.answer("같은 질문", 1120, draws)
    assert engine.collector.client.call_count == calls


def test_top_k_changes_evidence_for_same_question(environment, articles, draws):
    plan = SearchPlan(intent="statistics", filters=Filters(recent_rounds=100, numbers=[22]))
    engine = workflow(environment, articles, plan)
    question = "최근 100회 중 22번이 몇 번 나왔어?"
    engine.settings["retrieval"]["round_top_k"] = 1
    first = engine.answer(question, 1100, draws)
    engine.settings["retrieval"]["round_top_k"] = 5
    second = engine.answer(question, 1100, draws)
    assert first["status"] == second["status"] == "answered"
    assert len(first["audit"]["search"]["lotto_rounds"]) == 1
    assert len(second["audit"]["search"]["lotto_rounds"]) == 5
    assert {e["chunk_id"] for e in first["evidence"]} != {e["chunk_id"] for e in second["evidence"]}
    assert first["calculation"] == second["calculation"]


def test_missing_draw_does_not_collect_or_generate(environment, articles, draws):
    plan = SearchPlan(intent="tax", filters=Filters(round=9999))
    engine = workflow(environment, articles, plan)
    result = engine.answer("9999회 세금", 1120, draws)
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []
    assert engine.collector.client.call_count == 0


def test_unverified_generated_answer_is_not_shown(environment, articles, draws):
    def hallucination(*_):
        return GeneratedAnswer(status="answered", answer="99999999원", evidence_ids=["fake"])

    engine = workflow(
        environment,
        articles,
        SearchPlan(intent="claim_guide", need_external_docs=True),
        hallucination,
    )
    result = engine.answer("당첨자의 개인정보 보호 규정", 1120, draws)
    assert result["status"] == "insufficient_evidence"
    assert "99999999" not in result["answer"]
    assert not result["sources"]


def test_invalid_model_text_falls_back_to_verified_csv_only(environment, articles, draws):
    def hallucination(*_):
        return GeneratedAnswer(status="answered", answer="가짜 99999999원", evidence_ids=["fake"])

    engine = workflow(
        environment,
        articles,
        SearchPlan(intent="round_lookup", filters=Filters(round=1100)),
        hallucination,
    )
    result = engine.answer("1100회 당첨금", 1120, draws)
    assert result["status"] == "answered"
    assert "99999999" not in result["answer"]
    assert result["rows"][0]["first_prize_amount"] == next(
        draw.first_prize_amount for draw in draws if draw.round == 1100
    )
    assert result["limitations"]


def test_statistics_use_all_rows_instead_of_top_k(environment, articles, draws):
    engine = workflow(
        environment,
        articles,
        SearchPlan(intent="statistics", filters=Filters(recent_rounds=100, numbers=[22])),
    )
    result = engine.answer("최근 100회 22번 빈도", 1120, draws)
    assert result["status"] == "answered"
    assert result["calculation"]["row_count"] == 100
    assert result["calculation"]["result"] == {"22": 50}
    assert any(
        source["document_title"] == "CSV 전체 대상 회차 Python 집계" for source in result["sources"]
    )


def test_top_frequency_reaches_grounded_ai_answer(environment, articles, draws):
    from src.query import local_plan

    question = "최근 100회 중 가장 많이 나온 번호 6개는?"
    engine = workflow(environment, articles, local_plan(question, 1100))
    result = engine.answer(question, 1100, draws)
    assert result["status"] == "answered"
    assert result["calculation"]["row_count"] == 100
    assert list(result["calculation"]["result"].items()) == [(str(n), 50) for n in range(1, 7)]
    assert all(f"{n}번 50회" in result["answer"] for n in range(1, 7))
    assert any(
        source["document_title"] == "CSV 전체 대상 회차 Python 집계" for source in result["sources"]
    )


def test_overlap_dedup_does_not_depend_on_search_rank(articles):
    article = articles[0].model_copy(update={"text": "abcdefghijklmnopqrstuvwxyz"})
    batch = chunks(article, size=16, overlap=6)
    items = [
        {"chunk_id": item["id"], "text": item["text"], "metadata": item["metadata"]}
        for item in batch
    ]
    result = merge_evidence([list(reversed(items))])
    assert len(result) == 1 and result[0]["text"] == article.text
    assert set(result[0]["original_chunk_ids"]) == {item["id"] for item in batch}


def test_future_law_and_round_scopes_are_excluded():
    from src.retrieval import applicable

    assert not applicable({"scope": "round", "round_id": "lotto_1101"}, DAY, 1100)
    assert not applicable(
        {
            "document_type": "law_article",
            "effective_date": "2027-01-01",
            "verified_on": str(DAY),
        },
        DAY,
    )
    assert not applicable(
        {
            "document_type": "law_article",
            "effective_date": "2026-01-01",
            "verified_on": "2026-10-21",
        },
        DAY,
    )


def test_ai_chat_renders_tax_calculation_and_sources(
    environment, articles, raw_csv, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-not-sent")
    from streamlit.testing.v1 import AppTest

    from src.data import prepare_dataset
    from src.query import local_plan
    from src.settings import ROOT

    path = tmp_path / "app_rounds.csv"
    prepare_dataset(raw_csv, path, tmp_path / "registry.json")
    environment.settings["data"]["csv_path"] = str(path)
    plan = SearchPlan(intent="tax", prize_amount=1_000_000_000)
    engine = workflow(environment, articles, plan)
    monkeypatch.setattr("src.settings.load_settings", lambda: environment.settings)
    monkeypatch.setattr("src.planner.make_planner", lambda settings: local_plan)
    monkeypatch.setattr("src.workflow.make_workflow", lambda settings, planner=None: engine)
    events = [
        {
            "id": "tax",
            "action": "ask",
            "question": "10억 원의 예상 실수령액은?",
            "selected_round": 1100,
            "ai": True,
        }
    ]
    monkeypatch.setattr(
        "streamlit.components.v1.declare_component",
        lambda *args, **kwargs: lambda **props: events.pop(0) if events else None,
    )
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception
    response = app.session_state["custom_messages"][-1]["response"]
    assert "703,000,000원" in response["answer"]
    assert any("지방세법" in item["document_title"] for item in response["sources"])


def test_max_min_average_are_exact(draws):
    from src.query import prize_statistics

    result = prize_statistics(draws, Filters(recent_rounds=10), "average")
    assert result["row_count"] == 10
    assert result["result"]["first_prize_amount"] == "1114500000"
    assert (
        prize_statistics(draws, Filters(), "maximum")["result"]["first_prize_amount"]
        == 1_119_000_000
    )
    assert (
        prize_statistics(draws, Filters(), "minimum")["result"]["first_prize_amount"]
        == 1_000_000_000
    )


def test_search_results_have_csv_evidence_and_safe_fallback(environment, articles, draws):
    def invalid_generator(*_):
        return GeneratedAnswer(status="answered", answer="9999개", evidence_ids=["fake"])

    engine = workflow(
        environment,
        articles,
        SearchPlan(intent="search", filters=Filters(numbers=[22])),
        invalid_generator,
    )
    response = engine.answer("22번이 나온 회차", 1120, draws)
    assert response["status"] == "answered"
    assert len(response["rows"]) == 60
    assert response["calculation"]["row_count"] == 60
    assert "9999" not in response["answer"]
    assert response["sources"]


def test_selector_receives_resolved_date_and_required_law_queries(environment, articles, draws):
    plan = SearchPlan(
        intent="tax",
        prize_amount=1_000_000_000,
        search_queries=[
            "문장 검색 하나",
            "문장 검색 둘",
            "문장 검색 셋",
            "문장 검색 넷",
        ],
    )
    engine = workflow(environment, articles, plan)
    original = engine.collector.selector
    seen = []

    def dated_selector(question, actual_plan, candidates):
        assert actual_plan.as_of == DAY
        seen.append(actual_plan.search_queries)
        return original(question, actual_plan, candidates)

    engine.collector.selector = dated_selector
    response = engine.answer("10억원 실수령액", 1120, draws)
    assert response["status"] == "answered"
    assert seen and seen[0][:2] == ["소득세법", "지방세법"]
    assert response["audit"]["query_normalization"]["extracted"][0] == "문장 검색 하나"


def test_question_wording_variations_do_not_repeat_collection(environment, articles, draws):
    plans = iter(
        [
            SearchPlan(intent="tax", prize_amount=1_000_000_000, search_queries=["로또 세금"]),
            SearchPlan(
                intent="tax",
                prize_amount=1_000_000_000,
                search_queries=["복권 실수령액"],
            ),
        ]
    )
    engine = workflow(environment, articles, SearchPlan(intent="tax"))
    engine.planner = lambda *_: next(plans)
    first = engine.answer("10억원 세금", 1120, draws)
    count = engine.collector.client.call_count
    second = engine.answer("10억원 실수령액", 1120, draws)
    assert first["status"] == second["status"] == "answered"
    assert engine.collector.client.call_count == count


def test_tax_cites_two_external_source_types(environment, articles, draws):
    from src.guidance import NTS_URL

    guide = Article(
        document_id="nts_test",
        law_title="",
        article_no="",
        document_title="국세청 세율 안내",
        text="복권 당첨금의 소득금액이 3억원을 초과하는 경우\n30%",
        effective_date=None,
        source_url=NTS_URL,
        collected_at="test",
        verified_on=DAY,
        document_type="government_guidance",
        source_type="government_guidance",
        source_name="국세청",
    )

    class Guidance:
        def collect(self, as_of):
            ids = index_articles(
                [guide],
                environment.documents,
                environment.embeddings,
                environment.settings["chunking"],
            )
            return {"chunk_ids": ids, "status": "indexed"}

    engine = workflow(environment, articles, SearchPlan(intent="tax", prize_amount=1_000_000_000))
    engine.guidance = Guidance()
    result = engine.answer("10억원 실수령액", 1120, draws)
    assert result["status"] == "answered"
    assert {item["source_type"] for item in result["sources"]} == {
        "statute",
        "government_guidance",
    }
    assert "국세청 안내 원문" in result["answer"]
    assert result["calculation"]["net_amount"] == 703_000_000


def test_guidance_failure_does_not_replace_or_disable_statutes(environment, articles, draws):
    class Guidance:
        def collect(self, as_of):
            raise PublicError("guidance")

    engine = workflow(environment, articles, SearchPlan(intent="tax", prize_amount=1_000_000_000))
    engine.guidance = Guidance()
    result = engine.answer("10억원 실수령액", 1120, draws)
    assert result["status"] == "answered"
    assert result["calculation"]["net_amount"] == 703_000_000
    assert any("국세청" in note for note in result["limitations"])
