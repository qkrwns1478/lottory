import pytest
from pydantic import ValidationError

from src.planner import make_planner
from src.query import Filters, SearchPlan
from src.settings import load_settings


def test_missing_key_is_explicit(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_planner(load_settings())


@pytest.mark.parametrize(
    "question,expected",
    [
        ("1100회 당첨금은?", 1100),
        ("최근100회에서 22번 빈도", None),
        ("최근 100회에서 22번 빈도", None),
    ],
)
def test_structured_output_and_explicit_round_priority(monkeypatch, question, expected):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")

    class FakeChatModel:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "test-key-never-sent"

        def with_structured_output(self, schema, method):
            assert schema is SearchPlan
            return self

        def invoke(self, messages):
            assert len(messages) == 2
            return (
                SearchPlan(intent="round_lookup", filters=Filters(round=1120))
                if expected
                else SearchPlan(intent="statistics")
            )

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatModel)
    plan = make_planner(load_settings())(question, 1120, {"row_count": 200})
    assert plan.filters.round == expected


def test_invalid_explicit_round_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")

    class FakeChatModel:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            return SearchPlan(intent="round_lookup", filters=Filters(round=1120))

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatModel)
    with pytest.raises(ValidationError):
        make_planner(load_settings())("0회 당첨금", 1120, {})


def test_bad_model_json_uses_only_exact_supported_template(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline")

    class BadModel:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            return SearchPlan.model_validate({"intent": "invented"})

    monkeypatch.setattr("langchain_openai.ChatOpenAI", BadModel)
    planner = make_planner(load_settings())
    plan = planner("2026년 10월에 22번이 나온 회차는?", 1100, {})
    assert plan.filters.start_date.isoformat() == "2026-10-01"
    assert plan.filters.end_date.isoformat() == "2026-10-31"
    assert plan.filters.numbers == [22]
    with pytest.raises(ValidationError):
        planner("22번이 있으면서 3억원 넘는 회차", 1100, {})


def test_combined_tax_question_cannot_be_classified_as_plain_lookup(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test")

    class Model:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            return SearchPlan(intent="round_lookup", filters=Filters(round=1100))

    monkeypatch.setattr("langchain_openai.ChatOpenAI", Model)
    plan = make_planner(load_settings())("1100회 1등 당첨금과 세금은?", 1246, {})
    assert plan.intent == "tax"
    assert plan.need_external_docs and plan.need_exact_calculation
    assert plan.filters.round == 1100


@pytest.mark.parametrize("question,calculation,top_n", [
    ("최근 100회 중 가장 많이 나온 번호 6개는?", "frequency", 6),
    ("최근 100회 중 당첨금의 최대 금액과 최소 금액은?", "minmax", None),
])
def test_exact_statistics_override_incorrect_model_plan(monkeypatch, question, calculation, top_n):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test")

    class Model:
        def __init__(self, **kwargs):
            pass

        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            return SearchPlan(
                intent="statistics", calculation="maximum", filters=Filters(round=1100)
            )

    monkeypatch.setattr("langchain_openai.ChatOpenAI", Model)
    plan = make_planner(load_settings())(question, 1100, {})
    assert plan.calculation == calculation and plan.top_n == top_n
    assert plan.filters.recent_rounds == 100 and plan.filters.round is None
