from datetime import date

import pytest
from pydantic import ValidationError

from src.data import DataError
from src.query import Filters, SearchPlan, filter_draws, frequency, local_plan
from src.service import answer_question


@pytest.mark.parametrize(
    "values",
    [
        {"numbers": [46]},
        {"numbers": [0]},
        {"numbers": [22, 22]},
        {"round": -1},
        {"recent_rounds": 0},
        {"start_date": "2026-10-22", "end_date": "2026-01-01"},
        {"min_prize": 100, "max_prize": 99},
        {"odd_count": 7},
        {"surprise": True},
    ],
)
def test_invalid_filters(values):
    with pytest.raises(ValidationError):
        Filters(**values)


def test_window_before_number_filter_and_bonus_excluded(draws):
    result = frequency(draws, Filters(recent_rounds=100, numbers=[22, 7]))
    assert result["row_count"] == 100
    assert result["result"] == {"22": 50, "7": 0}
    assert result["data_range"] == {"from_round": 1021, "to_round": 1120}
    assert len(result["source_round_ids"]) == 100
    matches = filter_draws(draws, Filters(recent_rounds=10, numbers=[22]))
    assert len(matches) == 5
    assert min(draw.round for draw in matches) >= 1111


@pytest.mark.parametrize(
    "question",
    [
        "최근 100회 중 가장 많이 나온 번호 6개는?",
        "최근100회에서 가장 자주 나온 번호6개 알려줘",
        "최근 100회차 중 가장 많이 출현한 번호 6개를 알려줘",
    ],
)
def test_top_frequency_plan_keeps_window_and_requested_count(question):
    plan = local_plan(question, 1100)
    assert plan.intent == "statistics" and plan.calculation == "frequency"
    assert plan.filters.recent_rounds == 100 and plan.filters.round is None
    assert plan.filters.numbers == [] and plan.top_n == 6


def test_top_frequency_uses_full_window_and_breaks_ties_by_number(draws):
    response = answer_question("최근 100회 중 가장 많이 나온 번호 6개는?", 1100, draws)
    assert response["status"] == "answered"
    calculation = response["calculation"]
    assert list(calculation["result"].items()) == [(str(n), 50) for n in range(1, 7)]
    assert calculation["row_count"] == 100
    assert calculation["data_range"] == {"from_round": 1021, "to_round": 1120}
    assert len(calculation["source_round_ids"]) == 100
    assert "보너스 제외" in response["answer"] and "동률" in response["answer"]
    assert response["sources"]
    with pytest.raises(DataError, match="부족"):
        answer_question("최근 100회 중 가장 많이 나온 번호 6개는?", 1100, draws[:99])


@pytest.mark.parametrize("count", [0, 46, -1])
def test_top_frequency_rejects_invalid_counts(draws, count):
    with pytest.raises(ValidationError):
        SearchPlan(intent="statistics", top_n=count)
    with pytest.raises(ValueError):
        frequency(draws, Filters(recent_rounds=100), top_n=count)


def test_top_frequency_does_not_silently_drop_extra_conditions():
    assert (
        local_plan("최근 100회 중 22번을 제외하고 가장 많이 나온 번호 6개는?").intent == "unknown"
    )


def test_missing_or_insufficient_recent_data(draws):
    with pytest.raises(DataError, match="부족"):
        frequency(draws, Filters(recent_rounds=121))
    with pytest.raises(DataError, match="누락"):
        frequency([draw for draw in draws if draw.round != 1110], Filters(recent_rounds=100))


@pytest.mark.parametrize("question", [
    "최근 100회 중 당첨금의 최대 금액과 최소 금액은?",
    "최근100회차에서 1등 당첨금 최대와 최소를 알려줘",
    "최근 100회 당첨금 최소 및 최대는?",
])
def test_recent_prize_extremes_answer_both_values(question, draws):
    # 범위 밖의 큰 금액과 화면 선택 회차가 최근 100회 집계에 섞이지 않아야 한다.
    draws[0] = draws[0].model_copy(update={"first_prize_amount": 9_000_000_000})
    response = answer_question(question, 1100, draws)
    assert response["status"] == "answered"
    assert response["plan"]["calculation"] == "minmax"
    assert response["plan"]["filters"]["round"] is None
    calculation = response["calculation"]
    assert calculation["result"]["first_prize_amount"] == {
        "maximum": 1_119_000_000, "minimum": 1_020_000_000,
    }
    assert calculation["row_count"] == 100
    assert calculation["data_range"] == {"from_round": 1021, "to_round": 1120}
    assert len(calculation["source_round_ids"]) == 100
    assert "최대 금액: 1,119,000,000원" in response["answer"]
    assert "최소 금액: 1,020,000,000원" in response["answer"]
    assert "1게임당 1등" in response["answer"] and response["sources"]
    with pytest.raises(DataError, match="부족"):
        answer_question(question, 1100, draws[:99])
    with pytest.raises(DataError, match="누락"):
        answer_question(question, 1100, [d for d in draws if d.round != 1110])


@pytest.mark.parametrize("label,operation", [("최대", "maximum"), ("최소", "minimum"), ("평균", "average")])
def test_recent_single_prize_statistic(label, operation, draws):
    response = answer_question(f"최근 100회 중 당첨금의 {label} 금액은?", 1100, draws)
    assert response["status"] == "answered"
    assert response["plan"]["calculation"] == operation
    assert response["calculation"]["row_count"] == 100


def test_prize_template_does_not_drop_unsupported_conditions():
    assert local_plan("최근 100회 중 22번을 제외한 당첨금의 최대 금액과 최소 금액은?").intent == "unknown"


def test_filters_are_combined(draws):
    filters = Filters(
        start_date=date(2024, 6, 1),
        end_date=date(2024, 12, 31),
        numbers=[22],
        min_prize=1_025_000_000,
        max_prize=1_045_000_000,
        odd_count=1,
        min_sum=160,
        max_sum=170,
        min_consecutive_pairs=0,
    )
    result = filter_draws(draws, filters)
    assert result
    assert all(
        1026 <= draw.round <= 1046 and 22 in draw.numbers and draw.odd_count == 1 for draw in result
    )
    assert filter_draws(draws, filters.model_copy(update={"min_consecutive_pairs": 1})) == []


@pytest.mark.parametrize(
    "question,round_number",
    [
        ("1100회 당첨금은?", 1100),
        ("당첨번호는?", 1120),
        ("1100회 1등 당첨금과 1등 당첨 게임 수는?", 1100),
    ],
)
def test_lookup_templates_and_selected_round(question, round_number):
    plan = local_plan(question, 1120)
    assert plan.intent == "round_lookup"
    assert plan.filters.round == round_number


def test_local_month_and_recent_templates():
    plan = local_plan("2026년 9월에 22번이 나온 회차는?", 1120)
    assert plan.filters.start_date == date(2026, 9, 1)
    assert plan.filters.end_date == date(2026, 9, 30)
    assert plan.filters.numbers == [22]
    assert plan.filters.round is None
    plan = local_plan("최근 100회 중 22번이 몇 번 나왔어?", 1120)
    assert plan.intent == "statistics"
    assert plan.filters.recent_rounds == 100


def test_unsupported_conditions_are_not_silently_dropped():
    for question in (
        "최근 100회 중 22번이 몇 번 나왔어? 단, 홀수 3개",
        "1100회 최대 당첨금",
        "아무 질문",
    ):
        assert local_plan(question, 1120).intent == "unknown"


def test_exact_service_and_unavailable_draw(draws):
    response = answer_question("1100회 당첨금은?", 1120, draws)
    expected = next(draw for draw in draws if draw.round == 1100)
    assert response["status"] == "answered"
    assert f"{expected.first_prize_amount:,}원" in response["answer"]
    assert response["sources"][0]["evidence_id"] == "lotto_1100"
    absent = answer_question("9999회 당첨금은?", 1120, draws)
    assert absent["status"] == "insufficient_evidence"
    assert absent["sources"] == []


def test_statistical_service_and_legal_deferral(draws):
    response = answer_question("최근 100회 중 22번이 몇 번 나왔어?", 1120, draws)
    assert response["calculation"]["result"] == {"22": 50}
    assert response["calculation"]["row_count"] == 100
    for question in ("10억 원의 예상 실수령액은?", "당첨금 수령기한은?"):
        response = answer_question(question, 1120, draws)
        assert response["status"] == "insufficient_evidence"
        assert response["sources"] == []
        assert "보류" in response["answer"]


def test_injected_structured_planner(draws):
    def planner(question, selected_round, bounds):
        assert bounds["row_count"] == 120
        return SearchPlan(intent="search", filters=Filters(numbers=[22], min_prize=1_100_000_000))

    result = answer_question("22번 포함, 11억 이상", 1120, draws, planner=planner)
    assert result["status"] == "answered"
    assert all(
        row["first_prize_amount"] >= 1_100_000_000 and 22 in row["numbers"]
        for row in result["rows"]
    )


def test_probability_explanation_is_a_labeled_calculation(draws):
    result = answer_question("자주 나온 번호를 고르면 다음에 더 유리해?", 1120, draws)
    assert result["status"] == "answered"
    assert result["calculation"]["combinations"] == 8_145_060
    assert "가정" in result["answer"]
    assert result["sources"][0]["document_type"] == "structured_summary"
