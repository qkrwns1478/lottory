# 검색 계획 검증과 벡터 검색에 의존하지 않는 정확한 필터·집계

import calendar
import math
import re
from collections import Counter
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data import DataError, Draw


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round: int | None = Field(default=None, gt=0)
    start_date: date | None = None
    end_date: date | None = None
    numbers: list[int] = Field(default_factory=list)
    min_prize: int | None = Field(default=None, ge=0)
    max_prize: int | None = Field(default=None, ge=0)
    odd_count: int | None = Field(default=None, ge=0, le=6)
    min_sum: int | None = Field(default=None, ge=21, le=255)
    max_sum: int | None = Field(default=None, ge=21, le=255)
    min_consecutive_pairs: int | None = Field(default=None, ge=0, le=5)
    recent_rounds: int | None = Field(default=None, gt=0, le=10000)

    @model_validator(mode="after")
    def validate_ranges(self):
        if any(not 1 <= number <= 45 for number in self.numbers):
            raise ValueError("번호는 1~45 사이여야 합니다.")
        if len(set(self.numbers)) != len(self.numbers):
            raise ValueError("번호 조건이 중복됩니다.")
        for lower, upper in (
            (self.start_date, self.end_date),
            (self.min_prize, self.max_prize),
            (self.min_sum, self.max_sum),
        ):
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("범위의 시작은 끝보다 클 수 없습니다.")
        return self


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Literal[
        "round_lookup", "statistics", "search", "claim_guide", "tax", "probability", "unknown"
    ]
    filters: Filters = Field(default_factory=Filters)
    search_queries: list[str] = Field(default_factory=list, max_length=4)
    need_external_docs: bool = False
    need_exact_calculation: bool = False
    as_of: date | None = None
    prize_amount: int | None = Field(default=None, ge=0)
    calculation: Literal["frequency", "maximum", "minimum", "minmax", "average"] = "frequency"
    top_n: int | None = Field(default=None, ge=1, le=45)

    @model_validator(mode="after")
    def require_lookup_round(self):
        if self.intent == "round_lookup" and self.filters.round is None:
            raise ValueError("정확 회차 조회에는 회차 번호가 필요합니다.")
        if self.top_n is not None and (
            self.intent != "statistics" or self.calculation != "frequency"
        ):
            raise ValueError("상위 번호 개수는 번호 빈도 통계에서만 사용할 수 있습니다.")
        return self


def filter_draws(draws: list[Draw], filters: Filters) -> list[Draw]:
    ordered = sorted(draws, key=lambda draw: draw.round, reverse=True)
    # "최근 N회 중 조건"은 최근 N회를 먼저 선택한 뒤 다른 조건을 적용한다.
    if filters.recent_rounds:
        if len(ordered) < filters.recent_rounds:
            raise DataError(
                f"최근 {filters.recent_rounds}회 통계에 필요한 자료가 부족합니다. 현재 자료는 {len(ordered)}회분입니다."
            )
        ordered = ordered[: filters.recent_rounds]
        if any(a.round - b.round != 1 for a, b in zip(ordered, ordered[1:])):
            raise DataError("최근 회차 범위에 누락된 회차가 있어 계산을 보류합니다.")
    return [
        draw
        for draw in ordered
        if (
            (filters.round is None or draw.round == filters.round)
            and (filters.start_date is None or draw.draw_date >= filters.start_date)
            and (filters.end_date is None or draw.draw_date <= filters.end_date)
            and set(filters.numbers).issubset(draw.numbers)
            and (filters.min_prize is None or draw.first_prize_amount >= filters.min_prize)
            and (filters.max_prize is None or draw.first_prize_amount <= filters.max_prize)
            and (filters.odd_count is None or draw.odd_count == filters.odd_count)
            and (filters.min_sum is None or sum(draw.numbers) >= filters.min_sum)
            and (filters.max_sum is None or sum(draw.numbers) <= filters.max_sum)
            and (
                filters.min_consecutive_pairs is None
                or draw.consecutive_pairs >= filters.min_consecutive_pairs
            )
        )
    ]


def frequency(draws: list[Draw], filters: Filters, top_n: int | None = None):
    if top_n is not None and (
        isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 45
    ):
        raise ValueError("상위 번호 개수는 1~45의 정수여야 합니다.")
    # 번호 조건은 집계할 값이며 집계 대상 회차를 미리 거르는 조건이 아니다.
    population = filter_draws(draws, filters.model_copy(update={"numbers": []}))
    counts = Counter(number for draw in population for number in draw.numbers)
    numbers = filters.numbers or range(1, 46)
    if top_n is not None:
        numbers = sorted(numbers, key=lambda number: (-counts[number], number))[:top_n]
    result = {
        "calculation_type": "frequency",
        "filters": filters.model_dump(mode="json"),
        "row_count": len(population),
        "data_range": {
            "from_round": min((d.round for d in population), default=None),
            "to_round": max((d.round for d in population), default=None),
        },
        "result": {str(number): counts[number] for number in numbers},
        "source_round_ids": [f"lotto_{draw.round}" for draw in population],
    }
    if top_n is not None:
        result.update(top_n=top_n, tie_break="number_ascending")
    return result


def prize_statistics(draws, filters, operation):
    population = filter_draws(draws, filters)
    if not population:
        raise DataError("계산 대상 회차가 없습니다.")
    amounts = [draw.first_prize_amount for draw in population]
    if operation == "maximum":
        value = max(amounts)
    elif operation == "minimum":
        value = min(amounts)
    elif operation == "average":
        value = str(Decimal(sum(amounts)) / Decimal(len(amounts)))
    elif operation == "minmax":
        value = {"maximum": max(amounts), "minimum": min(amounts)}
    else:
        raise DataError("지원하지 않는 집계입니다.")
    return {
        "calculation_type": operation,
        "filters": filters.model_dump(mode="json"),
        "row_count": len(population),
        "result": {"first_prize_amount": value},
        "data_range": {
            "from_round": min(d.round for d in population),
            "to_round": max(d.round for d in population),
        },
        "source_round_ids": [f"lotto_{draw.round}" for draw in population],
    }


def probability_calculation():
    combinations = math.comb(45, 6)
    return {
        "calculation_type": "probability",
        "combinations": combinations,
        "assumptions": [
            "매 회차 1~45에서 서로 다른 6개를 균등하고 독립적으로 추첨한다는 수학적 가정"
        ],
        "text": (
            f"균등하고 독립적인 추첨을 가정합니다. 가능한 6개 번호 조합은 45C6 = {combinations:,}개입니다.\n"
            f"특정 조합의 다음 회차 1등 확률은 1/{combinations:,}이며 모든 조합의 확률이 같습니다.\n"
            "독립성 가정에서 과거에 자주 나온 번호를 선택해도 다음 추첨 확률은 높아지지 않습니다.\n"
            "이는 가정에 따른 수학적 설명이며 과거 빈도로 미래 결과를 예측한 값이 아닙니다."
        ),
    }


def local_plan(question: str, selected_round: int | None = None) -> SearchPlan:
    """지원하는 질문 형식은 로컬에서 해석하고 일반 자연어는 모델이 작성한 검색 계획을 사용한다."""
    question = question.strip()
    if any(word in question for word in ("세금", "세율", "실수령", "수령", "기한", "지급", "법령")):
        tax = any(w in question for w in ("세금", "세율", "실수령"))
        explicit = re.search(r"(\d+)회", question)
        amount = re.search(r"([\d,]+)\s*(억|만)?\s*원", question)
        prize = (
            int(amount.group(1).replace(",", ""))
            * {None: 1, "억": 100_000_000, "만": 10_000}[amount.group(2)]
            if amount
            else None
        )
        return SearchPlan(
            intent="tax" if tax else "claim_guide",
            need_external_docs=True,
            need_exact_calculation=tax,
            prize_amount=prize,
            filters=Filters(round=int(explicit.group(1)) if explicit else None),
            search_queries=["소득세법", "지방세법"] if tax else ["복권 및 복권기금법"],
        )
    if any(word in question for word in ("확률", "유리", "예측", "추천")):
        return SearchPlan(intent="probability")
    # 지원하는 질문 형식만 처리하고 알 수 없는 조건을 임의로 제거하지 않는다.
    prize = re.fullmatch(
        r"최근\s*(\d+)\s*회(?:차)?\s*(?:중|에서)?\s*(?:1등\s*)?당첨금(?:의)?\s*"
        r"((?:최대|최소|평균)(?:\s*금액)?(?:\s*(?:과|와|및|하고|,|·)\s*(?:최대|최소)(?:\s*금액)?)?)"
        r"\s*(?:은|는|을|를)?\s*(?:얼마(?:야|인가요)?|알려줘|알려주세요)?[?？]?",
        question,
    )
    if prize:
        operations = re.findall(r"최대|최소|평균", prize.group(2))
        if len(operations) == 1 or set(operations) == {"최대", "최소"}:
            return SearchPlan(
                intent="statistics",
                filters=Filters(recent_rounds=int(prize.group(1))),
                calculation="minmax" if len(operations) == 2 else {
                    "최대": "maximum", "최소": "minimum", "평균": "average"
                }[operations[0]],
                need_exact_calculation=True,
            )
    month = re.fullmatch(
        r"(\d{4})년\s*(\d{1,2})월에?\s*(\d+)번이?\s*(?:나온|포함된)\s*회차(?:는)?[?？]?", question
    )
    if month:
        year, month_number, number = map(int, month.groups())
        return SearchPlan(
            intent="search",
            filters=Filters(
                start_date=date(year, month_number, 1),
                end_date=date(year, month_number, calendar.monthrange(year, month_number)[1]),
                numbers=[number],
            ),
        )
    top_frequency = re.fullmatch(
        r"최근\s*(\d+)\s*회(?:차)?\s*(?:중|에서)?\s*가장\s*(?:많이|자주)\s*"
        r"(?:나온|출현한)\s*번호\s*(\d+)\s*개(?:는|를\s*알려줘|\s*알려줘)?[?？]?",
        question,
    )
    if top_frequency:
        count, top_n = map(int, top_frequency.groups())
        return SearchPlan(
            intent="statistics",
            filters=Filters(recent_rounds=count),
            need_exact_calculation=True,
            top_n=top_n,
        )
    recent = re.fullmatch(
        r"최근\s*(\d+)회(?:에서|\s*중)?\s*(\d+)번(?:은|이)?\s*(?:몇\s*번\s*(?:나왔어|나왔나요|나왔는지)|빈도)[?？]?",
        question,
    )
    if recent:
        count, number = map(int, recent.groups())
        return SearchPlan(
            intent="statistics",
            filters=Filters(recent_rounds=count, numbers=[number]),
            need_exact_calculation=True,
        )
    lookup = re.fullmatch(
        r"(?:(\d+)회\s*)?(?:당첨번호|1등\s*당첨금|당첨금|1등\s*당첨\s*게임\s*수)(?:와|과|및|\s|당첨번호|1등|당첨금|당첨|게임|수)*(?:는|은)?[?？]?",
        question,
    )
    if lookup:
        round_number = int(lookup.group(1)) if lookup.group(1) else selected_round
        if round_number:
            return SearchPlan(intent="round_lookup", filters=Filters(round=round_number))
    return SearchPlan(intent="unknown")
