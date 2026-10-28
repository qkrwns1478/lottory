# F015: 같은 질문에 대한 모델 단독/RAG 답변을 미리 정한 정답 기준으로 비교한다.

import json
import re
import statistics
import time

from pydantic import BaseModel

from src.data import atomic_write
from src.generation import make_model
from src.query import Filters, filter_draws, frequency
from src.settings import ROOT


class BaselineAnswer(BaseModel):
    status: str
    answer: str


def make_baseline(settings):
    model = make_model(settings).with_structured_output(BaselineAnswer, method="function_calling")

    def baseline(question, selected_round, draws):
        # 비교 기준 모델에는 CSV·검색 결과·출처·도구·계산 결과를 제공하지 않는다.
        result = model.invoke(
            [
                (
                    "system",
                    "대한민국 로또 질문에 한국어로 답하세요. 자료나 도구는 제공되지 않습니다. 확인할 수 없으면 insufficient_evidence로 답하세요. 출처를 만들어 내지 마세요. status와 answer를 반환하세요.",
                ),
                (
                    "human",
                    json.dumps(
                        {"question": question, "selected_round": selected_round},
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
        return {"status": result.status, "answer": result.answer, "sources": []}

    return baseline


def evaluation_cases(draws):
    ordered = sorted(draws, key=lambda draw: draw.round, reverse=True)
    cases = []
    for draw in ordered[:5]:
        cases.append(
            {
                "id": f"round_{draw.round}",
                "question": f"{draw.round}회 1등 당첨금과 1등 당첨 게임 수는?",
                "kind": "round",
                "round": draw.round,
                "expected": {
                    "first_prize_amount": draw.first_prize_amount,
                    "games": draw.first_prize_winning_games,
                },
            }
        )
    for number in (1, 7, 22, 40, 45):
        result = frequency(draws, Filters(recent_rounds=100, numbers=[number]))
        cases.append(
            {
                "id": f"frequency_{number}",
                "question": f"최근 100회 중 {number}번이 몇 번 나왔어?",
                "kind": "frequency",
                "number": number,
                "expected": {"count": result["result"][str(number)], "rows": 100},
            }
        )
    for draw in ordered[:4]:
        start = draw.draw_date.replace(day=1)
        import calendar

        end = start.replace(day=calendar.monthrange(start.year, start.month)[1])
        number = draw.numbers[0]
        matches = filter_draws(draws, Filters(start_date=start, end_date=end, numbers=[number]))
        cases.append(
            {
                "id": f"filter_{draw.round}",
                "question": f"{start.year}년 {start.month}월에 {number}번이 나온 회차는?",
                "kind": "filter",
                "expected": {"rounds": sorted(row.round for row in matches)},
            }
        )
    for number in (9999, 9998, 9997):
        cases.append(
            {
                "id": f"absent_{number}",
                "question": f"{number}회 당첨금은?",
                "kind": "absent",
                "expected": {},
            }
        )
    cases.extend(
        [
            {
                "id": "tax_billion",
                "question": "10억 원의 예상 실수령액은?",
                "kind": "tax",
                "expected": {"net": 703000000},
            },
            {
                "id": "claim_deadline",
                "question": "당첨금 수령기한은?",
                "kind": "claim",
                "expected": {"law": "복권 및 복권기금법", "article": "9"},
            },
            {
                "id": "probability",
                "question": "자주 나온 번호를 고르면 다음에 더 유리해?",
                "kind": "probability",
                "expected": {},
            },
        ]
    )
    return cases


def score(case, response):
    answer = response.get("answer", "")
    normalized = answer.replace(",", "")
    calculation = response.get("calculation") or {}
    expected, kind = case["expected"], case["kind"]
    answered = response.get("status") == "answered"
    if kind == "round":
        correct = answered and all(str(value) in normalized for value in expected.values())
    elif kind == "frequency":
        correct = answered and (
            calculation.get("result", {}).get(str(case["number"])) == expected["count"]
            if calculation
            else bool(re.search(rf"{case['number']}번\s*{expected['count']}회", normalized))
        )
    elif kind == "filter":
        rounds = sorted(row["round"] for row in response.get("rows", []))
        correct = answered and (
            rounds == expected["rounds"]
            if "rows" in response
            else sorted(set(map(int, re.findall(r"(\d+)회", answer)))) == expected["rounds"]
        )
    elif kind == "absent":
        correct = response.get("status") == "insufficient_evidence" and not response.get("sources")
    elif kind == "tax":
        # 현행 과세 조건에 따른 정답 기준이며 어느 모델에도 기대값을 제공하지 않는다.
        correct = answered and (
            calculation.get("net_amount") == expected["net"]
            if calculation
            else str(expected["net"]) in normalized
        )
    elif kind == "claim":
        correct = answered and "1년" in answer and "지급개시일" in answer
    else:
        # 회차 자료가 없어 보류했더라도 개념 설명은 맞을 수 있다.
        # 상태 표시와 구분해 실제 답변 내용으로 개념 정확성을 채점한다.
        correct = "높아지지" in answer or "유리하지" in answer
    sources = response.get("sources", [])
    evidence = {item["evidence_id"]: item for item in response.get("evidence", [])}
    valid = sum(
        item.get("evidence_id") in evidence
        and item.get("source_url") == evidence[item["evidence_id"]]["metadata"].get("source_url")
        for item in sources
    )
    return {
        "correct": bool(correct),
        "cited": bool(sources),
        "valid_citations": valid,
        "citation_count": len(sources),
        "abstention_correct": bool(correct) if kind == "absent" else None,
        "status": response.get("status", "error"),
    }


def evaluate(draws, baseline, rag, cases=None, progress=lambda _: None):
    cases = cases or evaluation_cases(draws)
    rows = []
    for index, case in enumerate(cases, 1):
        record = {"case": case}
        for name, runner in (("baseline", baseline), ("rag", rag)):
            progress(f"평가 {index}/{len(cases)} · {name}")
            started = time.perf_counter()
            try:
                response = runner(case["question"], draws[0].round, draws)
                metrics = score(case, response)
                record[name] = {
                    **metrics,
                    "elapsed_seconds": time.perf_counter() - started,
                    "answer": response["answer"],
                    "sources": response.get("sources", []),
                }
            except Exception:
                record[name] = {
                    "correct": False,
                    "cited": False,
                    "status": "error",
                    "valid_citations": 0,
                    "citation_count": 0,
                    "abstention_correct": False if case["kind"] == "absent" else None,
                    "elapsed_seconds": time.perf_counter() - started,
                    "answer": "평가 실행 오류",
                    "sources": [],
                }
        rows.append(record)
    return {
        "case_count": len(rows),
        "summary": summarize_results(rows),
        "results": rows,
        "scoring_version": "v2",
        "methodology": (
            "비교 조건: 동일 모델·동일 질문\n"
            "baseline 입력: 질문·선택 회차\n"
            "RAG 사용 자료: CSV·법령·검색·계산\n"
            "채점: 공개된 규칙 기반 oracle, 개념 질문은 실제 답변 내용으로 판정\n"
            "인용률 분모: 근거 없는 회차 3문항을 포함한 전체 질문\n"
            "Python 계산·검증의 효과를 포함한 전체 경로를 비교한 결과이며 "
            "포괄적인 의미 정확도나 순수한 검색 효과를 측정한 결과는 아닙니다."
        ),
    }


def summarize_results(rows):
    summary = {}
    for name in ("baseline", "rag"):
        results = [row[name] for row in rows]
        cited = sum(row["citation_count"] for row in results)
        absent = [row for row in results if row["abstention_correct"] is not None]
        summary[name] = {
            "accuracy": sum(row["correct"] for row in results) / len(results),
            "citation_rate": sum(row["cited"] for row in results) / len(results),
            "citation_precision": sum(row["valid_citations"] for row in results) / cited
            if cited
            else None,
            "abstention_accuracy": sum(row["abstention_correct"] for row in absent) / len(absent)
            if absent
            else None,
            "median_seconds": statistics.median(row["elapsed_seconds"] for row in results),
            "max_seconds": max(row["elapsed_seconds"] for row in results),
            "errors": sum(row["status"] == "error" for row in results),
        }
    return summary


def rescore_saved_concepts(report):
    """모델을 다시 호출하지 않고 저장된 답변으로 개념 문항을 재채점한다."""
    for row in report["results"]:
        if row["case"]["kind"] == "probability":
            for method in ("baseline", "rag"):
                row[method]["correct"] = score(row["case"], row[method])["correct"]
    report["summary"] = summarize_results(report["results"])
    report["scoring_version"] = "v2"
    report["rescored_from_saved_answers"] = True
    report["methodology"] += (
        " 개념 문항은 전역 status가 아닌 실제 문장의 정확성을 기준으로 저장된 답변을 재채점했습니다. 인용률은 근거 없는 회차 3문항도 포함한 전체 질문 기준입니다."
    )
    return report


def save_evaluation(report):
    atomic_write(
        ROOT / "data/evaluation_latest.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )
