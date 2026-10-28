# 모델 인증값 없이도 정확한 조회·계산과 서버가 연결한 출처를 제공한다.

from src.data import CSV_URL, OFFICIAL_URL
from src.query import (
    filter_draws,
    frequency,
    local_plan,
    prize_statistics,
    probability_calculation,
)

DISCLAIMER = (
    "과거 번호 빈도는 다음 회차의 당첨 확률을 높이지 않습니다. "
    "당첨금·세금·수령기한은 적용 법령과 동행복권 공식 안내에서 최종 확인하세요."
)


def coverage(draws):
    return {
        "from_round": min(draw.round for draw in draws),
        "to_round": max(draw.round for draw in draws),
        "row_count": len(draws),
        "start_date": min(draw.draw_date for draw in draws).isoformat(),
        "end_date": max(draw.draw_date for draw in draws).isoformat(),
    }


def source(title, evidence_id="csv_dataset"):
    return {
        "evidence_id": evidence_id,
        "document_title": title,
        "source_url": CSV_URL,
        "source_name": "캐시로지(CC BY 4.0 표시), 원천 발표 동행복권",
        "official_url": OFFICIAL_URL,
    }


def answer_question(
    question,
    selected_round,
    draws,
    planner=None,
    progress=None,
    engine=None,
    history=None,
):
    if engine is not None:
        return engine.answer(question, selected_round, draws, progress=progress, history=history)
    progress = progress or (lambda message: None)
    progress("질문 분석")
    bounds = coverage(draws)
    plan = (
        planner(question, selected_round, bounds)
        if planner
        else local_plan(question, selected_round)
    )
    response = {
        "status": "insufficient_evidence",
        "answer": "",
        "sources": [],
        "coverage": bounds,
        "rows": [],
        "calculation": None,
        "plan": plan.model_dump(mode="json"),
        "disclaimer": DISCLAIMER,
    }
    if plan.need_external_docs or plan.intent in ("tax", "claim_guide"):
        response["answer"] = (
            "제공된 근거에서 확인되지 않습니다. 법령 안내는 AI 분석 및 근거 기반 답변 모드에서 "
            "OPENAI_API_KEY와 LAW_API_OC를 설정한 후 이용하세요. 로컬 모드에서는 법령 안내를 보류합니다."
        )
        return response
    if plan.intent == "probability":
        calculation = probability_calculation()
        response.update(
            status="answered",
            answer=calculation["text"],
            calculation=calculation,
            sources=[
                {
                    **source("LOTTORY 작성 확률 계산 설명"),
                    "source_name": "LOTTORY 수학적 설명 (참고 회차 데이터: 캐시로지)",
                    "document_type": "structured_summary",
                }
            ],
        )
        return response
    if plan.intent == "unknown":
        response["answer"] = (
            "질문의 조건을 확인할 수 없습니다. ‘1100회 당첨금은?’, "
            "‘최근 100회 중 22번이 몇 번 나왔어?’, "
            "‘2026년 9월에 22번이 나온 회차는?’ 형식을 사용하거나 조건 탐색 화면을 이용하세요."
        )
        return response
    progress("CSV 근거 조회 및 정확 계산")
    if plan.intent == "statistics":
        calculation = (
            frequency(draws, plan.filters, top_n=plan.top_n)
            if plan.calculation == "frequency"
            else prize_statistics(draws, plan.filters, plan.calculation)
        )
        response["calculation"] = calculation
        if not calculation["row_count"]:
            response["answer"] = "요청한 범위의 데이터가 없어 통계를 계산할 수 없습니다."
            return response
        if plan.calculation == "frequency":
            result = ", ".join(
                f"{number}번 {count}회" for number, count in calculation["result"].items()
            )
            response["answer"] = (
                f"실제 {calculation['row_count']}개 회차의 당첨번호 6개를 집계했습니다. (보너스 제외)\n{result}"
            )
            if plan.top_n is not None:
                span = calculation["data_range"]
                response["answer"] = (
                    f"{span['from_round']}~{span['to_round']}회, 총 {calculation['row_count']}개 회차의 "
                    f"본번호를 집계한 빈도 상위 {len(calculation['result'])}개 번호 (보너스 제외)\n"
                    f"{result}\n동률이면 번호 오름차순으로 정렬해 요청한 개수만 표시합니다.\n"
                    "과거 빈도가 다음 회차의 당첨 확률을 높이지는 않습니다."
                )
        elif plan.calculation == "minmax":
            span = calculation["data_range"]
            amounts = calculation["result"]["first_prize_amount"]
            response["answer"] = (
                f"{span['from_round']}~{span['to_round']}회, 총 {calculation['row_count']}개 회차의 "
                "1게임당 1등 당첨금 기준입니다.\n"
                f"최대 금액: {amounts['maximum']:,}원\n"
                f"최소 금액: {amounts['minimum']:,}원"
            )
        else:
            label = {"maximum": "최대", "minimum": "최소", "average": "평균"}[plan.calculation]
            response["answer"] = (
                f"{calculation['row_count']}개 회차의 1게임당 1등 당첨금 {label}: {calculation['result']['first_prize_amount']}원"
            )
        response["sources"] = [source("회차 CSV 및 Python 통계 집계")]
    else:
        matches = filter_draws(draws, plan.filters)
        if not matches:
            response["answer"] = "현재 적재한 CSV 범위에서 조건에 맞는 회차가 확인되지 않습니다."
            return response
        response["rows"] = [draw.model_dump(mode="json") for draw in matches]
        if plan.intent == "round_lookup":
            response["answer"] = "\n\n".join(draw.search_text() for draw in matches)
            response["sources"] = [
                source(f"로또 6/45 제{draw.round}회 회차 기록", f"lotto_{draw.round}")
                for draw in matches
            ]
        else:
            response["answer"] = (
                f"조건에 일치하는 회차는 {len(matches)}개입니다. "
                "아래 결과는 추출한 날짜·번호·금액 등의 조건을 모두 만족합니다."
            )
            response["sources"] = [source("조건 탐색 대상 회차 CSV")]
    progress("출처 확인 및 답변 작성")
    response["status"] = "answered"
    return response
