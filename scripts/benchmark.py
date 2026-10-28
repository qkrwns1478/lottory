# 대표 질문 20개 이상의 응답시간을 측정하고 결과를 data/에 저장한다.

import argparse
import json
import math
import statistics
import time

from dotenv import load_dotenv

from src.data import atomic_write, load_draws
from src.service import answer_question
from src.settings import ROOT, load_settings, resolve_path

QUESTIONS = [
    "1100회 당첨금은?",
    "1101회 당첨번호는?",
    "1102회 1등 당첨 게임 수는?",
    "1246회 당첨금은?",
    "당첨번호는?",
    "최근 100회 중 22번이 몇 번 나왔어?",
    "최근 50회 중 1번이 몇 번 나왔어?",
    "최근 200회 중 45번이 몇 번 나왔어?",
    "최근 10회 중 7번이 몇 번 나왔어?",
    "최근 100회 중 6번이 몇 번 나왔어?",
    "2026년 9월에 22번이 나온 회차는?",
    "2026년 10월에 22번이 나온 회차는?",
    "2025년 1월에 1번이 나온 회차는?",
    "2024년 3월에 7번이 나온 회차는?",
    "9999회 당첨금은?",
    "당첨금 수령기한은?",
    "10억 원의 예상 실수령액은?",
    "1100회 1등 당첨금과 세금은?",
    "자주 나온 번호를 고르면 다음에 더 유리해?",
    "2026년 9월에 45번이 나온 회차는?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag", action="store_true", help="실제 OpenAI/법령 API를 사용하는 RAG 측정")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    settings = load_settings()
    draws = load_draws(resolve_path(settings["data"]["csv_path"]))
    engine = None
    if args.rag:
        from src.workflow import make_workflow

        engine = make_workflow(settings)
    cases = []
    for question in QUESTIONS:
        started = time.perf_counter()
        try:
            result = answer_question(question, 1100, draws, engine=engine)
            status = result["status"]
        except Exception as exc:
            status = "error"
            result = {"error_type": type(exc).__name__}
        cases.append(
            {
                "question": question,
                "status": status,
                "elapsed_seconds": time.perf_counter() - started,
                "plan": result.get("plan"),
                "error_type": result.get("error_type"),
            }
        )
    times = sorted(case["elapsed_seconds"] for case in cases)
    report = {
        "mode": "live_rag" if args.rag else "local_csv",
        "count": len(cases),
        "median_seconds": statistics.median(times),
        "p95_seconds": times[math.ceil(len(times) * 0.95) - 1],
        "max_seconds": max(times),
        "cases": cases,
    }
    atomic_write(
        ROOT / f"data/benchmark_{report['mode']}.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
