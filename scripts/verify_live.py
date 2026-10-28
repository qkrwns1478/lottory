# 실제 API의 대표 경로를 검증하며 민감한 내부 예외는 출력하지 않는다.

import argparse
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.contracts import PublicError
from src.data import atomic_write, load_draws
from src.settings import ROOT, load_settings, resolve_path
from src.workflow import make_workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", action="append", help="지정한 질문만 검증 (여러 번 지정 가능)")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    settings = load_settings()
    report = {
        "checked_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "cases": [],
        "credentials_present": {key: bool(os.getenv(key)) for key in ("OPENAI_API_KEY", "LAW_API_OC")},
    }
    try:
        draws = load_draws(resolve_path(settings["data"]["csv_path"]))
        engine = make_workflow(settings)
        questions = [
            "1100회 1등 당첨금과 1등 당첨 게임 수는?",
            "최근 100회 중 22번이 몇 번 나왔어?",
            "9999회 당첨금은?",
        ]
        if os.getenv("LAW_API_OC"):
            questions.extend(
                ["당첨금 수령기한은?", "10억 원의 예상 실수령액은?", "10억 원의 예상 실수령액은?"]
            )
        if args.question:
            questions = args.question
        for question in questions:
            before = engine.collector.client.call_count
            try:
                result = engine.answer(question, 1100, draws)
                report["cases"].append(
                    {
                        "question": question,
                        "status": result["status"],
                        "answer": result["answer"],
                        "source_count": len(result["sources"]),
                        "elapsed_seconds": result["elapsed_seconds"],
                        "law_api_calls": engine.collector.client.call_count - before,
                        "calculation": result["calculation"],
                        "limitations": result.get("limitations", []),
                        "plan": result["plan"],
                        "sources": result["sources"],
                    }
                )
            except Exception as exc:
                report["cases"].append(
                    {
                        "question": question,
                        "status": "error",
                        "error_code": exc.code
                        if isinstance(exc, PublicError)
                        else "api_or_configuration_error",
                        "error_type": type(exc).__name__,
                        "http_status": getattr(exc, "status_code", None),
                    }
                )
            atomic_write(
                ROOT / "data/verification_live.json",
                json.dumps(report, ensure_ascii=False, indent=2),
            )
            print(
                json.dumps(
                    {"case_done": question, "status": report["cases"][-1]["status"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    except Exception as exc:
        report["setup_error"] = (
            exc.code if isinstance(exc, PublicError) else "api_or_configuration_error"
        )
    atomic_write(
        ROOT / "data/verification_live.json", json.dumps(report, ensure_ascii=False, indent=2)
    )
    stamp = report["checked_at"].replace(":", "-").replace("+", "_")
    atomic_write(
        ROOT / f"data/verification_history/{stamp}.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    summary = {
        **report,
        "cases": [
            {
                key: value
                for key, value in case.items()
                if key
                in {
                    "question",
                    "status",
                    "source_count",
                    "elapsed_seconds",
                    "law_api_calls",
                    "error_code",
                }
            }
            for case in report["cases"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
