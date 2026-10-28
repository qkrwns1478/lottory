# 실제 API로 F2014/F2016을 검증하고 인증값을 제외한 결과를 저장한다.

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.data import atomic_write, load_draws
from src.retrieval import compare_search
from src.settings import ROOT, load_settings, resolve_path
from src.workflow import make_workflow


def main():
    load_dotenv(ROOT / ".env")
    settings = load_settings()
    draws = load_draws(resolve_path(settings["data"]["csv_path"]))
    engine = make_workflow(settings)
    report = {}
    for method in ("similarity", "mmr"):
        engine.settings["retrieval"]["search_type"] = method
        result = engine.answer("10억 원의 예상 실수령액은?", 1100, draws)
        assert result["status"] == "answered", f"{method}: 답변 실패"
        assert result["calculation"]["net_amount"] == 703000000, f"{method}: 계산 불일치"
        assert {"statute", "government_guidance"} <= {
            source.get("source_type") for source in result["sources"]
        }, f"{method}: 복수 출처 인용 누락"
        report[method] = {
            key: result.get(key)
            for key in (
                "status",
                "answer",
                "sources",
                "calculation",
                "audit",
                "elapsed_seconds",
                "limitations",
            )
        }
        print(
            json.dumps(
                {
                    "method": method,
                    "status": result["status"],
                    "source_types": sorted(
                        {source.get("source_type", "round_csv") for source in result["sources"]}
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    report["comparison"] = compare_search(
        "복권 당첨금 세금과 원천징수", engine, datetime.now(ZoneInfo("Asia/Seoul")).date()
    )
    atomic_write(
        ROOT / "data/verification_advanced.json", json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                name: {
                    method: {
                        key: values[method].get(key)
                        for key in ("selected_ids", "diversity", "source_type_count")
                    }
                    for method in ("similarity", "mmr")
                }
                for name, values in report["comparison"]["collections"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
