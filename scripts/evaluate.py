# 실행: python -m scripts.evaluate
# 20문항에 대해 두 경로에서 실제 모델을 호출해 비교한다.

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.data import load_draws
from src.evaluation import evaluate, make_baseline, save_evaluation
from src.settings import ROOT, load_settings, resolve_path
from src.workflow import make_workflow


def main():
    load_dotenv(ROOT / ".env")
    settings = load_settings()
    draws = load_draws(resolve_path(settings["data"]["csv_path"]))
    engine = make_workflow(settings)
    report = evaluate(
        draws, make_baseline(settings), engine.answer, progress=lambda step: print(step, flush=True)
    )
    report.update(
        model=settings["llm"]["model"],
        retrieval=settings["retrieval"],
        evaluated_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        data_range={
            "from_round": min(d.round for d in draws),
            "to_round": max(d.round for d in draws),
        },
    )
    save_evaluation(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
