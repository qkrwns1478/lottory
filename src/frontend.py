# 기존 Streamlit 실행 방식으로 직접 구현한 화면과 Python을 양방향 연결한다.

import copy
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from src.contracts import PublicError
from src.evaluation import evaluate, make_baseline, save_evaluation
from src.query import Filters, filter_draws
from src.recommendation import recommend_numbers
from src.retrieval import compare_search
from src.service import answer_question
from src.settings import ROOT


class FrontendEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=100)
    action: str
    question: str = Field(default="", max_length=2000)
    selected_round: int | None = Field(default=None, gt=0)
    ai: bool = False
    filters: Filters = Field(default_factory=Filters)
    search_type: str = "similarity"
    fetch_k: int = Field(default=20, ge=5, le=100)
    lambda_mult: float = Field(default=0.5, ge=0, le=1)
    recommendation_window: int = Field(default=100)
    game_count: int = Field(default=5, ge=1, le=5)


def handle_event(value, settings, draws, engine_factory, history=None, progress=lambda _: None):
    event = FrontendEvent.model_validate(value)
    if event.action not in {
        "ask",
        "filter",
        "compare",
        "evaluate",
        "clear",
        "recommend",
    } or event.search_type not in {"similarity", "mmr"}:
        raise ValueError("화면 요청을 확인하세요.")
    if event.selected_round is not None and event.selected_round not in {
        draw.round for draw in draws
    }:
        raise ValueError("선택한 회차가 적재 범위에 없습니다.")
    config = copy.deepcopy(settings)
    config["retrieval"].update(
        search_type=event.search_type,
        mmr_fetch_k=event.fetch_k,
        mmr_lambda_mult=event.lambda_mult,
    )
    if event.action == "clear":
        return {"id": event.id, "kind": "clear"}
    if event.action == "recommend":
        return {
            "id": event.id,
            "kind": "recommendation",
            "recommendation": recommend_numbers(
                draws, window=event.recommendation_window, game_count=event.game_count
            ),
        }
    if event.action == "filter":
        rows = filter_draws(draws, event.filters)
        return {
            "id": event.id,
            "kind": "filter",
            "rows": [draw.model_dump(mode="json") for draw in rows],
            "filters": event.filters.model_dump(mode="json"),
        }
    if event.action == "ask":
        if not event.question.strip():
            raise ValueError("질문을 입력하세요.")
        if event.ai and not os.getenv("OPENAI_API_KEY"):
            raise PublicError("openai_key")
        engine = engine_factory(config) if event.ai else None
        result = answer_question(
            event.question,
            event.selected_round,
            draws,
            engine=engine,
            history=history,
            progress=progress,
        )
        return {
            "id": event.id,
            "kind": "answer",
            "question": event.question,
            "response": result,
        }
    engine = engine_factory(config)
    if event.action == "compare":
        if not event.question.strip():
            raise ValueError("비교할 검색어를 입력하세요.")
        result = compare_search(
            event.question,
            engine,
            engine.today(),
            fetch_k=event.fetch_k,
            lambda_mult=event.lambda_mult,
        )
        return {"id": event.id, "kind": "comparison", "comparison": result}
    report = evaluate(draws, make_baseline(config), engine.answer, progress=progress)
    report.update(
        model=config["llm"]["model"],
        retrieval=config["retrieval"],
        evaluated_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        data_range={
            "from_round": min(draw.round for draw in draws),
            "to_round": max(draw.round for draw in draws),
        },
    )
    save_evaluation(report)
    return {"id": event.id, "kind": "evaluation", "evaluation": report}


def render_frontend(settings, draws, engine_factory):
    import streamlit as st
    import streamlit.components.v1 as components

    component = components.declare_component("lottory_frontend", path=str(ROOT / "frontend"))
    messages = st.session_state.setdefault("custom_messages", [])
    report_path = ROOT / "data/evaluation_latest.json"
    try:
        evaluation = (
            json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        )
    except (ValueError, OSError):
        evaluation = None
    event = component(
        rounds=[draw.round for draw in draws],
        messages=messages,
        selected_round=st.session_state.get("custom_round", draws[0].round),
        result=st.session_state.get("custom_result"),
        evaluation=evaluation,
        recommendation=st.session_state.get("custom_recommendation"),
        default=None,
        key="custom_frontend",
    )
    if event and event.get("id") != st.session_state.get("custom_last_event"):
        st.session_state.custom_last_event = event.get("id")
        try:
            if event.get("action") in {"ask", "clear"} and event.get("selected_round") not in {
                None,
                st.session_state.get("custom_round", draws[0].round),
            }:
                messages.clear()
                st.session_state.custom_round = event["selected_round"]
            with st.status("처리 중", expanded=True) as status:
                result = handle_event(
                    event,
                    settings,
                    draws,
                    engine_factory,
                    history=[
                        {
                            "question": m["question"],
                            "answer": m["response"]["answer"][:3000],
                            "plan": m["response"]["plan"],
                        }
                        for m in messages[-6:]
                    ],
                    progress=lambda step: st.write(step),
                )
                status.update(label="처리 완료", state="complete", expanded=False)
            if result["kind"] == "answer":
                messages.append({"question": result["question"], "response": result["response"]})
                target = result["response"]["plan"]["filters"].get("round") or event.get(
                    "selected_round"
                )
                if target:
                    st.session_state.custom_round = target
            if result["kind"] == "clear":
                messages.clear()
                st.session_state.custom_round = event.get("selected_round") or draws[0].round
            if result["kind"] == "recommendation":
                st.session_state.custom_recommendation = result["recommendation"]
            st.session_state.custom_result = result
        except Exception as exc:
            st.session_state.custom_result = {
                "id": event.get("id"),
                "kind": "error",
                "message": str(exc)
                if isinstance(exc, PublicError)
                else "요청을 처리하지 못했습니다. 입력 조건과 API 연결을 확인하세요.",
            }
        st.rerun()
