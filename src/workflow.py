# F003~F012: 계획 → 수집·색인 → 검색·계산 → 생성 → 검증

import json
import time
from datetime import datetime
from typing import TypedDict
from zoneinfo import ZoneInfo

from langgraph.graph import END, START, StateGraph

from src.contracts import GeneratedAnswer, PublicError
from src.data import CSV_URL
from src.documents import DocumentCollector
from src.generation import conflicts, make_generator, make_selector, verify_answer
from src.guidance import GuidanceCollector
from src.indexing import index_rounds, make_embeddings, open_collection
from src.law import LawClient
from src.planner import make_planner
from src.query import SearchPlan
from src.retrieval import merge_evidence, retrieve
from src.service import DISCLAIMER, answer_question, coverage
from src.settings import resolve_path
from src.tax import calculate_tax, tax_text


class State(TypedDict, total=False):
    question: str
    selected_round: int | None
    draws: list
    history: list
    extracted_queries: list
    plan: SearchPlan
    as_of: object
    collection_audit: dict
    search_audit: dict
    evidence: list
    response: dict
    canonical: str
    generated: GeneratedAnswer
    warnings: list
    halt: bool


class LottoWorkflow:
    def __init__(
        self,
        settings,
        planner,
        collector,
        round_collection,
        document_collection,
        embeddings,
        generator,
        today=None,
        guidance=None,
    ):
        self.settings, self.planner, self.collector = settings, planner, collector
        self.round_collection, self.document_collection = (
            round_collection,
            document_collection,
        )
        self.embeddings, self.generator = embeddings, generator
        self.guidance = guidance
        self.today = today or (lambda: datetime.now(ZoneInfo("Asia/Seoul")).date())
        graph = StateGraph(State)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        self.graph = graph.compile()

    def _retrieve_node(self, state, config):
        state = dict(state)
        progress = config.get("configurable", {}).get("progress", lambda _: None)
        progress("질문 분석")
        state.update(self._plan(state))
        progress("법령 후보 검색·선별·수집·색인")
        state.update(self._collect(state, config))
        progress("두 컬렉션 근거 검색·병합 및 정확 계산")
        state.update(self._retrieve(state))
        return state

    def _generate_node(self, state, config):
        state = dict(state)
        progress = config.get("configurable", {}).get("progress", lambda _: None)
        progress("근거 기반 답변 생성")
        state.update(self._generate(state))
        progress("인용·수치 검증")
        state.update(self._verify(state))
        return state

    def _plan(self, state):
        context = coverage(state["draws"])
        context["today"] = str(self.today())
        context["recent_conversation"] = state.get("history", [])[-6:]
        plan = SearchPlan.model_validate(
            self.planner(state["question"], state["selected_round"], context)
        )
        # 기준일이 없으면 오늘 지급을 가정하며 추첨일을 지급일로 간주하지 않는다.
        as_of = plan.as_of or self.today()
        plan.as_of = as_of
        extracted_queries = list(plan.search_queries)
        if plan.intent in {"tax", "claim_guide"}:
            plan.need_external_docs = True
            required_queries = (
                ["소득세법", "지방세법"] if plan.intent == "tax" else ["복권 및 복권기금법"]
            )
            # 목록 API는 법령명으로 검색한다. 검색어를 고정해 후속 질문의
            # 표현 차이로 같은 법령을 다시 수집하지 않도록 한다.
            plan.search_queries = required_queries
        response = answer_question(
            state["question"],
            state["selected_round"],
            state["draws"],
            planner=lambda *_: plan,
        )
        # 법령 안내를 보류하더라도 검증된 CSV 조회 결과는 제공한다.
        if plan.intent in {"tax", "claim_guide"} and plan.filters.round:
            matches = [draw for draw in state["draws"] if draw.round == plan.filters.round]
            response["rows"] = [draw.model_dump(mode="json") for draw in matches]
        halt = plan.intent == "unknown" or (
            plan.filters.round is not None
            and not any(draw.round == plan.filters.round for draw in state["draws"])
        )
        if plan.filters.round is not None and not any(
            draw.round == plan.filters.round for draw in state["draws"]
        ):
            response["answer"] = "현재 적재한 CSV 범위에서 요청한 회차가 확인되지 않습니다."
        if response["status"] == "insufficient_evidence" and not plan.need_external_docs:
            halt = True
        if as_of > self.today():
            halt = True
            response["answer"] = "미래 지급일의 적용 법령은 확인할 수 없어 답변을 보류합니다."
            response["status"] = "insufficient_evidence"
        return {
            "plan": plan,
            "as_of": as_of,
            "response": response,
            "halt": halt,
            "warnings": [],
            "extracted_queries": extracted_queries,
        }

    def _collect(self, state, config):
        if state["halt"] or not state["plan"].need_external_docs:
            return {"collection_audit": {"queries": []}}
        try:
            progress = config.get("configurable", {}).get("progress", lambda _: None)
            audit = self.collector.collect(
                state["question"], state["plan"], state["as_of"], progress=progress
            )
            if self.guidance is not None and state["plan"].intent == "tax":
                progress("국세청 안내문 수집·색인")
                try:
                    audit["guidance"] = self.guidance.collect(state["as_of"])
                except PublicError as exc:
                    return {"collection_audit": audit, "warnings": [str(exc)]}
            return {"collection_audit": audit}
        except PublicError as exc:
            return {
                "collection_audit": {"error_code": exc.code},
                "warnings": [str(exc)],
            }

    def _retrieve(self, state):
        if state["halt"]:
            return {
                "evidence": [],
                "search_audit": {},
                "canonical": state["response"]["answer"],
            }
        try:
            registry_path = resolve_path(self.settings["data"]["source_registry"])
            registry = (
                json.loads(registry_path.read_text(encoding="utf-8"))
                if registry_path.exists()
                else {}
            )
            collected = registry.get("cashlogy", {}).get("collected_at", "")
            index_report = index_rounds(
                state["draws"],
                self.round_collection,
                self.embeddings,
                collected,
                self.settings["embedding"]["batch_size"],
            )
            evidence, audit = retrieve(
                state["question"],
                state["plan"],
                state["draws"],
                self.round_collection,
                self.document_collection,
                self.embeddings,
                self.settings["retrieval"],
                state["as_of"],
                collected,
            )
        except Exception:
            raise PublicError("index") from None
        response = state["response"]
        canonical = response["answer"] if not state["plan"].need_external_docs else ""
        for row in response["rows"]:
            if state["plan"].need_external_docs:
                canonical += (
                    next(d.search_text() for d in state["draws"] if d.round == row["round"]) + "\n"
                )
        # 집계는 검색 상위 일부가 아니라 실제 계산에 사용한 전체 CSV 범위를 인용한다.
        if state["plan"].intent == "search" and response["rows"]:
            matching = response["rows"]
            canonical += "\n일치 회차: " + ", ".join(f"{row['round']}회" for row in matching) + "."
            response["calculation"] = {
                "calculation_type": "filter",
                "filters": state["plan"].filters.model_dump(mode="json"),
                "row_count": len(matching),
                "result": {"rounds": [row["round"] for row in matching]},
                "data_range": {
                    "from_round": min(row["round"] for row in matching),
                    "to_round": max(row["round"] for row in matching),
                },
                "source_round_ids": [f"lotto_{row['round']}" for row in matching],
            }
        if response["calculation"]:
            dataset = {
                "chunk_id": "csv_calculation",
                "text": canonical,
                "metadata": {
                    "document_id": "csv_calculation",
                    "document_title": "CSV 전체 대상 회차 Python 집계",
                    "source_url": CSV_URL,
                    "source_name": "캐시로지 / 원천 발표 동행복권",
                    "document_type": "verified_calculation",
                    "collected_at": collected,
                },
                "exact": True,
            }
            if response["calculation"]["calculation_type"] == "probability":
                dataset["metadata"].update(
                    document_type="structured_summary",
                    document_title="LOTTORY 작성 확률 계산 설명",
                    source_name="LOTTORY 수학적 설명 (참고 회차 데이터: 캐시로지)",
                )
            evidence = merge_evidence([[dataset], evidence], self.settings["retrieval"]["rrf_k"])
        if state["plan"].intent == "tax":
            amount = state["plan"].prize_amount
            if amount is None and response["rows"]:
                amount = response["rows"][0]["first_prize_amount"]
            calculation = (
                calculate_tax(amount, evidence, state["as_of"])
                if amount is not None and not conflicts(evidence)
                else None
            )
            response["calculation"] = calculation
            canonical += tax_text(calculation) if calculation else ""
            if calculation:
                for item in evidence:
                    if item["metadata"].get("document_type") != "government_guidance":
                        continue
                    lines = item["text"].splitlines()
                    for i, line in enumerate(lines):
                        if "복권" in line:
                            end = next(
                                (j for j in range(i, min(len(lines), i + 8)) if "30%" in lines[j]),
                                None,
                            )
                            if end is not None:
                                quote = "\n".join(lines[i : end + 1])
                                canonical += "\n국세청 안내 원문(보조 근거):\n" + quote
                                item["exact"] = True
                                break
        if state["plan"].intent == "claim_guide" and any(
            word in state["question"] for word in ("기한", "언제까지", "소멸시효")
        ):
            for item in evidence:
                if (
                    item["metadata"].get("law_title") == "복권 및 복권기금법"
                    and item["metadata"].get("article_no") == "9"
                ):
                    # 지급개시일과 복권 유형 조건이 유지되도록 수집한 조문을 그대로 인용한다.
                    canonical = f"적용 기준일: {state['as_of']}\n" + item["text"]
                    item["exact"] = True
                    break
        audit["round_index"] = index_report
        return {
            "response": response,
            "canonical": canonical,
            "evidence": evidence,
            "search_audit": audit,
        }

    def _generate(self, state):
        response, evidence, plan = (
            state["response"],
            state.get("evidence", []),
            state["plan"],
        )
        collision = conflicts(evidence)
        if collision:
            return {
                "generated": GeneratedAnswer(
                    status="conflicting_sources",
                    answer="근거 문서가 서로 충돌해 답변을 보류합니다.",
                    limitations=collision,
                )
            }
        has_law = any(item["metadata"].get("document_type") == "law_article" for item in evidence)
        if plan.intent == "claim_guide":
            has_law = any(
                item["metadata"].get("law_title") == "복권 및 복권기금법"
                and item["metadata"].get("article_no") in {"8", "9", "10"}
                for item in evidence
            )
        if (
            state["halt"]
            or not evidence
            or (plan.need_external_docs and not has_law)
            or (plan.intent == "tax" and response["calculation"] is None)
        ):
            message = (
                response["answer"]
                if state["halt"]
                else "제공된 근거에서 확인되지 않습니다. 적용 법령 또는 계산 조건이 부족해 안내를 보류합니다."
            )
            return {
                "generated": GeneratedAnswer(
                    status="insufficient_evidence",
                    answer=message,
                    limitations=state["warnings"],
                )
            }
        try:
            generated = self.generator(
                state["question"],
                plan,
                evidence,
                response["calculation"],
                state["canonical"],
            )
            checked = verify_answer(generated, evidence, state["canonical"])
            if checked.status == "insufficient_evidence" and state["canonical"]:
                raise PublicError("generation")
            return {"generated": checked}
        except Exception:
            # 모델 답변 검증에 실패해도 독립적으로 확인한 조회·계산 결과는 보존한다.
            safe = [
                item
                for item in evidence
                if item.get("exact")
                or item["evidence_id"]
                in ((response["calculation"] or {}).get("source_evidence_ids", []))
            ]
            if state["canonical"] and safe:
                fallback = GeneratedAnswer(
                    status="answered",
                    answer=state["canonical"],
                    evidence_ids=[item["evidence_id"] for item in safe],
                    citations=[
                        {"evidence_id": item["evidence_id"], "quote": item["text"]} for item in safe
                    ],
                    limitations=["생성 문장을 검증하지 못해 확인된 조회·계산 결과만 제공합니다."],
                )
                return {"generated": verify_answer(fallback, evidence, state["canonical"])}
            return {
                "generated": GeneratedAnswer(
                    status="insufficient_evidence",
                    answer=str(PublicError("generation")),
                )
            }

    def _verify(self, state):
        generated = state["generated"]
        response = state["response"]
        response.update(
            status=generated.status,
            answer=generated.answer,
            sources=[],
            limitations=list(dict.fromkeys(generated.limitations + state["warnings"])),
        )
        if generated.status == "answered":
            # 모델이 생략하더라도 서버가 확정한 수치 문장은 항상 표시한다.
            if state["canonical"] and state["canonical"] not in response["answer"]:
                response["answer"] += "\n\n" + state["canonical"]
            required = set(generated.evidence_ids)
            if response["calculation"] and response["calculation"].get("calculation_type") == "tax":
                required.update(response["calculation"]["source_evidence_ids"])
            required.update(item["evidence_id"] for item in state["evidence"] if item.get("exact"))
            response["sources"] = [
                {"evidence_id": item["evidence_id"], **item["metadata"]}
                for item in state["evidence"]
                if item["evidence_id"] in required
            ]
        response.update(
            plan=state["plan"].model_dump(mode="json"),
            disclaimer=DISCLAIMER,
            as_of=str(state["as_of"]),
            evidence=state.get("evidence", []),
            audit={
                "collection": state.get("collection_audit", {}),
                "search": state.get("search_audit", {}),
                "query_normalization": {
                    "extracted": state["extracted_queries"],
                    "executed": state["plan"].search_queries,
                },
            },
        )
        return {"response": response}

    def answer(self, question, selected_round, draws, progress=None, history=None):
        progress = progress or (lambda _: None)
        started, final = time.perf_counter(), None
        for update in self.graph.stream(
            {
                "question": question,
                "selected_round": selected_round,
                "draws": draws,
                "history": history or [],
            },
            stream_mode="updates",
            config={"configurable": {"progress": progress}},
        ):
            for value in update.values():
                if "response" in value:
                    final = value["response"]
        final["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        return final


def make_workflow(settings, planner=None):
    embeddings = make_embeddings(settings)
    rounds, documents = open_collection(settings), open_collection(settings, "document")
    if rounds.count() == 0:
        raise PublicError("index")
    collector = DocumentCollector(
        settings, LawClient(settings), make_selector(settings), documents, embeddings
    )
    return LottoWorkflow(
        settings,
        planner or make_planner(settings),
        collector,
        rounds,
        documents,
        embeddings,
        make_generator(settings),
        guidance=GuidanceCollector(settings, documents, embeddings)
        if settings.get("guidance", {}).get("enabled")
        else None,
    )
