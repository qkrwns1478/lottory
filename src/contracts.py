# 문서 수집과 RAG 단계가 공유하는 입력·출력 검증 스키마

from datetime import date
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


def law_url(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"law.go.kr", "www.law.go.kr"}
        or parsed.username
    ):
        raise ValueError("국가법령정보센터 HTTPS 원문 URL이 필요합니다.")
    if "oc=" in parsed.query.lower():
        raise ValueError("인증값이 포함된 URL은 저장하지 않습니다.")
    return value


class Candidate(Contract):
    candidate_id: str
    law_title: str
    effective_date: date
    source_url: str

    _url = field_validator("source_url")(law_url)


class Decision(Contract):
    candidate_id: str
    decision: Literal["index", "reject", "review_usage"]
    reason: str = Field(min_length=1)
    priority: int = Field(ge=1, le=100)


class Selection(Contract):
    decisions: list[Decision]


class Article(Contract):
    document_id: str
    law_title: str
    article_no: str
    document_title: str
    text: str = Field(min_length=1)
    effective_date: date | None
    source_url: str
    collected_at: str
    # 목록 API가 제공한 버전이 이후에도 계속 유효하다고 가정하지 않는다.
    verified_on: date

    document_type: Literal["law_article", "government_guidance"] = "law_article"
    source_type: Literal["statute", "government_guidance"] = "statute"
    source_name: str = "국가법령정보센터"

    @model_validator(mode="after")
    def validate_source(self):
        if self.document_type == "law_article":
            law_url(self.source_url)
            if self.effective_date is None or self.source_type != "statute":
                raise ValueError("법령의 시행일과 출처 유형이 필요합니다.")
        else:
            parsed = urlsplit(self.source_url)
            if (
                parsed.scheme != "https"
                or parsed.hostname not in {"d.nts.go.kr", "www.nts.go.kr"}
                or parsed.username
            ):
                raise ValueError("국세청 HTTPS 원문 URL이 필요합니다.")
            if self.source_type != "government_guidance":
                raise ValueError("안내문 출처 유형을 확인하세요.")
        return self


class Citation(Contract):
    evidence_id: str
    quote: str = Field(min_length=1)


class GeneratedAnswer(Contract):
    status: Literal["answered", "insufficient_evidence", "conflicting_sources"]
    answer: str
    evidence_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    disclaimer_required: bool = True


class PublicError(Exception):
    """미리 정한 공개 안내만 화면에 표시하고 민감한 내부 오류는 숨긴다."""

    MESSAGES = {
        "openai_key": "OPENAI_API_KEY가 없습니다. .env에 API 키를 설정하거나 AI 근거 답변을 끄고 로컬 조회를 이용하세요.",
        "law_key": "LAW_API_OC가 없습니다. .env에 법령 API 인증값을 설정하세요.",
        "law_network": "법령 API에 연결하지 못했습니다. 잠시 후 다시 시도하세요.",
        "law_response": "법령 API 응답을 검증하지 못했습니다.",
        "selection": "법령 후보 선별 결과를 검증하지 못했습니다.",
        "generation": "생성한 답변의 근거 또는 수치를 검증하지 못해 답변을 보류합니다.",
        "index": "근거 색인이 없거나 확인하지 못했습니다. python -m scripts.prepare_data --index 실행 후 다시 시도하세요. 로컬 조회는 AI 근거 답변을 끄고 이용할 수 있습니다.",
        "guidance": "국세청 안내문 수집을 완료하지 못했습니다. 확인된 법령 근거로 처리합니다.",
    }

    def __init__(self, code):
        self.code = code
        super().__init__(self.MESSAGES[code])
