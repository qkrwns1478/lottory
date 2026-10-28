# 법령 목록·본문 XML API를 호출하고 인증값과 공급자 내부 오류를 숨긴다.

import os
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone

import requests

from src.contracts import Article, Candidate, PublicError


def iso_day(value):
    compact = "".join(c for c in value if c.isdigit())
    return date.fromisoformat(f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}")


class LawClient:
    def __init__(self, settings, session=None, sleep=time.sleep, monotonic=time.monotonic):
        self.config = settings["law"]
        self.oc = os.getenv("LAW_API_OC", "")
        self.session = session or requests.Session()
        self.sleep, self.monotonic = sleep, monotonic
        self.last_request = None
        self.call_count = 0

    def _xml(self, endpoint, params):
        if not self.oc:
            raise PublicError("law_key")
        params = {"target": "law", **params, "OC": self.oc, "type": "XML"}
        for attempt in range(self.config["max_retries"] + 1):
            if self.last_request is not None:
                self.sleep(
                    max(0, self.config["request_interval"] - (self.monotonic() - self.last_request))
                )
            self.last_request = self.monotonic()
            self.call_count += 1
            try:
                response = self.session.get(
                    f"https://www.law.go.kr/DRF/{endpoint}",
                    params=params,
                    timeout=self.config["timeout"],
                    headers={"User-Agent": "LOTTORY educational research/1.0"},
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.config["max_retries"]:
                        self.sleep(self.config["backoff_seconds"] * 2**attempt)
                        continue
                    raise PublicError("law_network")
                if response.status_code != 200:
                    raise PublicError("law_response")
                payload = response.content
                if (
                    len(payload) > 20_000_000
                    or b"<!DOCTYPE" in payload.upper()
                    or b"<!ENTITY" in payload.upper()
                ):
                    raise PublicError("law_response")
                return ET.fromstring(payload)
            except (requests.RequestException, ET.ParseError):
                if attempt == self.config["max_retries"]:
                    raise PublicError("law_network") from None
                self.sleep(self.config["backoff_seconds"] * 2**attempt)
        raise PublicError("law_network")

    def search(self, query, as_of):
        root = self._xml(
            "lawSearch.do",
            {
                "query": query,
                "display": self.config["candidate_limit"],
                "efYd": f"00010101~{as_of:%Y%m%d}",
                "sort": "efdes",
            },
        )
        if root.tag != "LawSearch":
            raise PublicError("law_response")
        candidates = []
        for item in root.findall("law"):
            try:
                serial = item.findtext("법령일련번호", "").strip()
                if not serial.isdigit():
                    raise ValueError("invalid serial")
                candidate = Candidate(
                    candidate_id=serial,
                    law_title=item.findtext("법령명한글", "").strip(),
                    effective_date=iso_day(item.findtext("시행일자", "")),
                    source_url=f"https://www.law.go.kr/법령/{serial}",
                )
                # 인증값이 포함된 API 주소 대신 해당 법령 버전의 공식 원문 주소를 저장한다.
                candidate.source_url = f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={serial}&efYd={candidate.effective_date:%Y%m%d}"
                if candidate.effective_date <= as_of:
                    candidates.append(candidate)
            except ValueError:
                continue
        return candidates

    def articles(self, candidate, as_of):
        root = self._xml(
            "lawService.do",
            {
                "MST": candidate.candidate_id,
                "target": "eflaw",
                "efYd": f"{candidate.effective_date:%Y%m%d}",
            },
        )
        info = root.find("기본정보")
        if root.tag != "법령" or info is None:
            raise PublicError("law_response")
        title = info.findtext("법령명_한글", "").strip()
        effective = iso_day(info.findtext("시행일자", ""))
        if (
            title != candidate.law_title
            or effective != candidate.effective_date
            or effective > as_of
        ):
            raise PublicError("law_response")
        collected = datetime.now(timezone.utc).isoformat()
        articles = []
        for item in root.findall("./조문/조문단위"):
            if item.findtext("조문여부", "조문") != "조문":
                continue
            number = item.findtext("조문번호", "").strip()
            branch = item.findtext("조문가지번호", "").strip()
            if not number.isdigit():
                continue
            number = str(int(number)) + (
                f"의{int(branch)}" if branch.isdigit() and int(branch) else ""
            )
            # 본문 필드만 추출해 개정 메타데이터와 관련 없는 부록을 제외한다.
            pieces = [
                "".join(node.itertext()).strip()
                for node in item.iter()
                if node.tag.endswith("내용") and "".join(node.itertext()).strip()
            ]
            text = "\n".join(pieces)
            if not text or text.startswith("삭제"):
                continue
            article_title = item.findtext("조문제목", "").strip()
            articles.append(
                Article(
                    document_id=f"law_{candidate.candidate_id}_{number}",
                    law_title=title,
                    article_no=number,
                    document_title=f"{title} 제{number}조 {article_title}",
                    text=text,
                    effective_date=effective,
                    source_url=candidate.source_url,
                    collected_at=collected,
                    verified_on=as_of,
                )
            )
        if not articles:
            raise PublicError("law_response")
        return articles
