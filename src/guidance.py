# 별도 인증 없이 허용한 국세청 공개 안내 페이지의 관련 본문을 수집한다.

import json
import time
from datetime import datetime
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

import requests

from src.contracts import Article, PublicError
from src.data import atomic_write
from src.documents import digest, index_articles
from src.settings import resolve_path

NTS_URL = "https://d.nts.go.kr/nts/cm/cntnts/cntntsView.do?cntntsId=7893&mi=6457"


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ignored = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.ignored += 1
        if tag in {"p", "li", "tr", "td", "th", "h2", "h3", "h4", "div", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.ignored = max(0, self.ignored - 1)
        if tag in {"p", "li", "tr", "td", "th", "h2", "h3", "h4", "div"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def extract_guidance(html):
    parser = TextParser()
    parser.feed(html)
    text = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
    start = text.rfind("\n원천징수세율\n")
    if start >= 0:
        start += 1
    elif text.startswith("원천징수세율\n"):
        start = 0
    if start < 0:
        raise PublicError("guidance")
    section = text[start:]
    for marker in ("콘텐츠 만족도", "이 페이지", "담당부서", "매우만족"):
        index = section.find(marker)
        if index > 0:
            section = section[:index]
    if "복권" not in section or "30%" not in section or len(section) > 15000:
        raise PublicError("guidance")
    return section.strip()


class GuidanceCollector:
    def __init__(self, settings, collection, embeddings, session=None):
        self.settings, self.collection, self.embeddings = settings, collection, embeddings
        self.session = session or requests.Session()

    def collect(self, as_of):
        today = datetime.now(ZoneInfo("Asia/Seoul")).date()
        if as_of != today:
            return {"status": "historical_unavailable", "chunk_ids": []}
        config = self.settings["guidance"]
        if config["source_url"] != NTS_URL:
            raise PublicError("guidance")
        path = resolve_path(config["cache_path"])
        signature = digest(
            json.dumps(
                {
                    "url": NTS_URL,
                    "date": str(as_of),
                    "chunking": self.settings["chunking"],
                    "embedding": self.settings["embedding"],
                },
                sort_keys=True,
            )
        )
        try:
            cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (ValueError, OSError):
            cache = {}
        entry = cache.get(signature, {})
        ids = entry.get("chunk_ids", [])
        if ids and entry.get("expires_at", 0) > time.time():
            stored = self.collection.get(ids=ids, include=["metadatas"])
            if set(stored["ids"]) == set(ids) and all(
                m.get("verified_on") == str(as_of) for m in stored["metadatas"]
            ):
                return {"status": "indexed", "cache": "hit", "chunk_ids": ids, "http_calls": 0}
        calls = 0
        try:
            for attempt in range(config["max_retries"] + 1):
                calls += 1
                try:
                    response = self.session.get(
                        NTS_URL,
                        timeout=config["timeout"],
                        allow_redirects=False,
                        headers={"User-Agent": "LOTTORY educational research/1.0"},
                    )
                except requests.RequestException:
                    if attempt < config["max_retries"]:
                        time.sleep(2**attempt)
                        continue
                    raise PublicError("guidance") from None
                if (
                    response.status_code in {429, 500, 502, 503, 504}
                    and attempt < config["max_retries"]
                ):
                    time.sleep(2**attempt)
                    continue
                if response.status_code != 200 or len(response.content) > 5_000_000:
                    raise PublicError("guidance")
                response.encoding = "utf-8"
                text = extract_guidance(response.text)
                break
            article = Article(
                document_id="nts_other_income_rates",
                law_title="",
                article_no="",
                document_title="국세청 기타소득 원천징수세율 안내",
                text=text,
                effective_date=None,
                source_url=NTS_URL,
                collected_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                verified_on=as_of,
                document_type="government_guidance",
                source_type="government_guidance",
                source_name="국세청",
            )
            ids = index_articles(
                [article], self.collection, self.embeddings, self.settings["chunking"]
            )
            registry_path = resolve_path(self.settings["data"]["source_registry"])
            registry = (
                json.loads(registry_path.read_text(encoding="utf-8"))
                if registry_path.exists()
                else {}
            )
            registry[article.document_id] = {
                **article.model_dump(mode="json", exclude={"text"}),
                "license": "공공기관 공개 안내문; 개별 이용조건 원문 확인",
                "usage_scope": "비영리 교육용 관련 본문 발췌·색인, 출처와 원문 링크 표시",
                "content_hash": digest(text),
            }
            atomic_write(registry_path, json.dumps(registry, ensure_ascii=False, indent=2))
            cache[signature] = {
                "chunk_ids": ids,
                "expires_at": time.time() + config["cache_ttl_seconds"],
            }
            atomic_write(path, json.dumps(cache, ensure_ascii=False, indent=2))
            return {"status": "indexed", "cache": "miss", "chunk_ids": ids, "http_calls": calls}
        except Exception:
            raise PublicError("guidance") from None
