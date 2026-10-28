from datetime import date
from types import SimpleNamespace

import pytest
import requests

from src.contracts import PublicError
from src.law import LawClient
from src.settings import load_settings


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.params = []

    def get(self, url, **kwargs):
        self.params.append(kwargs["params"])
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


def response(content, status=200):
    return SimpleNamespace(status_code=status, content=content.encode())


def test_official_xml_list_and_article_body_fields(monkeypatch):
    monkeypatch.setenv("LAW_API_OC", "secret-test-never-sent")
    session = Session(
        [
            response(
                "<LawSearch><law><법령일련번호>123</법령일련번호><법령명한글>소득세법</법령명한글><시행일자>20260101</시행일자></law></LawSearch>"
            ),
            response(
                "<법령><기본정보><법령명_한글>소득세법</법령명_한글>"
                "<시행일자>20260101</시행일자></기본정보><조문><조문단위>"
                "<조문여부>조문</조문여부><조문번호>84</조문번호><조문제목>테스트</조문제목>"
                "<조문내용>본문</조문내용><항><항내용>항 내용</항내용>"
                "<호><호내용>호 내용</호내용></호></항></조문단위></조문></법령>"
            ),
        ]
    )
    client = LawClient(load_settings(), session, sleep=lambda _: None)
    candidate = client.search("소득세법", date(2026, 10, 22))[0]
    article = client.articles(candidate, date(2026, 10, 22))[0]
    assert article.text == "본문\n항 내용\n호 내용"
    assert article.article_no == "84"
    assert "OC=" not in article.source_url
    assert "secret-test" not in article.model_dump_json()
    assert session.params[1]["MST"] == "123"
    assert session.params[1]["target"] == "eflaw"
    assert session.params[1]["efYd"] == "20260101"


def test_retries_and_backoff(monkeypatch):
    monkeypatch.setenv("LAW_API_OC", "not-logged")
    session = Session(
        [response("unavailable", 429), response("unavailable", 503), response("<LawSearch />")]
    )
    sleeps = []
    client = LawClient(load_settings(), session, sleep=sleeps.append)
    assert client.search("소득세법", date(2026, 10, 22)) == []
    assert client.call_count == 3
    assert 1.0 in sleeps and 2.0 in sleeps


def test_network_failure_is_sanitized(monkeypatch):
    monkeypatch.setenv("LAW_API_OC", "private-oc")
    session = Session(
        [requests.ConnectionError("https://endpoint?OC=private-oc") for _ in range(3)]
    )
    client = LawClient(load_settings(), session, sleep=lambda _: None)
    with pytest.raises(PublicError) as caught:
        client.search("세금", date(2026, 10, 22))
    assert "private-oc" not in str(caught.value)


def test_key_missing_and_unsafe_xml_rejected(monkeypatch):
    monkeypatch.delenv("LAW_API_OC", raising=False)
    with pytest.raises(PublicError, match="LAW_API_OC"):
        LawClient(load_settings()).search("세금", date(2026, 10, 22))
    monkeypatch.setenv("LAW_API_OC", "test")
    client = LawClient(
        load_settings(), Session([response("<!DOCTYPE foo><LawSearch/>")]), sleep=lambda _: None
    )
    with pytest.raises(PublicError):
        client.search("세금", date(2026, 10, 22))
