# 기존 Streamlit 실행 진입점의 화면을 실제 브라우저로 검증한다.

import argparse
import json

from src.data import atomic_write
from src.settings import ROOT


def main():
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8511")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--layout-only", action="store_true")
    parser.add_argument("--disclosures-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / "data/screenshots"
    if not args.disclosures_only:
        output.mkdir(parents=True, exist_ok=True)
    report = {"checks": [], "console_errors": []}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda exc: report["console_errors"].append(str(exc)))
        page.goto(args.url)
        assert page.get_by_role("tab").count() == 0
        assert page.locator('[data-testid="stSidebar"]').count() == 0
        iframe = page.locator('iframe[title="src.frontend.lottory_frontend"]')
        iframe.wait_for()
        frame = iframe.element_handle().content_frame()
        frame.locator("#round-select option").first.wait_for(state="attached")

        def capture(name):
            # 부모 화면이 짧으면 iframe 내부 캡처가 잘릴 수 있다.
            # 반응형 너비는 유지하고 전체 콘텐츠 높이에 맞춰 캡처한다.
            height = frame.locator(".workspace").bounding_box()["height"]
            page.set_viewport_size(
                {"width": page.viewport_size["width"], "height": int(height) + 1000}
            )
            iframe.scroll_into_view_if_needed()
            frame.locator(".workspace").screenshot(path=str(output / name))

        if args.disclosures_only:

            def assert_stays_open(detail):
                detail.locator("summary").click()
                # 높이가 바뀌어 부모 화면이 다시 전송한 뒤에도 열린 상태를 유지해야 한다.
                frame.wait_for_timeout(800)
                assert detail.evaluate("detail => detail.open")

            frame.locator('[data-panel="evaluation"]').click()
            evaluation_detail = frame.locator("#evaluation-results details").first
            assert_stays_open(evaluation_detail)
            page.set_viewport_size({"width": 430, "height": 950})
            frame.wait_for_timeout(800)
            assert evaluation_detail.evaluate("detail => detail.open")
            frame.locator('[data-panel="chat"]').click()
            frame.locator("#question").fill("최근 100회 중 가장 많이 나온 번호 6개는?")
            frame.locator("#ask-form button").click()
            frame.locator("#chat-log details").wait_for()
            chat_detail = frame.locator("#chat-log details").first
            assert_stays_open(chat_detail)
            chat_detail.evaluate("detail => { window.disclosureNode = detail; }")
            args_snapshot = frame.evaluate("state.args")
            args_snapshot["disclosure_check"] = True
            iframe.evaluate(
                "(element, args) => element.contentWindow.postMessage("
                "{type: 'streamlit:render', args}, '*')",
                args_snapshot,
            )
            frame.wait_for_function("state.args.disclosure_check === true")
            assert chat_detail.evaluate("detail => detail === window.disclosureNode && detail.open")
            frame.locator("#question").fill("당첨번호는?")
            frame.locator("#ask-form button").click()
            frame.wait_for_function(
                "document.querySelectorAll('#chat-log .answer-text').length === 2"
            )
            assert chat_detail.evaluate("detail => detail.open")
            chat_detail.locator("summary").focus()
            chat_detail.locator("summary").press("Enter")
            frame.wait_for_function("!document.querySelector('#chat-log details').open")
            chat_detail.locator("summary").press("Space")
            frame.wait_for_function("document.querySelector('#chat-log details').open")
            frame.locator('[data-panel="evaluation"]').click()
            assert evaluation_detail.evaluate("detail => detail.open")
            assert not report["console_errors"], report["console_errors"]
            browser.close()
            print("접기·펼치기 검증 통과: 클릭·키보드·모바일·재전송·대화 추가·화면 전환")
            return
        if args.layout_only:
            frame.locator('[data-panel="filter"]').click()
            frame.locator("#numbers").fill("22")
            frame.locator("#filter-form button").click()
            frame.locator("#filter-results table").wait_for()
            page.set_viewport_size({"width": 430, "height": 950})
            frame.wait_for_timeout(500)
            print(
                json.dumps(
                    {
                        "iframe": iframe.bounding_box(),
                        "content": frame.locator(".workspace").bounding_box(),
                        "inner_height": frame.evaluate("innerHeight"),
                    }
                )
            )
            capture("frontend_mobile.png")
            browser.close()
            return
        if args.evaluation_only:
            frame.locator('[data-panel="evaluation"]').click()
            frame.locator("#evaluation-results table").wait_for()
            assert frame.locator("#evaluation-results tbody tr").count() == 20
            with page.expect_download() as download:
                frame.locator("#download-evaluation").click()
            download.value.save_as(str(output / "evaluation_download.json"))
            capture("frontend_evaluation.png")
            report["checks"].append("20문항 실평가 표·상세 답변·JSON 다운로드")
            report["passed"] = True
            atomic_write(
                ROOT / "data/verification_frontend_evaluation.json",
                json.dumps(report, ensure_ascii=False, indent=2),
            )
            browser.close()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return
        frame.locator("#round-select").select_option("1100")
        frame.locator("#question").fill("1100회 당첨금은?")
        frame.locator("#ask-form button").click()
        frame.locator("#chat-log .answer-text").first.wait_for(timeout=60000)
        assert "2,207,575,472" in frame.locator("#chat-log").inner_text()
        report["checks"].append("직접 구현한 채팅 → Python 회차 조회 → 출처 표시")
        frame.locator("#question").fill("당첨번호는?")
        frame.locator("#ask-form button").click()
        page.wait_for_function("document.querySelectorAll('iframe').length > 0")
        frame.wait_for_function("document.querySelectorAll('#chat-log .answer-text').length === 2")
        assert "제1100회" in frame.locator("#chat-log .answer-text").last.inner_text()
        report["checks"].append("회차 유지·후속 질문")
        capture("frontend_chat.png")
        frame.locator('[data-panel="filter"]').click()
        frame.locator("#numbers").fill("22")
        frame.locator("#filter-form button").click()
        frame.locator("#filter-results table").wait_for(timeout=60000)
        assert frame.locator("#filter-results tbody tr").count() > 0
        report["checks"].append("JavaScript 조건 입력 → Python 필터 → 표 렌더링")
        capture("frontend_filter.png")
        frame.locator('[data-panel="compare"]').click()
        frame.locator("#compare-form button").click()
        frame.locator("#compare-results .comparison-card").first.wait_for(timeout=60000)
        assert frame.locator("#compare-results .comparison-card").count() == 4
        report["checks"].append("실제 similarity/MMR 비교 화면·두 컬렉션 결과")
        capture("frontend_mmr.png")
        frame.locator("#search-type").select_option("mmr")
        frame.locator('[data-panel="chat"]').click()
        frame.locator("#ai-mode").check()
        frame.locator("#question").fill("10억 원의 예상 실수령액은?")
        frame.locator("#ask-form button").click()
        frame.wait_for_function(
            "document.querySelectorAll('#chat-log .answer-text').length === 3", timeout=120000
        )
        last = frame.locator("#chat-log .chat-message:not(.user)").last
        assert "703,000,000" in last.inner_text()
        assert "국세청 안내" in last.inner_text() and "법령" in last.inner_text()
        report["checks"].append("MMR 기반 실제 AI 질의·세금 계산·법령/국세청 출처 표시")
        capture("frontend_sources.png")
        page.set_viewport_size({"width": 430, "height": 950})
        frame.locator('[data-panel="filter"]').click()
        frame.wait_for_timeout(300)
        assert frame.evaluate(
            "document.querySelector('.workspace').scrollWidth <= window.innerWidth"
        )
        capture("frontend_mobile.png")
        report["checks"].append("모바일 폭에서 화면 넘침 없음")
        frame.locator("#min-sum").fill("255")
        frame.locator("#max-sum").fill("255")
        frame.locator("#filter-form button").click()
        frame.wait_for_function(
            "document.querySelector('#filter-results h3')?.textContent.includes('0개')"
        )
        report["checks"].append("직접 구현한 번호 합계 필터 적용")
        frame.locator("#clear-chat").click()
        frame.wait_for_function("document.querySelectorAll('#chat-log .chat-message').length === 0")
        report["checks"].append("대화 지우기·서버 대화 기록 초기화")
        assert not report["console_errors"], report["console_errors"]
        browser.close()
    report["passed"] = True
    atomic_write(
        ROOT / "data/verification_frontend.json", json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
