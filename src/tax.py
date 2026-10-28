# 검색한 법령에서 지원하는 과세 조건을 확인한 경우에만 세금을 계산한다.

import re
from decimal import ROUND_DOWN, Decimal


def calculate_tax(amount, evidence, as_of):
    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
        raise ValueError("당첨금은 0 이상의 정수여야 합니다.")
    statutes = {}
    for item in evidence:
        meta = item["metadata"]
        if (
            meta.get("document_type") == "law_article"
            and meta.get("effective_date", "9999") <= str(as_of)
            and meta.get("verified_on") == str(as_of)
        ):
            statutes[(meta.get("law_title"), meta.get("article_no"))] = item
    required = [("소득세법", "84"), ("소득세법", "129"), ("지방세법", "103의13")]
    if any(key not in statutes for key in required):
        return None
    exemption, rates, local = [re.sub(r"\s+", "", statutes[key]["text"]) for key in required]
    # 지원하는 법령 문구·수치만 인정하고 개정으로 달라지면 계산을 보류한다.
    exemption_valid = re.search(r"복권당첨금.{0,180}건별로200만원이하", exemption)
    higher_valid = re.search(r"3억원.{0,120}초과.{0,160}100분의30", rates)
    base_valid = re.search(r"그밖의기타소득.{0,120}100분의20", rates)
    local_valid = re.search(r"소득세.{0,240}100분의10", local)
    if not all((exemption_valid, higher_valid, base_valid, local_valid)):
        return None
    taxable = 0 if amount <= 2_000_000 else amount
    lower_base, upper_base = min(taxable, 300_000_000), max(taxable - 300_000_000, 0)
    income_tax = int(
        (
            Decimal(lower_base) * Decimal("0.20") + Decimal(upper_base) * Decimal("0.30")
        ).to_integral_value(rounding=ROUND_DOWN)
    )
    local_tax = int((Decimal(income_tax) * Decimal("0.10")).to_integral_value(rounding=ROUND_DOWN))
    return {
        "calculation_type": "tax",
        "as_of": str(as_of),
        "gross_amount": amount,
        "income_tax": income_tax,
        "local_income_tax": local_tax,
        "net_amount": amount - income_tax - local_tax,
        "lower_tax_base": lower_base,
        "upper_tax_base": upper_base,
        "exempt_limit": 2_000_000,
        "progressive_threshold": 300_000_000,
        "income_tax_rates": ["20%", "30%"],
        "local_tax_rate": "소득세의 10%",
        "source_evidence_ids": [statutes[key]["evidence_id"] for key in required],
        "assumptions": [
            "대한민국 거주자·로또 일시금·1게임의 당첨금 지급 가정",
            "입력 금액을 소득금액으로 가정하고 별도의 필요경비는 공제하지 않음",
            "200만원 이하는 과세하지 않으며 이를 초과하면 초과액만이 아닌 전체 소득금액을 기준으로 계산",
            "원 미만을 버린 예상액. 실제 지급기관의 원천징수·끝수 처리와 다를 수 있음",
            "추첨일과 지급일은 다를 수 있으며 적용 기준일에 지급받는 것으로 가정",
        ],
    }


def tax_text(result):
    return (
        f"적용 기준일: {result['as_of']}\n"
        f"당첨금 {result['gross_amount']:,}원\n"
        f"소득세 = {result['lower_tax_base']:,}원 × 20% + {result['upper_tax_base']:,}원 × 30% = {result['income_tax']:,}원\n"
        f"지방소득세 = 소득세의 10% = {result['local_income_tax']:,}원\n"
        f"예상 실수령액 = {result['gross_amount']:,}원 - {result['income_tax']:,}원 - {result['local_income_tax']:,}원 = {result['net_amount']:,}원\n"
        + "\n".join(result["assumptions"])
    )
