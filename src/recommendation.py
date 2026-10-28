import hashlib
import random
from collections import Counter
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

DISCLAIMER = (
    "추천 번호는 오락·참고용 정보입니다. 과거 데이터의 통계를 활용하며 "
    "당첨을 예측하거나 수익을 보장하지 않습니다. 각 추첨은 독립적이며 모든 6개 번호 "
    "조합의 1등 당첨 확률은 1/8,145,060으로 같습니다. 무리한 구매를 피하세요."
)


def recommend_numbers(draws, window=100, game_count=5, today=None):
    if window not in {20, 50, 100} or not 1 <= game_count <= 5:
        raise ValueError("분석 회차와 추천 게임 수를 확인하세요.")
    today = today or datetime.now(ZoneInfo("Asia/Seoul")).date()
    target_date = today + timedelta(days=(5 - today.weekday()) % 7)
    # 당일 추첨 결과와 미래 데이터는 추천에 사용하지 않는다.
    history = sorted(
        (draw for draw in draws if draw.draw_date < min(today, target_date)),
        key=lambda draw: (draw.draw_date, draw.round),
        reverse=True,
    )[:window]
    if not history:
        raise ValueError("추천에 사용할 과거 추첨 데이터가 없습니다.")
    counts = Counter(number for draw in history for number in draw.numbers)
    recent = history[:20]
    recent_counts = Counter(number for draw in recent for number in draw.numbers)
    # 전체 분석 기간과 최근 20회의 빈도를 같은 비중으로 반영한다.
    weights = {
        number: 1 + counts[number] / len(history) + recent_counts[number] / len(recent)
        for number in range(1, 46)
    }
    fingerprint = "|".join(
        f"{draw.round}:{draw.draw_date}:{','.join(map(str, sorted(draw.numbers)))}"
        for draw in history
    )
    seed = hashlib.sha256(f"{target_date}:{window}:{fingerprint}".encode()).hexdigest()
    rng = random.Random(seed)
    games = []
    seen = set()
    while len(games) < game_count:
        pool = list(range(1, 46))
        chosen = []
        for _ in range(6):
            number = rng.choices(pool, weights=[weights[n] for n in pool], k=1)[0]
            chosen.append(number)
            pool.remove(number)
        numbers = tuple(sorted(chosen))
        if numbers in seen:
            continue
        seen.add(numbers)
        games.append({"numbers": list(numbers)})
    latest = history[0]
    missing_weeks = max(0, (target_date - latest.draw_date).days // 7 - 1)
    return {
        "target_date": target_date.isoformat(),
        "generated_on": today.isoformat(),
        "requested_window": window,
        "row_count": len(history),
        "recent_count": len(recent),
        "from_round": history[-1].round,
        "to_round": latest.round,
        "from_date": history[-1].draw_date.isoformat(),
        "to_date": latest.draw_date.isoformat(),
        "games": games,
        "frequencies": [
            {
                "number": number,
                "count": counts[number],
                "recent_count": recent_counts[number],
            }
            for number in sorted(range(1, 46), key=lambda n: (-counts[n], n))
        ],
        "method": (
            "번호별 가중치 = 1 + 분석 기간 출현 횟수/분석 회차 수 "
            "+ 최근 20회(데이터가 적으면 전체) 출현 횟수/해당 회차 수. "
            "이 가중치로 1~45 중 중복 없이 6개를 추출합니다. 보너스 번호는 제외합니다. "
            "가중치는 조합 생성 기준이며 당첨 확률을 의미하지 않습니다. "
            "같은 주·데이터·분석 기간에는 같은 조합을 표시합니다."
        ),
        "warning": (
            f"최신 적재 데이터 이후 {missing_weeks}주 분량의 추첨 기록이 없습니다. "
            "데이터를 갱신하면 더 최근 기록을 반영할 수 있습니다."
            if missing_weeks
            else ""
        ),
        "disclaimer": DISCLAIMER,
    }
