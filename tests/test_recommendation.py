from collections import Counter
from datetime import date, timedelta

import pytest

from src.frontend import handle_event
from src.recommendation import recommend_numbers
from src.settings import load_settings


def test_analysis_and_reproducible_valid_games(draws):
    today = draws[-1].draw_date + timedelta(days=5)
    report = recommend_numbers(draws, today=today)
    expected = Counter(n for draw in draws[-100:] for n in draw.numbers)
    assert report["row_count"] == 100
    assert report["target_date"] == (today + timedelta(days=2)).isoformat()
    assert {row["number"]: row["count"] for row in report["frequencies"]} == {
        n: expected[n] for n in range(1, 46)
    }
    assert sum(row["recent_count"] for row in report["frequencies"]) == 120
    assert len({tuple(game["numbers"]) for game in report["games"]}) == 5
    for game in report["games"]:
        assert len(set(game["numbers"])) == 6
        assert all(1 <= n <= 45 for n in game["numbers"])
        assert game["numbers"] == sorted(game["numbers"])
    assert report == recommend_numbers(list(reversed(draws)), today=today)
    assert (
        recommend_numbers(draws, game_count=1, today=today)["games"]
        == report["games"][:1]
    )
    assert "당첨을 예측하거나 수익을 보장하지 않습니다" in report["disclaimer"]


def test_excludes_today_and_future_and_reports_short_history(draws):
    today = draws[10].draw_date
    report = recommend_numbers(draws, today=today)
    assert report["row_count"] == 10
    assert report["to_round"] == draws[9].round
    assert report["target_date"] == today.isoformat()
    with pytest.raises(ValueError):
        recommend_numbers(draws, today=draws[0].draw_date)


def test_stale_data_and_sunday_target(draws):
    today = draws[-1].draw_date + timedelta(days=15)
    report = recommend_numbers(draws, today=today)
    assert date.fromisoformat(report["target_date"]).weekday() == 5
    assert report["warning"]


@pytest.mark.parametrize("window,count", [(10, 5), (100, 0), (100, 6)])
def test_invalid_options(draws, window, count):
    with pytest.raises(ValueError):
        recommend_numbers(draws, window=window, game_count=count)


def test_frontend_event_uses_local_data_without_engine(draws):
    def no_engine(_):
        pytest.fail("추천은 외부 API를 호출하지 않아야 합니다.")

    result = handle_event(
        {
            "id": "recommend",
            "action": "recommend",
            "recommendation_window": 20,
            "game_count": 2,
        },
        load_settings(),
        draws,
        no_engine,
    )
    assert result["kind"] == "recommendation"
    assert result["recommendation"]["row_count"] == 20
    assert len(result["recommendation"]["games"]) == 2
