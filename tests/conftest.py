import csv
from datetime import date, timedelta

import pytest

from src.data import Draw


@pytest.fixture
def draws():
    # 테스트 전용 합성 데이터이며 실제 앱에서는 사용하지 않는다.
    return [
        Draw(
            round=1001 + i,
            draw_date=date(2024, 1, 6) + timedelta(weeks=i),
            numbers=(1, 2, 3, 4, 5, 6) if i % 2 else (10, 20, 22, 30, 40, 45),
            bonus_number=7,
            first_prize_amount=1_000_000_000 + i * 1_000_000,
            first_prize_winning_games=10 + i % 5,
        )
        for i in range(120)
    ]


@pytest.fixture
def raw_csv(tmp_path, draws):
    path = tmp_path / "raw.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "drwNo",
                "drwNoDate",
                *[f"n{i}" for i in range(1, 7)],
                "bonus",
                "firstWinamnt",
                "firstPrzwnerCo",
            ]
        )
        for draw in reversed(draws):
            writer.writerow(
                [
                    draw.round,
                    draw.draw_date,
                    *draw.numbers,
                    draw.bonus_number,
                    draw.first_prize_amount,
                    draw.first_prize_winning_games,
                ]
            )
    return path
