import csv
import json

import pytest

from src.data import DataError, Draw, load_draws, prepare_dataset


def test_prepare_and_roundtrip(raw_csv, tmp_path):
    destination, registry = tmp_path / "rounds.csv", tmp_path / "registry.json"
    draws = prepare_dataset(raw_csv, destination, registry, limit=100)
    assert len(draws) == 100
    assert draws[0].round == 1120 and draws[-1].round == 1021
    assert load_draws(destination) == draws
    provenance = json.loads(registry.read_text(encoding="utf-8"))["cashlogy"]
    assert provenance["acquisition_method"] == "local_file_import"
    assert provenance["original_downloaded_at"] is None
    prepare_dataset(raw_csv, destination, registry, limit=100)
    assert json.loads(registry.read_text(encoding="utf-8"))["cashlogy"] == provenance


@pytest.mark.parametrize(
    "column,value",
    [
        ("n1", "46"),
        ("n2", "10"),
        ("bonus", "10"),
        ("drwNoDate", "2026-02-30"),
        ("firstWinamnt", "-1"),
        ("firstPrzwnerCo", "abc"),
        ("n1", "1.5"),
        ("drwNo", "0"),
    ],
)
def test_reject_invalid_csv_rows(raw_csv, column, value):
    with raw_csv.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    # 첫 행은 1120회이므로 10·20·22·30·40·45가 있는 두 번째 행을 사용한다.
    rows[1][column] = value
    with raw_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DataError, match="검증 실패"):
        load_draws(raw_csv)


def test_duplicate_is_rejected(raw_csv):
    lines = raw_csv.read_text(encoding="utf-8-sig").splitlines()
    with raw_csv.open("a", encoding="utf-8") as stream:
        stream.write(lines[1] + "\n")
    with pytest.raises(DataError, match="검증 실패"):
        load_draws(raw_csv)


def test_minimum_and_missing_header(raw_csv, tmp_path):
    with pytest.raises(DataError, match="최소 200"):
        load_draws(raw_csv, minimum=200)
    bad = tmp_path / "bad.csv"
    bad.write_text("round,draw_date\n1,2026-10-17\n", encoding="utf-8")
    with pytest.raises(DataError, match="필수 열"):
        load_draws(bad)


def test_bonus_is_not_in_derived_statistics():
    draw = Draw(
        round=1,
        draw_date="2026-10-17",
        numbers=(1, 3, 5, 10, 20, 21),
        bonus_number=7,
        first_prize_amount=100,
        first_prize_winning_games=1,
    )
    assert draw.odd_count == 4
    assert draw.consecutive_pairs == 1
    assert draw.metadata("test")["number_sum"] == 60
