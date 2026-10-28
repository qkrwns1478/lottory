# 회차 데이터를 검증하고 수치 답변의 기준 데이터를 제공한다.

import csv
import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

CSV_URL = "https://cashlogy.kr/lotto/excel.csv"
OFFICIAL_URL = "https://www.dhlottery.co.kr/lt645/winNumber"


class DataError(ValueError):
    pass


class Draw(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    round: int = Field(gt=0)
    draw_date: date
    numbers: tuple[int, int, int, int, int, int]
    bonus_number: int = Field(ge=1, le=45)
    first_prize_amount: int = Field(ge=0)
    first_prize_winning_games: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_numbers(self):
        if len(set(self.numbers)) != 6 or any(not 1 <= n <= 45 for n in self.numbers):
            raise ValueError("당첨번호는 1~45의 서로 다른 정수 6개여야 합니다.")
        if self.bonus_number in self.numbers:
            raise ValueError("보너스 번호가 당첨번호와 중복됩니다.")
        return self

    @property
    def odd_count(self):
        return sum(n % 2 for n in self.numbers)

    @property
    def consecutive_pairs(self):
        numbers = sorted(self.numbers)
        return sum(b - a == 1 for a, b in zip(numbers, numbers[1:]))

    def search_text(self):
        return (
            f"로또 6/45 제{self.round}회 회차 기록\n"
            f"추첨일 {self.draw_date}\n"
            f"당첨번호 {', '.join(map(str, self.numbers))}\n"
            f"보너스 {self.bonus_number}\n"
            f"1게임당 1등 당첨금 {self.first_prize_amount:,}원\n"
            f"1등 당첨 게임 수 {self.first_prize_winning_games}게임\n"
            f"홀수 {self.odd_count}개, 짝수 {6 - self.odd_count}개\n"
            f"번호 합계 {sum(self.numbers)}, 연속 번호 쌍 {self.consecutive_pairs}개"
        )

    def metadata(self, collected_at):
        return {
            "round_id": f"lotto_{self.round}",
            "round": self.round,
            "draw_date": self.draw_date.isoformat(),
            "bonus_number": self.bonus_number,
            "first_prize_amount": self.first_prize_amount,
            "first_prize_winning_games": self.first_prize_winning_games,
            "odd_count": self.odd_count,
            "even_count": 6 - self.odd_count,
            "number_sum": sum(self.numbers),
            "has_consecutive": self.consecutive_pairs > 0,
            "document_title": f"로또 6/45 제{self.round}회 회차 기록",
            "source_url": CSV_URL,
            "source_name": "캐시로지 / 원천 발표 동행복권",
            "collected_at": collected_at,
        }


ALIASES = {
    "round": "drwNo",
    "draw_date": "drwNoDate",
    "bonus_number": "bonus",
    "first_prize_amount": "firstWinamnt",
    "first_prize_winning_games": "firstPrzwnerCo",
}
FIELDS = [
    "round",
    "draw_date",
    *(f"n{i}" for i in range(1, 7)),
    "bonus_number",
    "first_prize_amount",
    "first_prize_winning_games",
]


def load_draws(path: str | Path, minimum=100) -> list[Draw]:
    """잘못된 행이나 중복 회차가 있으면 값이 같아도 입력 전체를 거부한다."""
    draws = {}
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = set(reader.fieldnames or [])
            mapping = {key: key if key in headers else ALIASES.get(key, key) for key in FIELDS}
            if any(column not in headers for column in mapping.values()):
                raise DataError(
                    "CSV에 필수 열이 없습니다. 제공된 CSV 또는 정규화 CSV를 사용하세요."
                )
            for line, row in enumerate(reader, start=2):
                try:
                    values = {key: row[column].strip() for key, column in mapping.items()}
                    numbers = tuple(int(values.pop(f"n{i}")) for i in range(1, 7))
                    draw = Draw(numbers=numbers, **values)
                    if draw.round in draws:
                        raise ValueError("중복 회차")
                    draws[draw.round] = draw
                except (ValueError, TypeError, AttributeError) as exc:
                    raise DataError(
                        f"CSV {line}행 검증 실패: 번호·날짜·금액·중복 회차를 확인하세요."
                    ) from exc
    except (UnicodeError, csv.Error) as exc:
        raise DataError("UTF-8 CSV 파일 형식을 확인하세요.") from exc
    if len(draws) < minimum:
        raise DataError(f"고유 회차 {len(draws)}건: 최소 {minimum}건이 필요합니다.")
    return sorted(draws.values(), key=lambda item: item.round, reverse=True)


def atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as stream:
            name = stream.name
            stream.write(content)
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def prepare_dataset(source: Path, destination: Path, registry_path: Path, limit=200, minimum=100):
    import io

    if limit < minimum:
        raise DataError("수집 목표 건수는 최소 검증 건수 이상이어야 합니다.")
    draws = load_draws(source, minimum=minimum)[:limit]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    for draw in draws:
        row = draw.model_dump(exclude={"numbers"})
        row.update({f"n{i}": number for i, number in enumerate(draw.numbers, 1)})
        writer.writerow(row)
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    )
    dataset_hash = hashlib.sha256(output.getvalue().encode("utf-8")).hexdigest()
    raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    previous = registry.get("cashlogy", {})
    collected_at = (
        previous.get("collected_at")
        if previous.get("dataset_sha256") == dataset_hash and previous.get("raw_sha256") == raw_hash
        else None
    )
    collected_at = collected_at or datetime.now(timezone.utc).isoformat()
    registry["cashlogy"] = {
        "source_id": "cashlogy",
        "source_name": "캐시로지 / 원천 발표 동행복권",
        "source_url": CSV_URL,
        "official_url": OFFICIAL_URL,
        "license": "CC BY 4.0 (제공 안내 페이지 표시)",
        "usage_scope": "비영리 교육용 CSV 로컬 가공·색인",
        "acquisition_method": "local_file_import",
        "collected_at": collected_at,
        "original_downloaded_at": None,
        "raw_sha256": raw_hash,
        "dataset_sha256": dataset_hash,
        "row_count": len(draws),
        "from_round": draws[-1].round,
        "to_round": draws[0].round,
    }
    atomic_write(destination, output.getvalue())
    atomic_write(registry_path, json.dumps(registry, ensure_ascii=False, indent=2))
    return draws
