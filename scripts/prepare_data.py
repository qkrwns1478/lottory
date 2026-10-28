import argparse
import json

from dotenv import load_dotenv

from src.data import prepare_dataset
from src.settings import ROOT, load_settings, resolve_path


def main():
    parser = argparse.ArgumentParser(description="회차 CSV를 검증·정규화하고 필요에 따라 색인")
    parser.add_argument("--source", help="로컬 CSV 경로 (기본: config.yaml)")
    parser.add_argument("--index", action="store_true", help="OpenAI 임베딩 API를 호출해 Chroma에 색인")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    settings = load_settings()
    config = settings["data"]
    try:
        registry_path = resolve_path(config["source_registry"])
        draws = prepare_dataset(
            resolve_path(args.source or config["raw_csv"]),
            resolve_path(config["csv_path"]),
            registry_path,
            limit=config["target_rounds"],
            minimum=config["minimum_rounds"],
        )
        report = {
            "validated_rounds": len(draws),
            "from_round": draws[-1].round,
            "to_round": draws[0].round,
        }
        if args.index:
            from src.indexing import index_rounds, make_embeddings, open_collection

            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            report["index"] = index_rounds(
                draws,
                open_collection(settings),
                make_embeddings(settings),
                registry["cashlogy"]["collected_at"],
                settings["embedding"]["batch_size"],
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception:
        # 공급자 내부 오류·인증값·전체 프롬프트·예외 추적을 콘솔과 화면에 노출하지 않는다.
        parser.exit(1, "데이터 준비 또는 색인에 실패했습니다. CSV·설정·API 연결을 확인하세요.\n")


if __name__ == "__main__":
    main()
