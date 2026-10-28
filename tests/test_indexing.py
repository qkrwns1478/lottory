import chromadb
import pytest
from chromadb.config import Settings

from src.indexing import index_rounds, open_collection, query_collection, search_rounds
from src.settings import load_settings


class TestEmbeddings:
    """외부 API 없이 재현할 수 있는 테스트 전용 벡터이며 실제 앱에서는 사용하지 않는다."""

    __test__ = False

    def __init__(self):
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.5]


def test_real_chroma_upsert_query_and_replacement(tmp_path, draws):
    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection("test_rounds", embedding_function=None)
    embeddings = TestEmbeddings()
    first = index_rounds(draws, collection, embeddings, "test-date", batch_size=64)
    assert first["collection_count"] == 120 and first["ids_verified"]
    assert first["changed_count"] == 120
    second = index_rounds(draws, collection, embeddings, "test-date")
    assert second["changed_count"] == 0
    assert embeddings.document_calls == 2
    result = search_rounds("1100회 당첨금", collection, embeddings, round_number=1100)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "lotto_1100"
    assert result[0]["metadata"]["round"] == 1100
    assert search_rounds("9999회", collection, embeddings, round_number=9999) == []
    replacement = index_rounds(draws[-100:], collection, embeddings, "test-date")
    assert replacement["collection_count"] == 100
    assert replacement["removed_count"] == 20


def test_failed_embedding_does_not_delete_previous_data(tmp_path, draws):
    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection("test_rounds", embedding_function=None)
    index_rounds(draws, collection, TestEmbeddings(), "old-date")

    class FailingEmbeddings:
        def embed_documents(self, texts):
            raise RuntimeError("simulated outage")

    with pytest.raises(RuntimeError):
        index_rounds(draws[:100], collection, FailingEmbeddings(), "new-date")
    assert collection.count() == 120


def test_embedding_model_change_requires_new_collection(tmp_path):
    settings = load_settings()
    settings["vector_store"]["path"] = str(tmp_path / "chroma")
    collection = open_collection(settings)
    assert collection.metadata["embedding_fingerprint"]
    settings["embedding"]["model"] = "different-model"
    with pytest.raises(ValueError, match="임베딩 설정"):
        open_collection(settings)


def test_transient_segment_read_retries_without_reembedding(monkeypatch):
    from chromadb.errors import InternalError

    class TransientCollection:
        calls = 0

        def query(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise InternalError("Error creating hnsw segment reader: Nothing found on disk")
            return {"ids": [["ok"]]}

    monkeypatch.setattr("src.indexing.time.sleep", lambda _: None)
    collection = TransientCollection()
    assert query_collection(collection, query_embeddings=[[1, 2, 3]]) == {"ids": [["ok"]]}
    assert collection.calls == 3
