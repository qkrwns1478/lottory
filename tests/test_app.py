from streamlit.testing.v1 import AppTest

from src.data import prepare_dataset
from src.settings import ROOT, load_settings


def component_events(monkeypatch):
    events = []
    monkeypatch.setattr(
        "streamlit.components.v1.declare_component",
        lambda *args, **kwargs: lambda **props: events.pop(0) if events else None,
    )
    return events


def test_chat_followup_and_selection(raw_csv, tmp_path, monkeypatch):
    destination = tmp_path / "rounds.csv"
    prepare_dataset(raw_csv, destination, tmp_path / "registry.json")
    settings = load_settings()
    settings["data"]["csv_path"] = str(destination)
    monkeypatch.setattr("src.settings.load_settings", lambda: settings)
    events = component_events(monkeypatch)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception
    assert not app.tabs and not app.chat_input and not app.sidebar.selectbox
    events.append(
        {
            "id": "1",
            "action": "ask",
            "question": "1100회 당첨금은?",
            "selected_round": 1120,
        }
    )
    app.run()
    assert not app.exception
    assert app.session_state["custom_round"] == 1100
    events.append({"id": "2", "action": "ask", "question": "당첨번호는?", "selected_round": 1100})
    app.run()
    assert "제1100회" in app.session_state["custom_messages"][-1]["response"]["answer"]
    events.append({"id": "3", "action": "ask", "question": "당첨번호는?", "selected_round": 1110})
    app.run()
    assert "제1110회" in app.session_state["custom_messages"][-1]["response"]["answer"]
    assert not app.exception
    events.append({"id": "4", "action": "clear", "selected_round": 1110})
    app.run()
    assert app.session_state["custom_messages"] == []


def test_missing_data_has_setup_instructions(tmp_path, monkeypatch):
    settings = load_settings()
    settings["data"]["csv_path"] = str(tmp_path / "missing.csv")
    monkeypatch.setattr("src.settings.load_settings", lambda: settings)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not app.exception
    assert "검증된 회차 데이터가 없습니다" in app.warning[0].value


def test_filter_form(raw_csv, tmp_path, monkeypatch):
    destination = tmp_path / "rounds.csv"
    prepare_dataset(raw_csv, destination, tmp_path / "registry.json")
    settings = load_settings()
    settings["data"]["csv_path"] = str(destination)
    monkeypatch.setattr("src.settings.load_settings", lambda: settings)
    events = component_events(monkeypatch)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    events.append({"id": "filter", "action": "filter", "filters": {"numbers": [22]}})
    app.run()
    assert not app.exception
    assert len(app.session_state["custom_result"]["rows"]) == 60


def test_provider_errors_are_not_exposed(raw_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-not-sent")
    destination = tmp_path / "rounds.csv"
    prepare_dataset(raw_csv, destination, tmp_path / "registry.json")
    settings = load_settings()
    settings["data"]["csv_path"] = str(destination)
    monkeypatch.setattr("src.settings.load_settings", lambda: settings)

    def fail(settings):
        raise RuntimeError("sensitive-test-key and full prompt and traceback")

    monkeypatch.setattr("src.planner.make_planner", fail)
    events = component_events(monkeypatch)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    events.append(
        {
            "id": "ai",
            "action": "ask",
            "question": "1100회 당첨금은?",
            "selected_round": 1100,
            "ai": True,
        }
    )
    app.run()
    assert not app.exception
    result = app.session_state["custom_result"]
    assert result["kind"] == "error"
    assert "API 연결" in result["message"]
    assert "sensitive-test-key" not in str(result)


def test_round_change_clears_messages_before_ai_receives_history(raw_csv, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.service import answer_question

    destination = tmp_path / "rounds.csv"
    prepare_dataset(raw_csv, destination, tmp_path / "registry.json")
    settings = load_settings()
    settings["data"]["csv_path"] = str(destination)
    monkeypatch.setattr("src.settings.load_settings", lambda: settings)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    monkeypatch.setattr("src.planner.make_planner", lambda _: None)
    histories = []

    def answer(question, selected_round, draws, **kwargs):
        histories.append(kwargs["history"])
        return answer_question(question, selected_round, draws)

    monkeypatch.setattr(
        "src.workflow.make_workflow", lambda *a, **k: SimpleNamespace(answer=answer)
    )
    events = component_events(monkeypatch)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    for i, selected in enumerate([1100, 1100, 1110]):
        events.append(
            {
                "id": str(i),
                "action": "ask",
                "question": "당첨번호는?",
                "selected_round": selected,
                "ai": True,
            }
        )
        app.run()
        assert not app.exception
    assert histories[0] == []
    assert len(histories[1]) == 1
    assert histories[2] == []
    assert len(app.session_state["custom_messages"]) == 1
    assert "제1110회" in app.session_state["custom_messages"][0]["response"]["answer"]


def test_missing_ai_key_has_actionable_message(raw_csv, tmp_path, monkeypatch):
    destination = tmp_path / "rounds.csv"
    prepare_dataset(raw_csv, destination, tmp_path / "registry.json")
    settings = load_settings()
    settings["data"]["csv_path"] = str(destination)
    monkeypatch.setattr("src.settings.load_settings", lambda: settings)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    events = component_events(monkeypatch)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    events.append(
        {
            "id": "missing-key",
            "action": "ask",
            "question": "당첨번호는?",
            "selected_round": 1100,
            "ai": True,
        }
    )
    app.run()
    assert not app.exception
    result = app.session_state["custom_result"]
    assert result["kind"] == "error" and "OPENAI_API_KEY가 없습니다" in result["message"]
    assert "로컬 조회" in result["message"]


def test_empty_index_has_actionable_message(raw_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-not-sent")
    destination = tmp_path / "rounds.csv"
    prepare_dataset(raw_csv, destination, tmp_path / "registry.json")
    settings = load_settings()
    settings["data"]["csv_path"] = str(destination)
    settings["vector_store"]["path"] = str(tmp_path / "empty-index")
    monkeypatch.setattr("src.settings.load_settings", lambda: settings)
    monkeypatch.setattr("src.planner.make_planner", lambda _: None)
    monkeypatch.setattr("src.workflow.make_embeddings", lambda _: None)
    events = component_events(monkeypatch)
    app = AppTest.from_file(str(ROOT / "app.py")).run()
    events.append(
        {
            "id": "empty-index",
            "action": "ask",
            "question": "당첨번호는?",
            "selected_round": 1100,
            "ai": True,
        }
    )
    app.run()
    assert not app.exception
    result = app.session_state["custom_result"]
    assert result["kind"] == "error"
    assert "python -m scripts.prepare_data --index" in result["message"]
    events.append(
        {
            "id": "local",
            "action": "ask",
            "question": "당첨번호는?",
            "selected_round": 1100,
        }
    )
    app.run()
    assert app.session_state["custom_result"]["response"]["status"] == "answered"
