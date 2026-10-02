import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


def make_stream(text="  KYC means Know Your Customer.  ", tokens=20, duration_ns=2_000_000_000):
    return [
        {"message": {"content": text[:10]}, "done": False},
        {"message": {"content": text[10:]}, "done": False},
        {"message": {"content": ""}, "done": True,
         "eval_count": tokens, "eval_duration": duration_ns},
    ]


def fake_chat(**kwargs):
    return make_stream()


def broken_ollama(**kwargs):
    raise ConnectionError("Ollama is not running")


@pytest.fixture(autouse=True)
def temp_log(tmp_path, monkeypatch):
    log_file = tmp_path / "requests.jsonl"
    monkeypatch.setattr(app_module, "LOG_FILE", log_file)
    return log_file


# ---------- Good requests ----------

def test_ask_returns_answer(monkeypatch):
    monkeypatch.setattr(app_module.ollama, "chat", fake_chat)
    r = client.post("/ask", json={"question": "What is KYC?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "KYC means Know Your Customer."
    assert body["model"] == app_module.DEFAULT_MODEL
    assert body["tokens"] == 20
    assert body["tokens_per_second"] == 10.0
    assert body["ttft_seconds"] >= 0


def test_ask_sends_system_prompt_and_settings(monkeypatch):
    captured = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return make_stream()

    monkeypatch.setattr(app_module.ollama, "chat", spy)
    client.post("/ask", json={"question": "What is AML?"})
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["content"] == "What is AML?"
    assert captured["options"]["temperature"] == 0
    assert captured["stream"] is True


# ---------- Bad requests are stopped at the door ----------

def test_bad_requests_never_reach_the_model(monkeypatch, temp_log):
    calls = []
    monkeypatch.setattr(app_module.ollama, "chat", lambda **kw: calls.append(kw))

    bad_requests = [
        {"question": ""},
        {},
        {"question": "a" * 2001},
        {"question": "Hello", "model": "gpt-4"},
    ]
    for body in bad_requests:
        r = client.post("/ask", json=body)
        assert r.status_code == 422, f"Expected 422 for {body}"

    assert calls == []
    assert not temp_log.exists()


# ---------- When Ollama is switched off ----------

def test_ask_returns_503_when_ollama_down(monkeypatch):
    monkeypatch.setattr(app_module.ollama, "chat", broken_ollama)
    r = client.post("/ask", json={"question": "Hello"})
    assert r.status_code == 503


def test_health_ok(monkeypatch):
    fake_list = lambda: SimpleNamespace(models=[SimpleNamespace(model=app_module.DEFAULT_MODEL)])
    monkeypatch.setattr(app_module.ollama, "list", fake_list)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["default_model_installed"] is True


def test_health_returns_503_when_ollama_down(monkeypatch):
    monkeypatch.setattr(app_module.ollama, "list", broken_ollama)
    r = client.get("/health")
    assert r.status_code == 503


# ---------- Request logging ----------

def test_successful_request_is_logged(monkeypatch, temp_log):
    monkeypatch.setattr(app_module.ollama, "chat", fake_chat)
    question = "What is KYC?"
    client.post("/ask", json={"question": question})

    lines = temp_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "ok"
    assert record["tokens"] == 20
    assert record["tokens_per_second"] == 10.0
    assert record["question_chars"] == len(question)
    assert "question" not in record


def test_failed_request_is_logged(monkeypatch, temp_log):
    monkeypatch.setattr(app_module.ollama, "chat", broken_ollama)
    client.post("/ask", json={"question": "Hello"})

    record = json.loads(temp_log.read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "error"
    assert record["error"] == "ConnectionError"