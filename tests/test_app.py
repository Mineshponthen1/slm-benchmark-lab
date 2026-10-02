from types import SimpleNamespace

from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


def fake_chat(**kwargs):
    return {
        "message": {"content": "  KYC means Know Your Customer.  "},
        "eval_count": 20,
        "eval_duration": 2_000_000_000,
    }


def broken_ollama(**kwargs):
    raise ConnectionError("Ollama is not running")


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


def test_ask_sends_system_prompt_and_settings(monkeypatch):
    captured = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return fake_chat()

    monkeypatch.setattr(app_module.ollama, "chat", spy)
    client.post("/ask", json={"question": "What is AML?"})
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["content"] == "What is AML?"
    assert captured["options"]["temperature"] == 0


# ---------- Bad requests are stopped at the door ----------

def test_bad_requests_never_reach_the_model(monkeypatch):
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