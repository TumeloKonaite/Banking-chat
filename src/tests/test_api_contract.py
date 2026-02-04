from fastapi.testclient import TestClient


class _Doc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class _FakePipeline:
    def answer_question(self, question, history, doc_type):  # pragma: no cover - simple stub
        docs = [
            _Doc(
                page_content="ATM withdrawal fees may apply based on card type.",
                metadata={"source": "fees.pdf", "doc_type": "product_terms"},
            )
        ]
        return "ATM withdrawal fees depend on the account and card.", docs


def _build_client(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "_check_artifacts", lambda: (True, "Ready"))
    monkeypatch.setattr(app_module, "get_pipeline", lambda: _FakePipeline())
    return TestClient(app_module.app)


def test_health_endpoint_contract(app_module, monkeypatch):
    with _build_client(app_module, monkeypatch) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers.get("X-Request-ID")


def test_request_id_is_forwarded_when_provided(app_module, monkeypatch):
    with _build_client(app_module, monkeypatch) as client:
        response = client.get("/health", headers={"X-Request-ID": "req-123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-123"


def test_ready_endpoint_returns_503_when_not_ready(app_module, monkeypatch):
    with _build_client(app_module, monkeypatch) as client:
        app_module._READY_STATE = (False, "Artifacts missing for test")
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "Artifacts missing for test"


def test_ask_request_schema_validation(app_module, monkeypatch):
    with _build_client(app_module, monkeypatch) as client:
        payload = {
            "question": "",
            "doc_type": "product_terms",
            "history": [],
        }
        response = client.post("/ask", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "Invalid request. Check question length, history length, and payload size."
    assert "errors" in body


def test_ask_response_contract(app_module, monkeypatch):
    with _build_client(app_module, monkeypatch) as client:
        app_module._RATE_LIMIT_ENABLED = False
        payload = {
            "question": "What ATM fees apply?",
            "doc_type": "product_terms",
            "history": [{"role": "user", "content": "Help me with card fees"}],
        }
        response = client.post("/ask", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["answer"], str)
    assert isinstance(body["sources"], list)
    assert body["sources"][0]["source"] == "fees.pdf"
    assert body["sources"][0]["doc_type"] == "product_terms"
    assert "ATM withdrawal fees" in body["sources"][0]["preview"]


def test_ask_rate_limit_returns_429(app_module, monkeypatch):
    with _build_client(app_module, monkeypatch) as client:
        app_module._RATE_LIMIT_ENABLED = True
        app_module._RATE_LIMIT_MAX_REQUESTS = 1
        app_module._RATE_LIMIT_WINDOW_SECONDS = 60
        app_module._RATE_LIMIT_BUCKETS.clear()
        payload = {
            "question": "What ATM fees apply?",
            "doc_type": "product_terms",
            "history": [],
        }
        first = client.post("/ask", json=payload)
        second = client.post("/ask", json=payload)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers.get("Retry-After")
