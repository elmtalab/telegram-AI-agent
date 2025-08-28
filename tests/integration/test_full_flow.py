from fastapi.testclient import TestClient
from apps.chat_adapter.main import app as chat_app

client = TestClient(chat_app)

def test_full_flow():
    response = client.post("/ingest", json={})
    assert response.status_code == 200
