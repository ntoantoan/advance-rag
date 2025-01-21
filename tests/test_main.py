from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "active"

def test_create_document():
    test_doc = {
        "title": "Test Document",
        "content": "This is a test document"
    }
    response = client.post("/documents/", json=test_doc)
    assert response.status_code == 200
    assert response.json()["title"] == test_doc["title"]
    assert response.json()["content"] == test_doc["content"]
    assert "id" in response.json()

def test_get_documents():
    response = client.get("/documents/")
    assert response.status_code == 200
    assert isinstance(response.json(), list) 