from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["status"] == "healthy"


def test_ready_endpoint():
    response = client.get("/api/ready")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["status"] == "ready"


def test_analyze_rejects_invalid_sensitive_payload():
    csv_content = b"Loan Approved,Gender,Income\n1,Male,50000\n0,Female,45000\n"
    response = client.post(
        "/api/analyze",
        data={
            "target_column": "Loan Approved",
            "sensitive_features": '{"not":"a-list"}',
        },
        files={"dataset": ("sample.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 400
    assert "sensitive_features" in response.json()["detail"]


def test_mitigate_rejects_invalid_fixes_payload():
    csv_content = b"Loan Approved,Gender,Income\n1,Male,50000\n0,Female,45000\n"
    response = client.post(
        "/api/mitigate",
        data={
            "target_column": "Loan Approved",
            "sensitive_features": '["Gender"]',
            "fixes": "[]",
        },
        files={"dataset": ("sample.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 400
    assert "fixes" in response.json()["detail"]


def test_analyze_supports_multiple_sensitive_columns():
    csv_content = b"outcome,sex,region,age,experience\n1,Male,Urban,30,5\n0,Female,Rural,25,3\n1,Male,Rural,35,10\n0,Female,Urban,28,4\n1,Male,Urban,40,8\n0,Female,Rural,22,1\n1,Male,Urban,38,7\n0,Female,Rural,27,2\n"
    response = client.post(
        "/api/analyze",
        data={
            "target_column": "outcome",
            "sensitive_features": '["sex", "region"]',
        },
        files={"dataset": ("sample.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("group_metrics"), dict)
    assert any("Male | Urban" in key or "Female | Rural" in key for key in payload["group_metrics"].keys())
