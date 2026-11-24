import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from PIL import Image


@pytest.fixture(scope="module")
def client():
    from src.atdl.web.app import create_app

    with TestClient(create_app(device="cpu")) as instance:
        yield instance


@pytest.fixture(scope="module")
def frame():
    image = Image.new("RGB", (360, 200), (120, 120, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_case_study_uses_final_selection_and_saved_evidence(client):
    data = client.get("/api/project").json()
    assert data["default_model"] == "combined"
    assert data["models"]["combined"]["seed"] == 42
    assert data["models"]["standard320"]["size"] == 320
    assert data["dataset"]["image_counts"] == {"train": 1257, "val": 266, "test": 276}
    assert len(data["classes"]) == 6
    assert len(data["training"]["combined"]) == 20
    assert data["test"]["mAP50"] == pytest.approx(0.7842, abs=0.0001)
    assert data["thresholds"]["crazing"] == 0.14
    assert client.get("/api/figures/final_test_confusion_matrix.png").status_code == 200


def test_predict_validates_media_and_model_key(client, frame):
    assert client.post("/api/predict", content=b"invalid").status_code == 422
    assert client.post("/api/predict?model=unregistered", content=frame).status_code == 422


def test_real_prediction_exports_input_geometry_and_activations(client, frame):
    response = client.post("/api/predict?model=combined&trace=1&activations=1", content=frame)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["device"] == "cpu"
    assert result["model"] == "combined"
    assert result["source_size"] == [360, 200]
    assert result["input"]["shape"] == [1, 3, 384, 640]
    assert result["input"]["preview"].startswith("data:image/png;base64,")
    assert result["input"]["padding"] == [0, 14, 0, 14]
    assert len(result["activations"]) >= 2
    for box in result["boxes"]:
        x1, y1, x2, y2 = box["xyxy"]
        assert 0 <= x1 <= x2 <= 360 and 0 <= y1 <= y2 <= 200
        assert 0.01 <= box["score"] <= 1


def test_stream_replies_with_correlated_results_after_an_invalid_frame(client, frame):
    with client.websocket_connect("/api/stream") as socket:
        socket.send_json({"id": 41, "source_time": 1.0, "model": "combined"})
        socket.send_bytes(b"invalid")
        assert socket.receive_json()["error"]
        socket.send_json({"id": 42, "source_time": 1.25, "model": "combined"})
        socket.send_bytes(frame)
        result = socket.receive_json()
    assert result["id"] == 42
    assert result["source_time"] == 1.25
    assert result["result"]["source_size"] == [360, 200]
    assert result["result"]["input"]["preview"] is None


def test_sample_has_visible_variation_and_labeled_geometry(client):
    first = client.get("/api/sample/image?position=0").content
    second = client.get("/api/sample/image?position=8").content
    assert first != second
    frame = cv2.imdecode(np.frombuffer(first, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert frame.shape == (360, 640, 3)
    assert client.get("/api/sample/clip").content[:4] == b"\x1a\x45\xdf\xa3"
    assert client.get("/api/sample/annotation").json()["boxes"]


def test_cpu_is_reported_explicitly(client):
    status = client.get("/api/health").json()
    assert status["ready"]
    assert status["device"] == "cpu"
    assert "combined" in status["loaded_models"]



def test_training_logs_for_all_choices_are_available(client):
    for model in ("combined", "standard", "standard320"):
        for seed in (42, 43, 44):
            response = client.get(f"/api/training/{model}/{seed}")
            assert response.status_code == 200
            data = response.json()
            assert data["seed"] == seed and len(data["history"]) > 0


def test_stream_rejects_unexpected_message_types(client):
    with client.websocket_connect("/api/stream") as socket:
        socket.send_bytes(b"metadata must be text")
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()
        assert error.value.code == 1003


def test_required_cuda_does_not_fall_back_to_cpu(monkeypatch):
    import torch
    from src.atdl.web.app import create_app

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with TestClient(create_app(device="cuda")) as client:
        status = client.get("/api/health").json()
        assert not status["ready"]
        assert "CUDA was requested" in status["error"]
        assert client.post("/api/predict", content=b"frame").status_code == 503
        assert client.get("/api/project").status_code == 200
