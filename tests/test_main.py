import json
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

# Import the FastAPI application. Adjust the import path if the app is defined elsewhere.
# The project is expected to expose a variable named `app` in a module called `main`.
from main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_code():
    """
    Provide a small Python code snippet that can be used for testing the review endpoint.
    """
    return """def add(a, b):
    return a + b
"""


def test_root_endpoint_returns_200_and_openapi():
    """
    Ensure that the root endpoint ("/") is reachable and returns a JSON response
    containing basic API information.
    """
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # The root should provide a simple health check or description.
    assert isinstance(data, dict)
    assert "message" in data
    assert data["message"] == "LLM-Powered Code Review Assistant is running."


def test_review_endpoint_success(sample_code):
    """
    Test that posting a valid code snippet to the /review endpoint returns a review.
    """
    payload = {"code": sample_code}
    response = client.post("/review", json=payload)

    assert response.status_code == status.HTTP_200_OK, f"Unexpected status: {response.text}"
    result = response.json()
    # The response should contain a 'review' field with a non-empty string.
    assert "review" in result
    review_text = result["review"]
    assert isinstance(review_text, str)
    assert len(review_text.strip()) > 0
    # Basic sanity check: the review should mention the function name.
    assert "add" in review_text.lower()


def test_review_endpoint_missing_code():
    """
    Verify that the endpoint returns a 422 Unprocessable Entity error when the required
    'code' field is missing from the request body.
    """
    response = client.post("/review", json={})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    error_detail = response.json()
    assert "detail" in error_detail
    # FastAPI includes validation errors in the 'detail' list.
    assert any(
        err.get("loc") == ["body", "code"] and err.get("msg") == "field required"
        for err in error_detail["detail"]
    )


def test_review_endpoint_invalid_json():
    """
    Ensure that sending malformed JSON results in a 400 Bad Request response.
    """
    # Simulate sending raw data that is not valid JSON.
    response = client.post("/review", data="not-a-json")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    error_detail = response.json()
    assert "detail" in error_detail
    assert "JSON decode error" in error_detail["detail"]


def test_model_loading_is_lazy(monkeypatch, tmp_path):
    """
    Confirm that the model is loaded lazily (i.e., only when the first review request
    is made). This test monkeypatches the model loading function to track calls.
    """
    # Assume the project defines a function `load_model` in `review.py`.
    # We'll replace it with a stub that records invocation.
    import importlib

    review_module = importlib.import_module("review")
    load_calls = []

    def fake_load_model():
        load_calls.append(True)
        # Return a dummy object that mimics the interface used in the review function.
        class DummyModel:
            def __call__(self, *args, **kwargs):
                return "Dummy review."

        return DummyModel()

    monkeypatch.setattr(review_module, "load_model", fake_load_model)

    # Ensure no model is loaded at import time.
    assert len(load_calls) == 0

    # Trigger a review request.
    payload = {"code": "print('hello')"}
    response = client.post("/review", json=payload)
    assert response.status_code == status.HTTP_200_OK
    # The fake loader should have been called exactly once.
    assert len(load_calls) == 1
    result = response.json()
    assert result["review"] == "Dummy review."


def test_openapi_schema_is_valid():
    """
    Retrieve the OpenAPI schema and perform a minimal validation to ensure it
    contains expected paths and components.
    """
    response = client.get("/openapi.json")
    assert response.status_code == status.HTTP_200_OK
    schema = response.json()
    assert isinstance(schema, dict)
    # The schema should define the /review path.
    assert "/review" in schema.get("paths", {})
    # Verify that the request body schema for /review expects a 'code' property.
    review_path = schema["paths"]["/review"]["post"]
    request_body = review_path["requestBody"]["content"]["application/json"]["schema"]
    assert "properties" in request_body
    assert "code" in request_body["properties"]
    # The response schema should contain a 'review' property.
    response_schema = review_path["responses"]["200"]["content"]["application/json"]["schema"]
    assert "properties" in response_schema
    assert "review" in response_schema["properties"]


def test_static_files_served_correctly():
    """
    If the application serves static files (e.g., a simple HTML UI), verify that the
    index page is accessible.
    """
    # Assuming the static files are mounted at the root and an index.html exists.
    static_path = Path(__file__).parents[2] / "static" / "index.html"
    if static_path.is_file():
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        # The content type should be HTML.
        assert response.headers["content-type"].startswith("text/html")
    else:
        pytest.skip("Static index.html not present in the repository.")