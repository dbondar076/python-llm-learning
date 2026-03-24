import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.services.llm_service import reset_runtime_state


@pytest.fixture
def client() -> TestClient:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_llm_runtime_state() -> None:
    reset_runtime_state()