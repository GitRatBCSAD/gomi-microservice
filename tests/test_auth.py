"""ML endpoints must reject requests without the shared ML_API_KEY."""
import pytest
from fastapi import HTTPException

from app.api.routes.analyze import require_api_key


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("ML_API_KEY", "secret")


def test_accepts_matching_bearer_token():
    assert require_api_key("Bearer secret") is None


@pytest.mark.parametrize("header", ["", "secret", "Bearer wrong", "Bearer secret2"])
def test_rejects_missing_or_wrong_token(header):
    with pytest.raises(HTTPException) as exc:
        require_api_key(header)
    assert exc.value.status_code == 401
