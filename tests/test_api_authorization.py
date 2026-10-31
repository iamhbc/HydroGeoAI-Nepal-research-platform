from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hydrogeoai.api.routers import admin, experiments


@pytest.fixture
def api(monkeypatch):
    submit = Mock(return_value="fixture-job")
    state = SimpleNamespace(admin_key="private-test-key", jobs=SimpleNamespace(submit=submit),
                            registry=SimpleNamespace(list_experiments=lambda: []))
    monkeypatch.setattr(admin, "get_state", lambda: state)
    monkeypatch.setattr(experiments, "get_state", lambda: state)
    app = FastAPI()
    app.include_router(admin.router)
    app.include_router(experiments.router)
    with TestClient(app) as client:
        yield client, state, submit


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "incorrect"}])
def test_launch_requires_researcher_key(api, headers):
    client, _, submit = api
    response = client.post("/experiments", json={"profile": "quick"}, headers=headers)
    assert response.status_code == 401
    submit.assert_not_called()


@pytest.mark.parametrize("configured", ["change-me", "", "   "])
def test_default_or_empty_key_disables_researcher_actions(api, configured):
    client, state, submit = api
    state.admin_key = configured
    headers = {"X-API-Key": "change-me"}
    assert client.get("/admin/logs", headers=headers).status_code == 503
    assert client.post("/experiments", json={"profile": "quick"}, headers=headers).status_code == 503
    submit.assert_not_called()


def test_valid_key_launches_job_and_public_listing_stays_accessible(api):
    client, _, submit = api
    assert client.get("/experiments").status_code == 200
    response = client.post("/experiments", json={"profile": "quick", "modes": ["temporal"]},
                           headers={"X-API-Key": "private-test-key"})
    assert response.status_code == 200
    assert response.json()["job_id"] == "fixture-job"
    submit.assert_called_once()
    assert submit.call_args.args[0] == "experiment:quick"
    assert submit.call_args.args[2:] == ("configs/experiments/quick.yaml", ["temporal"])
