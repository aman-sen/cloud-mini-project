import json

import app as svc
import pytest
from fastapi.testclient import TestClient

client = TestClient(svc.app)
GOOD = '{"intent": "weather_query", "date": null, "time": null, "place": "london"}'


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v1")
    monkeypatch.setenv("POD_NAME", "pod-a")
    monkeypatch.delenv("FAULT_INJECT", raising=False)
    svc.STATE.update(ready=False, generator=None)


def test_probes():
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 503
    svc.STATE.update(ready=True, generator=lambda t: GOOD)
    assert client.get("/readyz").status_code == 200


def test_parse_ok_has_headers_and_exact_body():
    svc.STATE.update(ready=True, generator=lambda t: GOOD)
    r = client.post("/parse", json={"text": "what is the weather in london"})
    assert r.status_code == 200 and r.json() == json.loads(GOOD)
    assert r.headers["X-Model-Version"] == "v1" and r.headers["X-Served-By"] == "pod-a"
    assert list(r.json()) == ["intent", "date", "time", "place"]


def test_fault_inject_returns_500_exact_body(monkeypatch):
    monkeypatch.setenv("FAULT_INJECT", "1")
    svc.STATE.update(ready=True, generator=lambda t: GOOD)
    r = client.post("/parse", json={"text": "x"})
    assert r.status_code == 500 and r.json() == {"error": "invalid_model_output"}


@pytest.mark.parametrize("bad", ["", "not json", '{"intent": "alarm_set"}', '[1]',
                                 '{"intent": "nope", "date": null, "time": null, "place": null}',
                                 '{"intent": "alarm_set", "date": 1, "time": null, "place": null}',
                                 '{"intent": "alarm_set", "date": null, "time": null, "place": null, "x": 1}',
                                 '{"intent": "alarm_set", "date": null, "time": null, "pla'])
def test_invalid_model_output_never_200(bad):
    svc.STATE.update(ready=True, generator=lambda t: bad)
    r = client.post("/parse", json={"text": "x"})
    assert r.status_code == 500 and r.json() == {"error": "invalid_model_output"}


def test_not_ready_parse_is_503():
    assert client.post("/parse", json={"text": "x"}).status_code == 503
