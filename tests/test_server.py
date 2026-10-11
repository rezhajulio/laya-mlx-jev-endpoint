import asyncio
from unittest.mock import MagicMock

import httpx
import pytest

from server import app


class FakeAgent:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    def predict(self, state, questions):
        import time

        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.delay > 0:
            time.sleep(self.delay)

        answers = {}
        for qid, q in questions.items():
            qtype = q["type"]
            if qtype == "choice":
                answers[qid] = {
                    "type": "choice",
                    "choice": "opt1",
                    "probabilities": {"opt1": 0.9, "opt2": 0.1},
                    "confidence": 0.9,
                }
            elif qtype == "score":
                answers[qid] = {
                    "type": "score",
                    "score": 2.0,
                    "legend": {"0": "low", "1": "med", "2": "high"},
                    "probabilities": {"0": 0.05, "1": 0.15, "2": 0.8},
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": 0.95,
                    "confidence": 0.95,
                }
        self.in_flight -= 1
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": 15, "output_tokens": 0},
        }


@pytest.fixture(autouse=True)
def setup_app_state():
    app.state.agent = FakeAgent()
    app.state.lock = asyncio.Lock()
    yield
    app.state.agent = None


@pytest.mark.anyio
async def test_health_ready():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert "model" in data


@pytest.mark.anyio
async def test_health_not_ready():
    app.state.agent = None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 503


@pytest.mark.anyio
async def test_list_models():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        names = [m["name"] for m in data["models"]]
        assert "jev-latest" in names


@pytest.mark.anyio
async def test_systemone_noul_string_state():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "state": "User requested cancellation.",
            "questions": {
                "cancel": {
                    "type": "noul",
                    "instructions": "Is user cancelling?",
                }
            },
        }
        resp = await client.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "cancel" in data["answers"]
        assert data["answers"]["cancel"]["type"] == "noul"
        assert data["answers"]["cancel"]["noul"] == 0.95
        assert data["usage"]["input_tokens"] == 15


@pytest.mark.anyio
async def test_systemone_choice_dict_state():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "model": "jev-latest",
            "state": {"action": "refund", "amount": 100},
            "questions": {
                "dept": {
                    "type": "choice",
                    "instructions": "Route to team",
                    "criteria": ["billing", "sales"],
                }
            },
        }
        resp = await client.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "dept" in data["answers"]
        assert data["answers"]["dept"]["type"] == "choice"


@pytest.mark.anyio
async def test_systemone_unknown_model_fails():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "model": "non-existent-model",
            "state": "sample",
            "questions": {"q": {"type": "noul", "instructions": "test"}},
        }
        resp = await client.post("/v1/systemone", json=payload)
        assert resp.status_code == 400
        assert "Unknown model" in resp.json()["detail"]


@pytest.mark.anyio
async def test_systemone_extra_fields_forbidden():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "state": "sample",
            "unsupported_field": 123,
            "questions": {"q": {"type": "noul", "instructions": "test"}},
        }
        resp = await client.post("/v1/systemone", json=payload)
        assert resp.status_code == 400


@pytest.mark.anyio
async def test_systemone_empty_questions_fails():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "state": "sample",
            "questions": {},
        }
        resp = await client.post("/v1/systemone", json=payload)
        assert resp.status_code == 400


@pytest.mark.anyio
async def test_systemone_value_error_returns_400():
    mock_agent = MagicMock()
    mock_agent.predict.side_effect = ValueError("Question budget exceeded")
    app.state.agent = mock_agent

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "state": "sample",
            "questions": {"q": {"type": "noul", "instructions": "test"}},
        }
        resp = await client.post("/v1/systemone", json=payload)
        assert resp.status_code == 400
        assert "Question budget exceeded" in resp.json()["detail"]


@pytest.mark.anyio
async def test_concurrent_requests_serialized():
    fake = FakeAgent(delay=0.05)
    app.state.agent = fake

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "state": "sample",
            "questions": {"q": {"type": "noul", "instructions": "test"}},
        }
        tasks = [client.post("/v1/systemone", json=payload) for _ in range(5)]
        responses = await asyncio.gather(*tasks)
        for r in responses:
            assert r.status_code == 200

        # Assert forward passes were serialized by the lock
        assert fake.max_in_flight == 1
