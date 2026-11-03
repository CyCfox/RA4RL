import asyncio
import json

import httpx
import numpy as np
import pytest

from ra4rl.config import load_config
from ra4rl.data.llm import LLMClient
from ra4rl.envs.minigrid import RawState


def test_multimodal_payload_and_retry_repair(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "test-only-key")
    cfg = load_config()["llm"]
    cfg["max_retries"] = 2
    requests = []
    sessions = []
    delays = []

    def handler(request):
        requests.append(json.loads(request.content))
        sessions.append(request.headers["x-opencode-session"])
        if len(requests) == 1:
            return httpx.Response(503)
        if len(requests) == 2:
            return httpx.Response(200, json={"choices": [{"message": {"content": "bad json"}}]})
        return httpx.Response(200, json={"choices": [{"message": {
            "content": 'reason {"action_probs":[2,-1,0,0,0,0,0]}'}}]})

    async def no_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("ra4rl.data.llm.asyncio.sleep", no_sleep)

    async def scenario():
        client = LLMClient(cfg)
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        state = RawState(np.zeros((7, 7, 3), dtype=np.uint8), "go to the red door", 0, None, None)
        probs = await client.predict("MiniGrid-GoToDoor-6x6-v0", state, b"local-png-only")
        await client.close()
        return probs

    assert np.array_equal(asyncio.run(scenario()), [1, 0, 0, 0, 0, 0, 0])
    assert delays == [5, 1]
    assert len(set(sessions)) == 1 and sessions[0].startswith("ses_ra4rl_")
    payload = requests[-1]
    assert payload["thinking"] == {"type": "disabled"}
    content = payload["messages"][0]["content"]
    assert len(content) == 2 and content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "done to declare arrival" in content[0]["text"]


def test_credentials_missing_and_permanent_http_failures(monkeypatch):
    cfg = load_config()["llm"]
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        LLMClient(cfg)
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401)

    async def scenario():
        client = LLMClient(cfg, transport_policy={"http_retry_limit": 0})
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await client.predict("MiniGrid-DoorKey-6x6-v0", RawState(
                np.zeros((7, 7, 3), dtype=np.uint8), "reach goal", 0, None, None, False), b"png")
        finally:
            await client.close()

    with pytest.raises(RuntimeError, match="HTTP 401"):
        asyncio.run(scenario())
    assert len(calls) == 1
