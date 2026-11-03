import asyncio
from collections import defaultdict
import json

import httpx

from ra4rl.config import load_config
from ra4rl.data.collect import collect
from ra4rl.data.llm import LLMClient
from ra4rl.data.store import Store


def test_one_hundred_concurrent_episodes_have_independent_stable_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "test-only-key")
    cfg = load_config(smoke=True)
    cfg["max_episode_steps"] = 2
    by_session = defaultdict(int)
    active = peak = 0

    async def scenario():
        ready = asyncio.Event()

        async def handle(request):
            nonlocal active, peak
            session = request.headers["x-opencode-session"]
            by_session[session] += 1
            assert json.loads(request.content)["thinking"] == {"type": "disabled"}
            active += 1
            peak = max(peak, active)
            if active == 100:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=5)
            active -= 1
            return httpx.Response(200, json={"choices": [{"message": {
                "content": '{"action_probs":[1,0,0,0,0,0,0]}'}}]})

        client = LLMClient(cfg["llm"])
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            return await collect(cfg, tmp_path / "facts.sqlite", client, range(100), synthetic=True)
        finally:
            await client.close()

    result = asyncio.run(scenario())
    assert result["episode_count"] == 100 and peak == 100
    assert len(by_session) == 100 and set(by_session.values()) == {2}
    store = Store(tmp_path / "facts.sqlite", cfg["env_id"])
    assert len(store.rows()) == 200 and store.meta["complete"] == "true"
    store.close()
    progress = json.loads((tmp_path / "facts.sqlite.progress.json").read_text())
    assert progress["phase"] == "collected" and progress["saved_transitions"] == 200
    assert progress["api"]["max_in_flight"] == 100
