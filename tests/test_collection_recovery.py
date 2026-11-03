import asyncio
import json
import sqlite3

import httpx
import numpy as np
import pytest

from ra4rl.config import load_config
from ra4rl.data.collect import collect
from ra4rl.data.llm import EPISODE_SESSION, LLMClient
from ra4rl.data.store import Store, sha256
from ra4rl.envs.minigrid import RawState


def test_429_waits_do_not_exhaust_transient_retry_budget(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "private-test-key")
    cfg = dict(load_config()["llm"], max_retries=1)

    async def scenario():
        calls = []
        waits = []

        async def advance(delay):
            waits.append(delay)

        def handler(request):
            calls.append(request)
            if len(calls) <= 3:
                return httpx.Response(429, headers={"Retry-After": "120"}, text="private-test-key quota")
            return httpx.Response(200, json={"choices": [{"message": {
                "content": '{"action_probs":[1,0,0,0,0,0,0]}'}}]})

        client = LLMClient(cfg)
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr("ra4rl.data.llm.asyncio.sleep", advance)
        try:
            await client.predict("MiniGrid-DoorKey-6x6-v0", RawState(
                np.zeros((7, 7, 3), dtype=np.uint8), "reach goal", 0, None, None, False), b"png")
            assert len(calls) == 4 and waits == [5, 5, 5]
            assert client.stats["http_statuses"] == {"429": 3, "200": 1}
            assert "private-test-key" not in json.dumps(client.stats)
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("http_status", [401, 403, 429, 500])
def test_http_error_sleeps_only_that_request_and_releases_slot(monkeypatch, http_status, tmp_path):
    monkeypatch.setenv("OPENCODE_API_KEY", "private-test-key")
    cfg = dict(load_config()["llm"], concurrency=1, max_retries=0)

    async def scenario():
        sleeping = asyncio.Event()
        release = asyncio.Event()
        counts = {}
        waits = []

        async def sleep_local(delay):
            waits.append(delay)
            sleeping.set()
            await release.wait()

        def handler(request):
            session = request.headers["x-opencode-session"]
            counts[session] = counts.get(session, 0) + 1
            if session == "broken" and counts[session] <= 2:
                return httpx.Response(http_status, text="private-test-key temporarily denied")
            return httpx.Response(200, json={"choices": [{"message": {
                "content": '{"action_probs":[1,0,0,0,0,0,0]}'}}]})

        monkeypatch.setattr("ra4rl.data.llm.asyncio.sleep", sleep_local)
        client = LLMClient(cfg, error_log=tmp_path / "errors.jsonl")
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        state = RawState(np.zeros((7, 7, 3), dtype=np.uint8), "reach goal", 0, None, None, False)

        async def predict(session):
            token = EPISODE_SESSION.set(session)
            try:
                return await client.predict("MiniGrid-DoorKey-6x6-v0", state, b"png")
            finally:
                EPISODE_SESSION.reset(token)

        task = asyncio.create_task(predict("broken"))
        try:
            await asyncio.wait_for(sleeping.wait(), timeout=1)
            assert not task.done()
            await asyncio.wait_for(predict("healthy"), timeout=1)
            assert counts["healthy"] == 1 and not task.done()
            release.set()
            await asyncio.wait_for(task, timeout=1)
            assert waits == [5, 5] and counts["broken"] == 3
            assert client.progress_stats()["global_throttle"] is False
            assert "private-test-key" not in (tmp_path / "errors.jsonl").read_text()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await client.close()

    asyncio.run(scenario())


class LeftClient:
    def __init__(self, fail_after=None, interleave=False):
        self.calls = 0
        self.fail_after = fail_after
        self.interleave = interleave
        self.sessions = set()
        self.previous_actions = []

    async def predict(self, *args):
        self.calls += 1
        self.sessions.add(EPISODE_SESSION.get())
        self.previous_actions.append((EPISODE_SESSION.get(), args[1].previous_action))
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated interruption")
        if self.interleave:
            await asyncio.sleep(0)
        return np.array([1, 0, 0, 0, 0, 0, 0], dtype=float)


@pytest.mark.parametrize("interleave", [False, True])
def test_resume_copies_replays_and_only_requests_missing_steps(tmp_path, interleave):
    cfg = load_config(smoke=True)
    cfg.update(collection_seeds=[0, 2], max_episode_steps=3)
    original = tmp_path / "original.sqlite"
    first = LeftClient(fail_after=4, interleave=interleave)
    with pytest.raises(RuntimeError, match="simulated"):
        asyncio.run(collect(cfg, original, first, synthetic=True))
    digest = sha256(original)
    source = Store(original, cfg["env_id"])
    prefix = source.rows()
    sessions = json.loads(source.meta["collection_protocol"])["collection_id"]
    source.close()
    target = tmp_path / "resumed.sqlite"
    with sqlite3.connect(f"file:{original}?mode=ro", uri=True) as src:
        with sqlite3.connect(target) as dst:
            src.backup(dst)
    second = LeftClient()
    summary = asyncio.run(collect(cfg, target, second, synthetic=True, resume=True))
    assert summary["episode_count"] == 2
    assert second.calls == 6 - len(prefix)
    assert all(s.startswith(f"ses_ra4rl_{sessions}_ep") for s in second.sessions)
    # At the first missing step for each episode, restore its own saved action.
    seen = set()
    for session, action in second.previous_actions:
        if session not in seen:
            seed = int(session.rsplit("_ep", 1)[1])
            saved = [row for row in prefix if row["episode_id"] == seed]
            assert action == (saved[-1]["action_id"] if saved else None)
            seen.add(session)
        else:
            assert action == 0
    assert sha256(original) == digest
    finished = Store(target, cfg["env_id"])
    final = finished.rows()
    assert len(final) == 6 and finished.meta["complete"] == "true"
    by_id = {row["experience_id"]: row for row in final}
    assert all(by_id[row["experience_id"]] == row for row in prefix)
    finished.close()


def test_resume_rejects_corrupt_prefix_before_request(tmp_path):
    cfg = load_config(smoke=True)
    cfg.update(collection_seeds=[0, 1], max_episode_steps=3)
    path = tmp_path / "broken.sqlite"
    with pytest.raises(RuntimeError):
        asyncio.run(collect(cfg, path, LeftClient(fail_after=1), synthetic=True))
    with sqlite3.connect(path) as db:
        db.execute("UPDATE experiences SET reward=999")
    client = LeftClient()
    with pytest.raises(ValueError, match="replay"):
        asyncio.run(collect(cfg, path, client, synthetic=True, resume=True))
    assert client.calls == 0
