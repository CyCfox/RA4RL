import asyncio
from dataclasses import replace
import json

import httpx
import numpy as np
import pytest

from ra4rl.config import load_config
from ra4rl.data.collect import collect
from ra4rl.data.doorkey_prompt import FIRST_STEP_TEXT, PREVIOUS_ACTION_TEXT
from ra4rl.data.llm import EPISODE_SESSION, LLMClient, prompt
from ra4rl.data.translator import state_text
from ra4rl.envs.minigrid import make_env, snapshot


def test_status_and_previous_action_use_actual_state_even_when_forward_is_blocked():
    from minigrid.core.world_object import Door, Key, Wall
    env = make_env("MiniGrid-DoorKey-6x6-v0", 360, flat=False)
    try:
        obs, _ = env.reset(seed=0)
        assert FIRST_STEP_TEXT in prompt(env.unwrapped.spec.id, snapshot(env, obs))
        base = env.unwrapped
        base.agent_pos, base.agent_dir = (1, 1), 3
        base.grid.set(*base.front_pos, Wall())
        position = tuple(base.agent_pos)
        obs, *_ = env.step(2)
        assert tuple(base.agent_pos) == position
        state = snapshot(env, obs, previous_action=2)
        assert PREVIOUS_ACTION_TEXT[2] in prompt(env.unwrapped.spec.id, state)
        assert "moved forward" not in prompt(env.unwrapped.spec.id, state)
        # Door scalar stays accurate even when the door is outside the local view.
        door = next(o for o in base.grid.grid if isinstance(o, Door))
        door.is_open = True
        base.carrying = Key("yellow")
        state = snapshot(env, base.gen_obs(), previous_action=1)
        text = prompt(env.unwrapped.spec.id, state)
        assert f"{state_text(state)}\nThe door currently is opened.\n" in text
        assert "The key currently is picked up (carried by the agent)." in text
        assert "The agent executed a right turn in the previous step." in text
        # After dropping the key, current possession must no longer say picked up.
        base.carrying = None
        state = snapshot(env, base.gen_obs(), previous_action=4)
        assert "The key currently is not picked up" in prompt(env.unwrapped.spec.id, state)
        for action, sentence in enumerate(PREVIOUS_ACTION_TEXT):
            assert sentence in prompt(env.unwrapped.spec.id, replace(state, previous_action=action))
        with pytest.raises(ValueError, match="door open state"):
            prompt(env.unwrapped.spec.id, replace(state, door_open=None))
        obs, _ = env.reset(seed=1)
        assert FIRST_STEP_TEXT in prompt(env.unwrapped.spec.id, snapshot(env, obs))
    finally:
        env.close()


def test_three_interleaved_environments_send_their_own_action_history_on_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "test-only-key")
    cfg = load_config(smoke=True)
    cfg.update(max_episode_steps=3, collection_seeds=[0, 3])
    cfg["llm"]["concurrency"] = 3
    counts = {}
    retries = {}
    submitted = {}

    async def scenario():
        ready = asyncio.Event()

        async def handle(request):
            session = request.headers["x-opencode-session"]
            seed = int(session.rsplit("_ep", 1)[1])
            step = counts.get(seed, 0)
            body = json.loads(request.content)
            text = body["messages"][0]["content"][0]["text"]
            assert FIRST_STEP_TEXT in text if step == 0 else PREVIOUS_ACTION_TEXT[seed] in text
            if step == 0:
                submitted[seed] = True
                if len(submitted) == 3:
                    ready.set()
                await asyncio.wait_for(ready.wait(), timeout=5)
            if seed == 1 and step == 1 and not retries:
                retries[seed] = body
                return httpx.Response(429, text="retry this request")
            if seed in retries and step == 1:
                assert body == retries[seed]
            counts[seed] = step + 1
            probabilities = np.eye(7)[seed].tolist()
            return httpx.Response(200, json={"choices": [{"message": {
                "content": json.dumps({"action_probs": probabilities})}}]})

        client = LLMClient(cfg["llm"])
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        actual_sleep = asyncio.sleep

        # The collector heartbeat uses the same asyncio module; preserve its delay.
        async def sleep(delay):
            if delay == 5:
                await actual_sleep(0)
            else:
                await actual_sleep(delay)
        monkeypatch.setattr("ra4rl.data.llm.asyncio.sleep", sleep)
        try:
            return await collect(cfg, tmp_path / "facts.sqlite", client=client, synthetic=True)
        finally:
            await client.close()

    summary = asyncio.run(scenario())
    assert summary["episode_count"] == 3 and counts == {0: 3, 1: 3, 2: 3}
    assert summary["api"]["retries"] == 1
    records = [json.loads(line) for line in (tmp_path / "facts.sqlite.steps.jsonl").read_text().splitlines()]
    for record in records:
        assert record["previous_action"] == (None if record["step_idx"] == 0 else record["seed"])
    assert EPISODE_SESSION.get() is None
