import asyncio
import base64
import copy
import io
import json

import httpx
import numpy as np
from PIL import Image
import pytest

from ra4rl.config import load_config
from ra4rl.data.collect import collect, replay
from ra4rl.data.llm import LLMClient, prompt
from ra4rl.data.store import Store
from ra4rl.data.translator import LEGACY_TRANSLATOR_VERSION, state_text
from ra4rl.envs.minigrid import (
    GLOBAL_UP_RGB, LOCAL_RGB, global_agent_up_png, llm_png, local_png,
    make_env, rgb_input_protocol, snapshot,
)


def decoded(png):
    return np.asarray(Image.open(io.BytesIO(png)))


@pytest.mark.parametrize("direction", range(4))
def test_global_native_pixels_rotate_to_up_without_highlight_or_state_change(direction):
    env = make_env("MiniGrid-DoorKey-6x6-v0", 360, flat=False)
    try:
        env.reset(seed=0)
        base = env.unwrapped
        base.agent_dir = direction
        before = (base.grid.encode().copy(), tuple(base.agent_pos), base.step_count,
                  copy.deepcopy(base.np_random.bit_generator.state))
        native = base.get_frame(agent_pov=False, highlight=False, tile_size=32)
        rgb = decoded(global_agent_up_png(env))
        assert np.array_equal(rgb, np.rot90(native, (direction + 1) % 4))
        assert rgb.shape == (base.height * 32, base.width * 32, 3)
        red = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] < 40) & (rgb[:, :, 2] < 40)
        ys, _ = np.where(red)
        # A triangle pointing upward has more area below its vertical midpoint.
        assert ys.mean() > (ys.min() + ys.max()) / 2
        assert ((rgb[:, :, 1] > 200) & (rgb[:, :, 0] < 40)).any()  # global goal is visible
        assert np.array_equal(base.grid.encode(), before[0])
        assert tuple(base.agent_pos) == before[1] and base.step_count == before[2]
        assert base.np_random.bit_generator.state == before[3]
        assert base.agent_dir == direction and base.gen_obs()["image"].shape == (7, 7, 3)
    finally:
        env.close()


def test_prompt_uses_local_description_and_other_env_stays_local():
    from minigrid.core.world_object import Key
    env = make_env("MiniGrid-DoorKey-6x6-v0", 360, flat=False)
    other = make_env("MiniGrid-Fetch-6x6-N2-v0", flat=False)
    try:
        obs, _ = env.reset(seed=0)
        initial = snapshot(env, obs)
        text = prompt(env.unwrapped.spec.id, initial)
        assert state_text(initial) in text and "carrying nothing" in text
        assert "{description}" not in text and "{door_status}" not in text
        assert "The door currently is unopened." in text
        assert "The key currently is not picked up" in text
        env.unwrapped.carrying = Key("yellow")
        door = next(o for o in env.unwrapped.grid.grid if o is not None and o.type == "door")
        door.is_open = True
        obs = env.unwrapped.gen_obs()
        state = snapshot(env, obs)
        assert state_text(state) in prompt(env.unwrapped.spec.id, state)
        assert "carrying yellow key" in prompt(env.unwrapped.spec.id, state)
        assert "The door currently is opened." in prompt(env.unwrapped.spec.id, state)
        assert "The key currently is picked up (carried by the agent)." in prompt(env.unwrapped.spec.id, state)
        assert "{\"action_probs\": [7 probabilities]}" in prompt(env.unwrapped.spec.id, snapshot(env, obs))
        other.reset(seed=0)
        assert rgb_input_protocol(other.unwrapped.spec.id) == LOCAL_RGB
        assert llm_png(other) == local_png(other)
        assert llm_png(env) == global_agent_up_png(env)
    finally:
        env.close()
        other.close()


def test_mock_api_receives_exact_global_png_and_saved_image_keeps_local_symbols(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "test-only-key")
    cfg = load_config(smoke=True)
    cfg.update(max_episode_steps=2, collection_seeds=[0, 1])
    received = []
    descriptions = []

    def handle(request):
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        assert body["thinking"] == {"type": "disabled"}
        assert "MiniGrid DoorKey environment" in content[0]["text"]
        assert "{description}" not in content[0]["text"]
        descriptions.append(content[0]["text"])
        png = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])
        assert decoded(png).shape == (192, 192, 3)
        received.append(png)
        return httpx.Response(200, json={"choices": [{"message": {
            "content": '{"action_probs":[1,0,0,0,0,0,0]}'}}]})

    async def scenario():
        client = LLMClient(cfg["llm"])
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            await collect(cfg, tmp_path / "facts.sqlite", client=client, synthetic=True)
        finally:
            await client.close()

    asyncio.run(scenario())
    store = Store(tmp_path / "facts.sqlite", cfg["env_id"])
    try:
        rows = store.rows()
        assert json.loads(store.meta["collection_protocol"])["rgb_input"] == GLOBAL_UP_RGB
        assert [row["rgb_png"] for row in rows] == received
        assert all(row["env_description"] in text for row, text in zip(rows, descriptions))
        assert all(len(row["obs_img"]) == 147 for row in rows)
        for row in rows:
            assert (tmp_path / "images" / f"seed_000_step_{row['step_idx']:06d}.png").read_bytes() == row["rgb_png"]
    finally:
        store.close()


def test_historical_local_rgb_replay_remains_supported():
    env = make_env("MiniGrid-DoorKey-6x6-v0", 360, flat=False)
    try:
        obs, _ = env.reset(seed=0)
        state = snapshot(env, obs)
        png = local_png(env)
        _, reward, terminated, truncated, _ = env.step(0)
        row = dict(step_idx=0, obs_img=state.image.tobytes(), rgb_png=png, mission=state.mission,
                   direction=state.direction, carrying_type=state.carrying_type, carrying_color=state.carrying_color,
                   env_description=state_text(state, version=LEGACY_TRANSLATOR_VERSION), action_probs_json="[1,0,0,0,0,0,0]",
                   action_id=0, reward=reward, terminated=terminated, truncated=truncated)
        obs, _ = env.reset(seed=0)
        replay(env, obs, [row], translator_version=LEGACY_TRANSLATOR_VERSION)
        obs, _ = env.reset(seed=0)
        with pytest.raises(ValueError, match="replay"):
            replay(env, obs, [row], GLOBAL_UP_RGB, LEGACY_TRANSLATOR_VERSION)
    finally:
        env.close()
