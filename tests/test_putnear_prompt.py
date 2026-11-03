import asyncio
import base64
import copy
from dataclasses import replace
import io
import json
import sqlite3

import httpx
import numpy as np
from PIL import Image
import pytest

from ra4rl.config import load_config
from ra4rl.data.collect import collect, replay
from ra4rl.data.llm import LLMClient, prompt
from ra4rl.data.prompts import custom_prompt
from ra4rl.data.putnear_prompt import FIRST_STEP_TEXT, PREVIOUS_ACTION_TEXT, prompt_protocol
from ra4rl.data.store import Store
from ra4rl.envs.minigrid import (
    GLOBAL_NATIVE_RGB, GLOBAL_UP_RGB, LOCAL_RGB, llm_png, local_png,
    make_env, rgb_input_protocol, snapshot,
)

ENV_ID = "MiniGrid-PutNear-6x6-N2-v0"


@pytest.mark.parametrize("direction", range(4))
def test_putnear_global_native_pixels_preserve_orientation_and_environment(direction):
    env = make_env(ENV_ID, 50, flat=False)
    try:
        env.reset(seed=0)
        base = env.unwrapped
        base.agent_dir = direction
        before = (base.grid.encode().copy(), tuple(base.agent_pos), base.step_count,
                  copy.deepcopy(base.np_random.bit_generator.state))
        actual = np.asarray(Image.open(io.BytesIO(llm_png(env))))
        native = base.get_frame(agent_pov=False, highlight=False, tile_size=32)
        assert actual.shape == (192, 192, 3) and np.array_equal(actual, native)
        assert rgb_input_protocol(ENV_ID) == GLOBAL_NATIVE_RGB
        assert llm_png(env, LOCAL_RGB) == local_png(env)
        assert np.asarray(Image.open(io.BytesIO(local_png(env)))).shape == (224, 224, 3)
        assert base.agent_dir == direction and tuple(base.agent_pos) == before[1]
        assert base.step_count == before[2] and np.array_equal(base.grid.encode(), before[0])
        assert base.np_random.bit_generator.state == before[3]
        assert base.gen_obs()['image'].shape == (7, 7, 3)
    finally:
        env.close()


@pytest.mark.parametrize("kind,color,expected", [
    (None, None, "nothing"), ("key", "yellow", "a yellow key"),
    ("ball", "red", "a red ball"), ("box", "blue", "a blue box"),
])
def test_putnear_prompt_fills_live_carrying_and_only_substitutes_placeholders(kind, color, expected):
    env = make_env(ENV_ID, 50, flat=False)
    try:
        obs, _ = env.reset(seed=0)
        state = replace(snapshot(env, obs), carrying_type=kind, carrying_color=color,
                        mission="put the {pre_action} box near the {mission} ball")
        text = prompt(ENV_ID, state)
        assert f"当前任务： {state.mission}" in text
        assert f"当前携带状态： {expected}" in text
        assert f"上一步执行动作：{FIRST_STEP_TEXT}" in text
        assert '{"action_probs": [7 probabilities]}' in text
        assert "{a yellow key / a red ball / nothing}" not in text
        assert "整个环境的全局渲染图" in text and "8 邻域" in text
        assert "除去墙外网格大小是4×4，最左上角黑色格子为第一行第一列。" in text
        assert "物体会和智能体所在格子重叠" not in text
        assert "正前方" in text and "Only the current native 7x7" not in text
        template, protocol = custom_prompt(ENV_ID)
        assert protocol == prompt_protocol() and "{mission}" in template
        for action, sentence in enumerate(PREVIOUS_ACTION_TEXT):
            assert f"上一步执行动作：{sentence}" in prompt(ENV_ID, replace(state, previous_action=action))
        assert "不保证移动成功" in prompt(ENV_ID, replace(state, previous_action=2))
    finally:
        env.close()


@pytest.mark.parametrize("action", [-1, 7, True, 1.5, "left"])
def test_putnear_prompt_rejects_invalid_previous_action(action):
    env = make_env(ENV_ID, 50, flat=False)
    try:
        obs, _ = env.reset(seed=0)
        with pytest.raises(ValueError, match="Previous action"):
            prompt(ENV_ID, replace(snapshot(env, obs), previous_action=action))
    finally:
        env.close()


@pytest.mark.parametrize('thinking_enabled', [False, True])
def test_three_putnear_workers_send_exact_native_global_images_and_retry_own_history(tmp_path, monkeypatch,
                                                                                    thinking_enabled):
    monkeypatch.setenv("OPENCODE_API_KEY", "test-only-key")
    cfg = load_config('configs/putnear_6x6_n2_50.yaml', smoke=True)
    cfg.update(max_episode_steps=3, collection_seeds=[0, 3])
    cfg['llm']['concurrency'] = 3
    if thinking_enabled:
        cfg['llm'].update(thinking={'type': 'enabled'}, reasoning_effort='low')
    references = {seed: make_env(ENV_ID, 3, flat=False) for seed in range(3)}
    for seed, env in references.items():
        env.reset(seed=seed)
    counts, received = dict.fromkeys(range(3), 0), {}
    retried = []

    async def scenario():
        ready = asyncio.Event()
        first_requests = set()

        async def handler(request):
            seed = int(request.headers['x-opencode-session'].rsplit('_ep', 1)[1])
            step = counts[seed]
            body = json.loads(request.content)
            assert body['thinking'] == {'type': 'enabled' if thinking_enabled else 'disabled'}
            assert body.get('reasoning_effort') == ('low' if thinking_enabled else None)
            text, image = body['messages'][0]['content']
            assert f"上一步执行动作：{FIRST_STEP_TEXT if step == 0 else PREVIOUS_ACTION_TEXT[seed]}" in text['text']
            png = base64.b64decode(image['image_url']['url'].split(',', 1)[1])
            assert png == llm_png(references[seed])
            if step == 0:
                first_requests.add(seed)
                if len(first_requests) == 3:
                    ready.set()
                await asyncio.wait_for(ready.wait(), timeout=5)
            if seed == 1 and step == 1:
                if not retried:
                    retried.append(body)
                    return httpx.Response(429)
                assert body == retried[0]
            received[seed, step] = png
            counts[seed] += 1
            references[seed].step(seed)
            probs = [.01] * 7
            probs[seed] = .94
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
                'reasoning_content': 'test reasoning metadata' if thinking_enabled else '',
                'content': '先分析位置与朝向。' + json.dumps({'action_probs': probs})}}],
                'usage': {'completion_tokens_details': {'reasoning_tokens': 10 if thinking_enabled else 0}}})

        client = LLMClient(cfg['llm'])
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        actual_sleep = asyncio.sleep

        async def sleep(delay):
            await actual_sleep(0 if delay == 5 else delay)

        monkeypatch.setattr('ra4rl.data.llm.asyncio.sleep', sleep)
        try:
            summary = await collect(cfg, tmp_path / 'facts.sqlite', client, synthetic=True)
            assert client.stats['max_in_flight'] == 3
            assert client.stats['retries'] == 1
            return summary
        finally:
            await client.close()

    try:
        summary = asyncio.run(scenario())
        assert summary['episode_count'] == 3 and counts == {0: 3, 1: 3, 2: 3}
        store = Store(tmp_path / 'facts.sqlite', ENV_ID)
        try:
            protocol = json.loads(store.meta['collection_protocol'])
            assert protocol['prompt'] == prompt_protocol() and protocol['rgb_input'] == GLOBAL_NATIVE_RGB
            assert protocol['thinking'] == {'type': 'enabled' if thinking_enabled else 'disabled'}
            assert protocol.get('reasoning_effort') == ('low' if thinking_enabled else None)
            assert summary['thinking'] == protocol['thinking']
            assert summary.get('reasoning_effort') == protocol.get('reasoning_effort')
            assert summary['api']['reasoning_tokens'] == (90 if thinking_enabled else 0)
            rows = store.rows()
        finally:
            store.close()
        for row in rows:
            key = row['episode_id'], row['step_idx']
            assert row['rgb_png'] == received[key] and len(row['obs_img']) == 147
            assert (tmp_path / 'images' / f'seed_{key[0]:03d}_step_{key[1]:06d}.png').read_bytes() == received[key]
        trace = [json.loads(line) for line in (tmp_path / 'facts.sqlite.steps.jsonl').read_text().splitlines()]
        assert all(r['rotation_ccw_quarter_turns'] == 0 for r in trace)
        assert all(r['previous_action'] == (None if r['step_idx'] == 0 else r['seed']) for r in trace)
        for seed, env in references.items():
            obs, _ = env.reset(seed=seed)
            replay(env, obs, [r for r in rows if r['episode_id'] == seed], GLOBAL_NATIVE_RGB)
        assert rgb_input_protocol('MiniGrid-DoorKey-6x6-v0') == GLOBAL_UP_RGB
    finally:
        for env in references.values():
            env.close()


def test_changed_putnear_prompt_cannot_resume_old_collection(tmp_path):
    cfg = load_config('configs/putnear_6x6_n2_50.yaml', smoke=True)
    cfg.update(collection_seeds=[0, 1], max_episode_steps=3)

    class BrokenClient:
        async def predict(self, *args):
            raise RuntimeError('test interruption')

    database = tmp_path / 'facts.sqlite'
    with pytest.raises(RuntimeError, match='test interruption'):
        asyncio.run(collect(cfg, database, BrokenClient(), synthetic=True))
    with sqlite3.connect(database) as db:
        protocol = json.loads(db.execute("SELECT value FROM meta WHERE key='collection_protocol'").fetchone()[0])
        protocol['prompt']['sha256'] = 'old-template'
        db.execute("UPDATE meta SET value=? WHERE key='collection_protocol'", (json.dumps(protocol),))
    with pytest.raises(ValueError, match='protocol mismatch'):
        asyncio.run(collect(cfg, database, BrokenClient(), synthetic=True, resume=True))
