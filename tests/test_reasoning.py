import asyncio
import json

import httpx
import numpy as np
import pytest

from ra4rl.config import load_config
from ra4rl.data.llm import LLMClient
from ra4rl.data.reasoning import reasoning_options
from ra4rl.envs.minigrid import make_env, snapshot


@pytest.mark.parametrize('options', [
    {'thinking': True}, {'thinking': {'type': 'auto'}},
    {'thinking': {'type': 'enabled', 'budget': 1}},
    {'reasoning_effort': 'low'},
    {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'typo'},
])
def test_reasoning_options_reject_ambiguous_or_conflicting_settings(options):
    with pytest.raises(ValueError):
        reasoning_options(options)


def test_reasoning_defaults_are_legacy_disabled_and_new_putnear_config_is_explicit_low():
    assert reasoning_options({}) == {'thinking': {'type': 'disabled'}}
    cfg = load_config('configs/putnear_llm_thinking_low.yaml')
    assert reasoning_options(cfg['llm']) == {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'}
    assert cfg['max_episode_steps'] == 50 and cfg['llm']['max_tokens'] == 4096
    assert reasoning_options(load_config()['llm']) == {'thinking': {'type': 'disabled'}}


@pytest.mark.parametrize('truncated_first', [False, True])
def test_enabled_thinking_parses_final_content_only_and_retries_truncation(monkeypatch, truncated_first):
    monkeypatch.setenv('OPENCODE_API_KEY', 'test-only-key')
    cfg = load_config('configs/putnear_llm_thinking_low.yaml')['llm']
    cfg.update(max_retries=1)
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body['thinking'] == {'type': 'enabled'} and body['reasoning_effort'] == 'low'
        assert body['max_tokens'] == 4096
        truncated = truncated_first and len(requests) == 1
        return httpx.Response(200, json={'choices': [{
            'finish_reason': 'length' if truncated else 'stop',
            'message': {'reasoning_content': 'test fixture {"action_probs":[1,0,0,0,0,0,0]}',
                        'content': None if truncated else '{"action_probs":[0,0,1,0,0,0,0]}'}}],
            'usage': {'completion_tokens_details': {'reasoning_tokens': 40}}})

    async def scenario():
        env = make_env('MiniGrid-PutNear-6x6-N2-v0', 50, flat=False)
        client = LLMClient(cfg)
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            obs, _ = env.reset(seed=0)
            probs = await client.predict('MiniGrid-PutNear-6x6-N2-v0', snapshot(env, obs), b'png')
            assert np.array_equal(probs, [0, 0, 1, 0, 0, 0, 0])
            assert client.stats['reasoning_tokens'] == 40 * len(requests)
            assert client.stats['reasoning_content_responses'] == len(requests)
            assert client.stats['truncated_responses'] == int(truncated_first)
            assert client.stats['successful_requests'] == 1
            assert client.stats['retries'] == int(truncated_first)
        finally:
            env.close()
            await client.close()

    asyncio.run(scenario())
    if truncated_first:
        assert len(requests) == 2 and requests[0] == requests[1]


def test_disabled_thinking_still_rejects_provider_reasoning(monkeypatch):
    monkeypatch.setenv('OPENCODE_API_KEY', 'test-only-key')
    cfg = load_config()['llm']

    def handler(request):
        body = json.loads(request.content)
        assert body['thinking'] == {'type': 'disabled'} and 'reasoning_effort' not in body
        return httpx.Response(200, json={'choices': [{'message': {
            'reasoning_content': 'test reasoning', 'content': '{"action_probs":[1,0,0,0,0,0,0]}'}}]})

    async def scenario():
        env = make_env('MiniGrid-PutNear-6x6-N2-v0', 50, flat=False)
        client = LLMClient(cfg)
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            obs, _ = env.reset(seed=0)
            with pytest.raises(RuntimeError, match='thinking was disabled'):
                await client.predict('MiniGrid-PutNear-6x6-N2-v0', snapshot(env, obs), b'png')
        finally:
            env.close()
            await client.close()

    asyncio.run(scenario())
