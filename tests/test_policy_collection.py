import copy
import json

import numpy as np
from PIL import Image
import pytest
from stable_baselines3 import PPO

from ra4rl.config import config_hash, load_config
from ra4rl.data.policy_collect import CheckpointPolicy, collect_policy, validate_collection
from ra4rl.data.store import Store
from ra4rl.envs.minigrid import LOCAL_RGB, make_env, make_vec
from ra4rl.retrieval.teacher import RetrievalTeacher


class FixturePolicy:
    def __init__(self, cfg):
        self.training_config = cfg
        self.source = {'test_fixture': True}
        self.batch_sizes = []

    def predict_proba(self, obs):
        self.batch_sizes.append(len(obs))
        return np.tile([.4, .1, .1, .1, .1, .1, .1], (len(obs), 1))

    def verify_source(self):
        pass


def test_batches_stop_completed_slots_and_keep_schema_and_local_images(tmp_path, monkeypatch):
    cfg = load_config(smoke=True)
    cfg['max_episode_steps'] = 3
    resets, steps = {}, {}

    def env_factory(*args, **kwargs):
        env = make_env(*args, **kwargs)
        original_reset, original_step = env.reset, env.step
        current_seed = None

        def reset(*, seed, **kw):
            nonlocal current_seed
            current_seed = seed
            resets[seed] = resets.get(seed, 0) + 1
            env.unwrapped.max_steps = 1 + seed % 3
            return original_reset(seed=seed, **kw)

        def step(action):
            assert steps.get(current_seed, 0) < 1 + current_seed % 3
            steps[current_seed] = steps.get(current_seed, 0) + 1
            return original_step(action)

        env.reset, env.step = reset, step
        return env

    monkeypatch.setattr('ra4rl.data.policy_collect.make_env', env_factory)
    policy = FixturePolicy(cfg)
    database = tmp_path / 'facts.sqlite'
    summary = collect_policy(cfg, database, policy, seeds=range(5), batch_size=2)
    assert policy.batch_sizes == [2, 1, 2, 1, 1, 1, 1]
    assert resets == dict.fromkeys(range(5), 1)
    assert steps == {seed: 1 + seed % 3 for seed in range(5)}
    assert [row['seeds'] for row in summary['batches']] == [[0, 1], [2, 3], [4]]
    assert summary['api_requests'] == 0 and summary['episode_count'] == 5
    store = Store(database, cfg['env_id'])
    try:
        assert store.meta['provenance'] == 'trained_policy' and store.meta['complete'] == 'true'
        protocol = json.loads(store.meta['collection_protocol'])
        assert protocol['rgb_input'] == LOCAL_RGB and 'llm' not in protocol
        for row in store.rows():
            assert len(row['obs_img']) == 147 and row['action_id'] == 0
            assert json.loads(row['action_probs_json']) == [.4, .1, .1, .1, .1, .1, .1]
            image = tmp_path / 'images' / f"seed_{row['episode_id']:03d}_step_{row['step_idx']:06d}.png"
            assert image.read_bytes() == row['rgb_png'] and Image.open(image).size == (224, 224)
    finally:
        store.close()


def test_policy_trajectory_replay_and_invalid_batches_remain_incomplete(tmp_path):
    cfg = load_config(smoke=True)
    cfg['max_episode_steps'] = 2
    path = tmp_path / 'facts.sqlite'
    collect_policy(cfg, path, FixturePolicy(cfg), seeds=[0, 1], batch_size=2)
    assert validate_collection(path, cfg)['experience_count'] == 4
    trace = tmp_path / 'facts.sqlite.steps.jsonl'
    original = trace.read_text()
    lines = original.splitlines()
    record = json.loads(lines[0])
    record['previous_action'] = 1
    lines[0] = json.dumps(record)
    trace.write_text('\n'.join(lines) + '\n')
    with pytest.raises(ValueError, match='Trace input or action'):
        validate_collection(path, cfg)
    trace.write_text(original)
    with pytest.raises(FileExistsError):
        collect_policy(cfg, path, FixturePolicy(cfg), seeds=[0])

    class InvalidPolicy(FixturePolicy):
        def predict_proba(self, obs):
            return np.zeros((len(obs), 7))

    invalid = tmp_path / 'bad.sqlite'
    with pytest.raises(ValueError, match='sum to one'):
        collect_policy(cfg, invalid, InvalidPolicy(cfg), seeds=[0])
    store = Store(invalid, cfg['env_id'])
    assert store.meta['complete'] == 'false'
    assert store.db.execute('SELECT COUNT(*) FROM experiences').fetchone()[0] == 0
    store.close()


def test_checkpoint_uses_matching_frozen_normalization_and_soft_distribution(tmp_path):
    cfg = load_config(smoke=True)
    env = make_vec(cfg)
    checkpoint = tmp_path / 'checkpoints/best'
    checkpoint.mkdir(parents=True)
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    try:
        model = PPO('MlpPolicy', env, device='cpu', **cfg['ppo'])
        obs = env.reset()
        obs, *_ = env.step(np.array([0, 1]))
        raw_obs = env.get_original_obs()
        expected = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0]).distribution.probs.detach().numpy()
        model.save(checkpoint / 'model.zip')
        env.save(checkpoint / 'vecnormalize.pkl')
        (checkpoint / 'metadata.json').write_text(json.dumps(dict(
            algorithm='ppo', train_env_steps=0, config_hash=config_hash(cfg))))
        policy = CheckpointPolicy(checkpoint, device='cpu')
        try:
            result = policy.predict_proba(raw_obs)
            assert np.array_equal(result, expected.astype(np.float64))
            assert np.all((result > 0) & (result < 1))
            assert np.array_equal(result.argmax(axis=1), model.predict(obs, deterministic=True)[0])
            assert policy.normalizer.training is False and policy.normalizer.norm_reward is False
            policy.verify_source()
            bad_cfg = copy.deepcopy(cfg)
            bad_cfg['max_episode_steps'] += 1
            with pytest.raises(ValueError, match='horizon'):
                collect_policy(bad_cfg, tmp_path / 'wrong.sqlite', policy, seeds=[0])
            assert not (tmp_path / 'wrong.sqlite').exists()
        finally:
            policy.close()
        (tmp_path / 'config.json').write_text(json.dumps(bad_cfg))
        with pytest.raises(ValueError, match='hash mismatch'):
            CheckpointPolicy(checkpoint, device='cpu')
    finally:
        env.close()


def test_policy_source_cannot_be_silently_used_as_llm(tmp_path):
    cfg = load_config()
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(env_id=cfg['env_id'], provenance='trained_policy')))
    with pytest.raises(ValueError, match='experience_source'):
        RetrievalTeacher(tmp_path / 'facts.sqlite', tmp_path, cfg, device='cpu')
