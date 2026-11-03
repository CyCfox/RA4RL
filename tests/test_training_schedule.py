import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ra4rl.config import load_config
from ra4rl.envs.minigrid import make_env, make_vec
from ra4rl.evaluation_log import load_episodes
from ra4rl.training import train


@pytest.mark.parametrize('interval,total,expected', [
    (5, 112, [80, 112]), (5, 160, [80, 160]),
    (50, 816, [800, 816]), (50, 1600, [800, 1600]),
    (20, 336, [320, 336]), (20, 640, [320, 640]),
    (30, 496, [480, 496]), (30, 960, [480, 960]),
    (100, 1616, [1600, 1616]), (100, 3200, [1600, 3200]),
])
def test_evaluate_at_requested_interval_and_once_at_final_step(tmp_path, interval, total, expected):
    cfg = load_config(smoke=True)
    cfg.update(total_timesteps=total, max_episode_steps=2, eval_interval_updates=interval)
    output = tmp_path / 'run'
    assert train(cfg, 'ppo', output, device='cpu') == total
    evaluations = [json.loads(line) for line in (output / 'evaluations.jsonl').read_text().splitlines()]
    assert [r['train_env_steps'] for r in evaluations] == expected
    assert all(next(iter(r)) == 'train_env_steps' and 'episodes' not in r and 'episodes_file' not in r
               for r in evaluations)
    details = [json.loads(line) for line in (output / 'evaluation_details.jsonl').read_text().splitlines()]
    assert [r['train_env_steps'] for r in details] == expected
    assert all([e['seed'] for e in load_episodes(output, r)] == [100, 101] for r in evaluations)
    assert not (output / 'evaluation_details').exists()
    assert sorted(int(p.name[5:]) for p in (output / 'checkpoints').glob('step_*')) == expected
    training = [json.loads(line) for line in (output / 'progress.json').read_text().splitlines()]
    assert [r['time/total_timesteps'] for r in training] == list(range(16, total + 1, 16))
    assert (output / 'plots/training_curves.png').is_file()


@pytest.mark.parametrize('interval', [0, -1, 1.5, True])
def test_invalid_evaluation_interval_is_rejected(tmp_path, interval):
    cfg = load_config()
    cfg['eval_interval_updates'] = interval
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='positive integer'):
        load_config(path)


def test_gotodoor_6x6_3m_changes_only_budget_and_evaluation_interval():
    cfg = load_config('configs/gotodoor_6x6_3m.yaml')
    previous = load_config('configs/gotodoor_6x6_zoo_native.yaml')
    assert {key for key in cfg if cfg[key] != previous[key]} == {'total_timesteps', 'eval_interval_updates'}
    assert cfg['env_id'] == 'MiniGrid-GoToDoor-6x6-v0' and cfg['max_episode_steps'] == 200
    assert cfg['total_timesteps'] == 3000000 and cfg['eval_interval_updates'] == 30
    rollout = cfg['n_envs'] * cfg['ppo']['n_steps']
    updates = int(np.ceil(cfg['total_timesteps'] / rollout))
    assert rollout == 1024 and updates == 2930 and updates * rollout == 3000320
    assert updates // 30 + bool(updates % 30) == 98


def test_gotodoor_5x5_2m_changes_only_budget_and_evaluation_interval():
    cfg = load_config('configs/gotodoor_5x5_2m.yaml')
    previous = load_config('configs/gotodoor_5x5_zoo_native.yaml')
    assert {key for key in cfg if cfg[key] != previous[key]} == {'total_timesteps', 'eval_interval_updates'}
    assert cfg['env_id'] == 'MiniGrid-GoToDoor-5x5-v0' and cfg['max_episode_steps'] == 200
    assert cfg['total_timesteps'] == 2000000 and cfg['eval_interval_updates'] == 20
    rollout = cfg['n_envs'] * cfg['ppo']['n_steps']
    updates = int(np.ceil(cfg['total_timesteps'] / rollout))
    assert rollout == 1024 and updates == 1954 and updates * rollout == 2000896
    assert updates // 20 + bool(updates % 20) == 98


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('horizon', [30, 50, 100, 200, 400])
def test_putnear_zoo_configuration_and_requested_horizon(training, horizon):
    path = ('configs/putnear_6x6_n2_zoo_native.yaml' if horizon == 100
            else f'configs/putnear_6x6_n2_{horizon}.yaml')
    cfg = load_config(path)
    if horizon != 100:
        previous_paths = {30: 'configs/putnear_6x6_n2_50.yaml',
                          50: 'configs/putnear_6x6_n2_400.yaml',
                          200: 'configs/putnear_6x6_n2_zoo_native.yaml',
                          400: 'configs/putnear_6x6_n2_200.yaml'}
        previous = load_config(previous_paths[horizon])
        assert {k for k in cfg if cfg[k] != previous[k]} == {'protocol_revision', 'max_episode_steps'}
    zoo = yaml.safe_load(Path('rl-baselines3-zoo/hyperparams/ppo.yml').read_text())[cfg['env_id']]
    assert zoo['policy'] == 'MlpPolicy' and zoo['normalize'] is True
    assert zoo['env_wrapper'] == 'minigrid.wrappers.FlatObsWrapper'
    assert cfg['total_timesteps'] == zoo['n_timesteps'] == 10000000
    assert cfg['n_envs'] == zoo['n_envs'] == 8 and cfg['eval_interval_updates'] == 100
    for key in ('n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'gae_lambda',
                'clip_range', 'ent_coef'):
        assert cfg['ppo'][key] == zoo[key]
    updates = int(np.ceil(cfg['total_timesteps'] / 1024))
    assert updates == 9766 and updates * 1024 == 10000384
    assert updates // 100 + bool(updates % 100) == 98
    env = make_vec(cfg, training=training)
    try:
        obs = env.reset()
        assert obs.shape == (8 if training else 10, 2835)
        assert all(slot.unwrapped.width == slot.unwrapped.height == 6 for slot in env.venv.envs)
        assert all(slot.unwrapped.numObjs == 2 and slot.unwrapped.max_steps == horizon for slot in env.venv.envs)
        for step in range(horizon):
            _, rewards, dones, infos = env.step(np.zeros(env.num_envs, dtype=int))
            assert np.all(rewards == 0) and np.all(dones == (step == horizon - 1))
        assert all(info['TimeLimit.truncated'] for info in infos)
    finally:
        env.close()


@pytest.mark.parametrize('outcome', ['success_diagonal', 'wrong_drop', 'wrong_pickup'])
@pytest.mark.parametrize('horizon', [30, 50, 100, 200, 400])
def test_putnear_native_success_reward_and_failure_are_preserved(outcome, horizon):
    env = make_env('MiniGrid-PutNear-6x6-N2-v0', max_steps=horizon, flat=False)
    try:
        env.reset(seed=42)
        base = env.unwrapped
        moving = base.grid.get(*base.move_pos)
        target = base.grid.get(*base.target_pos)
        for x in range(1, 5):
            for y in range(1, 5):
                base.grid.set(x, y, None)
        base.agent_pos, base.agent_dir = np.array([2, 1]), 1
        base.target_pos = np.array([4, 4] if outcome == 'wrong_drop' else [3, 3])
        if outcome == 'wrong_pickup':
            base.target_pos = np.array([2, 2])
            base.grid.set(2, 2, target)
            base.grid.set(4, 4, moving)
        else:
            base.grid.set(*base.target_pos, target)
            base.grid.set(2, 2, moving)
        _, reward, terminated, truncated, info = env.step(3)
        if outcome == 'wrong_pickup':
            assert terminated and not truncated and reward == 0 and not info['is_success']
        else:
            assert not terminated and not truncated and reward == 0
            _, reward, terminated, truncated, info = env.step(4)
            assert terminated and not truncated
            assert info['is_success'] == (outcome == 'success_diagonal')
            assert reward == pytest.approx(1 - .9 * 2 / horizon if outcome == 'success_diagonal' else 0)
    finally:
        env.close()


@pytest.mark.parametrize('variant', ['s3r1', 's3r2'])
def test_keycorridor_native_horizon_and_zoo_configuration(variant):
    cfg = load_config(f'configs/keycorridor_{variant}_zoo_native.yaml')
    reference = load_config('configs/doorkey_zoo_native.yaml')
    assert cfg['ppo'] == reference['ppo'] and cfg['n_envs'] == 8
    assert cfg['total_timesteps'] == 500000 and cfg['eval_interval_updates'] == 5
    assert int(np.ceil(cfg['total_timesteps'] / 1024)) * 1024 == 500736
    env = make_env(cfg['env_id'], cfg['max_episode_steps'])
    try:
        obs, _ = env.reset(seed=42)
        assert env.unwrapped.spec.id == f'MiniGrid-KeyCorridor{variant.upper()}-v0'
        assert obs.shape == (2835,) and env.unwrapped.max_steps == 270
        for step in range(270):
            _, reward, terminated, truncated, _ = env.step(6)
            assert not terminated and reward == 0
            assert truncated == (step == 269)
    finally:
        env.close()


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('size,total,interval,updates,actual', [
    (5, 5000000, 50, 4883, 5000192), (6, 2000000, 20, 1954, 2000896),
])
def test_gotodoor_zoo_config_and_200_step_horizon(training, size, total, interval, updates, actual):
    cfg = load_config(f'configs/gotodoor_{size}x{size}_zoo_native.yaml')
    zoo = yaml.safe_load(Path('rl-baselines3-zoo/hyperparams/ppo.yml').read_text())['MiniGrid-GoToDoor-5x5-v0']
    assert zoo['policy'] == 'MlpPolicy' and zoo['normalize'] is True
    assert zoo['env_wrapper'] == 'minigrid.wrappers.FlatObsWrapper'
    assert cfg['n_envs'] == zoo['n_envs'] == 8
    assert cfg['total_timesteps'] == total and zoo['n_timesteps'] == 5000000
    for key in ('n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'gae_lambda',
                'clip_range', 'ent_coef'):
        assert cfg['ppo'][key] == zoo[key]
    assert cfg['eval_interval_updates'] == interval
    assert int(np.ceil(cfg['total_timesteps'] / 1024)) == updates
    assert updates * 1024 == actual
    assert updates // interval + bool(updates % interval) == 98
    env = make_vec(cfg, training=training)
    try:
        obs = env.reset()
        assert obs.shape == (8 if training else 10, 2835)
        assert all(slot.unwrapped.spec.id == cfg['env_id'] for slot in env.venv.envs)
        assert all(slot.unwrapped.width == slot.unwrapped.height == size for slot in env.venv.envs)
        assert all(slot.unwrapped.max_steps == 200 for slot in env.venv.envs)
        for step in range(200):
            _, rewards, dones, infos = env.step(np.zeros(env.num_envs, dtype=int))
            assert np.all(rewards == 0) and np.all(dones == (step == 199))
        assert all(info['TimeLimit.truncated'] for info in infos)
    finally:
        env.close()


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('horizon', [360, 540])
def test_keycorridor_s3r2_horizon_applies_to_training_and_evaluation(training, horizon):
    cfg = load_config(f'configs/keycorridor_s3r2_{horizon}.yaml')
    old = load_config('configs/keycorridor_s3r2_zoo_native.yaml')
    assert {key for key in cfg if cfg[key] != old[key]} == {'protocol_revision', 'max_episode_steps'}
    assert cfg['max_episode_steps'] == horizon and cfg['eval_interval_updates'] == 5
    assert cfg['total_timesteps'] == 500000
    env = make_vec(cfg, training=training)
    try:
        env.seed(42 if training else 100)
        obs = env.reset()
        assert obs.shape == (8 if training else 10, 2835)
        assert all(slot.unwrapped.max_steps == horizon for slot in env.venv.envs)
        for step in range(horizon):
            _, rewards, dones, infos = env.step(np.full(env.num_envs, 6))
            assert np.all(rewards == 0)
            assert np.all(dones == (step == horizon - 1))
        assert all(info['TimeLimit.truncated'] for info in infos)
    finally:
        env.close()
