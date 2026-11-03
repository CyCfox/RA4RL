import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from ra4rl.algorithms.ppo import KDRolloutBuffer, RA4RLPPO, forward_kl, kd_weight
from ra4rl.config import load_config
from ra4rl.envs import make_vec
from ra4rl.envs.minigrid import make_env, snapshot


def test_kd_schedule_and_zero_safe_gradients():
    assert kd_weight(0) == 10 and kd_weight(50000) == 5
    assert kd_weight(100000) == 0 and kd_weight(100001) == 0
    assert kd_weight(0, anneal_steps=10000) == 10
    assert kd_weight(5000, anneal_steps=10000) == 5
    assert kd_weight(9216, anneal_steps=10000) > 0
    assert kd_weight(10000, anneal_steps=10000) == kd_weight(10240, anneal_steps=10000) == 0
    assert kd_weight(0, anneal_steps=50000) == 10
    assert kd_weight(25000, anneal_steps=50000) == 5
    assert kd_weight(49152, anneal_steps=50000) > 0
    assert kd_weight(50000, anneal_steps=50000) == kd_weight(50176, anneal_steps=50000) == 0
    assert kd_weight(0, anneal_steps=200000) == 10
    assert kd_weight(100000, anneal_steps=200000) == 5
    assert kd_weight(199680, anneal_steps=200000) > 0
    assert kd_weight(200000, anneal_steps=200000) == kd_weight(200704, anneal_steps=200000) == 0
    logits = torch.zeros((1, 7), requires_grad=True)
    teacher = torch.tensor([[1., 0, 0, 0, 0, 0, 0]])
    loss = forward_kl(teacher, logits.log_softmax(-1))
    assert loss.item() == pytest.approx(np.log(7))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.allclose(logits.grad, torch.softmax(logits.detach(), -1) - teacher)


def test_buffer_shuffle_preserves_teacher_alignment():
    buf = KDRolloutBuffer(4, spaces.Box(0, 100, (1,), dtype=np.float32), spaces.Discrete(7),
                          n_envs=2, device="cpu")
    teachers = np.zeros((4, 2, 7), dtype=np.float32)
    for step in range(4):
        code = np.array([2 * step, 2 * step + 1])
        teachers[step, np.arange(2), code % 7] = 1
        buf.add(code[:, None].astype(np.float32), np.zeros((2, 1)), np.zeros(2), np.zeros(2),
                torch.zeros(2), torch.zeros(2))
    buf.compute_returns_and_advantage(torch.zeros(2), np.ones(2))
    buf.set_teacher(teachers)
    for _ in range(3):
        for data in buf.get(3):
            assert torch.equal(data.observations.flatten().long() % 7, data.teacher_probs.argmax(-1))


def test_putnear_llm_ra4rl_one_million_anneal_protocol_and_rollout_boundaries():
    cfg = load_config('configs/putnear_ra4rl_llm_1m.yaml')
    source = load_config('configs/putnear_llm_thinking_low.yaml')
    assert {key for key in cfg if cfg[key] != source[key]} == {'kd'}
    assert cfg['kd'] == {'lambda_start': 10.0, 'anneal_steps': 1000000}
    assert cfg['env_id'] == 'MiniGrid-PutNear-6x6-N2-v0' and cfg['max_episode_steps'] == 50
    assert cfg['total_timesteps'] == 10000000 and cfg['eval_interval_updates'] == 100
    assert cfg['llm']['thinking'] == {'type': 'enabled'} and cfg['llm']['reasoning_effort'] == 'low'
    assert kd_weight(0, **cfg['kd']) == 10
    assert kd_weight(500000, **cfg['kd']) == 5
    assert kd_weight(999424, **cfg['kd']) > 0
    assert kd_weight(1000000, **cfg['kd']) == kd_weight(1000448, **cfg['kd']) == 0
    assert cfg['n_envs'] * cfg['ppo']['n_steps'] == 1024
    assert 9766 * 1024 == 10000384
    assert 9766 // 100 + bool(9766 % 100) == 98


def test_zero_lambda_is_exact_upstream_ppo_with_timeouts():
    cfg = load_config(smoke=True)
    cfg["max_episode_steps"] = 2
    env_a, env_b = make_vec(cfg), make_vec(cfg)
    common = dict(seed=42, device="cpu", **cfg["ppo"])
    a = PPO("MlpPolicy", env_a, **common)
    a.learn(16)

    def forbidden(_):
        pytest.fail("lambda=0 must not call teacher")

    b = RA4RLPPO("MlpPolicy", env_b, teacher=forbidden, lambda_start=0, **common)
    b.learn(16)
    for p, q in zip(a.policy.parameters(), b.policy.parameters()):
        assert torch.equal(p, q)
    assert np.array_equal(a.rollout_buffer.returns, b.rollout_buffer.returns)
    env_a.close()
    env_b.close()


def test_raw_alignment_auto_reset_and_one_optimizer_step_per_minibatch():
    cfg = load_config(smoke=True)
    cfg["max_episode_steps"] = 1
    env = make_vec(cfg)
    received = []

    def teacher(states):
        received.extend(states)
        return np.tile([1., 0, 0, 0, 0, 0, 0], (len(states), 1))

    model = RA4RLPPO("MlpPolicy", env, teacher=teacher, seed=42, device="cpu", **cfg["ppo"])
    calls = []
    original = model.policy.optimizer.step

    def step(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    model.policy.optimizer.step = step
    model.learn(16)
    assert len(received) == 16
    assert len(calls) == 4  # 16 samples / batch 8 * 2 epochs, no separate KD optimizer pass
    assert received[0].image.shape == (7, 7, 3)
    assert np.isfinite(model.logger.name_to_value["train/kd_loss"])
    # Independently replay each slot, including every TimeLimit auto-reset, against captured pre-step states.
    actions = model.rollout_buffer.actions.reshape(2, 8)
    for slot in range(2):
        reference = make_env(cfg["env_id"], max_steps=1, flat=False)
        obs, _ = reference.reset(seed=42 + slot)
        for t in range(8):
            expected = snapshot(reference, obs)
            captured = received[t * 2 + slot]
            assert np.array_equal(captured.image, expected.image)
            assert captured.mission == expected.mission and captured.direction == expected.direction
            obs, _, done, truncated, _ = reference.step(int(actions[slot, t]))
            assert done or truncated
            obs, _ = reference.reset()
        reference.close()
    env.close()
