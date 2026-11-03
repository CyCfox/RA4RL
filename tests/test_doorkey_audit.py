from collections import deque

import numpy as np
from stable_baselines3 import PPO

from ra4rl.config import load_config
from ra4rl.envs import make_vec
from ra4rl.envs.minigrid import make_env
from ra4rl.evaluation import evaluate


class DoorKeyOracle:
    """Privileged test oracle to validate success/reward reporting, never used for training."""
    def __init__(self, vec):
        self.envs = vec.venv.envs

    @staticmethod
    def action(base):
        positions = {kind: next((x, y) for x in range(base.width) for y in range(base.height)
                                if getattr(base.grid.get(x, y), "type", None) == kind)
                     for kind in ("door", "goal")}
        door = base.grid.get(*positions["door"])
        if base.carrying is None and not door.is_open:
            target = next((x, y) for x in range(base.width) for y in range(base.height)
                          if getattr(base.grid.get(x, y), "type", None) == "key")
            interaction = 3
        elif not door.is_open:
            target, interaction = positions["door"], 5
        else:
            target, interaction = positions["goal"], None
        directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        initial = (*map(int, base.agent_pos), int(base.agent_dir))
        queue, visited = deque([(initial, [])]), {initial}
        while queue:
            (x, y, direction), actions = queue.popleft()
            dx, dy = directions[direction]
            if interaction is None and (x, y) == target:
                return actions[0]
            if interaction is not None and (x + dx, y + dy) == target:
                return actions[0] if actions else interaction
            options = [(0, (x, y, (direction - 1) % 4)), (1, (x, y, (direction + 1) % 4))]
            obj = base.grid.get(x + dx, y + dy)
            if obj is None or obj.can_overlap():
                options.append((2, (x + dx, y + dy, direction)))
            for action, state in options:
                if state not in visited:
                    queue.append((state, actions + [action]))
                    visited.add(state)
        raise AssertionError("Oracle found no path")

    def predict(self, obs, deterministic=True):
        return np.array([self.action(env.unwrapped) for env in self.envs]), None


def test_evaluation_detects_real_successes_and_exact_raw_reward():
    cfg = load_config(smoke=True)
    train, evaluation = make_vec(cfg), make_vec(cfg, training=False)
    train.reset()
    result = evaluate(DoorKeyOracle(evaluation), train, evaluation, range(100, 110))
    assert result["success_count"] == 10 and result["success_rate_mean"] == 1
    for episode in result["episodes"]:
        assert np.isclose(episode["return_value"], 1 - .9 * episode["length"] / 200)
    train.close()
    evaluation.close()


def test_project_rollout_orchestration_matches_single_sb3_learn():
    import torch
    cfg = load_config(smoke=True)
    training = make_vec(cfg)
    reference = PPO("MlpPolicy", training, seed=42, device="cpu", **cfg["ppo"])
    reference.learn(48)
    expected = {k: v.clone() for k, v in reference.policy.state_dict().items()}
    means = training.obs_rms.mean.copy()
    training.close()
    training, evaluation = make_vec(cfg), make_vec(cfg, training=False)
    segmented = PPO("MlpPolicy", training, seed=42, device="cpu", **cfg["ppo"])
    for i in range(3):
        segmented.learn(16, reset_num_timesteps=i == 0)
        evaluate(segmented, training, evaluation, [100, 101])
    assert all(torch.equal(expected[k], v) for k, v in segmented.policy.state_dict().items())
    assert np.array_equal(means, training.obs_rms.mean)
    training.close()
    evaluation.close()


def test_revised_doorkey_config_retains_zoo_ppo_and_native_horizon():
    original = load_config()
    revised = load_config("configs/doorkey_zoo_native.yaml")
    assert original["max_episode_steps"] == 200  # Historical protocol remains reproducible.
    assert revised["ppo"] == original["ppo"]
    assert revised["max_episode_steps"] == 360 and revised["ppo"]["ent_coef"] == 0
    assert revised["total_timesteps"] == 100000
    env = make_env(revised["env_id"], revised["max_episode_steps"])
    env.reset(seed=42)
    assert env.unwrapped.max_steps == 10 * env.unwrapped.width ** 2 == 360
    env.unwrapped.step_count = 10
    assert env.unwrapped._reward() == 1 - .9 * 10 / 360
    env.close()
