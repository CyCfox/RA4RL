import numpy as np
from stable_baselines3 import PPO

from ra4rl.config import load_config
from ra4rl.envs import make_vec
from ra4rl.evaluation import evaluate, summarize


def test_metrics_no_and_single_success():
    empty = summarize([dict(success=False, return_value=0., length=200)])
    assert np.isnan(empty["successful_steps_mean"]) and np.isnan(empty["successful_steps_std"])
    one = summarize([dict(success=True, return_value=.8, length=12),
                     dict(success=False, return_value=0., length=200)])
    assert one["successful_steps_std"] == 0
    assert one["success_rate_mean"] == .5 and one["success_rate_std"] == .5


def test_evaluation_fixed_seeds_frozen_statistics():
    cfg = load_config(smoke=True)
    cfg["max_episode_steps"] = 3
    training, evaluation = make_vec(cfg), make_vec(cfg, training=False)
    model = PPO("MlpPolicy", training, seed=42, device="cpu", **cfg["ppo"])
    model.learn(16)
    count = training.obs_rms.count
    steps = model.num_timesteps
    first = evaluate(model, training, evaluation, [100, 101, 102])
    frozen_count = evaluation.obs_rms.count
    second = evaluate(model, training, evaluation, [100, 101, 102])
    assert first["episodes"] == second["episodes"]
    assert [e["seed"] for e in first["episodes"]] == [100, 101, 102]
    assert training.obs_rms.count == count == frozen_count == evaluation.obs_rms.count
    assert model.num_timesteps == steps
    assert evaluation.norm_reward is False and evaluation.training is False
    training.close()
    evaluation.close()
