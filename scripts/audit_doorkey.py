"""Bounded diagnostic: no formal training or mutation of prior experiment artifacts."""
import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO

from ra4rl.checkpoints import restore
from ra4rl.config import load_config
from ra4rl.envs import make_env, make_vec
from ra4rl.evaluation import evaluate, summarize


def snapshot_training_logs(run):
    evaluations = [json.loads(line) for line in (run / "evaluations.jsonl").read_text().splitlines()]
    progress = list(csv.DictReader((run / "progress.csv").open()))
    return dict(evaluations=len(evaluations), last_steps=evaluations[-1]["train_env_steps"],
                max_eval_success=max(e["success_rate_mean"] for e in evaluations),
                nonzero_eval_count=sum(e["success_count"] > 0 for e in evaluations),
                first_entropy=-float(progress[0]["train/entropy_loss"]),
                last_entropy=-float(progress[-1]["train/entropy_loss"]),
                last_value_loss=float(progress[-1]["train/value_loss"]))


def wrapper_parity():
    project = make_env("MiniGrid-DoorKey-6x6-v0")
    reference = FlatObsWrapper(gym.make("MiniGrid-DoorKey-6x6-v0", max_steps=200))
    rng = np.random.default_rng(7)
    comparisons = 0
    for seed in range(5):
        a, _ = project.reset(seed=seed)
        b, _ = reference.reset(seed=seed)
        assert np.array_equal(a, b)
        for action in rng.integers(0, 7, size=200):
            a, ar, at, ax, _ = project.step(int(action))
            b, br, bt, bx, _ = reference.step(int(action))
            assert np.array_equal(a, b) and (ar, at, ax) == (br, bt, bx)
            comparisons += 1
            if at or ax:
                break
    project.close()
    reference.close()
    return comparisons


def orchestration_parity(device):
    cfg = load_config()
    cfg["eval_n_envs"] = 2
    steps = 4096
    env = make_vec(cfg)
    standard = PPO("MlpPolicy", env, seed=42, device=device, **cfg["ppo"])
    standard.learn(steps)
    expected = {k: v.clone() for k, v in standard.policy.state_dict().items()}
    expected_obs, expected_ret = env.obs_rms.mean.copy(), env.ret_rms.mean.copy()
    env.close()
    env, evaluation = make_vec(cfg), make_vec(cfg, training=False)
    segmented = PPO("MlpPolicy", env, seed=42, device=device, **cfg["ppo"])
    for i in range(4):
        segmented.learn(1024, reset_num_timesteps=i == 0)
        evaluate(segmented, env, evaluation, [100, 101])
    delta = max((expected[k] - v).abs().max().item() for k, v in segmented.policy.state_dict().items())
    result = dict(parameter_max_abs_delta=delta, obs_rms_equal=np.array_equal(expected_obs, env.obs_rms.mean),
                  ret_rms_equal=np.array_equal(expected_ret, env.ret_rms.mean))
    env.close()
    evaluation.close()
    return result


def inspect_checkpoint(run, device):
    cfg = load_config()
    checkpoint = sorted((run / "checkpoints").glob("step_*"))[-1]
    model, normalized, _ = restore(checkpoint, make_vec(cfg).venv, device, training=False)
    report = dict(checkpoint=str(checkpoint), policy=str(model.policy), optimizer=str(model.policy.optimizer),
                  n_updates=model._n_updates, num_timesteps=model.num_timesteps,
                  norm_reward_train=True, obs_rms_count=normalized.obs_rms.count,
                  ret_rms_mean=float(normalized.ret_rms.mean), ret_rms_var=float(normalized.ret_rms.var))
    cfg["eval_n_envs"] = 10
    evaluation = make_vec(cfg, training=False)
    report["re_evaluation"] = {k: v for k, v in evaluate(model, normalized, evaluation, range(100, 200)).items()
                               if k != "episodes"}
    evaluation.close()
    # Trace sampled and argmax behavior with exactly the checkpoint's observation statistics.
    report["traces"] = {}
    for deterministic in (True, False):
        episodes = []
        counts = dict(picked_key=0, opened_door=0, reached_right_room=0)
        action_counts = np.zeros(7, dtype=int)
        torch.manual_seed(12345)
        for seed in range(100, 120):
            env = make_env(cfg["env_id"])
            obs, _ = env.reset(seed=seed)
            door_x = next(x for x in range(env.unwrapped.width) for y in range(env.unwrapped.height)
                          if getattr(env.unwrapped.grid.get(x, y), "type", None) == "door")
            total = 0
            had_key = opened = right = False
            for length in range(1, 201):
                obs_batch = normalized.normalize_obs(obs[None, :])
                action, _ = model.predict(obs_batch, deterministic=deterministic)
                action = int(action[0])
                action_counts[action] += 1
                obs, reward, done, truncated, _ = env.step(action)
                base = env.unwrapped
                had_key |= base.carrying is not None and base.carrying.type == "key"
                opened |= any(getattr(obj, "type", None) == "door" and obj.is_open
                              for obj in base.grid.grid if obj is not None)
                right |= base.agent_pos[0] > door_x
                total += reward
                if done or truncated:
                    episodes.append(dict(success=done and reward > 0, return_value=total, length=length))
                    break
            counts["picked_key"] += had_key
            counts["opened_door"] += opened
            counts["reached_right_room"] += right
            env.close()
        report["traces"]["argmax" if deterministic else "sampled"] = dict(
            **summarize(episodes), **counts, action_counts=action_counts.tolist())
    normalized.close()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(training_logs=snapshot_training_logs(args.run), wrapper_parity_steps=wrapper_parity(),
                  orchestration_parity=orchestration_parity(args.device))
    (args.output / "audit.json").write_text(json.dumps(report, indent=2))
    report["checkpoint"] = inspect_checkpoint(args.run, args.device)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, default=lambda x: x.item()))
    print(json.dumps(report, indent=2, default=lambda x: x.item()))


if __name__ == "__main__":
    main()
