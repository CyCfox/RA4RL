"""Fixed, bounded DoorKey diagnostics; separate from the paused formal experiment."""
import argparse
import json
from pathlib import Path
import time

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from ra4rl.config import load_config
from ra4rl.envs import make_vec
from ra4rl.evaluation import evaluate


class Probe(gym.Wrapper):
    def reset(self, **kwargs):
        self.had_key = self.opened = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        base = self.unwrapped
        key = base.carrying is not None and base.carrying.type == "key"
        opened = any(getattr(obj, "type", None) == "door" and obj.is_open for obj in base.grid.grid
                     if obj is not None)
        info["new_key"] = key and not self.had_key
        info["new_open"] = opened and not self.opened
        info["is_success"] = bool(done and reward > 0)
        self.had_key |= key
        self.opened |= opened
        return obs, reward, done, trunc, info


class Counts(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = self.successes = self.keys = self.opened = 0
        self.actions = np.zeros(7, dtype=int)
        self.first_success_step = None

    def _on_step(self):
        for info in self.locals["infos"]:
            self.keys += info["new_key"]
            self.opened += info["new_open"]
            if "episode" in info:
                self.episodes += 1
                success = info["episode"]["r"] > 0
                self.successes += success
                if success and self.first_success_step is None:
                    self.first_success_step = self.num_timesteps
        self.actions += np.bincount(self.locals["actions"].astype(int).flatten(), minlength=7)
        return True


def run_case(root, env_id, max_steps, ent_coef, seed, total, device):
    name = f"{env_id}_cap{max_steps}_ent{ent_coef}_seed{seed}"
    folder = root / name
    folder.mkdir()
    cfg = load_config()
    cfg.update(env_id=env_id, train_seed=seed, max_episode_steps=max_steps, profile="diagnostic")
    cfg["ppo"]["ent_coef"] = ent_coef
    (folder / "config.json").write_text(json.dumps(cfg, indent=2))
    # Independent native Zoo-style construction: SB3 make_vec_env wraps Monitor, FlatObsWrapper outside.
    env = make_vec_env(lambda: Probe(gym.make(env_id, max_steps=max_steps)), n_envs=8, seed=seed,
                       wrapper_class=FlatObsWrapper, monitor_kwargs={"info_keywords": ("is_success",)})
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)
    model = PPO("MlpPolicy", env, seed=seed, device=device, **cfg["ppo"])
    callback = Counts()
    started = time.monotonic()
    model.learn(total, callback=callback)
    evaluation = make_vec(cfg, training=False)
    result = evaluate(model, env, evaluation, range(100, 200))
    result = {k: v for k, v in result.items() if k != "episodes"}
    result.update(train_env_steps=model.num_timesteps, elapsed_seconds=time.monotonic() - started,
                  train_episodes=callback.episodes, train_successes=callback.successes,
                  first_train_success_step=callback.first_success_step,
                  train_key_episodes=callback.keys, train_door_open_episodes=callback.opened,
                  action_counts=callback.actions.tolist())
    model.save(folder / "model.zip")
    env.save(folder / "vecnormalize.pkl")
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    env.close()
    evaluation.close()
    return dict(case=name, **result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases", default="zoo5,project6,native6,entropy6")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = dict(zoo5=("MiniGrid-DoorKey-5x5-v0", 250, 0.0),
                 project6=("MiniGrid-DoorKey-6x6-v0", 200, 0.0),
                 native6=("MiniGrid-DoorKey-6x6-v0", 360, 0.0),
                 entropy6=("MiniGrid-DoorKey-6x6-v0", 200, 0.01))
    selected = [cases[name] for name in args.cases.split(",")]
    (args.output / "protocol.json").write_text(json.dumps(dict(
        purpose="finite diagnosis, not formal experiment", seed=42, steps=args.steps,
        cases=selected, other_ppo_parameters=load_config()["ppo"]), indent=2))
    results = []
    for env_id, cap, entropy in selected:
        results.append(run_case(args.output, env_id, cap, entropy, 42, args.steps, args.device))
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
