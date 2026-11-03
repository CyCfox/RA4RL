import json
from pathlib import Path
import shutil

from stable_baselines3.common.vec_env import VecNormalize

from ra4rl.algorithms.ppo import RA4RLPPO
from stable_baselines3 import PPO


class Checkpoints:
    def __init__(self, root, keep=10):
        self.root, self.keep = Path(root), keep
        self.root.mkdir(parents=True, exist_ok=True)
        self.best_score = None

    def save(self, model, env, evaluation, config_hash, index_signature=None):
        step = model.num_timesteps
        name = f"step_{step:09d}"
        target = self.root / name
        temporary = self.root / ("." + name)
        temporary.mkdir(exist_ok=False)
        model.save(temporary / "model.zip")
        env.save(temporary / "vecnormalize.pkl")
        metadata = dict(train_env_steps=step, config_hash=config_hash,
                        evaluation=evaluation, index_signature=index_signature,
                        algorithm="ra4rl" if isinstance(model, RA4RLPPO) else "ppo")
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2))
        if target.exists():
            raise FileExistsError(target)
        temporary.rename(target)
        score = (evaluation["success_rate_mean"], evaluation["return_mean"], -evaluation["episode_steps_mean"])
        if self.best_score is None or score > self.best_score:
            fresh = self.root / ".best_new"
            shutil.copytree(target, fresh)
            best = self.root / "best"
            if best.exists():
                shutil.rmtree(best)
            fresh.rename(best)
            self.best_score = score
        # Only delete checkpoints created within this run's own directory.
        for old in sorted(self.root.glob("step_*"))[:-self.keep]:
            shutil.rmtree(old)
        return target


def restore(path, vec_env, device="cuda", teacher=None, training=True):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    env = VecNormalize.load(path / "vecnormalize.pkl", vec_env)
    env.training, env.norm_reward = training, training
    cls = RA4RLPPO if metadata["algorithm"] == "ra4rl" else PPO
    model = cls.load(path / "model.zip", env=env, device=device)
    if isinstance(model, RA4RLPPO):
        model.teacher = teacher
    return model, env, metadata
