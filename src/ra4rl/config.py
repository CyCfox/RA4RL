import copy
import hashlib
import json
from pathlib import Path

import yaml

ENV_IDS = (
    "MiniGrid-DoorKey-6x6-v0", "MiniGrid-GoToDoor-6x6-v0",
    "MiniGrid-Fetch-6x6-N2-v0", "MiniGrid-KeyCorridorS3R3-v0",
    "MiniGrid-KeyCorridorS3R1-v0",
    "MiniGrid-KeyCorridorS3R2-v0",
    "MiniGrid-GoToDoor-5x5-v0",
    "MiniGrid-PutNear-6x6-N2-v0",
)


def load_config(path="configs/default.yaml", env_id=None, smoke=False):
    cfg = yaml.safe_load(Path(path).read_text())
    if env_id:
        cfg["env_id"] = env_id
    if cfg["env_id"] not in ENV_IDS:
        raise ValueError("Unsupported environment")
    cfg["profile"] = "smoke" if smoke else "formal"
    if smoke:
        cfg.update(total_timesteps=32, n_envs=2, eval_n_envs=2, eval_seeds=[100, 102])
        cfg["ppo"].update(n_steps=8, batch_size=8, n_epochs=2)
        cfg["retrieval"].update(embedding_batch_size=32, rerank_batch_size=32)
    if cfg["n_envs"] * cfg["ppo"]["n_steps"] % cfg["ppo"]["batch_size"]:
        raise ValueError("Rollout must contain complete mini-batches")
    interval = cfg.get("eval_interval_updates", 1)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("eval_interval_updates must be a positive integer")
    return copy.deepcopy(cfg)


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
