import asyncio
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoints import restore
from .config import ENV_IDS, load_config
from .data.collect import collect
from .envs import make_vec
from .evaluation_log import load_episodes
from .retrieval.index import build_index
from .retrieval.teacher import RetrievalTeacher
from .training import train


class SyntheticClient:
    """Explicit fixture, never an LLM baseline or a source for formal experiments."""
    def __init__(self):
        self.counter = 0

    async def predict(self, env_id, state, png):
        await asyncio.sleep(0)
        probs = np.full(7, 0.02)
        probs[self.counter % 3] = 0.88
        self.counter += 1
        return probs


def run_smoke(output, device="cuda"):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    report = {"profile": "smoke", "synthetic_experiences": True, "formal_experiments_run": False,
              "cuda_available": torch.cuda.is_available(), "environments": []}
    for env_id in ENV_IDS:
        folder = root / env_id
        folder.mkdir()
        cfg = load_config(env_id=env_id, smoke=True)
        fixture_cfg = load_config(env_id=env_id, smoke=True)
        fixture_cfg["max_episode_steps"] = 12
        db_path, index_path = folder / "experiences.sqlite", folder / "index"
        summary = asyncio.run(collect(fixture_cfg, db_path, SyntheticClient(), range(3), synthetic=True))
        build_index(db_path, index_path, cfg, device)
        teacher = RetrievalTeacher(db_path, index_path, cfg, device, allow_synthetic=True)
        vec = make_vec(cfg)
        vec.reset()
        raw = vec.get_attr("current_raw")
        p1 = teacher(raw)
        p2 = teacher(raw)
        assert np.array_equal(p1, p2)
        fresh = RetrievalTeacher(db_path, index_path, cfg, device, allow_synthetic=True)
        assert np.allclose(p1, fresh(raw), atol=1e-6)
        vec.close()
        del teacher, fresh
        results = dict(env_id=env_id, fixture_episodes=summary["episode_count"], algorithms={})
        for algorithm in ("ppo", "ra4rl"):
            run = folder / algorithm
            steps = train(cfg, algorithm, run,
                          db_path if algorithm == "ra4rl" else None,
                          index_path if algorithm == "ra4rl" else None, device)
            checkpoint = sorted((run / "checkpoints").glob("step_*"))[-1]
            vec = make_vec(cfg)
            # restore consumes a raw vector environment and reloads checkpoint-specific statistics.
            model, restored, metadata = restore(checkpoint, vec.venv, device, training=False)
            assert metadata["train_env_steps"] == steps == cfg["total_timesteps"]
            obs = restored.reset()
            action1, _ = model.predict(obs, deterministic=True)
            model2, restored2, _ = restore(checkpoint, make_vec(cfg).venv, device, training=False)
            assert np.array_equal(action1, model2.predict(obs, deterministic=True)[0])
            assert np.array_equal(restored.obs_rms.mean, restored2.obs_rms.mean)
            restored.close()
            restored2.close()
            logs = [json.loads(line) for line in (run / "evaluations.jsonl").read_text().splitlines()]
            assert len(logs) == 2
            assert all([e["seed"] for e in load_episodes(run, log)] == [100, 101] for log in logs)
            results["algorithms"][algorithm] = dict(train_env_steps=steps, evaluations=len(logs),
                                                     checkpoint_reload=True)
        report["environments"].append(results)
        (root / "smoke_report.json").write_text(json.dumps(report, indent=2))
    return report
