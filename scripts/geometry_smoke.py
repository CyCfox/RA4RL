"""One DoorKey rollout per algorithm with full formal update geometry, synthetic fixtures.

This checks 8*128 samples and 160 optimizer steps without running a final experiment.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ra4rl.config import load_config
from ra4rl.training import train
from ra4rl.checkpoints import restore
from ra4rl.envs import make_vec
from ra4rl.evaluation_log import load_episodes

parser = argparse.ArgumentParser()
parser.add_argument("--fixture-root", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=False)
cfg = load_config()
cfg.update(profile="smoke", total_timesteps=1024)
fixture = Path(args.fixture_root) / cfg["env_id"]
report = {"profile": "smoke", "formal_experiments_run": False, "algorithms": {}}
for algorithm in ("ppo", "ra4rl"):
    output = root / algorithm
    steps = train(cfg, algorithm, output,
                  fixture / "experiences.sqlite" if algorithm == "ra4rl" else None,
                  fixture / "index" if algorithm == "ra4rl" else None)
    model, env, _ = restore(output / "checkpoints" / "best", make_vec(cfg).venv, training=False)
    optimizer_steps = {int(state["step"].item()) for state in model.policy.optimizer.state.values()}
    assert optimizer_steps == {160}
    assert steps == 1024 and model._n_updates == 10
    evaluations = [json.loads(line) for line in (output / "evaluations.jsonl").read_text().splitlines()]
    assert len(evaluations) == 1
    assert [episode["seed"] for episode in load_episodes(output, evaluations[0])] == list(range(100, 200))
    logs = [json.loads(line) for line in (output / "progress.json").read_text().splitlines()]
    if algorithm == "ra4rl":
        assert np.isfinite(logs[-1]["train/kd_loss"]) and logs[-1]["train/lambda"] > 0
    env.close()
    report["algorithms"][algorithm] = {"steps": steps, "optimizer_steps": 160, "epochs": 10,
                                       "evaluation_episodes": 100, "evaluation_parallel_envs": 10}
(root / "geometry_report.json").write_text(json.dumps(report, indent=2))
