import json

from ra4rl.config import load_config
from ra4rl.training import train
from ra4rl.evaluation_log import load_episodes


def test_resume_preserves_step_accounting_and_rejects_changed_protocol(tmp_path):
    cfg = load_config(smoke=True)
    cfg["max_episode_steps"] = 2
    initial = tmp_path / "initial"
    assert train(cfg, "ppo", initial, device="cpu") == 32
    completion = json.loads((initial / "completion.json").read_text())
    assert completion["actual_train_env_steps"] == 32
    assert (initial / "plots" / "training_curves.png").is_file()
    monitor_files = list((initial / "monitor").glob("*.monitor.csv"))
    assert len(monitor_files) == cfg["n_envs"]
    assert all("is_success" in path.read_text() for path in monitor_files)
    progress = [json.loads(line) for line in (initial / "progress.json").read_text().splitlines()]
    assert all("rollout/ep_rew_mean" in row and "rollout/success_rate" in row for row in progress)
    checkpoint = initial / "checkpoints" / "step_000000016"
    resumed = tmp_path / "resumed"
    assert train(cfg, "ppo", resumed, device="cpu", resume=checkpoint) == 32
    logs = [json.loads(line) for line in (resumed / "evaluations.jsonl").read_text().splitlines()]
    assert len(logs) == 1 and logs[0]["train_env_steps"] == 32
    assert list(logs[0])[0] == "train_env_steps" and "episodes" not in logs[0]
    assert [item["seed"] for item in load_episodes(resumed, logs[0])] == [100, 101]
    cfg["ppo"]["learning_rate"] *= 2
    import pytest
    with pytest.raises(ValueError, match="configuration"):
        train(cfg, "ppo", tmp_path / "mismatch", device="cpu", resume=checkpoint)
