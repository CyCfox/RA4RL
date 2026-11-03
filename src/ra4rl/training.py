import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from .algorithms.ppo import RA4RLPPO, kd_weight
from .checkpoints import Checkpoints, restore
from .config import config_hash
from .envs import make_vec
from .evaluation import evaluate
from .evaluation_log import EvaluationLog
from .provenance import record_run
from .retrieval.teacher import RetrievalTeacher


def train(cfg, algorithm, output, db_path=None, index_path=None, device="cuda", resume=None):
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Run experimental Python inside Docker")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable")
    teacher = None
    if algorithm == "ra4rl":
        teacher = RetrievalTeacher(db_path, index_path, cfg, device, allow_synthetic=cfg["profile"] == "smoke")
    manifest = record_run(output, cfg, algorithm, db_path, index_path)
    train_env = make_vec(cfg, monitor_dir=Path(output) / "monitor")
    eval_env = make_vec(cfg, training=False)
    try:
        cls = PPO if algorithm == "ppo" else RA4RLPPO
        extra = {} if algorithm == "ppo" else dict(teacher=teacher, **cfg["kd"])
        if resume:
            raw_vec = train_env.venv
            model, train_env, metadata = restore(resume, raw_vec, device, teacher)
            if metadata["config_hash"] != config_hash(cfg) or metadata["algorithm"] != algorithm:
                raise ValueError("Resume configuration / algorithm mismatch")
            if metadata.get("index_signature") != manifest.get("index_manifest_sha256"):
                raise ValueError("Resume retrieval index changed")
        else:
            model = cls("MlpPolicy", train_env, seed=cfg["train_seed"], device=device,
                        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                                           activation_fn=torch.nn.Tanh, ortho_init=True),
                        **cfg["ppo"], **extra)
        model.set_logger(configure(str(output), ["csv", "json"]))
        checkpoints = Checkpoints(Path(output) / "checkpoints")
        rollout_steps = cfg["n_envs"] * cfg["ppo"]["n_steps"]
        first = resume is None
        with EvaluationLog(output) as evaluation_log:
            while model.num_timesteps < cfg["total_timesteps"]:
                model.learn(total_timesteps=rollout_steps, reset_num_timesteps=first, log_interval=None)
                first = False
                update = model.num_timesteps // rollout_steps
                if (update % cfg.get("eval_interval_updates", 1) == 0
                        or model.num_timesteps >= cfg["total_timesteps"]):
                    result = evaluate(model, train_env, eval_env, range(*cfg["eval_seeds"]))
                    summary = evaluation_log.write(model.num_timesteps, result,
                        0.0 if algorithm == "ppo" else kd_weight(model.num_timesteps, **cfg["kd"]))
                    checkpoints.save(model, train_env, summary, manifest["config_hash"],
                                     manifest.get("index_manifest_sha256"))
                model.logger.record("time/total_timesteps", model.num_timesteps)
                # learn(log_interval=None) suppresses SB3's rollout logs as well as automatic dumps.
                # Persist raw training outcomes so sparse-reward failure is visible separately from eval.
                if model.ep_info_buffer:
                    episodes = list(model.ep_info_buffer)
                    model.logger.record("rollout/ep_rew_mean", sum(e["r"] for e in episodes) / len(episodes))
                    model.logger.record("rollout/ep_len_mean", sum(e["l"] for e in episodes) / len(episodes))
                if model.ep_success_buffer:
                    model.logger.record("rollout/success_rate", sum(model.ep_success_buffer) / len(model.ep_success_buffer))
                model.logger.dump(step=model.num_timesteps)
        # Plot only after all updates, evaluations, and checkpoints have finished.
        from .plotting import plot_runs
        plots = plot_runs([output])
        completion = dict(status="completed", requested_train_env_steps=cfg["total_timesteps"],
                          actual_train_env_steps=model.num_timesteps, config_hash=manifest["config_hash"],
                          plots=plots)
        (Path(output) / "completion.json").write_text(json.dumps(completion, indent=2))
        return model.num_timesteps
    finally:
        train_env.close()
        eval_env.close()
