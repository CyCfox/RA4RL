import numpy as np
from stable_baselines3.common.vec_env import sync_envs_normalization


def summarize(episodes):
    result = {}
    successes = [e["length"] for e in episodes if e["success"]]
    for name, values in (
        ("success_rate", [e["success"] for e in episodes]),
        ("return", [e["return_value"] for e in episodes]),
        ("episode_steps", [e["length"] for e in episodes]),
        ("successful_steps", successes),
    ):
        result[name + "_mean"] = float(np.mean(values)) if values else float("nan")
        result[name + "_std"] = float(np.std(values, ddof=0)) if values else float("nan")
    result["success_count"] = len(successes)
    return result


def evaluate(model, train_env, eval_env, seeds):
    """Exactly one episode per fixed seed; auto-reset surplus transitions are ignored."""
    sync_envs_normalization(train_env, eval_env)
    eval_env.training, eval_env.norm_reward = False, False
    episodes = []
    seeds = list(seeds)
    count = eval_env.num_envs
    for start in range(0, len(seeds), count):
        batch = seeds[start:start + count]
        # SB3 reset seed queue is per slot. A base+slot seed would change the fixed scene assignment.
        for i in range(count):
            eval_env.venv._seeds[i] = batch[i] if i < len(batch) else batch[-1]
        obs = eval_env.reset()
        totals, lengths = np.zeros(count), np.zeros(count, dtype=int)
        active = np.arange(count) < len(batch)
        while active.any():
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(actions)
            totals[active] += rewards[active]
            lengths[active] += 1
            for i in np.flatnonzero(active & dones):
                episodes.append(dict(seed=batch[i], success=bool(infos[i]["is_success"]),
                                     return_value=float(totals[i]), length=int(lengths[i])))
                active[i] = False
    return {**summarize(episodes), "episodes": sorted(episodes, key=lambda x: x["seed"])}
