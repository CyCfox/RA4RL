import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import time
from uuid import uuid4

import gymnasium as gym
import numpy as np

from ra4rl.envs.minigrid import LOCAL_RGB, llm_png, rgb_input_protocol, snapshot
from .llm import LLMClient, EPISODE_SESSION
from .prompts import custom_prompt
from .reasoning import reasoning_options
from .store import Store
from .translator import TRANSLATOR_VERSION, state_text


def replay(env, obs, rows, rgb_protocol=None, translator_version=TRANSLATOR_VERSION):
    """Recreate a saved prefix and verify exact observations and transitions."""
    total = 0.0
    terminal = False
    for step, row in enumerate(rows):
        state = snapshot(env, obs)
        if terminal or row["step_idx"] != step or any((
            state.image.astype(np.uint8).tobytes() != row["obs_img"],
            llm_png(env, LOCAL_RGB if rgb_protocol is None else rgb_protocol) != row["rgb_png"],
            state.mission != row["mission"],
            state.direction != row["direction"], state.carrying_type != row["carrying_type"],
            state.carrying_color != row["carrying_color"],
            state_text(state, version=translator_version) != row["env_description"],
            int(np.argmax(json.loads(row["action_probs_json"]))) != row["action_id"],
        )):
            raise ValueError("Saved episode prefix does not match deterministic replay")
        obs, reward, terminated, truncated, _ = env.step(row["action_id"])
        if (float(reward), bool(terminated), bool(truncated)) != (
                row["reward"], bool(row["terminated"]), bool(row["truncated"])):
            raise ValueError("Saved transition does not match deterministic replay")
        total += float(reward)
        terminal = bool(terminated or truncated)
    return obs, total, terminal


def save_image(directory, seed, step, png):
    path = directory / f"seed_{seed:03d}_step_{step:06d}.png"
    if path.exists():
        if path.read_bytes() != png:
            raise ValueError(f"Existing screenshot differs: {path.name}")
    else:
        with path.open("xb") as file:
            file.write(png)
    return path


async def collect(cfg, path, client=None, seeds=None, synthetic=False, resume=False):
    if seeds is None:
        seeds = range(*cfg["collection_seeds"])
    seeds = list(seeds)
    own_client = client is None
    client = client or LLMClient(cfg["llm"])
    store = Store(path, cfg["env_id"], create=not resume, resume=resume,
                  provenance="synthetic" if synthetic else "llm")
    images = Path(path).parent / "images"
    images.mkdir(exist_ok=True)
    trace_path = Path(str(path) + ".steps.jsonl")
    collection_id = uuid4().hex
    rgb_protocol = rgb_input_protocol(cfg["env_id"])
    reasoning = reasoning_options(cfg['llm'])
    protocol = {"seeds": seeds, "max_episode_steps": cfg["max_episode_steps"], "llm": cfg["llm"],
                **reasoning, "collection_id": collection_id,
                "rgb_input": rgb_protocol, "translator_version": TRANSLATOR_VERSION}
    custom = custom_prompt(cfg["env_id"])
    if custom:
        protocol["prompt"] = custom[1]
    saved = {seed: [] for seed in seeds}
    results = []
    if resume:
        previous = json.loads(store.meta["collection_protocol"])
        if any(previous.get(key) != protocol[key] for key in protocol if key != "collection_id"):
            store.close()
            if own_client:
                await client.close()
            raise ValueError("Resume collection protocol mismatch")
        collection_id = previous["collection_id"]
        for row in store.rows():
            if row["episode_id"] not in saved:
                raise ValueError("Unexpected seed in resume database")
            saved[row["episode_id"]].append(row)
        for row in store.db.execute("SELECT * FROM episodes"):
            results.append(dict(seed=row["episode_id"], success=bool(row["success"]),
                                return_value=row["return_value"], length=row["length"]))
    else:
        store.db.execute("INSERT INTO meta VALUES (?,?)", ("collection_protocol", json.dumps(protocol, sort_keys=True)))
        store.db.commit()
    transitions = sum(map(len, saved.values()))
    if resume:
        # Validate every prefix before any episode can send a new API request.
        # The episode coroutine repeats replay to reconstruct its live environment.
        try:
            if len({r["seed"] for r in results}) != len(results) or any(r["seed"] not in saved for r in results):
                raise ValueError("Unexpected episode summary in resume database")
            for seed in seeds:
                env = gym.make(cfg["env_id"], max_steps=cfg["max_episode_steps"])
                try:
                    obs, _ = env.reset(seed=seed)
                    _, total, terminal = replay(env, obs, saved[seed], rgb_protocol)
                    existing = next((r for r in results if r["seed"] == seed), None)
                    if existing is not None:
                        if not terminal:
                            raise ValueError("Episode summary exists for an unfinished prefix")
                        row = saved[seed][-1]
                        expected = dict(seed=seed, success=bool(row["terminated"] and row["reward"] > 0),
                                        return_value=total, length=len(saved[seed]))
                        if existing != expected:
                            raise ValueError("Saved episode summary does not match replay")
                finally:
                    env.close()
            Path(str(path) + ".resume_validation.json").write_text(json.dumps(dict(
                status="passed", collection_id=collection_id, verified_seeds=seeds,
                verified_transitions=transitions, saved_completed_episodes=len(results),
                api_requests_during_replay=0), indent=2))
            # Materialize old screenshot bytes too, without new API calls.
            for rows in saved.values():
                for row in rows:
                    save_image(images, row["episode_id"], row["step_idx"], row["rgb_png"])
        except BaseException:
            store.close()
            if own_client:
                await client.close()
            raise
    started = time.monotonic()
    next_progress = started
    progress_path = Path(str(path) + ".progress.json")

    def progress(phase, error=None):
        nonlocal next_progress
        next_progress = time.monotonic() + 10
        record = dict(phase=phase, expected_episodes=len(seeds), completed_episodes=len(results),
                      saved_transitions=transitions, elapsed_seconds=time.monotonic() - started,
                      **reasoning,
                      updated_at=datetime.now(timezone.utc).isoformat(),
                      api=client.progress_stats() if hasattr(client, "progress_stats") else getattr(client, "stats", {}))
        if error:
            record["error"] = error
        temporary = Path(str(progress_path) + ".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(progress_path)

    async def episode(seed):
        nonlocal transitions
        session_token = EPISODE_SESSION.set(f"ses_ra4rl_{collection_id}_ep{seed:03d}")
        env = gym.make(cfg["env_id"], max_steps=cfg["max_episode_steps"])
        try:
            obs, _ = env.reset(seed=seed)
            obs, total, terminal = replay(env, obs, saved[seed], rgb_protocol)
            existing = next((r for r in results if r["seed"] == seed), None)
            if terminal:
                save_image(images, seed, len(saved[seed]), llm_png(env, rgb_protocol))
                row = saved[seed][-1]
                result = dict(seed=seed, success=bool(row["terminated"] and row["reward"] > 0),
                              return_value=total, length=len(saved[seed]))
                if existing is not None and existing != result:
                    raise ValueError("Saved episode summary does not match replay")
                if existing is None:
                    store.episode(seed, result["success"], total, result["length"])
                    results.append(result)
                return
            if existing is not None:
                raise ValueError("Episode summary exists for an unfinished prefix")
            previous_action = saved[seed][-1]["action_id"] if saved[seed] else None
            for step in range(len(saved[seed]), cfg["max_episode_steps"]):
                state = snapshot(env, obs, previous_action=previous_action)
                png = llm_png(env, rgb_protocol)
                image = save_image(images, seed, step, png)
                before_position = [int(x) for x in env.unwrapped.agent_pos]
                probs = await client.predict(cfg["env_id"], state, png)
                action = int(np.argmax(probs))
                obs, reward, terminated, truncated, _ = env.step(action)
                previous_action = action
                store.add(seed, step, state, png, probs, action, reward, state_text(state), terminated, truncated)
                record = dict(seed=seed, step_idx=step, image=f"images/{image.name}",
                              action_id=action, action_probs=probs.tolist(), reward=float(reward),
                              terminated=bool(terminated), truncated=bool(truncated),
                              agent_position=before_position,
                              next_agent_position=[int(x) for x in env.unwrapped.agent_pos],
                              agent_position_frame="native_world_top_left_x_right_y_down",
                              direction=state.direction, carrying_type=state.carrying_type,
                              carrying_color=state.carrying_color, door_open=state.door_open,
                              previous_action=state.previous_action,
                              rgb_input=rgb_protocol,
                              rotation_ccw_quarter_turns=(state.direction + 1) % 4
                              if rgb_protocol["view"] == "global_agent_up" else
                              0 if rgb_protocol["view"] == "global_native" else None)
                with trace_path.open("a") as file:
                    file.write(json.dumps(record) + "\n")
                transitions += 1
                if time.monotonic() >= next_progress:
                    progress("collecting")
                total += float(reward)
                if terminated or truncated:
                    save_image(images, seed, step + 1, llm_png(env, rgb_protocol))
                    success = bool(terminated and reward > 0)
                    store.episode(seed, success, total, step + 1)
                    results.append(dict(seed=seed, success=success, return_value=total, length=step + 1))
                    progress("collecting")
                    return
            raise RuntimeError("Environment did not enforce episode limit")
        finally:
            env.close()
            EPISODE_SESSION.reset(session_token)

    async def heartbeat():
        while True:
            await asyncio.sleep(10)
            progress("collecting")

    tasks = [asyncio.create_task(episode(seed)) for seed in seeds]
    heartbeat_task = asyncio.create_task(heartbeat())
    progress("collecting")
    try:
        await asyncio.gather(*tasks)
        store.finish()
        summary = {
            "provenance": "synthetic" if synthetic else "llm", "env_id": cfg["env_id"],
            "episode_count": len(results), "episodes": sorted(results, key=lambda x: x["seed"]),
            "success_rate_mean": float(np.mean([r["success"] for r in results])),
            "return_mean": float(np.mean([r["return_value"] for r in results])),
            "episode_steps_mean": float(np.mean([r["length"] for r in results])),
            "api": getattr(client, "stats", {}), **reasoning,
            "rgb_input": rgb_protocol,
            "screenshots": dict(directory="images", naming="seed_{seed:03d}_step_{step:06d}.png",
                                step_semantics="zero-based pre-action observation; step=length is terminal frame"),
        }
        Path(str(path) + ".summary.json").write_text(json.dumps(summary, indent=2))
        progress("collected")
        return summary
    except BaseException as exc:
        progress("incomplete", str(exc))
        raise
    finally:
        heartbeat_task.cancel()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(heartbeat_task, *tasks, return_exceptions=True)
        store.close()
        if own_client:
            await client.close()
