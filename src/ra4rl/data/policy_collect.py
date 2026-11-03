"""Collect soft policy targets from a frozen project checkpoint, without an API."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from ra4rl.checkpoints import restore
from ra4rl.config import config_hash
from ra4rl.envs.minigrid import LOCAL_RGB, local_png, make_env, snapshot
from ra4rl.evaluation import summarize
from .collect import replay, save_image
from .store import Store, probabilities, sha256
from .translator import TRANSLATOR_VERSION, state_text


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class CheckpointPolicy:
    def __init__(self, checkpoint, device='cuda', config_path=None):
        self.checkpoint = Path(checkpoint).resolve()
        config_path = Path(config_path) if config_path else self.checkpoint.parent.parent / 'config.json'
        self.training_config = json.loads(config_path.read_text())
        metadata = json.loads((self.checkpoint / 'metadata.json').read_text())
        if metadata['config_hash'] != config_hash(self.training_config):
            raise ValueError('Checkpoint and training configuration hash mismatch')
        if metadata['algorithm'] not in {'ppo', 'ra4rl'}:
            raise ValueError('Unsupported checkpoint algorithm')
        self.source = dict(checkpoint=str(self.checkpoint), algorithm=metadata['algorithm'],
                           train_env_steps=metadata['train_env_steps'],
                           training_config_hash=metadata['config_hash'],
                           files={name: sha256(self.checkpoint / name) for name in
                                  ('model.zip', 'vecnormalize.pkl', 'metadata.json')})
        cfg = self.training_config
        raw = DummyVecEnv([lambda: make_env(cfg['env_id'], cfg['max_episode_steps'])])
        try:
            self.model, self.normalizer, _ = restore(self.checkpoint, raw, device=device, training=False)
            self.model.policy.set_training_mode(False)
            if (self.model.action_space.n != 7 or self.normalizer.training
                    or self.normalizer.norm_reward or not self.normalizer.norm_obs):
                raise ValueError('Expected seven actions and frozen observation-only normalization')
        except BaseException:
            raw.close()
            raise
        self._rms = copy.deepcopy(self.normalizer.obs_rms)

    def predict_proba(self, observations):
        normalized = self.normalizer.normalize_obs(np.asarray(observations))
        tensor, _ = self.model.policy.obs_to_tensor(normalized)
        with torch.inference_mode():
            distribution = self.model.policy.get_distribution(tensor).distribution
            result = distribution.probs.cpu().numpy().astype(np.float64)
        if result.shape != (len(observations), 7):
            raise ValueError('Expected a batch of seven-action policy distributions')
        return np.stack([probabilities(row) for row in result])

    def verify_source(self):
        if any(sha256(self.checkpoint / name) != digest for name, digest in self.source['files'].items()):
            raise ValueError('Source checkpoint changed during collection')
        current = self.normalizer.obs_rms
        if (current.count != self._rms.count or not np.array_equal(current.mean, self._rms.mean)
                or not np.array_equal(current.var, self._rms.var)):
            raise ValueError('Observation normalization statistics changed during inference')

    def close(self):
        self.normalizer.close()


def collect_policy(cfg, path, policy, seeds=None, batch_size=10):
    """Each batch holds at most batch_size live episodes; finished slots stop."""
    seeds = list(range(*cfg['collection_seeds'])) if seeds is None else list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or batch_size < 1:
        raise ValueError('Require unique episode seeds and a positive batch size')
    if any(cfg[key] != policy.training_config[key] for key in ('env_id', 'max_episode_steps')):
        raise ValueError('Collection environment or horizon differs from source checkpoint')
    path = Path(path)
    schedule = [seeds[start:start + batch_size] for start in range(0, len(seeds), batch_size)]
    protocol = dict(source='trained_policy', seeds=seeds, max_episode_steps=cfg['max_episode_steps'],
                    batch_size=batch_size, batches=schedule, policy_source=policy.source,
                    action_selection='argmax_lowest_index_tie', action_targets='policy_softmax',
                    rgb_input=LOCAL_RGB, translator_version=TRANSLATOR_VERSION,
                    observation='FlatObsWrapper(maxStrLen=96) + checkpoint frozen VecNormalize')
    store = Store(path, cfg['env_id'], create=True, provenance='trained_policy')
    images = path.parent / 'images'
    results, batch_results = [], []
    transitions = 0
    started = time.monotonic()

    def progress(phase, batch, error=None):
        write_json(str(path) + '.progress.json', dict(
            phase=phase, source='trained_policy', expected_episodes=len(seeds),
            completed_episodes=len(results), saved_transitions=transitions,
            batch_idx=batch, batch_count=len(schedule), batch_size=batch_size,
            elapsed_seconds=time.monotonic() - started,
            updated_at=datetime.now(timezone.utc).isoformat(), error=error, api_requests=0))

    try:
        images.mkdir(exist_ok=True)
        store.db.execute('INSERT INTO meta VALUES (?,?)', ('collection_protocol', json.dumps(protocol, sort_keys=True)))
        store.db.commit()
        with Path(str(path) + '.steps.jsonl').open('x') as trace:
            for batch_idx, batch in enumerate(schedule):
                progress('collecting', batch_idx)
                envs, observations = [], []
                lengths, totals = [0] * len(batch), [0.0] * len(batch)
                previous = [None] * len(batch)
                active = list(range(len(batch)))
                batch_start = len(results)
                try:
                    for seed in batch:
                        env = make_env(cfg['env_id'], cfg['max_episode_steps'])
                        envs.append(env)
                        obs, _ = env.reset(seed=seed)
                        observations.append(obs)
                    while active:
                        targets = policy.predict_proba(np.stack([observations[i] for i in active]))
                        if targets.shape != (len(active), 7):
                            raise ValueError('Policy batch shape mismatch')
                        # Validate the entire batch before stepping any environment.
                        targets = np.stack([probabilities(row) for row in targets])
                        next_active = []
                        for i, probs in zip(active, targets):
                            env, seed, step = envs[i], batch[i], lengths[i]
                            state = snapshot(env, env.unwrapped.gen_obs(), previous_action=previous[i])
                            png = local_png(env)
                            image = save_image(images, seed, step, png)
                            position = list(map(int, env.unwrapped.agent_pos))
                            action = int(np.argmax(probs))
                            observations[i], reward, terminated, truncated, _ = env.step(action)
                            store.add(seed, step, state, png, probs, action, reward, state_text(state),
                                      terminated, truncated)
                            trace.write(json.dumps(dict(
                                seed=seed, step_idx=step, batch_idx=batch_idx, image=f'images/{image.name}',
                                action_id=action, action_probs=probs.tolist(), reward=float(reward),
                                terminated=bool(terminated), truncated=bool(truncated),
                                direction=state.direction, carrying_type=state.carrying_type,
                                carrying_color=state.carrying_color, door_open=state.door_open,
                                previous_action=state.previous_action, agent_position=position,
                                next_agent_position=list(map(int, env.unwrapped.agent_pos)),
                                agent_position_frame='native_world_top_left_x_right_y_down',
                                rgb_input=LOCAL_RGB, source='trained_policy')) + '\n')
                            trace.flush()
                            transitions += 1
                            lengths[i] += 1
                            totals[i] += float(reward)
                            previous[i] = action
                            if terminated or truncated:
                                save_image(images, seed, lengths[i], local_png(env))
                                success = bool(terminated and reward > 0)
                                store.episode(seed, success, totals[i], lengths[i])
                                results.append(dict(seed=seed, success=success,
                                                    return_value=totals[i], length=lengths[i]))
                            elif lengths[i] >= cfg['max_episode_steps']:
                                raise RuntimeError('Environment did not enforce episode limit')
                            else:
                                next_active.append(i)
                        active = next_active
                    batch_results.append(dict(batch_idx=batch_idx, seeds=batch,
                                              completed_episodes=len(results) - batch_start))
                    progress('batch_completed', batch_idx)
                finally:
                    for env in envs:
                        env.close()
        policy.verify_source()
        store.finish()
        summary = dict(provenance='trained_policy', env_id=cfg['env_id'], episode_count=len(results),
                       episodes=sorted(results, key=lambda r: r['seed']), **summarize(results),
                       experience_count=transitions, batches=batch_results, rgb_input=LOCAL_RGB,
                       translator_version=TRANSLATOR_VERSION, policy_source=policy.source, api_requests=0,
                       screenshots=dict(directory='images', naming='seed_{seed:03d}_step_{step:06d}.png',
                           step_semantics='zero-based pre-action local observation; step=length is terminal frame'))
        write_json(str(path) + '.summary.json', summary)
        progress('collected', len(schedule) - 1)
        return summary
    except BaseException as exc:
        progress('incomplete', len(batch_results), f'{type(exc).__name__}: {exc}')
        raise
    finally:
        store.close()


def validate_collection(path, cfg):
    """Replay every episode and check every screenshot without policy inference."""
    store = Store(path, cfg['env_id'])
    try:
        if store.meta['complete'] != 'true':
            raise ValueError('Cannot validate incomplete collection')
        protocol = json.loads(store.meta['collection_protocol'])
        rows = store.rows()
        episodes = [dict(row) for row in store.db.execute('SELECT * FROM episodes ORDER BY episode_id')]
    finally:
        store.close()
    if [row['episode_id'] for row in episodes] != sorted(protocol['seeds']):
        raise ValueError('Episode seeds do not match protocol')
    records = [json.loads(line) for line in Path(str(path) + '.steps.jsonl').read_text().splitlines()]
    traces = {(r['seed'], r['step_idx']): r for r in records}
    if len(records) != len(rows) or len(traces) != len(rows):
        raise ValueError('Trace count differs from SQLite')
    batch_for_seed = {seed: i for i, batch in enumerate(protocol['batches']) for seed in batch}
    for episode in episodes:
        seed = episode['episode_id']
        trajectory = [row for row in rows if row['episode_id'] == seed]
        env = make_env(cfg['env_id'], cfg['max_episode_steps'], flat=False)
        try:
            obs, _ = env.reset(seed=seed)
            _, total, terminal = replay(env, obs, trajectory, LOCAL_RGB, protocol['translator_version'])
            if not terminal or total != episode['return_value'] or len(trajectory) != episode['length']:
                raise ValueError('Saved episode summary differs from deterministic replay')
            last = trajectory[-1]
            if bool(last['terminated'] and last['reward'] > 0) != bool(episode['success']):
                raise ValueError('Saved success differs from replay')
            previous = None
            for row in trajectory:
                record = traces.get((seed, row['step_idx']), {})
                if (record.get('previous_action') != previous
                        or record.get('batch_idx') != batch_for_seed[seed]
                        or record.get('action_id') != row['action_id']
                        or record.get('action_probs') != json.loads(row['action_probs_json'])
                        or record.get('rgb_input') != LOCAL_RGB
                        or record.get('source') != 'trained_policy'):
                    raise ValueError('Trace input or action differs from SQLite')
                previous = row['action_id']
                image = Path(path).parent / 'images' / f"seed_{seed:03d}_step_{row['step_idx']:06d}.png"
                if image.read_bytes() != row['rgb_png']:
                    raise ValueError('Screenshot differs from SQLite')
            terminal_image = Path(path).parent / 'images' / f"seed_{seed:03d}_step_{episode['length']:06d}.png"
            if terminal_image.read_bytes() != local_png(env):
                raise ValueError('Terminal screenshot mismatch')
        finally:
            env.close()
    return dict(status='passed', replay_validation='passed', episode_count=len(episodes),
                experience_count=len(rows), screenshot_count=len(rows) + len(episodes),
                screenshots_match_database=True, traces_match_database=True, api_requests=0)
