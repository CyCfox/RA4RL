"""Collect and index a frozen checkpoint's experience. Run inside Docker."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from ra4rl.config import config_hash
from ra4rl.data.policy_collect import CheckpointPolicy, collect_policy, validate_collection, write_json
from ra4rl.data.store import sha256
from ra4rl.envs.minigrid import LOCAL_RGB, make_env, snapshot
from ra4rl.data.translator import TRANSLATOR_VERSION


def main():
    if not Path('/.dockerenv').exists():
        raise RuntimeError('Run experimental Python inside Docker')
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--training-config', type=Path,
                        help='Original training config.json; default checkpoint/../../config.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true', help='Three episodes, batches of two, then index/query')
    parser.add_argument('--smoke-episodes', type=int,
                        help='Override smoke episode count for short tasks needing at least Top-K transitions')
    args = parser.parse_args()
    if args.smoke_episodes is not None and (not args.smoke or args.smoke_episodes < 1):
        parser.error('--smoke-episodes requires --smoke and a positive episode count')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable')
    policy = CheckpointPolicy(args.checkpoint, args.device, args.training_config)
    cfg = copy.deepcopy(policy.training_config)
    cfg.update(experience_source='trained_policy', policy_source=policy.source,
               collection_seeds=[0, args.smoke_episodes or 3] if args.smoke else [0, 100],
               collection_batch_size=2 if args.smoke else 10, profile='smoke' if args.smoke else 'formal')
    manifest = dict(source='trained_policy', status='collecting', env_id=cfg['env_id'],
                    mode='smoke' if args.smoke else 'build', config_hash=config_hash(cfg),
                    seeds=list(range(*cfg['collection_seeds'])),
                    max_episode_steps=cfg['max_episode_steps'], batch_size=cfg['collection_batch_size'],
                    rgb_input=LOCAL_RGB, translator_version=TRANSLATOR_VERSION,
                    policy_source=policy.source, runtime_image_id=os.environ.get('RA4RL_IMAGE_ID'),
                    source_sha256={str(p): sha256(p) for p in Path('src/ra4rl').rglob('*.py')},
                    entrypoint_sha256=sha256(__file__), api_requests=0)
    created = False
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(args.output / 'config.json', cfg)
        write_json(args.output / 'collection_manifest.json', manifest)
        database = args.output / 'experiences.sqlite'
        summary = collect_policy(cfg, database, policy, batch_size=cfg['collection_batch_size'])
        validation = validate_collection(database, cfg)
        write_json(args.output / 'validation.json', validation)
        manifest['status'] = 'indexing'
        write_json(args.output / 'collection_manifest.json', manifest)
        from ra4rl.retrieval.index import build_index
        from ra4rl.retrieval.teacher import RetrievalTeacher
        index = build_index(database, args.output / 'index', cfg, device=args.device)
        teacher = RetrievalTeacher(database, args.output / 'index', cfg, device=args.device)
        env = make_env(cfg['env_id'], cfg['max_episode_steps'], flat=False)
        try:
            obs, _ = env.reset(seed=0)
            result = teacher([snapshot(env, obs)])
            assert result.shape == (1, 7) and np.isfinite(result).all() and np.allclose(result.sum(axis=1), 1)
        finally:
            env.close()
        policy.verify_source()
        completion = dict(status='completed', source='trained_policy', episode_count=summary['episode_count'],
                          experience_count=index['count'], success_count=summary['success_count'],
                          success_rate_mean=summary['success_rate_mean'], return_mean=summary['return_mean'],
                          episode_steps_mean=summary['episode_steps_mean'], batch_count=len(summary['batches']),
                          batch_size=cfg['collection_batch_size'], sqlite_sha256=index['sqlite_sha256'],
                          config_hash=config_hash(cfg), index_manifest_sha256=sha256(args.output / 'index/manifest.json'),
                          validation=validation, retrieval_query_validation='passed',
                          checkpoint_unchanged=True, normalization_frozen=True, api_requests=0)
        manifest['status'] = 'completed'
        write_json(args.output / 'collection_manifest.json', manifest)
        write_json(args.output / 'completion.json', completion)
        print(json.dumps(completion, indent=2))
    except BaseException as exc:
        if created:
            manifest.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
            write_json(args.output / 'collection_manifest.json', manifest)
        raise
    finally:
        policy.close()


if __name__ == '__main__':
    main()
