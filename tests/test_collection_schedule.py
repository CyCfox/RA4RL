import runpy

import pytest

from ra4rl.config import load_config


validate_build_schedule = runpy.run_path('scripts/collect_experience.py')['validate_build_schedule']


@pytest.mark.parametrize('bounds', [[0, 100], [0, 500], [500, 1000]])
def test_formal_collection_accepts_explicit_parallel_seed_count(bounds):
    cfg = load_config('configs/putnear_llm_thinking_low.yaml')
    cfg['collection_seeds'] = bounds
    cfg['llm']['concurrency'] = bounds[1] - bounds[0]
    validate_build_schedule(cfg)


@pytest.mark.parametrize('bounds,concurrency', [
    ([0, 500], 100), ([0, 100], 500), ([0, 500], True),
    ([0, 500], 500.0), ([0, 0], 0), ([-1, 499], 500),
    ([0, True], 1), ([0, 500.0], 500), ([0], 500), (None, 500),
])
def test_formal_collection_rejects_ambiguous_or_inconsistent_schedule(bounds, concurrency):
    cfg = load_config('configs/putnear_llm_thinking_low.yaml')
    cfg['collection_seeds'] = bounds
    cfg['llm']['concurrency'] = concurrency
    with pytest.raises(ValueError):
        validate_build_schedule(cfg)
