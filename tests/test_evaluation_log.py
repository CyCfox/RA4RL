import json

import pytest

from ra4rl.evaluation import summarize
from ra4rl.evaluation_log import EvaluationLog, load_episodes


def test_summary_first_key_and_one_shared_detail_file(tmp_path):
    expected = {}
    with EvaluationLog(tmp_path) as log:
        for step, length in ((1024, 20), (2048, 30)):
            episodes = [dict(seed=100, success=False, return_value=0., length=360),
                        dict(seed=101, success=True, return_value=.95, length=length)]
            expected[step] = episodes
            result = dict(**summarize(episodes), episodes=episodes)
            summary = log.write(step, result, 0.)
            assert 'episodes' not in summary and 'episodes_file' not in summary
            # Both files must be visible before the writer is closed.
            assert load_episodes(tmp_path, summary) == episodes
    summaries = (tmp_path / 'evaluations.jsonl').read_text().splitlines()
    assert len(summaries) == 2 and all(line.startswith('{"train_env_steps":') for line in summaries)
    details = [json.loads(line) for line in (tmp_path / 'evaluation_details.jsonl').read_text().splitlines()]
    assert [row['train_env_steps'] for row in details] == [1024, 2048]
    assert not (tmp_path / 'evaluation_details').exists()
    for line in summaries:
        record = json.loads(line)
        assert load_episodes(tmp_path, record) == expected[record['train_env_steps']]


def test_read_legacy_inline_and_per_evaluation_files(tmp_path):
    episodes = [dict(seed=100, success=True, return_value=.95, length=20)]
    assert load_episodes(tmp_path, dict(episodes=episodes)) == episodes
    path = tmp_path / 'evaluation_details/step_000001024.json'
    path.parent.mkdir()
    details = dict(train_env_steps=1024, episodes=episodes)
    path.write_text(json.dumps(details))
    summary = dict(train_env_steps=1024, evaluation_episodes=1,
                   episodes_file='evaluation_details/step_000001024.json')
    assert load_episodes(tmp_path, summary) == episodes
    details['train_env_steps'] = 0
    path.write_text(json.dumps(details))
    with pytest.raises(ValueError, match='different training step'):
        load_episodes(tmp_path, summary)


@pytest.mark.parametrize('problem,message', [
    ('missing', 'No evaluation details'), ('duplicate', 'Duplicate evaluation details'),
    ('count', 'episode count differs'),
])
def test_shared_details_reject_missing_duplicate_and_mismatched_records(tmp_path, problem, message):
    details = dict(train_env_steps=1024, episodes=[dict(seed=100)])
    rows = [] if problem == 'missing' else [details] * (2 if problem == 'duplicate' else 1)
    (tmp_path / 'evaluation_details.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    summary = dict(train_env_steps=1024, evaluation_episodes=2 if problem == 'count' else 1)
    with pytest.raises(ValueError, match=message):
        load_episodes(tmp_path, summary)
