import csv
import json

import pytest
from PIL import Image

from ra4rl.plotting import load_curve, plot_runs
from ra4rl.evaluation_log import EvaluationLog


def fixture_run(path, horizon=360):
    path.mkdir()
    cfg = dict(env_id="MiniGrid-DoorKey-6x6-v0", max_episode_steps=horizon,
               eval_seeds=[100, 200], train_seed=42)
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "manifest.json").write_text(json.dumps(dict(algorithm="ppo")))
    rows = []
    for step, success in ((1024, 0.), (2048, .5), (3072, 1.)):
        rows.append(dict(train_env_steps=step, success_rate_mean=success, success_rate_std=.1,
                         return_mean=success * .95, return_std=.05,
                         episode_steps_mean=360 - success * 345, episode_steps_std=10.,
                         episodes=[dict(seed=100 + i) for i in range(100)]))
    (path / "evaluations.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return rows


def test_three_panel_exports_preserve_exact_recorded_values(tmp_path):
    run = tmp_path / "run"
    rows = fixture_run(run)
    paths = plot_runs([run])
    assert len(paths) == 5
    with Image.open(run / "plots" / "training_curves.png") as image:
        assert image.width > image.height * 3
    assert (run / "plots" / "training_curves.pdf").read_bytes().startswith(b"%PDF")
    assert "<svg" in (run / "plots" / "training_curves.svg").read_text()
    with (run / "plots" / "curve_data.csv").open() as file:
        exported = list(csv.DictReader(file))
    assert [int(row["train_env_steps"]) for row in exported] == [1024, 2048, 3072]
    assert [float(row["return_mean"]) for row in exported] == [row["return_mean"] for row in rows]
    metadata = json.loads((run / "plots" / "plot_metadata.json").read_text())
    assert metadata["smoothing"] == "none" and metadata["synthetic_points"] is False
    assert "not a CI" in metadata["band"]


def test_refuse_incomparable_horizons_and_nonmonotonic_steps(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    rows = fixture_run(a)
    fixture_run(b, horizon=200)
    with pytest.raises(ValueError, match="horizons"):
        plot_runs([a, b], tmp_path / "comparison")
    rows[-1]["train_env_steps"] = 1024
    (a / "evaluations.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="strictly increasing"):
        load_curve(a)


def test_plotting_reads_new_compact_summary_format(tmp_path):
    run = tmp_path / "new_format"
    rows = fixture_run(run)
    (run / "evaluations.jsonl").unlink()
    with EvaluationLog(run) as log:
        for row in rows:
            log.write(row["train_env_steps"], row, 0.)
    plot_runs([run])
    with (run / "plots" / "curve_data.csv").open() as file:
        exported = list(csv.DictReader(file))
    assert all(int(row["evaluation_episodes"]) == 100 for row in exported)
