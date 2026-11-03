import csv
import json
from pathlib import Path

from .evaluation_log import episode_count


def analyze(runs, output):
    """Export comparable final observations; a single training seed supports no significance claim."""
    rows = []
    for run in runs:
        run = Path(run)
        cfg = json.loads((run / "config.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        evaluations = [json.loads(line) for line in (run / "evaluations.jsonl").read_text().splitlines()]
        if not evaluations:
            raise ValueError(f"No completed evaluations in {run}")
        last = evaluations[-1]
        rows.append(dict(run=str(run), profile=cfg["profile"], env_id=cfg["env_id"],
                         algorithm=manifest["algorithm"], train_seed=cfg["train_seed"],
                         evaluation_episodes=episode_count(last),
                         **{k: v for k, v in last.items() if k not in {
                             "episodes", "episodes_file", "evaluation_episodes"}}))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
