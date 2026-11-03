import asyncio
import json

from ra4rl.config import load_config
from ra4rl.data.collect import collect
from ra4rl.data.review import review_collection
from ra4rl.smoke import SyntheticClient


def test_task_review_replays_episodes_and_links_exact_local_frames(tmp_path):
    cfg = load_config(smoke=True)
    cfg["max_episode_steps"] = 3
    database = tmp_path / "facts.sqlite"
    asyncio.run(collect(cfg, database, SyntheticClient(), [0, 1], synthetic=True))
    report = review_collection(database, tmp_path)
    assert report["episode_count"] == 2 and report["success_count"] == 0
    assert report["replay_validation"] == "passed" and report["screenshot_count"] == 8
    assert all(r["truncated"] and not r["terminated"] for r in report["episodes"])
    assert len(list((tmp_path / "images").glob("*.png"))) == 8
    assert (tmp_path / "seed_000_overview.png").is_file()
    assert "images/seed_001_step_000003.png" in (tmp_path / "replay.html").read_text()
    assert json.loads((tmp_path / "task_review.json").read_text()) == report
