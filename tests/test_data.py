import asyncio
import json

import numpy as np
import pytest
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.world_object import Key

from ra4rl.config import ENV_IDS, load_config
from ra4rl.data.llm import parse_probs, prompt
from ra4rl.data.store import Store
from ra4rl.data.translator import (
    AGENT_RELATIVE_TRANSLATOR_VERSION, FRONT_WALL_TRANSLATOR_VERSION,
    LEGACY_TRANSLATOR_VERSION, translator, tokenize,
)
from ra4rl.envs.minigrid import make_env, snapshot, local_png


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_native_observation_coordinates_and_flatten(env_id):
    env = make_env(env_id)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (2835,)
    base = env.unwrapped
    x, y = base.front_pos
    base.grid.set(x, y, Key("blue"))
    raw_obs = base.gen_obs()
    assert raw_obs["image"][3, 5].tolist() == [OBJECT_TO_IDX["key"], COLOR_TO_IDX["blue"], 0]
    state = snapshot(env, raw_obs)
    assert "a blue key at agent's 1 cells forward" in translator(raw_obs)
    assert local_png(env).startswith(b"\x89PNG")
    if "DoorKey" in env_id:
        assert "MiniGrid DoorKey environment" in prompt(env_id, state)
        assert translator(raw_obs) in prompt(env_id, state)
    else:
        assert state.mission in prompt(env_id, state)
    env.close()


def test_translator_visibility_stability_and_state():
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[3, 5] = [OBJECT_TO_IDX["door"], COLOR_TO_IDX["yellow"], 2]
    image[2, 5] = [OBJECT_TO_IDX["key"], COLOR_TO_IDX["blue"], 0]
    image[3, 6] = [OBJECT_TO_IDX["wall"], 0, 0]
    text = translator(image, "open the door", "key", "yellow")
    assert text == translator(image.copy(), "open the door", "key", "yellow")
    assert text.startswith("Mission: open the door")
    assert "carrying yellow key" in text
    assert "locked yellow door at agent's 1 cells forward" in text
    assert "1 cells left, 1 cells forward" in text
    assert "unseen" not in text and "wall" not in text
    assert tokenize("Door-20 LEFT") == ["door", "20", "left"]


def test_translator_describes_three_adjacent_walls_and_preserves_older_versions():
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    wall_positions = ((3, 5), (2, 6), (4, 6), (3, 4), (2, 5), (4, 5), (1, 6))
    for x, y in wall_positions:
        image[x, y] = [OBJECT_TO_IDX["wall"], COLOR_TO_IDX["grey"], 0]
    image[4, 4] = [OBJECT_TO_IDX["key"], COLOR_TO_IDX["yellow"], 0]
    image[5, 3] = [OBJECT_TO_IDX["door"], COLOR_TO_IDX["yellow"], 2]
    text = translator(image, "open the door")
    assert text.count("wall") == 3
    assert "grey wall at agent's 1 cells forward" in text
    assert "grey wall at agent's 1 cells left" in text
    assert "grey wall at agent's 1 cells right" in text
    assert "center" not in text and "0 cells forward" not in text
    assert "yellow key at agent's 1 cells right, 2 cells forward" in text
    assert "locked yellow door at agent's 2 cells right, 3 cells forward" in text
    assert text.count("at agent's ") == 5
    assert "adjacent" not in text
    old = translator(image, "open the door", version=LEGACY_TRANSLATOR_VERSION)
    assert old.count("wall") == len(wall_positions) and "adjacent cell in front of it" in old
    front_only = translator(image, "open the door", version=FRONT_WALL_TRANSLATOR_VERSION)
    assert front_only.count("wall") == 1 and "at agent's " not in front_only
    assert "grey wall at 0 cells center, 1 cells forward" in front_only
    assert "yellow key at 1 cells right, 2 cells forward" in front_only
    agent_relative = translator(image, "open the door", version=AGENT_RELATIVE_TRANSLATOR_VERSION)
    assert agent_relative.count("wall") == 3
    assert "grey wall at agent's 0 cells center, 1 cells forward" in agent_relative
    assert "grey wall at agent's 1 cells left, 0 cells forward" in agent_relative
    assert "grey wall at agent's 1 cells right, 0 cells forward" in agent_relative
    image[3, 5] = [OBJECT_TO_IDX["empty"], 0, 0]
    assert translator(image, "open the door").count("wall") == 2
    image[2, 6] = image[4, 6] = [OBJECT_TO_IDX["empty"], 0, 0]
    assert "wall" not in translator(image, "open the door")


@pytest.mark.parametrize("position,where", [
    ((3, 5), "1 cells forward"),
    ((3, 2), "4 cells forward"),
    ((1, 6), "2 cells left"),
    ((5, 6), "2 cells right"),
    ((1, 3), "2 cells left, 3 cells forward"),
    ((5, 3), "2 cells right, 3 cells forward"),
])
def test_translator_omits_only_zero_distance_axes(position, where):
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[position] = [OBJECT_TO_IDX["key"], COLOR_TO_IDX["yellow"], 0]
    text = translator(image, "pick up the key")
    assert text.endswith(f"visible: a yellow key at agent's {where}.")
    assert "center" not in text and "0 cells forward" not in text


@pytest.mark.parametrize("content", [
    'reason ```json\n{"action_probs":[-1,2,0,0,0,0,0]}\n```',
    '{bad} then {"action_probs":[0,1,0,0,0,0,0]}',
])
def test_parse_repair(content):
    assert np.array_equal(parse_probs(content), [0, 1, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("values", [[], [0] * 7, [float("nan")] * 7, [True] * 7, ["1"] * 7])
def test_parse_reject(values):
    with pytest.raises(ValueError):
        parse_probs(json.dumps({"action_probs": values}))


def test_store_isolation_and_exclusive_creation(tmp_path):
    path = tmp_path / "facts.sqlite"
    store = Store(path, ENV_IDS[0], create=True)
    store.close()
    with pytest.raises(FileExistsError):
        Store(path, ENV_IDS[0], create=True)
    with pytest.raises(ValueError, match="Cross-environment"):
        Store(path, ENV_IDS[1])


def test_formal_configuration():
    cfg = load_config()
    assert cfg["n_envs"] * cfg["ppo"]["n_steps"] == 1024
    assert cfg["ppo"]["n_epochs"] * 1024 // cfg["ppo"]["batch_size"] == 160
    assert int(np.ceil(cfg["total_timesteps"] / 1024)) * 1024 == 1000448
    assert cfg["collection_seeds"] == [0, 100] and cfg["eval_seeds"] == [100, 200]


def test_collection_roundtrip_complete_and_failure(tmp_path):
    from ra4rl.data.collect import collect
    from ra4rl.smoke import SyntheticClient
    cfg = load_config(smoke=True)
    cfg["max_episode_steps"] = 2
    path = tmp_path / "facts.sqlite"
    summary = asyncio.run(collect(cfg, path, SyntheticClient(), [0, 1], synthetic=True))
    assert summary["episode_count"] == 2 and summary["success_rate_mean"] == 0
    store = Store(path, cfg["env_id"])
    assert store.meta["complete"] == "true" and store.meta["provenance"] == "synthetic"
    rows = store.rows()
    assert len(rows) == 4
    assert len(rows[0]["obs_img"]) == 147 and rows[0]["rgb_png"].startswith(b"\x89PNG")
    assert np.argmax(json.loads(rows[0]["action_probs_json"])) == rows[0]["action_id"]
    for row in rows:
        screenshot = tmp_path / "images" / f"seed_{row['episode_id']:03d}_step_{row['step_idx']:06d}.png"
        assert screenshot.read_bytes() == row["rgb_png"]
    assert len(list((tmp_path / "images").glob("*.png"))) == 6  # 4 action inputs + 2 terminal frames
    trace = [json.loads(line) for line in (tmp_path / "facts.sqlite.steps.jsonl").read_text().splitlines()]
    assert len(trace) == 4 and trace[0]["image"].startswith("images/seed_")
    store.close()

    class BrokenClient:
        async def predict(self, *args):
            raise RuntimeError("fixture failure")

    broken = tmp_path / "broken.sqlite"
    with pytest.raises(RuntimeError, match="fixture failure"):
        asyncio.run(collect(cfg, broken, BrokenClient(), [0, 1], synthetic=True))
    store = Store(broken, cfg["env_id"])
    assert store.meta["complete"] == "false"
    store.close()
