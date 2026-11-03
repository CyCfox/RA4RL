import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE episodes (episode_id INTEGER PRIMARY KEY, success INTEGER NOT NULL,
 return_value REAL NOT NULL, length INTEGER NOT NULL);
CREATE TABLE experiences (
 experience_id TEXT PRIMARY KEY, episode_id INTEGER NOT NULL, step_idx INTEGER NOT NULL,
 env_id TEXT NOT NULL, obs_img BLOB NOT NULL, rgb_png BLOB NOT NULL, mission TEXT NOT NULL,
 direction INTEGER NOT NULL, carrying_type TEXT, carrying_color TEXT,
 action_probs_json TEXT NOT NULL, action_id INTEGER NOT NULL, reward REAL NOT NULL,
 env_description TEXT NOT NULL, terminated INTEGER NOT NULL, truncated INTEGER NOT NULL,
 UNIQUE(episode_id, step_idx));
"""


def probabilities(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (7,) or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("Invalid seven-action probabilities")
    if not np.isclose(arr.sum(), 1, atol=1e-6):
        raise ValueError("Action probabilities must sum to one")
    return arr


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Store:
    def __init__(self, path, env_id, create=False, provenance="llm", resume=False):
        path = Path(path)
        if create and resume:
            raise ValueError("Cannot create and resume simultaneously")
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive reservation prevents accidental overwriting of existing experiences.
            with path.open("xb"):
                pass
        elif not path.is_file():
            raise FileNotFoundError(path)
        self.path = path
        writable = create or resume
        self.db = sqlite3.connect(path if writable else f"file:{path.resolve()}?mode=ro", uri=not writable)
        self.db.row_factory = sqlite3.Row
        if create:
            self.db.executescript(SCHEMA)
            self.db.executemany("INSERT INTO meta VALUES (?,?)", [
                ("env_id", env_id), ("provenance", provenance), ("complete", "false")])
            self.db.commit()
        self.meta = dict(self.db.execute("SELECT key,value FROM meta"))
        if self.meta["env_id"] != env_id:
            self.close()
            raise ValueError("Cross-environment experience store")
        if resume and (self.meta["complete"] != "false" or self.meta["provenance"] != provenance):
            self.close()
            raise ValueError("Resume requires an incomplete store with matching provenance")

    def add(self, episode, step, state, png, probs, action, reward, text, terminated, truncated):
        probs = probabilities(probs)
        if int(np.argmax(probs)) != action:
            raise ValueError("Stored action must be argmax of source distribution")
        self.db.execute("INSERT INTO experiences VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            f"{episode:03d}:{step:03d}", episode, step, self.meta["env_id"],
            state.image.astype(np.uint8).tobytes(), png, state.mission, state.direction,
            state.carrying_type, state.carrying_color, json.dumps(probs.tolist()),
            action, float(reward), text, int(terminated), int(truncated)))
        self.db.commit()

    def episode(self, seed, success, reward, length):
        self.db.execute("INSERT INTO episodes VALUES (?,?,?,?)", (seed, int(success), reward, length))
        self.db.commit()

    def finish(self):
        self.db.execute("UPDATE meta SET value='true' WHERE key='complete'")
        self.db.commit()

    def rows(self):
        rows = [dict(row) for row in self.db.execute("SELECT * FROM experiences ORDER BY experience_id")]
        if not rows or any(row["env_id"] != self.meta["env_id"] for row in rows):
            raise ValueError("Empty or mixed experience store")
        for row in rows:
            probabilities(json.loads(row["action_probs_json"]))
        return rows

    def close(self):
        self.db.close()
