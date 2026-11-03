import json
from pathlib import Path

import pytest

from ra4rl.checkpoints import Checkpoints
from ra4rl.config import load_config
from ra4rl.data.store import Store, sha256
from ra4rl.retrieval.index import build_index
from ra4rl.retrieval.teacher import RetrievalTeacher


def test_incomplete_collection_rejected_before_model_load(tmp_path):
    cfg = load_config(smoke=True)
    path = tmp_path / "partial.sqlite"
    store = Store(path, cfg["env_id"], create=True)
    store.close()
    with pytest.raises(ValueError, match="incomplete"):
        build_index(path, tmp_path / "index", cfg, device="cpu")
    assert not (tmp_path / "index").exists()


def test_index_integrity_and_formal_synthetic_rejection(tmp_path):
    cfg = load_config(smoke=True)
    db = tmp_path / "facts.sqlite"
    db.write_bytes(b"fixture")
    (tmp_path / "bm25.json").write_text("{}")
    manifest = dict(env_id=cfg["env_id"], provenance="synthetic", sqlite_sha256=sha256(db),
                    bm25_sha256=sha256(tmp_path / "bm25.json"))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="synthetic"):
        RetrievalTeacher(db, tmp_path, cfg, device="cpu")
    db.write_bytes(b"changed")
    with pytest.raises(ValueError, match="database changed"):
        RetrievalTeacher(db, tmp_path, cfg, device="cpu", allow_synthetic=True)
    db.write_bytes(b"fixture")
    (tmp_path / "bm25.json").write_text("changed")
    with pytest.raises(ValueError, match="BM25 index changed"):
        RetrievalTeacher(db, tmp_path, cfg, device="cpu", allow_synthetic=True)


def test_checkpoint_retention_and_lexicographic_best(tmp_path):
    class Model:
        num_timesteps = 0

        def save(self, path):
            Path(path).write_text(str(self.num_timesteps))

    class Env:
        def save(self, path):
            Path(path).write_text("normalization")

    model, env = Model(), Env()
    manager = Checkpoints(tmp_path, keep=10)
    for step in range(12):
        model.num_timesteps = step
        metrics = dict(success_rate_mean=.5, return_mean=1. if step == 0 else .5,
                       episode_steps_mean=200 - step)
        manager.save(model, env, metrics, "fixture-hash")
    assert len(list(tmp_path.glob("step_*"))) == 10
    assert (tmp_path / "best" / "model.zip").read_text() == "0"
    assert (tmp_path / "best" / "vecnormalize.pkl").is_file()
    model.num_timesteps = 12
    manager.save(model, env, dict(success_rate_mean=.6, return_mean=0., episode_steps_mean=200), "fixture-hash")
    assert (tmp_path / "best" / "model.zip").read_text() == "12"
