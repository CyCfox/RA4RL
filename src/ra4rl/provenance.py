import importlib.metadata
import json
import os
import platform
from pathlib import Path

import torch

from .config import config_hash
from .data.store import sha256


def git_commit(folder):
    """Read git metadata without requiring the git executable in the runtime image."""
    directory = Path(folder) / ".git"
    head = (directory / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    loose = directory / ref
    if loose.is_file():
        return loose.read_text().strip()
    packed = directory / "packed-refs"
    if packed.exists():
        for line in packed.read_text().splitlines():
            if not line.startswith(("#", "^")) and line.endswith(" " + ref):
                return line.split()[0]
    raise ValueError(f"Cannot resolve upstream revision: {folder}")


def record_run(output, cfg, algorithm, db_path=None, index_path=None):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.json").write_text(json.dumps(cfg, indent=2))
    sources = [p for folder in ("src", "configs", "scripts", "docker")
               for p in Path(folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    sources += [Path(p) for p in ("requirements.txt", "pyproject.toml", "RA4RL_Experiment_Plan.md")]
    packages = {p: importlib.metadata.version(p) for p in (
        "stable-baselines3", "minigrid", "gymnasium", "torch", "sentence-transformers",
        "transformers", "chromadb", "numpy", "rank-bm25")}
    manifest = dict(config_hash=config_hash(cfg), algorithm=algorithm, python=platform.python_version(),
                    packages=packages, source_sha256={str(p): sha256(p) for p in sources},
                    cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    manifest["runtime_image"] = os.environ.get("RA4RL_IMAGE", "unrecorded")
    manifest["runtime_image_id"] = os.environ.get("RA4RL_IMAGE_ID", "unrecorded")
    for folder in ("stable-baselines3", "rl-baselines3-zoo"):
        manifest[folder + "_commit"] = git_commit(folder)
    if db_path:
        manifest["sqlite_sha256"] = sha256(db_path)
        manifest["index_manifest"] = json.loads((Path(index_path) / "manifest.json").read_text())
        manifest["index_manifest_sha256"] = sha256(Path(index_path) / "manifest.json")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
