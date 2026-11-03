import json
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ra4rl.data.store import Store, sha256
from ra4rl.data.translator import LEGACY_TRANSLATOR_VERSION, translator


def chroma(path):
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def model_fingerprint(path):
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and ".cache" not in p.parts and not p.name.startswith(".")}


def build_index(db_path, index_path, cfg, device="cuda"):
    env_id = cfg["env_id"]
    store = Store(db_path, env_id)
    try:
        if store.meta["complete"] != "true":
            raise ValueError("Cannot index incomplete collection")
        rows, provenance = store.rows(), store.meta["provenance"]
        protocol = json.loads(store.meta.get("collection_protocol", "{}"))
        translator_version = protocol.get("translator_version", LEGACY_TRANSLATOR_VERSION)
    finally:
        store.close()
    path = Path(index_path)
    path.mkdir(parents=True, exist_ok=False)
    documents = []
    metadata = []
    for row in rows:
        image = np.frombuffer(row["obs_img"], dtype=np.uint8).reshape(7, 7, 3)
        text = translator(image, row["mission"], row["carrying_type"], row["carrying_color"],
                          version=translator_version)
        if text != row["env_description"]:
            raise ValueError("Translator mismatch with SQLite facts")
        documents.append(text)
        metadata.append({key: row[key] for key in (
            "mission", "action_probs_json", "action_id", "env_id", "episode_id", "step_idx")})
        metadata[-1]["obs_img_flat"] = json.dumps(image.flatten().tolist())
    model_path = cfg["retrieval"]["embedding_model"]
    model = SentenceTransformer(model_path, device=device, local_files_only=True)
    embeddings = model.encode(documents, normalize_embeddings=True,
                              batch_size=cfg["retrieval"]["embedding_batch_size"])
    if embeddings.shape != (len(rows), 384):
        raise ValueError("Embedding dimension must be 384")
    client = chroma(path / "chroma")
    collection = client.create_collection("experiences", metadata={"hnsw:space": "cosine", "env_id": env_id})
    ids = [row["experience_id"] for row in rows]
    for start in range(0, len(rows), 512):
        end = start + 512
        collection.add(ids=ids[start:end], documents=documents[start:end],
                       embeddings=embeddings[start:end].tolist(), metadatas=metadata[start:end])
    payload = dict(ids=ids, documents=documents)
    (path / "bm25.json").write_text(json.dumps(payload))
    if collection.count() != len(ids) or set(collection.get()["ids"]) != set(ids):
        raise ValueError("Index IDs disagree")
    manifest = dict(env_id=env_id, count=len(ids), ids=ids, sqlite_sha256=sha256(db_path),
                    bm25_sha256=sha256(path / "bm25.json"), provenance=provenance,
                    translator_version=translator_version,
                    embedding_fingerprint=model_fingerprint(model_path),
                    reranker_fingerprint=model_fingerprint(cfg["retrieval"]["reranker_model"]))
    if provenance == "trained_policy":
        manifest["policy_source"] = protocol["policy_source"]
    # Completion manifest is written last; partial indexes cannot be opened.
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
