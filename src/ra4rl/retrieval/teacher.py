import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ra4rl.data.store import Store, probabilities, sha256
from ra4rl.data.prompts import custom_prompt
from ra4rl.data.reasoning import reasoning_options
from ra4rl.data.translator import LEGACY_TRANSLATOR_VERSION, TRANSLATOR_VERSION, state_text, tokenize
from ra4rl.envs.minigrid import LOCAL_RGB, rgb_input_protocol
from .index import chroma, model_fingerprint


def rrf(dense, sparse, k=60, limit=20):
    scores = {}
    for ranking in (dense, sparse):
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))[:limit]


def mix(scores, distributions):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite reranker scores")
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    result = weights @ np.stack([probabilities(p) for p in distributions])
    return probabilities(result).astype(np.float32)


class RetrievalTeacher:
    def __init__(self, db_path, index_path, cfg, device="cuda", allow_synthetic=False):
        self.cfg, self.path = cfg["retrieval"], Path(index_path)
        self.db_path = Path(db_path)
        manifest = json.loads((self.path / "manifest.json").read_text())
        if manifest["env_id"] != cfg["env_id"]:
            raise ValueError("Cross-environment index")
        source = manifest["provenance"]
        expected_source = cfg.get("experience_source", "llm")
        if source == "synthetic" and not allow_synthetic:
            raise ValueError("Formal training cannot use synthetic experiences")
        if source != expected_source and not (source == "synthetic" and allow_synthetic):
            raise ValueError("Experience source differs from explicitly configured experience_source")
        if source not in {"llm", "synthetic", "trained_policy"}:
            raise ValueError("Unknown experience provenance")
        if manifest["sqlite_sha256"] != sha256(db_path):
            raise ValueError("Experience database changed after index build")
        if manifest["bm25_sha256"] != sha256(self.path / "bm25.json"):
            raise ValueError("BM25 index changed")
        for key in ("embedding", "reranker"):
            if manifest[key + "_fingerprint"] != model_fingerprint(self.cfg[key + "_model"]):
                raise ValueError(f"{key} model changed since index build")
        store = Store(db_path, cfg["env_id"])
        try:
            if store.meta["complete"] != "true":
                raise ValueError("Incomplete experience collection")
            protocol = json.loads(store.meta.get("collection_protocol", "{}"))
            if store.meta['provenance'] != source:
                raise ValueError('Index and SQLite provenance differ')
            self.translator_version = protocol.get("translator_version", LEGACY_TRANSLATOR_VERSION)
            if manifest.get("translator_version", LEGACY_TRANSLATOR_VERSION) != self.translator_version:
                raise ValueError("Index and SQLite observation translator versions differ")
            if cfg["profile"] == "formal":
                if self.translator_version != TRANSLATOR_VERSION:
                    raise ValueError("Experience observation translator does not match current formal protocol")
                episode_ids = [row[0] for row in store.db.execute("SELECT episode_id FROM episodes ORDER BY episode_id")]
                if (episode_ids != list(range(*cfg["collection_seeds"]))
                        or protocol["seeds"] != list(range(*cfg["collection_seeds"]))
                        or protocol["max_episode_steps"] != cfg["max_episode_steps"]):
                    raise ValueError("Experience collection does not match frozen formal protocol")
                if source == 'trained_policy':
                    if (protocol.get('source') != source
                            or protocol.get('policy_source') != cfg.get('policy_source')
                            or manifest.get('policy_source') != protocol.get('policy_source')
                            or protocol.get('batch_size') != cfg.get('collection_batch_size')
                            or protocol.get('batches') != [episode_ids[i:i + 10] for i in range(0, len(episode_ids), 10)]
                            or protocol.get('batch_size') != 10
                            or protocol.get('action_selection') != 'argmax_lowest_index_tie'
                            or protocol.get('action_targets') != 'policy_softmax'
                            or protocol.get('rgb_input') != LOCAL_RGB):
                        raise ValueError('Policy experience does not match frozen formal protocol')
                elif protocol["llm"] != cfg["llm"]:
                    raise ValueError("Experience collection does not match frozen formal protocol")
                if source == 'llm' and any(
                        protocol.get(key) != value for key, value in reasoning_options(cfg['llm']).items()):
                    raise ValueError('Experience thinking settings do not match frozen formal protocol')
                custom = custom_prompt(cfg["env_id"])
                if source == 'llm' and custom and (
                        protocol.get("prompt") != custom[1]
                        or protocol.get("rgb_input") != rgb_input_protocol(cfg["env_id"])):
                    raise ValueError(f"{cfg['env_id']} experience prompt or RGB protocol does not match current protocol")
            rows = store.rows()
        finally:
            store.close()
        bm = json.loads((self.path / "bm25.json").read_text())
        self.ids, self.documents = bm["ids"], bm["documents"]
        self.collection = chroma(self.path / "chroma").get_collection("experiences")
        records = self.collection.get(include=["documents", "metadatas"])
        if (self.ids != manifest["ids"] or self.ids != [r["experience_id"] for r in rows]
                or self.collection.count() != len(self.ids) or set(records["ids"]) != set(self.ids)):
            raise ValueError("SQLite / BM25 / Chroma ID mismatch")
        fact = {r["experience_id"]: r for r in rows}
        self.lookup = dict(zip(self.ids, range(len(self.ids))))
        for item, text, meta in zip(records["ids"], records["documents"], records["metadatas"]):
            row = fact[item]
            if (text != row["env_description"] or text != self.documents[self.lookup[item]]
                    or any(meta[k] != row[k] for k in (
                        "env_id", "mission", "action_probs_json", "action_id", "episode_id", "step_idx"))
                    or json.loads(meta["obs_img_flat"]) != list(row["obs_img"])):
                raise ValueError("Index content differs from SQLite facts")
        if len(self.ids) < max(self.cfg[k] for k in ("dense_k", "sparse_k", "fusion_k", "rerank_k")):
            raise ValueError("Insufficient experiences for configured Top-K")
        self.bm25 = BM25Okapi([tokenize(text) for text in self.documents])
        self.probs = {r["experience_id"]: probabilities(json.loads(r["action_probs_json"])) for r in rows}
        self.encoder = SentenceTransformer(self.cfg["embedding_model"], device=device, local_files_only=True)
        self.reranker = CrossEncoder(self.cfg["reranker_model"], device=device, local_files_only=True)
        self.cache = {}
        self.query_count = 0
        self._db_stat = self.db_path.stat()

    def __call__(self, states):
        current = self.db_path.stat()
        if (current.st_mtime_ns, current.st_size) != (self._db_stat.st_mtime_ns, self._db_stat.st_size):
            raise ValueError("Static database changed during training")
        texts = [state_text(state, version=self.translator_version) for state in states]
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        chunk_size = self.cfg["embedding_batch_size"]
        for start in range(0, len(missing), chunk_size):
            queries = missing[start:start + chunk_size]
            embeddings = self.encoder.encode(queries, normalize_embeddings=True, batch_size=chunk_size)
            dense_k = self.cfg["dense_k"]
            initial_k = min(len(self.ids), dense_k + 1)
            dense = self.collection.query(query_embeddings=embeddings.tolist(), n_results=initial_k)
            all_candidates = []
            pairs = []
            for i, query in enumerate(queries):
                # Stable ID tie breaking for both sources.
                ranking = sorted(zip(dense["ids"][i], dense["distances"][i]), key=lambda x: (x[1], x[0]))
                # Expand a tie at the Top-K boundary before applying the ID tie break.
                while len(ranking) < len(self.ids) and ranking[dense_k - 1][1] == ranking[-1][1]:
                    expanded_k = min(len(self.ids), 2 * len(ranking))
                    expanded = self.collection.query(query_embeddings=[embeddings[i].tolist()], n_results=expanded_k)
                    ranking = sorted(zip(expanded["ids"][0], expanded["distances"][0]), key=lambda x: (x[1], x[0]))
                scores = self.bm25.get_scores(tokenize(query))
                sparse = sorted(self.ids, key=lambda item: (-scores[self.lookup[item]], item))[:self.cfg["sparse_k"]]
                candidates = rrf([item for item, _ in ranking[:dense_k]], sparse,
                                 self.cfg["rrf_k"], self.cfg["fusion_k"])
                if len(candidates) < self.cfg["fusion_k"]:
                    raise ValueError("Insufficient retrieval candidates")
                all_candidates.append(candidates)
                pairs.extend((query, self.documents[self.lookup[item]]) for item in candidates)
            # predict's default activation for a one-label CrossEncoder is sigmoid: explicitly use logits.
            import torch
            logits = np.asarray(self.reranker.predict(pairs, batch_size=self.cfg["rerank_batch_size"],
                                                      activation_fct=torch.nn.Identity(),
                                                      show_progress_bar=False)).reshape(-1)
            offset = 0
            for query, candidates in zip(queries, all_candidates):
                scores = logits[offset:offset + len(candidates)]
                offset += len(candidates)
                order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))[:self.cfg["rerank_k"]]
                self.cache[query] = mix(scores[order], [self.probs[candidates[i]] for i in order])
                self.query_count += 1
        return np.stack([self.cache[text] for text in texts])
