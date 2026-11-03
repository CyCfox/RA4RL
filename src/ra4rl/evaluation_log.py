"""Compact summaries and a single append-only JSONL file for all episode details."""
from contextlib import ExitStack
import json
from pathlib import Path


def load_episodes(run, record):
    """Read shared JSONL details as well as both historical detail formats."""
    if "episodes" in record:
        return record["episodes"]
    if "episodes_file" in record:
        details = json.loads((Path(run) / record["episodes_file"]).read_text())
    else:
        details = None
        with (Path(run) / "evaluation_details.jsonl").open() as file:
            for line in file:
                if not line.strip():
                    continue
                candidate = json.loads(line)
                if candidate["train_env_steps"] == record["train_env_steps"]:
                    if details is not None:
                        raise ValueError("Duplicate evaluation details for the same training step")
                    details = candidate
        if details is None:
            raise ValueError("No evaluation details for this training step")
    if details["train_env_steps"] != record["train_env_steps"]:
        raise ValueError("Evaluation details belong to a different training step")
    episodes = details["episodes"]
    if len(episodes) != record["evaluation_episodes"]:
        raise ValueError("Evaluation episode count differs from summary")
    return episodes


def episode_count(record):
    return record["evaluation_episodes"] if "evaluation_episodes" in record else len(record["episodes"])


class EvaluationLog:
    def __init__(self, run):
        self.run = Path(run)
        self.details = self.run / "evaluation_details.jsonl"

    def __enter__(self):
        with ExitStack() as stack:
            self.stream = stack.enter_context((self.run / "evaluations.jsonl").open("x"))
            self.detail_stream = stack.enter_context(self.details.open("x"))
            self._stack = stack.pop_all()
        return self

    def write(self, step, result, kd_lambda):
        episodes = result["episodes"]
        details = json.dumps(dict(train_env_steps=step, episodes=episodes), allow_nan=False)
        # Insertion order deliberately makes train_env_steps the first serialized field.
        summary = dict(train_env_steps=step)
        summary.update({k: v for k, v in result.items()
                        if k not in {"episodes", "episodes_file", "train_env_steps"}})
        summary.update(evaluation_episodes=len(episodes), **{"lambda": kd_lambda})
        encoded_summary = json.dumps(summary, allow_nan=True)
        # Flush all episode details before publishing the corresponding summary.
        self.detail_stream.write(details + "\n")
        self.detail_stream.flush()
        self.stream.write(encoded_summary + "\n")
        self.stream.flush()
        return summary

    def __exit__(self, *_):
        self._stack.close()
