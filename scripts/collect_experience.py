"""Real LLM preflight / collection / static indexing, confined to the project Docker runtime."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from ra4rl.config import load_config, config_hash
from ra4rl.data.collect import collect
from ra4rl.data.llm import LLMClient
from ra4rl.envs.minigrid import make_env, snapshot, llm_png, rgb_input_protocol
from ra4rl.data.store import sha256
from ra4rl.data.translator import TRANSLATOR_VERSION
from ra4rl.data.prompts import custom_prompt
from ra4rl.data.reasoning import reasoning_options


def validate_build_schedule(cfg):
    """A formal collection launches one worker per configured seed."""
    bounds = cfg.get('collection_seeds')
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) for x in bounds)
            or bounds[0] < 0 or bounds[1] <= bounds[0]):
        raise ValueError('collection_seeds must be nonnegative integer [start, exclusive_end]')
    count = bounds[1] - bounds[0]
    concurrency = cfg['llm'].get('concurrency')
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency != count:
        raise ValueError('Formal collection concurrency must equal the configured seed count')


def credentials(plan=None):
    if os.environ.get("OPENCODE_API_KEY"):
        return "environment"
    if plan:
        # Explicit launch option: consume the credential already supplied for this project.
        # Never print, copy to an artifact, or put the value on a command line.
        match = re.search(r"API Key:\s*([^，\s|]+)", Path(plan).read_text())
        if match:
            os.environ["OPENCODE_API_KEY"] = match.group(1)
            return "project_plan"
    raise ValueError("No configured API credential")


async def preflight(cfg, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    client_cfg = dict(cfg["llm"], max_retries=0)
    client = LLMClient(client_cfg, transport_policy={"http_retry_limit": 0})
    receipts = []

    async def receipt(response):
        await response.aread()
        request_body = json.loads(response.request.content)
        item = dict(http_status=response.status_code,
                    requested_thinking=request_body['thinking'],
                    requested_reasoning_effort=request_body.get('reasoning_effort'),
                    requested_max_tokens=request_body['max_tokens'])
        if response.is_success:
            body = response.json()
            message = body["choices"][0]["message"]
            usage = body.get("usage", {})
            item.update(model=body.get("model"), usage=usage,
                        reasoning_content_present=bool(message.get("reasoning_content")))
            if reasoning_options(cfg['llm'])['thinking']['type'] == 'disabled' and (
                    item["reasoning_content_present"]
                    or usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)):
                raise RuntimeError("Provider returned reasoning although thinking was disabled")
        else:
            item["error"] = response.text.replace(os.environ["OPENCODE_API_KEY"], "[REDACTED]")[:2000]
        receipts.append(item)
        (output / "http_receipts.json").write_text(json.dumps(receipts, indent=2))

    client.client.event_hooks["response"] = [receipt]
    env = make_env(cfg["env_id"], cfg["max_episode_steps"], flat=False)
    started = time.monotonic()
    try:
        obs, _ = env.reset(seed=0)
        predictions = []
        previous_action = None
        for step in range(3):
            state = snapshot(env, obs, previous_action=previous_action)
            png = llm_png(env)
            if step == 0:
                (output / "llm_view.png").write_bytes(png)
            probs = await client.predict(cfg["env_id"], state, png)
            action = int(probs.argmax())
            obs, reward, done, truncated, _ = env.step(action)
            predictions.append(dict(step=step, action_probs=probs.tolist(), action_id=action,
                                    previous_action=state.previous_action))
            previous_action = action
            if done or truncated:
                break
        result = dict(status="passed", env_id=cfg["env_id"], model=cfg["llm"]["model"],
                      rgb_input=rgb_input_protocol(cfg["env_id"]),
                      **reasoning_options(cfg['llm']), elapsed_seconds=time.monotonic() - started,
                      requests=receipts, predictions=predictions)
        (output / "receipt.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        await client.close()
        env.close()


def main():
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Run inside Docker only")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "test", "build"])
    parser.add_argument("--config", default="configs/doorkey_zoo_native.yaml")
    parser.add_argument("--credentials-plan")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume-from", type=Path, help="Copy and verify an incomplete SQLite store; never modify the source")
    parser.add_argument("--test-timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.test_timeout_seconds <= 0:
        parser.error("test-timeout-seconds must be positive")
    source = credentials(args.credentials_plan)
    cfg = load_config(args.config)
    if args.mode == "preflight":
        asyncio.run(preflight(cfg, args.output))
        return
    if args.mode == "test":
        if args.resume_from:
            parser.error("The three-environment test always starts fresh")
        cfg["profile"] = "llm_task_test"
        cfg["collection_seeds"] = [0, 3]
        cfg["llm"]["concurrency"] = 3
    else:
        validate_build_schedule(cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    manifest_path = args.output / "collection_manifest.json"
    provenance = dict(
        env_id=cfg["env_id"], config_hash=config_hash(cfg), **reasoning_options(cfg['llm']),
        mode=args.mode, concurrency=cfg["llm"]["concurrency"],
        seeds=list(range(*cfg["collection_seeds"])), max_episode_steps=cfg["max_episode_steps"],
        credentials_source=source, status="collecting",
        source_sha256={str(p): sha256(p) for p in Path("src/ra4rl").rglob("*.py")},
        entrypoint_sha256=sha256(__file__),
        runtime_image_id=os.environ.get("RA4RL_IMAGE_ID"))
    overlap = sorted(set(range(*cfg['collection_seeds'])) & set(range(*cfg['eval_seeds'])))
    provenance['configured_evaluation_seed_overlap'] = overlap
    provenance['configured_evaluation_is_independent'] = not overlap
    provenance["rgb_input"] = rgb_input_protocol(cfg["env_id"])
    provenance["translator_version"] = TRANSLATOR_VERSION
    transport_policy = dict(global_throttle=False, http_error_cooldown_seconds=5,
                            http_retry_limit=None, retry_scope="single_request")
    provenance["transport_policy"] = transport_policy
    custom = custom_prompt(cfg["env_id"])
    if custom:
        template, template_protocol = custom
        template_path = args.output / "prompt_template.txt"
        template_path.write_text(template)
        provenance["prompt"] = dict(**template_protocol, file=template_path.name)
    manifest_path.write_text(json.dumps(provenance, indent=2))
    database = args.output / "experiences.sqlite"
    if args.resume_from:
        original_hash = sha256(args.resume_from)
        with sqlite3.connect(f"file:{args.resume_from.resolve()}?mode=ro", uri=True) as original:
            with sqlite3.connect(database) as destination:
                original.backup(destination)
        provenance["resume_source"] = dict(path=str(args.resume_from), sha256=original_hash)
        manifest_path.write_text(json.dumps(provenance, indent=2))

    async def run():
        client = LLMClient(cfg["llm"], transport_policy=transport_policy,
                           error_log=args.output / "http_errors.jsonl")
        try:
            work = collect(cfg, database, client=client, resume=bool(args.resume_from))
            if args.mode == "test":
                return await asyncio.wait_for(work, timeout=args.test_timeout_seconds)
            return await work
        finally:
            await client.close()

    try:
        summary = asyncio.run(run())
    except BaseException as exc:
        provenance.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        manifest_path.write_text(json.dumps(provenance, indent=2))
        raise
    if args.mode == "test":
        from ra4rl.data.review import review_collection
        review = review_collection(database, args.output)
        complete = dict(status="completed", mode="test", episodes=summary["episode_count"],
                        success_count=review["success_count"], task_success=review["success_count"] == 3,
                        config_hash=config_hash(cfg), sqlite_sha256=sha256(database),
                        **reasoning_options(cfg['llm']), index_built=False)
        (args.output / "completion.json").write_text(json.dumps(complete, indent=2))
        provenance["status"] = "test_completed"
        manifest_path.write_text(json.dumps(provenance, indent=2))
        print(json.dumps(complete, indent=2))
        return
    provenance["status"] = "indexing"
    provenance["episodes"] = summary["episode_count"]
    manifest_path.write_text(json.dumps(provenance, indent=2))
    from ra4rl.retrieval.index import build_index
    manifest = build_index(database, args.output / "index", cfg, device="cuda")
    complete = dict(status="completed", episodes=summary["episode_count"],
                    experience_count=manifest["count"], config_hash=config_hash(cfg),
                    sqlite_sha256=manifest["sqlite_sha256"], **reasoning_options(cfg['llm']))
    (args.output / "completion.json").write_text(json.dumps(complete, indent=2))
    provenance["status"] = "completed"
    manifest_path.write_text(json.dumps(provenance, indent=2))
    print(json.dumps(complete, indent=2))


if __name__ == "__main__":
    main()
