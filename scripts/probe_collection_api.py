"""One request only, preserving sanitized rate-limit diagnostics."""
import asyncio
import json
import os
from pathlib import Path

from collect_experience import credentials
from ra4rl.config import load_config
from ra4rl.data.llm import LLMClient
from ra4rl.envs.minigrid import make_env, llm_png, snapshot


async def main():
    credentials("RA4RL_Experiment_Plan.md")
    cfg = load_config("configs/doorkey_zoo_native.yaml")
    client = LLMClient(dict(cfg["llm"], max_retries=0), transport_policy={"http_retry_limit": 0})
    receipts = []

    async def receipt(response):
        await response.aread()
        record = dict(status=response.status_code, headers={
            k: v for k, v in response.headers.items()
            if any(part in k.lower() for part in ("retry", "ratelimit", "rate-limit", "date"))})
        if response.status_code != 200:
            record["body"] = response.text.replace(os.environ["OPENCODE_API_KEY"], "[REDACTED]")[:3000]
        receipts.append(record)

    client.client.event_hooks["response"] = [receipt]
    env = make_env(cfg["env_id"], 360, flat=False)
    try:
        obs, _ = env.reset(seed=0)
        try:
            result = await client.predict(cfg["env_id"], snapshot(env, obs), llm_png(env))
            receipts[-1]["valid_probabilities"] = result.tolist()
        except RuntimeError as exc:
            receipts[-1]["result"] = str(exc)
        path = Path("outputs/diagnostics/doorkey_rate_limit_probe.json")
        with path.open("x") as file:
            json.dump(receipts, file, indent=2)
        print(json.dumps(receipts, indent=2))
    finally:
        await client.close()
        env.close()


if __name__ == "__main__":
    asyncio.run(main())
