import asyncio
import base64
from contextvars import ContextVar
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np

from .doorkey_prompt import render_prompt
from .putnear_prompt import render_prompt as putnear_prompt
from .reasoning import reasoning_options

EPISODE_SESSION = ContextVar("ra4rl_episode_session", default=None)


TASKS = {
    "DoorKey": "Find and pick up the key, toggle the matching locked door while carrying the key, then reach the green goal.",
    "GoToDoor": "Navigate adjacent to the door of the color specified in the mission, then use done to declare arrival. Toggling a door terminates with failure.",
    "Fetch": "Find the object of the exact color and type specified by the mission and pick it up. Picking up a different object ends the episode with failure.",
    "KeyCorridor": "Explore the connected rooms, find the matching key, toggle the locked door, and pick up the mission target object.",
    "PutNear": "Pick up the exact object named first in the mission, then drop it in an empty cell next to the second object (including diagonal neighbors). Picking up the wrong object or dropping unsuccessfully ends the episode with failure.",
}


def prompt(env_id, state):
    if "DoorKey" in env_id:
        return render_prompt(state)
    if "PutNear" in env_id:
        return putnear_prompt(state)
    task = next(value for key, value in TASKS.items() if key in env_id)
    carrying = f"{state.carrying_color} {state.carrying_type}" if state.carrying_type else "nothing"
    return (
        "You control an agent in MiniGrid. Only the current native 7x7 local RGB view is shown. "
        "Occluded cells are black; walls are gray; visible empty cells are dark gray. "
        "The red triangle at bottom center faces upward. Closed doors and objects block movement. "
        f"Task: {task}\nMission: {state.mission}\nCarrying: {carrying}\n"
        "Actions: 0 left (turn left), 1 right (turn right), 2 forward, "
        "3 pickup (adjacent object in front), 4 drop, 5 toggle (adjacent door in front), "
        "6 done (declare completion in GoToDoor; otherwise no operation). "
        'Reason briefly if needed, then output strict JSON: {"action_probs":[p0,p1,p2,p3,p4,p5,p6]}. '
        "Use exactly seven finite nonnegative probabilities summing to one."
    )


def parse_probs(content):
    decoder = json.JSONDecoder()
    for pos, char in enumerate(content):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(content[pos:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or set(obj) != {"action_probs"}:
            continue
        values = obj["action_probs"]
        if not isinstance(values, list) or len(values) != 7:
            continue
        if any(isinstance(x, bool) or not isinstance(x, (float, int)) for x in values):
            continue
        arr = np.asarray(values, dtype=np.float64)
        if not np.isfinite(arr).all():
            continue
        arr = np.clip(arr, 0, 1)
        if arr.sum() > 0:
            return arr / arr.sum()
    raise ValueError("No valid repairable action_probs JSON")


class LLMClient:
    def __init__(self, cfg, transport_policy=None, error_log=None):
        self.cfg = cfg
        self.reasoning_options = reasoning_options(cfg)
        key = os.environ.get("OPENCODE_API_KEY")
        if not key:
            raise ValueError("Set OPENCODE_API_KEY; credentials are never read from the plan")
        self.session_id = "ses_ra4rl_" + uuid4().hex
        self.client = httpx.AsyncClient(timeout=cfg["timeout"],
            headers={"Authorization": f"Bearer {key}", "User-Agent": "RA4RL-ResearchCollector/0.1"},
            limits=httpx.Limits(max_connections=cfg["concurrency"],
                               max_keepalive_connections=cfg["concurrency"]))
        self.semaphore = asyncio.Semaphore(cfg["concurrency"])
        self.stats = dict(requests=0, successful_requests=0, retries=0,
                          reasoning_tokens=0, reasoning_content_responses=0,
                          finish_reasons={}, truncated_responses=0,
                          http_statuses={}, max_in_flight=0)
        self.in_flight = 0
        self.transport_policy = dict(transport_policy or {})
        if "requests_per_minute" in self.transport_policy:
            raise ValueError("Global request throttling has been removed")
        self.http_retry_limit = self.transport_policy.get("http_retry_limit")
        self.error_log = Path(error_log) if error_log else None
        self.cooling_requests = 0

    def progress_stats(self):
        return dict(self.stats, global_throttle=False, http_error_cooldown_seconds=5,
                    cooling_requests=self.cooling_requests, in_flight=self.in_flight)

    async def predict(self, env_id, state, png):
        payload = {
            "model": self.cfg["model"], "max_tokens": self.cfg["max_tokens"],
            **self.reasoning_options,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt(env_id, state)},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}],
        }
        attempt = http_errors = 0
        while True:
            try:
                # Release the connection slot before any local retry sleep so
                # another environment can proceed even at concurrency=1.
                async with self.semaphore:
                    self.stats["requests"] += 1
                    self.in_flight += 1
                    self.stats["max_in_flight"] = max(self.stats["max_in_flight"], self.in_flight)
                    try:
                        response = await self.client.post(self.cfg["endpoint"], json=payload,
                            headers={"x-opencode-session": EPISODE_SESSION.get() or self.session_id})
                    finally:
                        self.in_flight -= 1
                statuses = self.stats["http_statuses"]
                status = str(response.status_code)
                statuses[status] = statuses.get(status, 0) + 1
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                message = choice["message"]
                reasoning = body.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                self.stats["reasoning_tokens"] += reasoning or 0
                self.stats["reasoning_content_responses"] += bool(message.get("reasoning_content"))
                if self.reasoning_options['thinking']['type'] == 'disabled' and (
                        message.get("reasoning_content") or reasoning):
                    raise RuntimeError("Provider returned reasoning although thinking was disabled")
                finish = choice.get('finish_reason')
                if finish:
                    counts = self.stats['finish_reasons']
                    counts[finish] = counts.get(finish, 0) + 1
                if finish == 'length':
                    self.stats['truncated_responses'] += 1
                    raise ValueError('Provider output truncated before completion')
                # Never parse probabilities from the provider's reasoning field.
                result = parse_probs(message.get("content") or '')
                self.stats["successful_requests"] += 1
                return result
            except httpx.HTTPStatusError as exc:
                http_errors += 1
                detail = exc.response.text.replace(os.environ["OPENCODE_API_KEY"], "[REDACTED]")
                error = dict(at=datetime.now(timezone.utc).isoformat(),
                             session=EPISODE_SESSION.get() or self.session_id,
                             status=exc.response.status_code, error=detail[:1500], cooldown_seconds=5)
                self.stats["last_http_error"] = error
                if self.error_log:
                    with self.error_log.open("a") as file:
                        file.write(json.dumps(error) + "\n")
                if self.http_retry_limit is not None and http_errors > self.http_retry_limit:
                    raise RuntimeError(f"LLM rejected request: HTTP {exc.response.status_code}: {detail[:1500]}") from None
                # All HTTP failures, including 401/403/429/5xx, retry only this
                # unchanged request. Other episode workers remain independent.
                self.stats["retries"] += 1
                self.cooling_requests += 1
                try:
                    await asyncio.sleep(5)
                finally:
                    self.cooling_requests -= 1
                continue
            except (httpx.TimeoutException, httpx.TransportError, ValueError, KeyError, IndexError):
                pass
            if attempt == self.cfg["max_retries"]:
                raise RuntimeError("LLM retries exhausted; collection remains incomplete")
            await asyncio.sleep(min(2 ** attempt, 30))
            self.stats["retries"] += 1
            attempt += 1

    async def close(self):
        await self.client.aclose()
