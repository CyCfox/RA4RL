"""Replay completed collections for inspection; no information is sent to the LLM."""
from collections import Counter
import json
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .collect import replay, save_image
from .store import Store
from .translator import LEGACY_TRANSLATOR_VERSION
from ra4rl.envs.minigrid import LOCAL_RGB, llm_png

ACTIONS = ("left", "right", "forward", "pickup", "drop", "toggle", "done")

VIEWER = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>LLM 逐步图像输入</title><style>
body{font:16px system-ui;max-width:850px;margin:30px auto;background:#f4f6f8;color:#19212b}
img{width:448px;height:448px;image-rendering:pixelated;background:#111}
input{width:75%}pre{white-space:pre-wrap}button,select{padding:8px;margin:5px}
</style><h1>LLM 逐步图像输入</h1>
<p>截图是动作前发送给 LLM 的实际图像（原生全局、全局朝上或历史局部协议）；最后一帧为终止状态。步数从 0 开始。</p>
<select id="seed"></select><button id="previous">上一步</button><button id="next">下一步</button>
<p><input id="step" type="range" min="0" value="0"></p><img id="frame"><pre id="detail"></pre>
<script>const data=__DATA__, seed=document.querySelector('#seed'), step=document.querySelector('#step');
Object.keys(data).forEach(s=>seed.add(new Option('seed '+s,s)));
function draw(){const a=data[seed.value];step.max=a.length-1;const r=a[+step.value];
document.querySelector('#frame').src=r.image;document.querySelector('#detail').textContent=JSON.stringify(r,null,2)}
seed.onchange=()=>{step.value=0;draw()};step.oninput=draw;
document.querySelector('#previous').onclick=()=>{step.value=Math.max(0,+step.value-1);draw()};
document.querySelector('#next').onclick=()=>{step.value=Math.min(+step.max,+step.value+1);draw()};
document.onkeydown=e=>{if(e.key==='ArrowRight')document.querySelector('#next').click();
if(e.key==='ArrowLeft')document.querySelector('#previous').click()};draw();</script></html>"""


def review_collection(database, output):
    output = Path(output)
    images = output / "images"
    images.mkdir(exist_ok=True)
    import sqlite3
    with sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True) as db:
        meta = dict(db.execute("SELECT key,value FROM meta"))
        episodes = list(db.execute("SELECT episode_id,success,return_value,length FROM episodes ORDER BY episode_id"))
    if meta["complete"] != "true":
        raise ValueError("Review requires a completed collection")
    protocol = json.loads(meta["collection_protocol"])
    rgb_protocol = protocol.get("rgb_input", LOCAL_RGB)
    translator_version = protocol.get("translator_version", LEGACY_TRANSLATOR_VERSION)
    store = Store(database, meta["env_id"])
    try:
        rows = store.rows()
    finally:
        store.close()
    viewer = {}
    reports = []
    for seed, success, reward, length in episodes:
        trajectory = [r for r in rows if r["episode_id"] == seed]
        if len(trajectory) != length:
            raise ValueError("Episode length disagrees with stored transitions")
        env = gym.make(meta["env_id"], max_steps=protocol["max_episode_steps"])
        frames = []
        events = []
        positions, states, counts = set(), Counter(), Counter()
        blocked_forward = 0
        try:
            obs, _ = env.reset(seed=seed)
            for row in trajectory:
                step = row["step_idx"]
                position = tuple(map(int, env.unwrapped.agent_pos))
                positions.add(position)
                states[(position, row["direction"], row["obs_img"], row["carrying_type"])] += 1
                before_key = env.unwrapped.carrying is not None and env.unwrapped.carrying.type == "key"
                before_open = any(o and o.type == "door" and o.is_open for o in env.unwrapped.grid.grid)
                # Validate this prefix once below; here reconstruct for event analysis.
                action = row["action_id"]
                obs, _, _, _, _ = env.step(action)
                after_key = env.unwrapped.carrying is not None and env.unwrapped.carrying.type == "key"
                after_open = any(o and o.type == "door" and o.is_open for o in env.unwrapped.grid.grid)
                if after_key and not before_key:
                    events.append(dict(step_idx=step, event="picked_key"))
                if after_open and not before_open:
                    events.append(dict(step_idx=step, event="opened_door"))
                blocked_forward += int(action == 2 and tuple(env.unwrapped.agent_pos) == position)
                counts[ACTIONS[action]] += 1
                image = save_image(images, seed, step, row["rgb_png"])
                frames.append(dict(seed=seed, step_idx=step, image=f"images/{image.name}",
                                   action=ACTIONS[action], action_id=action,
                                   action_probs=json.loads(row["action_probs_json"]),
                                   carrying_type=row["carrying_type"], reward=row["reward"],
                                   terminated=bool(row["terminated"]), truncated=bool(row["truncated"])))
            positions.add(tuple(map(int, env.unwrapped.agent_pos)))
            final = save_image(images, seed, length, llm_png(env, rgb_protocol))
            frames.append(dict(seed=seed, step_idx=length, image=f"images/{final.name}",
                               action="terminal frame", success=bool(success)))
            initial, _ = env.reset(seed=seed)
            _, total, terminal = replay(env, initial, trajectory, rgb_protocol, translator_version)
            if not terminal or total != reward:
                raise ValueError("Episode summary disagrees with deterministic replay")
        finally:
            env.close()
        viewer[str(seed)] = frames
        reports.append(dict(seed=seed, success=bool(success), return_value=reward, length=length,
                            terminated=bool(trajectory[-1]["terminated"]),
                            truncated=bool(trajectory[-1]["truncated"]),
                            picked_key=any(e["event"] == "picked_key" for e in events),
                            opened_door=any(e["event"] == "opened_door" for e in events),
                            events=events, action_counts=dict(counts), unique_positions=len(positions),
                            blocked_forward_steps=blocked_forward,
                            most_repeated_state_visits=max(states.values())))
        chosen = np.linspace(0, len(frames) - 1, min(16, len(frames)), dtype=int)
        fig, axes = plt.subplots(4, 4, figsize=(10, 11))
        for ax in axes.flat:
            ax.axis("off")
        for ax, index in zip(axes.flat, chosen):
            frame = frames[index]
            ax.imshow(plt.imread(output / frame["image"]), interpolation="nearest")
            ax.set_title(f"step {frame['step_idx']}: {frame['action']}", fontsize=9)
        fig.suptitle(f"seed={seed} | success={bool(success)} | steps={length} | return={reward:.4f}")
        fig.tight_layout()
        fig.savefig(output / f"seed_{seed:03d}_overview.png", dpi=130)
        plt.close(fig)
    report = dict(env_id=meta["env_id"], episodes=reports, episode_count=len(reports),
                  success_count=sum(r["success"] for r in reports),
                  rgb_input=rgb_protocol,
                  translator_version=translator_version,
                  image_semantics="pre-action LLM input; step=episode length is terminal frame",
                  screenshot_count=sum(len(frames) for frames in viewer.values()),
                  replay_validation="passed")
    (output / "task_review.json").write_text(json.dumps(report, indent=2))
    (output / "replay.html").write_text(VIEWER.replace("__DATA__", json.dumps(viewer).replace("<", "\\u003c")))
    return report
