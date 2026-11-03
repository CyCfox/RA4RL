"""Offline native rendering preview: no LLM client, requests, or episode collection."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ra4rl.data.doorkey_prompt import DOORKEY_PROMPT, prompt_protocol
from ra4rl.data.llm import prompt
from ra4rl.data.translator import TRANSLATOR_VERSION, state_text
from ra4rl.envs.minigrid import llm_png, make_env, rgb_input_protocol, snapshot


def main():
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Run inside Docker only")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env_id = "MiniGrid-DoorKey-6x6-v0"
    env = make_env(env_id, 360, flat=False)
    records = []
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    try:
        env.reset(seed=0)
        base = env.unwrapped
        for direction, name in enumerate(("right", "down", "left", "up")):
            # Only vary the rendering fixture; no action or env.step is executed.
            base.agent_dir = direction
            native = base.get_frame(highlight=False, tile_size=32, agent_pov=False)
            Image.fromarray(native).save(args.output / f"native_direction_{direction}.png")
            png = llm_png(env)
            image_path = args.output / f"llm_direction_{direction}.png"
            image_path.write_bytes(png)
            rotated = np.asarray(Image.open(image_path))
            assert np.array_equal(rotated, np.rot90(native, (direction + 1) % 4))
            state = snapshot(env, base.gen_obs())
            (args.output / f"description_direction_{direction}.txt").write_text(state_text(state))
            (args.output / f"prompt_direction_{direction}.txt").write_text(prompt(env_id, state))
            axes[0, direction].imshow(native)
            axes[0, direction].set_title(f"Native facing {name}")
            axes[1, direction].imshow(rotated)
            axes[1, direction].set_title(f"LLM facing up | CCW {((direction + 1) % 4) * 90} deg")
            records.append(dict(native_direction=direction, native_facing=name,
                                rotation_ccw_quarter_turns=(direction + 1) % 4,
                                llm_image=image_path.name, sha256=hashlib.sha256(png).hexdigest()))
        assert base.step_count == 0
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.output / "orientation_preview.png", dpi=140)
        (args.output / "prompt_template.txt").write_text(DOORKEY_PROMPT)
        receipt = dict(status="passed", scope="offline_render_only", env_id=env_id, seed=0,
                       env_steps=0, llm_api_requests=0, collection_started=False,
                       translator_version=TRANSLATOR_VERSION,
                       rgb_input=rgb_input_protocol(env_id), prompt=prompt_protocol(), views=records,
                       known_difference="Native global rendering does not overlay carried keys; prompt carrying state is explicit")
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt, indent=2))
    finally:
        plt.close(fig)
        env.close()


if __name__ == "__main__":
    main()
