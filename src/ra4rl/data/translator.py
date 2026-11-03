import re

import numpy as np
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT

LEGACY_TRANSLATOR_VERSION = "local_all_walls_adjacent_v1"
FRONT_WALL_TRANSLATOR_VERSION = "local_front_wall_distance_v2"
AGENT_RELATIVE_TRANSLATOR_VERSION = "local_adjacent_walls_agent_relative_v3"
TRANSLATOR_VERSION = "local_adjacent_walls_compact_axes_v4"

WALL_POSITIONS = {
    FRONT_WALL_TRANSLATOR_VERSION: frozenset({(3, 5)}),
    AGENT_RELATIVE_TRANSLATOR_VERSION: frozenset({(3, 5), (2, 6), (4, 6)}),
    TRANSLATOR_VERSION: frozenset({(3, 5), (2, 6), (4, 6)}),
}


def translator(obs, mission=None, carrying_type=None, carrying_color=None, version=TRANSLATOR_VERSION):
    if version not in {LEGACY_TRANSLATOR_VERSION, *WALL_POSITIONS}:
        raise ValueError("Unsupported observation translator version")
    legacy = version == LEGACY_TRANSLATOR_VERSION
    image = obs["image"] if isinstance(obs, dict) else obs
    mission = obs.get("mission", mission) if isinstance(obs, dict) else mission
    image = np.asarray(image)
    if image.shape != (7, 7, 3) or not mission:
        raise ValueError("Expected (7,7,3) image and mission")
    objects = []
    for x in range(7):
        for y in range(7):
            kind, color_id, state = map(int, image[x, y])
            kind = IDX_TO_OBJECT[kind]
            if kind in {"unseen", "empty", "agent"} or (x, y) == (3, 6):
                continue
            if kind == "wall" and not legacy and (x, y) not in WALL_POSITIONS[version]:
                continue
            lateral, forward = x - 3, 6 - y
            color = IDX_TO_COLOR[color_id]
            label = f"{color} {kind}"
            if kind == "door":
                label = f"{['open', 'closed', 'locked'][state]} {label}"
            if legacy and (x, y) == (3, 5):
                where = "in the adjacent cell in front of it"
            else:
                side = "left" if lateral < 0 else "right" if lateral > 0 else "center"
                owner = "agent's " if version in {AGENT_RELATIVE_TRANSLATOR_VERSION, TRANSLATOR_VERSION} else ""
                if version == TRANSLATOR_VERSION and lateral == 0:
                    where = f"at {owner}{forward} cells forward"
                elif version == TRANSLATOR_VERSION and forward == 0:
                    where = f"at {owner}{abs(lateral)} cells {side}"
                else:
                    where = f"at {owner}{abs(lateral)} cells {side}, {forward} cells forward"
            objects.append((kind, x, y, f"a {label} {where}"))
    held = f"{carrying_color} {carrying_type}" if carrying_type else "nothing"
    visible = "; ".join(o[3] for o in sorted(objects)) or "no visible objects"
    return f"Mission: {mission} The agent is currently carrying {held}; visible: {visible}."


def state_text(state, version=TRANSLATOR_VERSION):
    return translator(state.image, state.mission, state.carrying_type, state.carrying_color, version=version)


def tokenize(text):
    return re.findall(r"[a-z]+|[0-9]+", text.lower())
