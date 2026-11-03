"""User's 2026-11-01 global-view prompt; missing JSON square bracket corrected.

Native global rendering does not draw carried objects. The current local
observation description reports what the agent carries.
"""

import hashlib
from numbers import Integral
import re

from .translator import state_text

PROMPT_VERSION = "doorkey_global_description_history_20261101_v4"

PREVIOUS_ACTION_TEXT = (
    "The agent executed a left turn in the previous step.",
    "The agent executed a right turn in the previous step.",
    "The agent executed a forward action in the previous step.",
    "The agent executed a pickup action in the previous step.",
    "The agent executed a drop action in the previous step.",
    "The agent executed a toggle action in the previous step.",
    "The agent executed the done action (no operation) in the previous step.",
)
FIRST_STEP_TEXT = "This is the first step; the agent has not executed any previous action in this episode."

DOORKEY_PROMPT = """You control the agent in the MiniGrid DoorKey environment.
Environment description:
MiniGrid is a grid-based environment composed of walls, empty cells, keys, doors, and goals. The agent can move forward, turn, pick up keys, and open doors. Walls, unopened doors, or keys that have not been picked up block movement.
The agent is a red triangle. The goal is a green cell. Walls are gray cells. Empty cells are black cells. An opened door appears as a thin rectangular strip, and at that point it can be passed through. When the agent successfully picks up the key, the key will overlap with the cell the agent is in and follow the agent’s movement.
{description}
{door_status}
{key_status}
{previous_action}
Task description:
The agent must pick up the key in the map, then use the key to open the door, pass through the door, and reach the goal cell; the task is then successfully completed.
The key must be located in the adjacent cell directly in front of the agent to be picked up; a locked door must be located in the adjacent cell directly in front of the agent, and the toggle action must be executed while carrying the key to open it.
Key example that must be understood:
The agent faces upward by default, and upward is the agent’s front. If the key or door is already adjacent to the agent but is not directly in front of the agent, prioritize turning so that the key or door is directly in front of the agent. Action rules:
All actions are relative to the agent’s current orientation:
0 = left: turn left 90° in place
1 = right: turn right 90° in place
2 = forward: move forward one cell along the current orientation
3 = pickup: pick up the key in the adjacent cell directly in front
4 = drop: drop an object
5 = toggle: use the key to open the door in the adjacent cell directly in front
6 = done: no operation:
Output format:
In the output, reason briefly. **You must** first analyze the grid coordinates of the agent, key, and door (the bottom-left cell is the origin (0,0), the y-axis increases upward, the x-axis increases to the right, and walls also count as coordinates), the agent’s current pickup state, the positions of the key and door in the grid, and the action decision. Reason only about the action for this step. Then return only strict JSON containing one field:
{"action_probs": [7 probabilities]}
Requirement: exactly 7 finite nonnegative numbers forming a soft probability distribution, and the sum is 1."""


def render_prompt(state):
    if state.door_open is None:
        raise ValueError("DoorKey prompt requires the current door open state")
    action = state.previous_action
    if action is not None and (isinstance(action, bool) or not isinstance(action, Integral)
                               or action not in range(7)):
        raise ValueError("Previous action must be None or a MiniGrid action in 0..6")
    values = {
        "description": state_text(state),
        "door_status": f"The door currently is {'opened' if state.door_open else 'unopened'}.",
        # This reports current possession, including after drop; it does not
        # claim the key has never been picked up earlier in the episode.
        "key_status": "The key currently is picked up (carried by the agent)."
        if state.carrying_type == "key" else
        "The key currently is not picked up (the agent is not carrying it).",
        "previous_action": FIRST_STEP_TEXT if action is None else PREVIOUS_ACTION_TEXT[action],
    }
    # Replace placeholders in one pass so braces in a mission stay literal.
    return re.sub(r"\{(description|door_status|key_status|previous_action)\}",
                  lambda match: values[match.group(1)], DOORKEY_PROMPT)


def prompt_protocol():
    return dict(version=PROMPT_VERSION, sha256=hashlib.sha256(DOORKEY_PROMPT.encode()).hexdigest())
