"""User's PutNear 4x4-interior global-view prompt and live action history."""
import hashlib
from numbers import Integral
import re

PROMPT_VERSION = "putnear_global_native_grid4_history_20261103_v2"
PREVIOUS_ACTION_TEXT = (
    "left（原地向左转，即将智能体逆时针转90度）",
    "right（原地向右转，即将智能体顺时针转90度）",
    "forward（执行向前移动一格动作，不保证移动成功）",
    "pickup（执行拾取正前方物体动作，不保证拾取成功）",
    "drop（执行放置携带物体到正前方空格动作，不保证放置成功）",
    "toggle（执行交互动作）",
    "done（执行无操作动作）",
)
FIRST_STEP_TEXT = "无（这是当前 episode 的第一步）"

PUTNEAR_PROMPT = """你控制一个 MiniGrid-PutNear 环境中的智能体。
环境描述：
MiniGrid-PutNear 是一个基于网格的物体操作环境，包含墙壁、空地以及不同颜色的钥匙、球和箱子。墙壁和物体会阻挡智能体移动。除去墙外网格大小是4×4，最左上角黑色格子为第一行第一列。
观察说明：
输入的图像是整个环境的全局渲染图，包含所有网格、墙壁、物体和智能体。
智能体由红色三角形表示，三角形尖端朝向是智能体朝向，也是智能体正前方。图片的灰色部分是墙壁，无法通行。没有放置物体的黑色格子是空地，可以通行。任务描述：
PutNear 要求智能体根据自然语言指令，将指定物体拾取，并放置到另一个指定物体附近。
任务执行过程：
根据指令识别待移动物体和目标物体。
找到待移动物体，使其位于智能体正前方相邻格子，执行 pickup。
携带物体寻找目标参照物体。
找到目标物体周围相邻的空闲格子，使其位于智能体正前方相邻格子，执行 drop。
邻近关系：
放置位置必须满足目标物体的 8 邻域条件，即上下左右及四个对角线方向：
放置位置必须为空闲格子。
错误拾取或放置会导致任务立即失败，因此必须确保放置位置满足邻近条件。
当前任务： {mission}
当前携带状态： {a yellow key / a red ball / nothing}
上一步执行动作：{pre_action}
动作空间：
请选择以下 7 个动作的概率分布：
0 = left（原地向左转，即将智能体逆时针转90度）
1 = right（原地向右转，即将智能体顺时针转90度）
2 = forward（向前移动一格）
3 = pickup（拾取正前方物体）
4 = drop（放置携带物体到正前方空格）
5 = toggle（不使用）
6 = done（不使用）
输出格式：
简要分析当前智能体和物体所在格子，分析智能体朝向，分析物体相对智能体方向，并完成动作决策，只需考虑当前步骤。然后返回严格 JSON：
{"action_probs": [7 probabilities]}
要求：恰好包含 7 个有限非负概率，依次对应动作 0～6，总和必须为 1。"""


def render_prompt(state):
    action = state.previous_action
    if action is not None and (isinstance(action, bool) or not isinstance(action, Integral)
                               or action not in range(7)):
        raise ValueError("Previous action must be None or a MiniGrid action in 0..6")
    values = {
        "mission": state.mission,
        "a yellow key / a red ball / nothing":
            f"a {state.carrying_color} {state.carrying_type}" if state.carrying_type else "nothing",
        "pre_action": FIRST_STEP_TEXT if action is None else PREVIOUS_ACTION_TEXT[action],
    }
    # One pass preserves JSON braces and any placeholder-like text in missions.
    return re.sub(r"\{(mission|a yellow key / a red ball / nothing|pre_action)\}",
                  lambda match: values[match.group(1)], PUTNEAR_PROMPT)


def prompt_protocol():
    return dict(version=PROMPT_VERSION, sha256=hashlib.sha256(PUTNEAR_PROMPT.encode()).hexdigest())
