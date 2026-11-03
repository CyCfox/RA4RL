"""Frozen custom prompt templates and versions shared by collection and validation."""
from . import doorkey_prompt, putnear_prompt


def custom_prompt(env_id):
    """Return (template, protocol), or None for the generic local-view prompt."""
    if "DoorKey" in env_id:
        return doorkey_prompt.DOORKEY_PROMPT, doorkey_prompt.prompt_protocol()
    if "PutNear" in env_id:
        return putnear_prompt.PUTNEAR_PROMPT, putnear_prompt.prompt_protocol()
    return None
