"""BYLW P5 Isaac Lab extension.

Core utilities deliberately remain importable without Isaac Lab so P5-0A can
be validated on CPU. Gym task registration is enabled only when the Isaac Lab
runtime is importable.
"""

import os
from importlib.util import find_spec


if (
    find_spec("isaaclab") is not None
    and os.environ.get("BYLW_P5_STATIC_IMPORT") != "1"
):
    from .tasks import *  # noqa: F401,F403
