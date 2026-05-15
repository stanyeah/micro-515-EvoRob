"""Set MUJOCO_GL before importing mujoco / MujocoEnv (platform defaults)."""

import os
import sys


def configure_mujoco_gl() -> None:
    """Pick a MuJoCo rendering backend if MUJOCO_GL is not already set.

    macOS/Windows: glfw (display or compatible context).
    Linux: egl (headless-friendly).
    """
    if "MUJOCO_GL" in os.environ:
        return
    if sys.platform == "darwin" or sys.platform == "win32":
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ["MUJOCO_GL"] = "egl"
