"""Stuck termination shared by final-project training envs (eval_flat/ice/hill)."""

import numpy as np

STUCK_SPEED_THRESHOLD_XY = 0.01  # m/s — ||v_xy|| below this counts as one stuck step
STUCK_TIME_SEC = 10.0


def advance_stuck_steps(stuck_steps: int, xy_velocity, dt: float) -> int:
    """Increment or reset the stuck counter from torso xy velocity (m/s)."""
    speed_xy = float(np.linalg.norm(xy_velocity))
    if speed_xy < STUCK_SPEED_THRESHOLD_XY:
        return stuck_steps + 1
    return 0


def is_stuck_terminated(stuck_steps: int, dt: float) -> bool:
    """True after ~STUCK_TIME_SEC of consecutive low-velocity steps."""
    return stuck_steps > STUCK_TIME_SEC / dt
