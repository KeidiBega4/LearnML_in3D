"""Stuck-recovery agent wrapper.

Wraps the default MLP policy with a stuck detector. Triggers recovery when:
  - Speed stays below STUCK_SPEED for STUCK_THRESHOLD consecutive frames, OR
  - Front ray drops below WALL_TRIGGER_DIST while nearly stopped.

During recovery the robot reverses and steers toward whichever side has more
open space (using the 90-degree side rays), so it actually escapes rather than
just bouncing off the same wall.

Usage:
    python 03_benchmark.py --tag clean --weights nav_clean.npz --module drive2win.agent
"""
from __future__ import annotations

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action

STUCK_SPEED        = 0.3   # m/s — below this counts as "not moving"
STUCK_THRESHOLD    = 20    # frames before recovery kicks in (1 s at 20 Hz)
RECOVERY_FRAMES    = 30    # frames of recovery maneuver (1.5 s at 20 Hz)
WALL_TRIGGER_DIST  = 3.0   # m — front ray below this + nearly stopped = instant recovery

# Sensor ray indices (matches normalize.py FEATURE_NAMES order)
IDX_RAY_FRONT  = 3   # ray_0_front
IDX_RAY_LEFT   = 5   # ray_2_+90
IDX_RAY_RIGHT  = 9   # ray_6_-90


def make_policy(weights_path: str):
    """Return a stateful policy function with stuck-recovery logic.

    Required by benchmark.py's --module hook.
    """
    w = nn_mod.load(weights_path)

    stuck_count   = [0]
    recovery_left = [0]
    recovery_steer = [0.0]

    def policy(state):
        sensors = state["sensors"]
        speed   = sensors.get("speed", 0.0)
        rays    = sensors.get("rays", [50.0] * 8)

        ray_front = rays[0] if len(rays) > 0 else 50.0
        ray_left  = rays[2] if len(rays) > 2 else 50.0
        ray_right = rays[6] if len(rays) > 6 else 50.0

        # Count consecutive stuck frames
        if speed < STUCK_SPEED:
            stuck_count[0] += 1
        else:
            stuck_count[0] = 0

        # Trigger: slow + near wall, OR stuck too long
        wall_stuck = ray_front < WALL_TRIGGER_DIST and speed < STUCK_SPEED
        time_stuck = stuck_count[0] >= STUCK_THRESHOLD

        if (wall_stuck or time_stuck) and recovery_left[0] == 0:
            print(f"  [agent] recovery triggered — front={ray_front:.1f}m  speed={speed:.2f}")
            recovery_left[0]  = RECOVERY_FRAMES
            stuck_count[0]    = 0
            # Steer toward the side with more open space
            recovery_steer[0] = 1.0 if ray_left >= ray_right else -1.0

        # Execute recovery: full reverse + smart steering
        if recovery_left[0] > 0:
            recovery_left[0] -= 1
            return -1.0, recovery_steer[0]

        # Normal MLP control
        x = sensors_to_input(sensors)
        return clip_action(nn_mod.forward(x, w))

    return policy
