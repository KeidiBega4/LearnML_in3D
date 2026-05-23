"""Stuck-recovery agent wrapper.

Wraps the default MLP policy with a stuck detector. When speed stays below
STUCK_SPEED for STUCK_THRESHOLD consecutive frames (at 20 Hz), the agent
switches to a recovery maneuver (reverse + alternating steering) for
RECOVERY_FRAMES frames before handing back to the MLP.

Usage:
    python 03_benchmark.py --tag v4 --weights nav_v4.npz --module drive2win.agent
"""
from __future__ import annotations

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action

STUCK_SPEED      = 0.3   # m/s — below this counts as "not moving"
STUCK_THRESHOLD  = 60    # frames before recovery kicks in (3 s at 20 Hz)
RECOVERY_FRAMES  = 40    # frames of recovery maneuver (2 s at 20 Hz)


def make_policy(weights_path: str):
    """Return a stateful policy function with stuck-recovery logic.

    Required by benchmark.py's --module hook.
    """
    w = nn_mod.load(weights_path)

    stuck_count    = [0]
    recovery_left  = [0]

    def policy(state):
        sensors = state["sensors"]
        speed   = sensors.get("speed", 0.0)

        # Count consecutive stuck frames
        if speed < STUCK_SPEED:
            stuck_count[0] += 1
        else:
            stuck_count[0] = 0

        # Trigger recovery when stuck too long
        if stuck_count[0] >= STUCK_THRESHOLD and recovery_left[0] == 0:
            print(f"  [agent] stuck for {stuck_count[0]} frames — starting recovery")
            recovery_left[0] = RECOVERY_FRAMES
            stuck_count[0]   = 0

        # Execute recovery: full reverse, steering alternates every 0.5 s (10 frames)
        if recovery_left[0] > 0:
            recovery_left[0] -= 1
            steer = 1.0 if (recovery_left[0] % 20) < 10 else -1.0
            return -1.0, steer

        # Normal MLP control
        x = sensors_to_input(sensors)
        return clip_action(nn_mod.forward(x, w))

    return policy
