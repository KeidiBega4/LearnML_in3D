"""Stuck-recovery agent wrapper.

When the robot gets near a wall and slows down, it reverses and steers
toward whichever side has more open space, then hands back to the MLP.
If it gets stuck on the same wall again shortly after, it flips direction.

Usage:
    python 03_benchmark.py --tag clean --weights nav_clean.npz --module drive2win.agent
"""
from __future__ import annotations
import numpy as np

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action

STUCK_SPEED        = 0.3   # m/s — below this counts as not moving
STUCK_THRESHOLD    = 10    # frames before recovery kicks in (0.5 s at 20 Hz)
RECOVERY_FRAMES    = 80    # frames of reversing (4 s at 20 Hz)
WALL_TRIGGER_DIST  = 3.0   # m — front ray below this + nearly stopped = instant recovery
WALL_ZERO_DIST     = 0.5   # m — front ray below this = instant recovery regardless of speed
SAME_WALL_WINDOW   = 150   # frames — retriggering within this flips the steer direction
ESCAPE_FRAMES      = 25    # frames of biased open-space steering after recovery ends


def make_policy(weights_path: str):
    """Return a stateful policy function with stuck-recovery logic."""
    w = nn_mod.load(weights_path)

    stuck_count         = [0]
    recovery_left       = [0]
    recovery_steer      = [0.0]
    escape_left         = [0]
    escape_steer        = [0.0]
    last_recovery_steer = [0.0]   # direction used in the most recent recovery
    frames_since_recovery = [SAME_WALL_WINDOW + 1]  # start as if long ago

    def policy(state):
        sensors   = state["sensors"]
        speed     = sensors.get("speed", 0.0)
        rays      = sensors.get("rays", [50.0] * 8)

        ray_front = rays[0] if len(rays) > 0 else 50.0
        ray_left  = rays[2] if len(rays) > 2 else 50.0
        ray_right = rays[6] if len(rays) > 6 else 50.0

        # Tick the since-recovery counter
        if recovery_left[0] == 0 and escape_left[0] == 0:
            frames_since_recovery[0] = min(
                frames_since_recovery[0] + 1, SAME_WALL_WINDOW + 1
            )

        # Count how long we've been slow
        if speed < STUCK_SPEED:
            stuck_count[0] += 1
        else:
            stuck_count[0] = 0

        # Trigger conditions
        wall_zero  = ray_front < WALL_ZERO_DIST               # touching wall — act immediately
        wall_stuck = ray_front < WALL_TRIGGER_DIST and speed < STUCK_SPEED
        time_stuck = stuck_count[0] >= STUCK_THRESHOLD

        if (wall_zero or wall_stuck or time_stuck) and recovery_left[0] == 0:
            # If we got stuck again soon after last recovery, flip direction
            if frames_since_recovery[0] <= SAME_WALL_WINDOW:
                steer = -last_recovery_steer[0]
                print(f"  [agent] same wall — flipping steer to {steer:+.1f}  front={ray_front:.1f}m")
            else:
                steer = 1.0 if ray_left >= ray_right else -1.0
                print(f"  [agent] reversing — front={ray_front:.1f}m  speed={speed:.2f}")

            recovery_left[0]        = RECOVERY_FRAMES
            recovery_steer[0]       = steer
            last_recovery_steer[0]  = steer
            escape_steer[0]         = steer
            stuck_count[0]          = 0
            frames_since_recovery[0] = 0

        # Recovery phase: reverse until clear of wall
        if recovery_left[0] > 0:
            recovery_left[0] -= 1
            if recovery_left[0] == 0:
                escape_left[0] = ESCAPE_FRAMES  # start escape steering once reversing ends
            if ray_front > WALL_TRIGGER_DIST:
                return 1.0, recovery_steer[0]   # open ahead — push forward
            return -1.0, recovery_steer[0]       # wall ahead — reverse

        # Escape phase: briefly steer toward open space before handing back to MLP
        if escape_left[0] > 0:
            escape_left[0] -= 1
            return 0.8, escape_steer[0]

        # Normal MLP
        x = sensors_to_input(sensors)
        return clip_action(nn_mod.forward(x, w))

    return policy
