"""Minimal stuck-recovery wrapper.

If the robot hasn't moved for STUCK_THRESHOLD frames, reverse for
REVERSE_FRAMES frames then hand straight back to the MLP.
That's it — the MLP handles all steering and navigation.

Usage:
    python 03_benchmark.py --tag v1 --weights nav_v1.npz --module drive2win.agent
"""
from __future__ import annotations

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action

STUCK_SPEED      = 0.5   # m/s — below this counts as not moving
STUCK_THRESHOLD  = 15    # frames before reversing (0.75 s at 20 Hz)
REVERSE_FRAMES   = 40    # frames to reverse (2.0 s at 20 Hz)
REVERSE_THROTTLE = -1.0  # full reverse throttle


def make_policy(weights_path: str):
    w = nn_mod.load(weights_path)

    stuck_count   = [0]
    reverse_left  = [0]

    def policy(state):
        sensors = state["sensors"]
        speed   = sensors.get("speed", 0.0)

        # Track slow frames — only count when not already reversing
        if reverse_left[0] > 0:
            stuck_count[0] = 0   # freeze counter during reverse
        elif speed < STUCK_SPEED:
            stuck_count[0] += 1
        else:
            stuck_count[0] = 0

        # Trigger reverse
        if stuck_count[0] >= STUCK_THRESHOLD and reverse_left[0] == 0:
            print(f"  [agent] stuck ({stuck_count[0]} frames) — reversing")
            reverse_left[0] = REVERSE_FRAMES
            stuck_count[0]  = 0

        # Reverse phase — steer away from whichever side has less space
        if reverse_left[0] > 0:
            reverse_left[0] -= 1
            rays      = sensors.get("rays", [50.0] * 8)
            ray_left  = rays[2] if len(rays) > 2 else 50.0
            ray_right = rays[6] if len(rays) > 6 else 50.0
            steer = 0.4 if ray_right < ray_left else -0.4
            return REVERSE_THROTTLE, steer

        # Normal MLP
        x = sensors_to_input(sensors)
        return clip_action(nn_mod.forward(x, w))

    return policy
