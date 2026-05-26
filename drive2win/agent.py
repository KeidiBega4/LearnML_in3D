"""Stuck-recovery agent wrapper.

Wraps the default MLP policy with a stuck detector and multi-level recovery:

  Level 1 — normal recovery:
    Reverse + steer toward open space for RECOVERY_FRAMES, then a cooldown
    so stuck detection doesn't re-fire before the robot has a chance to move.

  Level 2 — escape mode:
    Triggered when ESCAPE_TRIGGER consecutive recoveries pass with no new
    checkpoint. Alternates forward+steer / reverse+opposite-steer in 10-frame
    bursts to wiggle free from wall gaps and ramp geometry.

Usage:
    python 03_benchmark.py --tag clean --weights nav_clean.npz --module drive2win.agent
"""
from __future__ import annotations
import numpy as np

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action

STUCK_SPEED        = 0.3    # m/s — below this counts as not moving
STUCK_THRESHOLD    = 10     # frames before recovery kicks in (0.5 s at 20 Hz)
RECOVERY_FRAMES    = 50     # frames of level-1 recovery (2.5 s at 20 Hz)
RECOVERY_COOLDOWN  = 25     # frames after recovery where stuck detection is off
WALL_TRIGGER_DIST  = 3.0    # m — front ray below this + nearly stopped = instant recovery

U_TURN_TRIGGER     = 2.0    # radians (~115°) — checkpoint is behind, start U-turn
U_TURN_EXIT        = 1.2    # radians (~69°)  — checkpoint is ahead enough, hand back to MLP
U_TURN_FWD_CLEAR   = 5.0    # m — if front ray is beyond this, go forward during U-turn

ESCAPE_TRIGGER     = 3      # consecutive failed recoveries before escape mode
ESCAPE_FRAMES      = 80     # total frames of escape maneuver (4 s at 20 Hz)
ESCAPE_BURST       = 10     # frames per forward/reverse burst inside escape


def make_policy(weights_path: str):
    """Return a stateful policy function with multi-level stuck-recovery logic.

    Required by benchmark.py's --module hook.
    """
    w = nn_mod.load(weights_path)

    stuck_count        = [0]
    recovery_left      = [0]
    recovery_steer     = [0.0]
    cooldown_left      = [0]

    in_uturn           = [False]
    uturn_steer        = [0.0]

    # Escape mode state
    escape_left        = [0]
    escape_invocation  = [0]   # parity flips steer direction each new escape
    recovery_attempts  = [0]   # consecutive recoveries with no checkpoint progress
    cp_at_last_recovery = [0]  # checkpoint count when last recovery started

    def policy(state):
        sensors       = state["sensors"]
        speed         = sensors.get("speed", 0.0)
        heading_error = sensors.get("heading_error", 0.0)
        rays          = sensors.get("rays", [50.0] * 8)

        nav    = sensors.get("navigation") or {}
        curr_cp = nav.get("checkpoints_completed", 0) or 0

        ray_front = rays[0] if len(rays) > 0 else 50.0
        ray_left  = rays[2] if len(rays) > 2 else 50.0
        ray_right = rays[6] if len(rays) > 6 else 50.0

        # ── Level-2: escape mode ─────────────────────────────────────────
        if escape_left[0] > 0:
            escape_left[0] -= 1
            phase = (escape_left[0] // ESCAPE_BURST) % 2
            # Alternate steer direction each invocation so we don't always wiggle the same way
            base_steer = 1.0 if escape_invocation[0] % 2 == 0 else -1.0
            if phase == 0:
                return 0.8, base_steer
            else:
                return -0.8, -base_steer

        # ── Cooldown: suppress stuck detection after recovery ────────────
        if cooldown_left[0] > 0:
            cooldown_left[0] -= 1

        # ── Stuck counter — skip during U-turn and cooldown ─────────────
        if not in_uturn[0] and cooldown_left[0] == 0:
            if speed < STUCK_SPEED:
                stuck_count[0] += 1
            else:
                stuck_count[0] = 0

        # ── Level-1: recovery trigger ────────────────────────────────────
        wall_stuck = ray_front < WALL_TRIGGER_DIST and speed < STUCK_SPEED
        time_stuck = stuck_count[0] >= STUCK_THRESHOLD

        if (wall_stuck or time_stuck) and recovery_left[0] == 0 and cooldown_left[0] == 0:
            print(f"  [agent] recovery triggered — front={ray_front:.1f}m  speed={speed:.2f}")

            if curr_cp > cp_at_last_recovery[0]:
                recovery_attempts[0] = 0
            else:
                recovery_attempts[0] += 1

            cp_at_last_recovery[0] = curr_cp
            stuck_count[0] = 0

            if recovery_attempts[0] >= ESCAPE_TRIGGER:
                print(f"  [agent] ESCAPE MODE — {recovery_attempts[0]} failed recoveries")
                recovery_attempts[0] = 0
                escape_invocation[0] += 1
                escape_left[0] = ESCAPE_FRAMES
                return 0.8, (1.0 if ray_left >= ray_right else -1.0)

            recovery_left[0]  = RECOVERY_FRAMES
            recovery_steer[0] = 1.0 if ray_left >= ray_right else -1.0

        # ── Level-1: recovery maneuver ───────────────────────────────────
        if recovery_left[0] > 0:
            recovery_left[0] -= 1
            if recovery_left[0] == 0:
                cooldown_left[0] = RECOVERY_COOLDOWN
            if ray_front > WALL_TRIGGER_DIST:
                steer = float(np.clip(heading_error / np.pi, -1.0, 1.0))
                return 1.0, steer
            return -1.0, recovery_steer[0]

        # ── U-turn: checkpoint is behind ────────────────────────────────
        if abs(heading_error) > U_TURN_TRIGGER and not in_uturn[0]:
            in_uturn[0]    = True
            uturn_steer[0] = -1.0 if heading_error > 0 else 1.0
            print(f"  [agent] U-turn — steer={'left' if uturn_steer[0] < 0 else 'right'}")

        if in_uturn[0]:
            if abs(heading_error) < U_TURN_EXIT:
                in_uturn[0]    = False
                stuck_count[0] = 0  # don't carry accumulated slow frames into recovery
                print(f"  [agent] U-turn done — heading_error={heading_error:.2f}")
            else:
                throttle = 0.8 if ray_front > U_TURN_FWD_CLEAR else -1.0
                return throttle, uturn_steer[0]

        # ── Normal MLP control ───────────────────────────────────────────
        x = sensors_to_input(sensors)
        return clip_action(nn_mod.forward(x, w))

    return policy
