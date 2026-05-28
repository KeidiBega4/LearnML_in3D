"""Simple stuck-recovery agent.

Rules:
  1. Front ray low + slow  → wall hit, reverse + hard steer away
  2. Speed near zero, no wall → boost throttle forward
  3. Still stuck after boost  → reverse 1.25 s

Everything else is MLP.

Usage:
    python 03_benchmark.py --tag v1 --weights nav_v1.npz --module drive2win.agent
"""
from __future__ import annotations
import numpy as np

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action

OSCILLATION_WINDOW  = 140   # frames to look back (7 s at 20 Hz)
OSCILLATION_DIST    = 1.5   # m — net displacement must be less than this = oscillating
OSCILLATION_SAMPLE  = 20    # record position every N frames
OSCILLATION_MIN_PTS = 6     # minimum history points before triggering

WALL_DIST         = 2.0   # m — front ray below this = wall
WALL_SPEED        = 1.0   # m/s — must also be slow to count as wall hit
TERRAIN_SPEED     = 0.2   # m/s — below this with clear front = terrain stuck
TERRAIN_FRAMES    = 20    # frames stopped before boost kicks in (1.0 s at 20 Hz)
BOOST_FRAMES      = 20    # frames of forward boost before giving up (1.0 s)
REVERSE_FRAMES    = 25    # frames to reverse (1.25 s at 20 Hz)


def make_policy(weights_path: str):
    w = nn_mod.load(weights_path)

    terrain_count  = [0]   # frames stopped with clear front
    boost_left     = [0]   # frames of boost remaining
    reverse_left   = [0]   # frames of reverse remaining
    reverse_steer  = [0.0]
    frame_count    = [0]
    pos_history    = []    # (x, z) sampled every OSCILLATION_SAMPLE frames

    MODE_NORMAL  = 0
    MODE_WALL    = 1
    MODE_BOOST   = 2
    MODE_REVERSE = 3
    mode = [MODE_NORMAL]

    def policy(state):
        sensors   = state["sensors"]
        speed     = sensors.get("speed", 0.0)
        rays      = sensors.get("rays", [50.0] * 8)
        ray_front = rays[0] if len(rays) > 0 else 50.0
        ray_left  = rays[2] if len(rays) > 2 else 50.0
        ray_right = rays[6] if len(rays) > 6 else 50.0

        wall_hit    = ray_front < WALL_DIST and speed < WALL_SPEED
        front_clear = ray_front >= WALL_DIST
        frame_count[0] += 1

        # Record position every N frames to detect oscillation
        pos = state.get("position") if isinstance(state, dict) else None
        if pos and frame_count[0] % OSCILLATION_SAMPLE == 0:
            pos_history.append((pos.get("x", 0.0), pos.get("z", 0.0)))
            max_history = OSCILLATION_WINDOW // OSCILLATION_SAMPLE
            if len(pos_history) > max_history:
                pos_history.pop(0)

        # Oscillation check — robot moving but not going anywhere
        oscillating = False
        if len(pos_history) >= OSCILLATION_MIN_PTS and not wall_hit:
            xs = [p[0] for p in pos_history]
            zs = [p[1] for p in pos_history]
            net = np.sqrt((xs[-1]-xs[0])**2 + (zs[-1]-zs[0])**2)
            if net < OSCILLATION_DIST:
                oscillating = True
                print(f"  [agent] oscillating — net displacement {net:.1f}m over {len(pos_history)}s")

        # ── WALL HIT — reverse immediately, steer hard away ──────────────────
        if wall_hit and mode[0] != MODE_WALL and reverse_left[0] == 0:
            steer = 0.8 if ray_left >= ray_right else -0.8
            print(f"  [agent] wall hit front={ray_front:.1f}m — reversing steer={steer:+.1f}")
            mode[0]          = MODE_WALL
            reverse_left[0]  = REVERSE_FRAMES
            reverse_steer[0] = steer
            terrain_count[0] = 0
            boost_left[0]    = 0

        # ── REVERSE (wall or terrain fallback) ───────────────────────────────
        if reverse_left[0] > 0:
            reverse_left[0] -= 1
            if reverse_left[0] == 0:
                mode[0] = MODE_NORMAL
            return -1.0, reverse_steer[0]

        # ── TERRAIN — stopped but no wall ────────────────────────────────────
        if front_clear and speed < TERRAIN_SPEED:
            terrain_count[0] += 1
        else:
            terrain_count[0] = 0

        if terrain_count[0] >= TERRAIN_FRAMES and mode[0] == MODE_NORMAL:
            print(f"  [agent] terrain stuck — boosting forward")
            mode[0]          = MODE_BOOST
            boost_left[0]    = BOOST_FRAMES
            terrain_count[0] = 0

        # ── BOOST — push forward, if still stuck then reverse ────────────────
        if boost_left[0] > 0:
            boost_left[0] -= 1
            if speed > TERRAIN_SPEED:                # worked — back to normal
                boost_left[0] = 0
                mode[0] = MODE_NORMAL
                print(f"  [agent] boost worked")
            elif boost_left[0] == 0:                 # still stuck — reverse
                steer = 0.8 if ray_left >= ray_right else -0.8
                print(f"  [agent] boost failed — reversing")
                reverse_left[0]  = REVERSE_FRAMES
                reverse_steer[0] = steer
            else:
                return 1.0, 0.0                      # full throttle straight

        # ── NORMAL MLP ────────────────────────────────────────────────────────
        mode[0] = MODE_NORMAL
        x = sensors_to_input(sensors)
        throttle, steer = clip_action(nn_mod.forward(x, w))

        # If oscillating in open space — reverse and steer toward checkpoint
        if oscillating and front_clear and reverse_left[0] == 0:
            heading_error  = sensors.get("heading_error", 0.0)
            # heading_error > 0 = checkpoint is left → steer left (-1)
            # heading_error < 0 = checkpoint is right → steer right (+1)
            cp_steer = -1.0 if heading_error > 0 else 1.0
            print(f"  [agent] oscillating — reversing toward checkpoint (steer={cp_steer:+.1f})")
            reverse_left[0]  = REVERSE_FRAMES
            reverse_steer[0] = cp_steer
            pos_history.clear()

        return throttle, steer

    return policy
