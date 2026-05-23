"""Filter out wall-oscillation samples and combine all datasets.

The back-and-forth wall behavior is caused by training samples where the robot
is facing a wall (front ray short) and reversing (negative throttle). These
samples teach the network: "wall in front = go backward" without learning to
also steer away.

Usage:
    python filter_data.py
    python filter_data.py --out data_clean.npz
    python filter_data.py --front-threshold 8.0 --out data_clean.npz
"""
from __future__ import annotations
import argparse
import numpy as np

# Feature indices (matches normalize.py FEATURE_NAMES)
IDX_RAY_FRONT = 3   # ray_0_front, raw range 0-50m
IDX_THROTTLE  = 0   # action index 0


def filter_oscillation(states, actions, front_threshold=5.0, throttle_threshold=-0.2):
    """Remove samples where robot is backing up into a wall.

    Keeps samples where either:
    - Front ray is clear (robot not near a wall), OR
    - Throttle is positive (robot is driving forward, not reversing)
    """
    ray_front = states[:, IDX_RAY_FRONT]        # raw meters
    throttle  = actions[:, IDX_THROTTLE]

    # Drop: close to wall AND reversing
    bad_mask = (ray_front < front_threshold) & (throttle < throttle_threshold)
    keep = ~bad_mask

    n_before = len(states)
    n_removed = bad_mask.sum()
    print(f"  removed {n_removed:,} / {n_before:,} oscillation samples "
          f"({100 * n_removed / n_before:.1f}%)")
    return states[keep], actions[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+",
                    default=["data_v1.npz", "data_v2.npz", "data_v3.npz",
                             "data_v4.npz", "data_v5.npz"],
                    help="Raw dataset files to combine")
    ap.add_argument("--out", default="data_clean.npz")
    ap.add_argument("--front-threshold", type=float, default=5.0,
                    help="Front ray distance (meters) below which samples are suspect")
    ap.add_argument("--throttle-threshold", type=float, default=-0.2,
                    help="Throttle value below which the robot is considered reversing")
    args = ap.parse_args()

    all_states, all_actions = [], []

    for path in args.inputs:
        try:
            d = np.load(path, allow_pickle=False)
        except FileNotFoundError:
            print(f"  skipping {path} (not found)")
            continue

        states  = d["states"]
        actions = d["actions"]
        print(f"\n{path}: {len(states):,} samples before filter")
        states, actions = filter_oscillation(
            states, actions,
            front_threshold=args.front_threshold,
            throttle_threshold=args.throttle_threshold,
        )
        print(f"  kept {len(states):,} samples")
        all_states.append(states)
        all_actions.append(actions)

    if not all_states:
        print("No data loaded. Check your input files.")
        return

    states_out  = np.concatenate(all_states,  axis=0).astype(np.float32)
    actions_out = np.concatenate(all_actions, axis=0).astype(np.float32)

    print(f"\nFinal dataset: {len(states_out):,} samples")
    print(f"Throttle distribution:")
    print(f"  forward (>0.2) : {(actions_out[:, 0] >  0.2).sum():,}")
    print(f"  neutral        : {((actions_out[:, 0] >= -0.2) & (actions_out[:, 0] <= 0.2)).sum():,}")
    print(f"  reverse (<-0.2): {(actions_out[:, 0] < -0.2).sum():,}")

    np.savez(args.out, states=states_out, actions=actions_out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
