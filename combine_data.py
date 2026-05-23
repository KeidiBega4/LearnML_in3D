"""Combine multiple .npz datasets into one.

Usage:
    python combine_data.py data_v1.npz data_v2.npz data_v3.npz
    python combine_data.py data_v1.npz data_v2.npz --out data_combined.npz
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Two or more .npz files to merge")
    ap.add_argument("--out", default="data_combined.npz",
                    help="Output file name (default: data_combined.npz)")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        ap.error("Provide at least two input files.")

    state_frames, action_frames = [], []
    FEATURE_NAMES = [
        "speed", "heading_error", "checkpoint_distance",
        "ray_front", "ray_+45", "ray_+90", "ray_+135",
        "ray_back", "ray_-135", "ray_-90", "ray_-45",
        "ground_friction",
    ]

    for path in args.inputs:
        d = np.load(path, allow_pickle=False)
        states = d["states"]
        actions = d["actions"]
        assert states.shape[1] == 12, f"{path}: expected 12 state features, got {states.shape[1]}"
        assert actions.shape[1] == 2, f"{path}: expected 2 action columns, got {actions.shape[1]}"

        state_frames.append(pd.DataFrame(states, columns=FEATURE_NAMES))
        action_frames.append(pd.DataFrame(actions, columns=["throttle", "steering"]))
        print(f"  {path}: {len(states):,} samples")

    states_combined = pd.concat(state_frames, ignore_index=True)
    actions_combined = pd.concat(action_frames, ignore_index=True)

    print(f"\nCombined: {len(states_combined):,} total samples from {len(args.inputs)} files")
    print("\nFeature summary:")
    print(states_combined.describe().loc[["mean", "std", "min", "max"]].to_string())

    np.savez(
        args.out,
        states=states_combined.to_numpy(dtype=np.float32),
        actions=actions_combined.to_numpy(dtype=np.float32),
    )
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
