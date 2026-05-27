"""Combine multiple .npz datasets into one.

Usage:
    python combine_data.py data_v1.npz data_v2.npz data_v3.npz
    python combine_data.py data_v1.npz data_v2.npz --out data_combined.npz
    python combine_data.py data_v1.npz data_v2.npz --mirror --out data_combined_mirrored.npz
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

# Column indices (must match normalize.py FEATURE_NAMES order)
IDX_HEADING   = 1
IDX_RAY_P45   = 4;  IDX_RAY_M45  = 10
IDX_RAY_P90   = 5;  IDX_RAY_M90  = 9
IDX_RAY_P135  = 6;  IDX_RAY_M135 = 8
IDX_STEERING  = 1   # action column


def mirror_samples(states: np.ndarray, actions: np.ndarray):
    """Flip every sample left-right to double the dataset and balance turns."""
    s = states.copy()
    a = actions.copy()
    s[:, IDX_HEADING] *= -1
    s[:, IDX_RAY_P45],  s[:, IDX_RAY_M45]  = states[:, IDX_RAY_M45].copy(),  states[:, IDX_RAY_P45].copy()
    s[:, IDX_RAY_P90],  s[:, IDX_RAY_M90]  = states[:, IDX_RAY_M90].copy(),  states[:, IDX_RAY_P90].copy()
    s[:, IDX_RAY_P135], s[:, IDX_RAY_M135] = states[:, IDX_RAY_M135].copy(), states[:, IDX_RAY_P135].copy()
    a[:, IDX_STEERING] *= -1
    return s, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Two or more .npz files to merge")
    ap.add_argument("--out", default="data_combined.npz",
                    help="Output file name (default: data_combined.npz)")
    ap.add_argument("--mirror", action="store_true",
                    help="Append a left-right mirrored copy of every sample")
    args = ap.parse_args()

    if len(args.inputs) < 2 and not args.mirror:
        ap.error("Provide at least two input files (or one file with --mirror).")

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

    states_combined  = pd.concat(state_frames,  ignore_index=True).to_numpy(dtype=np.float32)
    actions_combined = pd.concat(action_frames, ignore_index=True).to_numpy(dtype=np.float32)

    if args.mirror:
        s_mir, a_mir = mirror_samples(states_combined, actions_combined)
        states_combined  = np.concatenate([states_combined,  s_mir],  axis=0)
        actions_combined = np.concatenate([actions_combined, a_mir], axis=0)
        left  = (actions_combined[:, IDX_STEERING] < -0.1).sum()
        right = (actions_combined[:, IDX_STEERING] >  0.1).sum()
        print(f"\nMirroring applied — left turns: {left:,}  right turns: {right:,}")

    print(f"\nCombined: {len(states_combined):,} total samples from {len(args.inputs)} files")
    print("\nFeature summary:")
    df = pd.DataFrame(states_combined, columns=FEATURE_NAMES)
    print(df.describe().loc[["mean", "std", "min", "max"]].to_string())

    np.savez(
        args.out,
        states=states_combined,
        actions=actions_combined,
    )
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
