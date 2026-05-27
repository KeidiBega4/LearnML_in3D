"""Phase-1 reward-weighted imitation loop.

Each iteration:
  1. Run --runs autonomous episodes with the current weights, collecting
     per-step (state, action) pairs via run_policy's on_step hook.
  2. Score each run: checkpoints * 10 + mean_speed * 2 - crashes * 5.
  3. Retrain on all collected samples weighted by their run's score.
     Runs with zero or negative score are excluded from training.
  4. Save the new weights and append metrics to benchmarks/<tag>.json.

Usage:
    python 04_rl_train.py --weights nav_combinedv1v2v3v4.npz --tag rl_v1
    python 04_rl_train.py --weights nav_combinedv1v2v3v4.npz --tag rl_v1 --iterations 10 --runs 8
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from game_client import GameClient
from drive2win import nn as nn_mod
from drive2win import agent as agent_mod
from drive2win.eval import run_policy, score_runs
from drive2win.normalize import sensors_to_input, clip_action, normalize_states

SERVER_URL          = "https://ml.ferit.tech"
API_KEY             = "None"
TARGET_CHECKPOINTS  = 8


# ── Data collection ──────────────────────────────────────────────────────────

_TEMP_WEIGHTS = "nav_rl_temp.npz"


def collect_episode(client, w, duration, player_name):
    """Run one autonomous episode and return per-step data + run summary.

    Uses the full agent policy (with stuck-recovery) so the robot doesn't
    sit against a wall for the entire episode producing useless data.
    """
    # Save current weights so agent.make_policy can load them
    nn_mod.save(w, _TEMP_WEIGHTS)
    policy = agent_mod.make_policy(_TEMP_WEIGHTS)

    steps_data: list[tuple[np.ndarray, np.ndarray]] = []

    def on_step(_step, state, action):
        sensors = state["sensors"]
        raw = np.array([
            sensors.get("speed", 0.0),
            sensors.get("heading_error", 0.0),
            sensors.get("checkpoint_distance", 100.0),
            *sensors.get("rays", [50.0] * 8)[:8],
            sensors.get("ground_friction", 1.0),
        ], dtype=np.float32)
        steps_data.append((raw, np.array(action, dtype=np.float32)))

    result = run_policy(client, policy, duration=duration, hz=20.0, on_step=on_step)
    return steps_data, result


# ── Reward ───────────────────────────────────────────────────────────────────

def compute_reward(result: dict, steps_data: list) -> float:
    cp      = result["checkpoints_passed"]
    crashes = result["crashes"]
    # mean_speed from the collected step data (column 0 = speed, raw m/s)
    mean_spd = float(np.mean([s[0][0] for s in steps_data])) if steps_data else 0.0
    return cp * 10.0 + mean_spd * 2.0 - crashes * 5.0


# ── Weighted training ─────────────────────────────────────────────────────────

def _backward_weighted(x, y_target, w, cache, sample_weights):
    """Standard backprop with per-sample weight on the MSE gradient."""
    n  = x.shape[0]
    y  = cache["y"]
    dy = 2.0 * (y - y_target) / (n * y.shape[1])
    dy = dy * sample_weights[:, None]   # scale each sample

    dz3 = dy * (1.0 - y * y)
    dW3 = cache["a2"].T @ dz3;  db3 = dz3.sum(axis=0)
    da2 = dz3 @ w["W3"].T
    dz2 = da2 * (cache["z2"] > 0)
    dW2 = cache["a1"].T @ dz2;  db2 = dz2.sum(axis=0)
    da1 = dz2 @ w["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    dW1 = x.T @ dz1;            db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def train_weighted(X, Y, W, w_init, epochs=100, lr=1e-3, batch=64, seed=0):
    """Retrain from w_init on (X, Y) with per-sample weights W."""
    rng  = np.random.default_rng(seed)
    w    = {k: v.copy() for k, v in w_init.items()}
    adam = nn_mod.init_adam(w)
    N    = len(X)

    # Normalise weights so mean=1 — keeps LR comparable across iterations
    W = W / (W.mean() + 1e-8)

    best_loss = float("inf")
    best_w    = {k: v.copy() for k, v in w.items()}

    for epoch in range(epochs):
        idx     = rng.permutation(N)
        Xs, Ys, Ws = X[idx], Y[idx], W[idx]
        ep_loss, n_b = 0.0, 0
        for i in range(0, N, batch):
            xb = Xs[i:i+batch]; yb = Ys[i:i+batch]; wb = Ws[i:i+batch]
            cache    = nn_mod.forward_all(xb, w)
            ep_loss += nn_mod.mse_loss(cache["y"], yb); n_b += 1
            grads    = _backward_weighted(xb, yb, w, cache, wb)
            nn_mod.adam_step(w, grads, adam, lr=lr)
        loss = ep_loss / max(1, n_b)
        if loss < best_loss:
            best_loss = loss
            best_w    = {k: v.copy() for k, v in w.items()}
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"      epoch {epoch:3d}  loss={loss:.4f}  best={best_loss:.4f}")

    return best_w


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase-1 reward-weighted imitation")
    ap.add_argument("--weights",    required=True, help="Starting .npz weights")
    ap.add_argument("--tag",        default="rl_v1", help="Output tag")
    ap.add_argument("--iterations", type=int,   default=5)
    ap.add_argument("--runs",       type=int,   default=5,    help="Episodes per iteration")
    ap.add_argument("--epochs",     type=int,   default=100,  help="Training epochs per iteration")
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--duration",   type=float, default=60.0, help="Seconds per episode")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--server",     default=SERVER_URL)
    ap.add_argument("--api-key",    default=API_KEY)
    args = ap.parse_args()

    Path("benchmarks").mkdir(exist_ok=True)
    log_path = Path(f"benchmarks/{args.tag}.json")

    weights = nn_mod.load(args.weights)
    log     = []

    for it in range(1, args.iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"  Iteration {it}/{args.iterations}   tag={args.tag}")
        print(f"{'=' * 60}")

        client = GameClient(args.server, args.api_key)
        all_states, all_actions, all_w = [], [], []
        run_results = []

        for r in range(args.runs):
            session = client.create_session(
                mode="time_trial",
                player_name=f"rl_{args.tag}_it{it}_r{r+1}",
                config={"seed": args.seed, "wind_enabled": False, "obstacles_enabled": True},
            )
            client.connect_ws()
            try:
                client.configure(obstacles_enabled=True)
            except Exception as e:
                print(f"  [warn] configure failed ({e})")
            time.sleep(0.6)

            print(f"\n  run {r+1}/{args.runs}  session={session['session_id'][:8]}…")
            steps_data, result = collect_episode(
                client, weights, args.duration,
                player_name=f"rl_{args.tag}_it{it}_r{r+1}",
            )

            reward = compute_reward(result, steps_data)
            print(f"    checkpoints={result['checkpoints_passed']}/{TARGET_CHECKPOINTS}"
                  f"  crashes={result['crashes']}"
                  f"  reward={reward:.1f}"
                  f"  steps={len(steps_data)}")

            run_results.append(result)

            if reward > 0 and steps_data:
                raw  = np.stack([s for s, _ in steps_data])
                acts = np.stack([a for _, a in steps_data])
                all_states.append(raw)
                all_actions.append(acts)
                all_w.append(np.full(len(steps_data), reward, dtype=np.float32))

            client.disconnect_ws()
            try: client.delete_session()
            except Exception: pass

        summary = score_runs(run_results, TARGET_CHECKPOINTS)
        print(f"\n  --- iteration {it} summary ---")
        print(f"  max_checkpoints : {summary['max_checkpoints']}/{TARGET_CHECKPOINTS}")
        print(f"  completion_rate : {int(summary['completion_rate'] * summary['n_runs'])}/{summary['n_runs']}")
        print(f"  mean_crashes    : {summary['mean_crashes']:.1f}")

        if not all_states:
            print("\n  No positive-reward runs — skipping retrain.")
            print("  Tip: try more --runs, a longer --duration, or reduce --seed variance.")
            log.append({"iteration": it, "summary": summary, "n_steps": 0, "retrained": False})
            log_path.write_text(json.dumps(log, indent=2))
            continue

        X = normalize_states(np.concatenate(all_states, axis=0))
        Y = np.concatenate(all_actions, axis=0)
        W = np.concatenate(all_w, axis=0)

        print(f"\n  Retraining on {len(X):,} steps from"
              f" {len(all_states)} run(s)  ({args.epochs} epochs)…")
        weights = train_weighted(X, Y, W, weights,
                                 epochs=args.epochs, lr=args.lr, seed=it)

        out = f"nav_{args.tag}_it{it}.npz"
        nn_mod.save(weights, out)
        print(f"  Saved {out}")

        log.append({"iteration": it, "summary": summary,
                    "n_steps": len(X), "retrained": True})
        log_path.write_text(json.dumps(log, indent=2))

    print(f"\nDone. Final weights: nav_{args.tag}_it{args.iterations}.npz")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
