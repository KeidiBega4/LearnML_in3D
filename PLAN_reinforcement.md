# Plan: Adding Reward Signals to the Robot

## Background

The robot currently learns by **behavior cloning** (imitation learning):
- A human drives and records `(state, action)` pairs
- The network is trained with MSE loss to copy those actions
- It has no concept of "good" or "bad" — it just mimics

This plan introduces **reward signals** so the robot can learn from what it does right or wrong,
not just from what the human did.

---

## Phase 1 — Reward-Weighted Imitation (Low Risk, Fast Results)

Keep the existing MLP + Adam + behavior-cloning pipeline intact.
Add a reward score per run and retrain with those weights.

### How it works

1. Run the robot on the track (already done via `03_benchmark.py`)
2. For each run, compute a scalar reward:
   ```
   reward = checkpoints_reached * 10
           + mean_speed * 2
           - crashes * 5
           - time_stuck_frames * 0.1
   ```
3. When retraining, multiply each sample's loss by the reward of the run it came from
4. Samples from good runs pull the weights harder; bad runs barely move them
5. Repeat: benchmark → score → retrain → benchmark

### Files to change

| File | Change |
|------|--------|
| `drive2win/benchmark.py` | Return per-step sensor+action log alongside the summary |
| `02_train.py` | Accept a `sample_weights` array; multiply MSE loss element-wise |
| New: `04_rl_train.py` | Orchestrate the benchmark → score → retrain loop |

### Reward function (starting point)

```python
def compute_reward(run: dict) -> float:
    cp     = run["checkpoints_reached"]
    speed  = run["mean_speed"]          # already in benchmark summary
    crash  = run["crashes"]
    return cp * 10.0 + speed * 2.0 - crash * 5.0
```

Tune the weights here first before touching anything else.

### Success looks like

- Runs with more checkpoints become the dominant training signal
- The robot gradually stops crashing as much without any new human data
- Benchmark `max_checkpoints` goes up over iterations

---

## Phase 2 — Online Reward Shaping (Medium Effort)

Add **per-step** reward signals during a live run so the robot gets immediate feedback,
not just end-of-run scores.

### Per-step signals available from `sensors`

| Signal | Reward idea |
|--------|-------------|
| `sensors["speed"]` | +speed bonus every frame |
| `sensors["rays"][0]` (front ray) | -penalty when very short (near wall) |
| `sensors["heading_error"]` | -penalty for large heading error |
| Checkpoint event | +big bonus on each checkpoint |
| Collision / stuck | -penalty (speed < threshold for N frames) |

### Implementation sketch

```python
def step_reward(sensors: dict, prev_cp: int, curr_cp: int) -> float:
    speed        = sensors.get("speed", 0.0)
    front_ray    = sensors.get("rays", [50.0])[0]
    heading_err  = abs(sensors.get("heading_error", 0.0))
    new_cp       = curr_cp - prev_cp

    r  = speed * 0.1                          # reward forward motion
    r += new_cp * 10.0                        # big bonus per checkpoint
    r -= max(0, 3.0 - front_ray) * 0.5       # penalty for being near walls
    r -= heading_err * 0.05                   # penalty for misalignment
    return r
```

### Files to change

| File | Change |
|------|--------|
| `drive2win/benchmark.py` | Yield per-step `(state, action, reward)` tuples |
| `04_rl_train.py` | Accumulate steps, apply discounting (γ = 0.95), then retrain |

---

## Phase 3 — True Reinforcement Learning (REINFORCE)

Replace behavior cloning with a proper policy gradient algorithm.
The robot generates its own data entirely — no human demos needed.

### How REINFORCE works

1. Run one full episode with the current policy
2. Collect trajectory: `[(s0, a0), (s1, a1), ..., (sT, aT)]`
3. Compute discounted return at each step: `G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...`
4. Update weights to make actions with high return more likely:
   ```
   loss = -log π(a_t | s_t) * G_t    (averaged over all steps)
   ```
5. Repeat thousands of times

### Key difference from current setup

| | Current (Behavior Cloning) | REINFORCE |
|---|---|---|
| Data source | Human demonstrations | Robot's own experience |
| Loss | MSE vs human action | -log_prob × return |
| Output layer | `tanh` → deterministic action | Mean + std → sample action |
| Training trigger | Once after data collection | After every episode |

### Output layer change needed

Switch from deterministic to **stochastic** output (Gaussian policy):
```python
# Current: 2 outputs (throttle, steer) via tanh
# New: 4 outputs — (mean_throttle, mean_steer, log_std_throttle, log_std_steer)
# Sample action: a ~ Normal(mean, exp(log_std))
# log_prob used in REINFORCE loss
```

### Files to change

| File | Change |
|------|--------|
| `drive2win/nn.py` | Add stochastic forward, add `log_prob()` |
| `04_rl_train.py` | Full REINFORCE loop: episode → returns → gradient → update |
| `drive2win/agent.py` | Use stochastic sampling during training, greedy during eval |

### Known challenges

- **High variance**: REINFORCE is noisy. Use a baseline (average return) to reduce it.
- **Exploration**: The robot must try random actions to discover what works.
- **Sample inefficiency**: Needs many episodes. Behavior cloning is much faster to start.
- **Sparse rewards**: If checkpoints are rare, the signal is very weak early on.

---

## Recommended Order

```
Phase 1 (reward-weighted imitation)
  → get reward signal working and visible in benchmark numbers
  → tune reward weights

Phase 2 (per-step shaping)
  → improve feedback granularity
  → helps even without going to true RL

Phase 3 (REINFORCE)
  → only if Phase 1+2 plateau and you want the robot to self-improve
  → significantly more effort and tuning required
```

---

## Quick Reference: Reward Signal Sources

```python
# After every frame (from sensors dict):
speed        = sensors["speed"]           # m/s
front_ray    = sensors["rays"][0]         # metres to nearest wall ahead
heading_err  = sensors["heading_error"]   # radians, 0 = pointing at checkpoint

# After every run (from benchmark summary):
checkpoints  = run["checkpoints_reached"]
crashes      = run["crashes"]
lap_time     = run["lap_time"]            # None if not completed
mean_speed   = run["mean_speed"]
```

---

## Notes

- Start with Phase 1 before any RL. It reuses everything already built.
- The reward function coefficients (10, 2, 5 etc.) need tuning — treat them like hyperparameters.
- Keep `nav_<tag>.npz` snapshots at each iteration so you can roll back.
- A robot that gets a -5 crash penalty will quickly learn that avoiding walls matters more than speed.
