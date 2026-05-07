# Drive2Win — Final Project

Train a neural network to drive the simulation's standard bot autonomously, then race it against your classmates' bots in a live tournament.

This is your only project from Week 10 to the end of the course. There are no separate weekly labs. **You work at your own pace.** What you submit is not a single "final model" — it is a *trail* of iterations: each one collected data or changed a model or fixed a bug, each one is benchmarked, each one is committed with a note explaining what you changed and why.

> Week 9's behavioral-cloning lab was a tutorial. **This project is from scratch** — your data, your network, your iteration loop, your grade.

---

## How you are graded

The grade is **50% process + 50% final performance**.

### 50% — process (graded from your git history and `benchmarks/` folder)

Your process grade is based on **what's in `benchmarks/`** and **what's in your git log**.

- Every iteration must produce a `benchmarks/<tag>.json` log and the matching PNG figures (the script `03_benchmark.py` does this for you — see below).
- Every iteration must be a commit with a message that says **what changed and why**. "v3-deepnet: deeper net, predicted +5% completion" is a good message. "updated stuff" is not.
- The instructor will read your git log front to back. Visible improvement curves and clear hypotheses score higher than a single lucky model.

A reasonable trail has **6–10 iterations**. Three iterations is too few. Twenty trivial commits without changing anything is also too few.

### 50% — final performance (graded from the live tournament)

A single number: how many checkpoints your bot passes in the tournament rounds (described below).

### Plus a tournament bonus on your overall course grade

| Place | Bonus |
|-------|-------|
| 1st | **+10%** to overall course grade |
| 2nd | **+5%** |
| 3rd | **+2%** |

---

## The tournament

On the final day, all 20 students compete simultaneously.

- **5 rounds × 5 minutes each.** The terrain seed changes between rounds.
- **3 rounds without obstacles, 2 rounds with obstacles.**
- **Pass / fail bar:** completing **one full lap** in any round is a pass.
- **Ranking:** total checkpoints passed across all 5 rounds.

Why this format matters for *how you train*:

- The terrain shifts every round. A model overfit to one specific map will drive itself into a wall on the next one. **Test your model on multiple `--seeds` before the tournament**, not just `seed 42`.
- Two rounds have obstacles. If you skipped recording obstacle-driving data, your bot will not pass those rounds. Plan for it.
- Five minutes is long. Stuck-against-a-wall is a 5-minute-long mistake. Recovery driving in your dataset matters more than smoothness.

---

## The unifying metric — `benchmark.py`

Every iteration is measured the same way. `03_benchmark.py` calls into the canonical evaluator and writes a JSON log + path PNGs to `benchmarks/`.

```
python 03_benchmark.py --tag v3-deepnet --seeds 42 7 99
```

What it reports per seed:

```
seed   42  complete=4/5  median_lap=51.2s  crashes=0.8  max_cp=8
seed    7  complete=2/5  median_lap=58.0s  crashes=1.4  max_cp=8
seed   99  complete=0/5  median_lap=—      crashes=2.6  max_cp=5
```

> **Do not edit `benchmark.py`.** It defines the comparison. If everyone uses a different evaluator, the leaderboard is meaningless.

---

## What ships in this folder

```
LearnML_in3D/
├── README.md                    ← this file
├── INSTRUCTOR_TODO.md           ← platform changes the project assumes
├── game_client.py               ← Python SDK for talking to the simulation server
├── 01_collect.py                ← drive, save data_<tag>.npz (with positions)
├── 02_train.py                  ← implement my_backward(), train, save nav_<tag>.npz
├── 03_benchmark.py              ← run benchmark across seeds, log to benchmarks/
├── 04_compare.py                ← table + plot across all your iterations
├── drive2win/                   ← the package your code lives in (grow it!)
│   ├── benchmark.py             — canonical evaluator (DO NOT EDIT)
│   ├── normalize.py             — input/output scaling. Single source of truth.
│   ├── nn.py                    — MLP forward pass + Adam. backward() is filled in
│   │                              for use by later iterations; the version YOU
│   │                              hand in for grading lives in 02_train.py.
│   ├── eval.py                  — run_policy + score_runs (used by benchmark)
│   └── viz.py                   — every plot you'll need (path overlays, action
│                                  histograms, loss curves, iteration history)
└── benchmarks/                  ← your iteration log (committed to git!)
    └── README.md                — naming and what each file is
```

You add files (new architectures, new training scripts, fix-ups) inside `drive2win/` as you iterate.

---

## Setup — once

```
pip install numpy matplotlib scikit-learn torch requests websocket-client
```

From the repo root:

```python
from game_client import GameClient
from drive2win.benchmark import run_benchmark
from drive2win import nn, viz, normalize
```

Run all scripts from the repo root (the directory that contains `drive2win/`).

---

## The iteration loop

Your project is one loop, run many times. Every pass through the loop is one commit, one `benchmarks/<tag>.json`, one slightly better (or sometimes worse — that's information too) model.

```
        ┌──────── 01_collect.py ──────────┐
        │                                  ▼
        │                              data_<tag>.npz
        │                                  │
        │                                  ▼
        │                          02_train.py  ──── my_backward()
        │                                  │
        │                                  ▼
        │                              nav_<tag>.npz
        │                                  │
        │                                  ▼
        │                          03_benchmark.py
        │                                  │
        │                                  ▼
        │                  benchmarks/<tag>.json + PNGs
        │                                  │
        │                                  ▼
        │                       look at the figures.
        │                       write down what failed.
        │                       form ONE hypothesis.
        │                                  │
        └──────────  pick a change to try  ┘
```

### The first iteration (everyone does this)

1. **Collect data.** `python 01_collect.py --tag v1 --seed 42` — five phases, ~6 minutes of careful driving including walls-and-recover. Output: `data_v1.npz`.
2. **Implement backprop.** Open `scripts/02_train.py`, replace each `...` in `my_backward()` with the right chain-rule expression. The script gradient-checks your code before training; if any param's max relative error is ≥ 1e-4, the assertion fires and you fix the bug.
3. **Train.** `python 02_train.py --data data_v1.npz --tag v1` — 300 epochs, Adam, batch 64, lr 1e-3. Output: `nav_v1.npz` plus `fig_loss_v1.png`, `fig_actions_v1.png`, `fig_heading_v1.png`.
4. **Benchmark.** `python 03_benchmark.py --tag v1 --data data_v1.npz` — 5 runs on seed 42. Output: `benchmarks/v1.json`, `v1_paths.png`, `v1_progress.png`, `v1_overlay.png`.
5. **Commit.** `git add data_v1.npz nav_v1.npz benchmarks/v1.* fig_*_v1.png && git commit -m "v1-bc: baseline behavioral cloning, completion X/5"`.

A typical first iteration: completion 1–2 / 5, median lap ~55 s, crashes 1–2.

### Iterations 2 through N — pick *one* hypothesis at a time

Now look at `v1_paths.png` and `v1_overlay.png`. **Where does the model fail?** Pick one thing, change it, retrain, re-benchmark.

You only learn from an iteration if you can say *what you predicted* and *what actually happened*. Change one thing per iteration so you know which change moved the needle.

#### Things students try in this project

These are not weeks. There is no required order. Pick whichever you think will help most given what you saw in the last iteration.

| Idea | What you change |
|------|-----------------|
| **Better data — recovery** | Re-record `01_collect.py` with more wall-recovery samples. The single most-effective fix in this project. |
| **Better data — DAgger-lite** | Watch your bot fail. Take over with WASD at the failure point, drive correct actions, save those frames, append to your dataset, retrain. |
| **Deeper / wider network** | Edit `drive2win/nn.py` to e.g. 12→128→64→32→2 (update H1, H2 and `init_weights`, then update `forward`/`forward_all`/`backward`). |
| **Different activation** | Try LeakyReLU instead of ReLU at hidden layers. Hint: only the activation derivative changes in `backward()`. |
| **Action smoothing** | Don't predict raw `(throttle, steering)` — predict the **delta** from the previous action, or low-pass filter the output at inference. |
| **Different normalization** | Edit `normalize.py`. e.g. divide rays by their per-channel std rather than `RAY_MAX`. |
| **CNN on the 32×32 terrain grid** | Add `drive2win/cnn.py` (PyTorch). Expose `make_policy(weights_path)`, then `--module drive2win.cnn` to benchmark. Needs `RecordingSystem` to capture `grid32` — see `INSTRUCTOR_TODO.md`. |
| **Hybrid CNN + MLP** | Concatenate the CNN features with the 12-vector before the final FC. Almost always beats either alone on obstacle rounds. |
| **sklearn `Pipeline`** | Wrap normalization + model in a pipeline so train and inference agree by construction. |
| **Ensemble** | Train two seeds of the same model, average their actions at inference. |
| **Test on multiple seeds early** | If your `seed=42` numbers look great but `seed=7` is dead, you don't have a model — you have a memorized map. The tournament will shred it. |

If you find yourself thinking *"none of these are interesting"*, look at `_history.png` and pick whichever is currently your worst metric. If completion is fine but crashes are high, you have a smoothness problem. If completion is low across all seeds, you have a coverage problem.

---

## Visualization — see what the model is doing

The hardest mistake in this project is **iterating with your eyes closed**. Numbers go down, you change something, numbers go up, you don't know why.

`drive2win/viz.py` exists so you don't have to. Every figure below is one function call.

| Function | When to look at it |
|----------|--------------------|
| `plot_path` / `plot_multi_run_paths` | Where did the bot drive in this run / across runs. |
| `plot_path_overlay` | Where YOU drove vs where the NN drove. The single most-revealing plot. Looks great when they overlap; tells you the data was thin where they diverge. |
| `plot_action_histograms` | Are your demonstrations symmetric, or did you only ever turn right? |
| `plot_heading_vs_steering` | Should slope downward. If it doesn't, your network can't learn to navigate from this data. |
| `plot_loss_curves` | Train + val. If val rises while train falls, you're overfit. |
| `plot_speed_profile` | Where is the bot stuck (speed near 0)? Often it's one specific stretch. |
| `plot_checkpoint_progress` | 5 bars, one per run. Variance tells you if the model is consistent or just lucky. |
| `plot_iteration_history` | All iterations side by side. Run `04_compare.py` to produce it. |

Use these in your iteration commit messages. A figure pasted into a commit body says more than five paragraphs.

---

## House rules

- **Your training data must come from your own driving.** No swapping recordings. The whole point is that *your* network learns from *your* hands.
- **No external pretrained models.** PyTorch is fine; copying weights from elsewhere is not.
- **Do not edit `benchmark.py`.** Same reason as above.
- **Commit per iteration.** Even if you don't push to GitHub, a local commit per iteration is what gets graded.
- **Test on more than one seed before the tournament.** Five rounds, terrain changes each time, two with obstacles. If `seed=42` is the only number you've ever seen, you have not tested yet.

---

## Quick start

```
# install deps
pip install numpy matplotlib scikit-learn torch requests websocket-client

# iteration 1
python 01_collect.py     --tag v1 --seed 42
python 02_train.py       --data data_v1.npz --tag v1
python 03_benchmark.py   --tag v1 --data data_v1.npz
git add data_v1.npz nav_v1.npz benchmarks/v1.* fig_*_v1.png
git commit -m "v1-bc: baseline behavioral cloning"

# look at benchmarks/v1_paths.png and benchmarks/v1_overlay.png.
# decide one thing to change.
# iteration 2 ...
```

Open `scripts/02_train.py`, find the `my_backward()` TODO, and start.
