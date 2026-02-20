
# Agri Robot Gridworld

https://medium.com/@evanhammerstein/bioengineering-a-brain-in-the-labor-force-of-the-future-agricultural-robots-via-innovating-74b00b68cc3c

Grid-based farming/fieldwork environment (Gymnasium) with a Pygame viewer, replay recording, and DQN-family baselines:

- **Environment**: `AgriRobotEnv` in `agri_env.py` (action masking, day-based battery, plants/weeds dynamics, optional feature toggles).
- **Viewer**: `main.py` for manual play, replay playback, and GIF export.
- **Agents**: Double DQN / Single DQN, optional **dueling** head, optional **n-step returns**.
- **Experiment tooling**: training runs under `runs/`, replay NPZs under `replays/`, GIFs under `gifs/`.

This repo is set up so you can:
1) train a suite of agents,
2) evaluate them,
3) save replays,
4) export side-by-side GIFs for comparisons.

## Requirements

- Python **3.10+** (uses modern type syntax)
- Dependencies in `requirements.txt` (notably: `gymnasium`, `pygame`, `torch`, `numpy`, `matplotlib`, `imageio`, `Pillow`).

Torch note:

- `main.py` (viewer) will still run if `torch` is missing (agent autoplay is just disabled).
- Training/evaluation scripts (`train_double_dqn.py`, `evaluate_policy.py`, batch eval scripts) **require** `torch`.

## Setup

Create a virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- If installing `torch` via `pip install -r requirements.txt` fails on your platform, install a compatible PyTorch build first, then re-run `pip install -r requirements.txt`.
- If you have a CUDA-capable GPU, install an appropriate PyTorch build for your system.
- On macOS, Pygame usually works via pip, but you may need to update Xcode Command Line Tools if you hit SDL-related issues.

## Quickstart

### 1) Run the interactive viewer

```bash
python main.py
```

Controls are displayed in the bottom UI panel. The important ones:

- **Move**: WASD / arrow keys
- **Spray**: IJKL
- **Plant**: 1/2/3/4
- **Harvest**: 5/6/7/8
- **Refill**: Z (weed), X (fert), C (water)
- **Dropoff**: V
- **Toggle fog**: F
- **End day**: N
- **Toggle agent autoplay** (if a checkpoint is present): P

Replay shortcuts:

- **L**: load `replays/last_episode.npz` (if present)
- **[ / ]**: cycle through `replays/*.npz`
- **Space**: play/pause replay
- **, / .**: step back / forward

### 2) Train a suite of agents

The main training entrypoint is:

```bash
python train_double_dqn.py
```

By default, this runs a **requested** suite and writes results under `runs/<timestamp>/...`.
Each experiment saves:

- `runs/<timestamp>/<experiment>/double_dqn_agri.pt` (final checkpoint)
- `runs/<timestamp>/<experiment>/metrics.csv` + `metrics.npz` + `*.png` plots
- a convenience copy in the project root: `double_dqn_agri_<experiment>.pt`

Common knobs:

```bash
# Fewer episodes, run on GPU if available
python train_double_dqn.py --episodes 400 --device cuda

# Run the smaller "auto" suite (driven by --n-step / --dueling)
python train_double_dqn.py --suite auto --n-step 5 --dueling --double-dqn

# Periodic greedy evaluation during training
python train_double_dqn.py --eval-every 50 --eval-episodes 20
```

Suite notes:

- `--suite requested` (default) runs: `baseline`, `singledqn`, `nstep3`, `nstep5`, `dueling`, `both`.
- `--suite auto` runs a smaller sweep driven by `--n-step`/`--dueling`.
- Use `--experiments baseline,nstep3` to filter which experiments within the chosen suite run.

Replay recording during training:

- Best-replay tracking is **enabled by default** once epsilon drops below a threshold.
- It writes: `replays/<run_id>_<experiment>_best.npz`.

### 3) Evaluate a trained policy

Evaluate greedily over several episodes:

```bash
python evaluate_policy.py --model double_dqn_agri_nstep3.pt --episodes 25 --seed 100 --vary-seed
```

Notes:
- `evaluate_policy.py` auto-detects whether a checkpoint used dueling (newer checkpoints save metadata). You can override with `--dueling` or `--no-dueling`.

## Replays and GIF export

### View a replay

```bash
python main.py --replay replays/20260111_182909_nstep3_best.npz
```

You can also load multiple replays at once (side-by-side) by passing `--replay` multiple times.

### Export a GIF from a replay (non-interactive)

```bash
python main.py \
	--replay replays/20260111_182909_nstep3_best.npz \
	--gif-out gifs/nstep3_best.gif \
	--gif-export \
	--gif-fps 15 \
	--gif-every 2
```

Multiple replays → one grid GIF:

```bash
python main.py \
	--replay replays/a.npz --replay replays/b.npz --replay replays/c.npz --replay replays/d.npz \
	--gif-grid 2x2 \
	--gif-out gifs/grid.gif \
	--gif-export
```

Useful flags:

- `--gif-include-ui`: include the bottom UI panel in the capture
- `--gif-stop-at min|max`: when multiple replays are loaded, stop at the shortest or longest
- `--gif-max-frames N`: hard cap frames

## Batch evaluation helpers

### Side-by-side method comparison GIFs

`batch_eval_run_gifs.py` loads several trained methods from a specific run ID, evaluates them on a set of seeds, saves replays, and exports GIFs.

Example:

```bash
python batch_eval_run_gifs.py --run-id 20260111_182909 --n-seeds 4
```

Defaults:

- Replays → `replays/eval_<run-id>/...`
- GIFs → `gifs/eval_<run-id>/...`

This script calls the repo’s virtualenv Python at `.venv/bin/python` to run `main.py` for GIF export.

### Seed sweep plots

`seed_sweep_eval.py` evaluates multiple methods across many base seeds (with repeats per seed) and writes a CSV + a publication-style plot.

```bash
python seed_sweep_eval.py --run-id 20260111_182909 --methods baseline,singledqn,nstep3,nstep5 --n-seeds 20 --repeats 20
```

## Configuration

Environment parameters and feature toggles live in `config.py`.
Training overrides (e.g., number of days / actions per day / reward shaping) are applied in `train_double_dqn.py`.

Notable environment behaviors:

- Episodes run for `NUM_DAYS`, with `BATTERY_ACTIONS_PER_DAY` actions per day.
- If the robot doesn’t end the day on the charger, it can be disabled the next day (see config toggles).
- Observations include a fixed **3x3 local view** plus a **memory map** of last-seen state.
- Rewards are primarily based on **delta net worth** (money + pending value).

## Repo layout

- `agri_env.py`: Gymnasium environment + action mask
- `main.py`: viewer + replay player + GIF exporter
- `dqn_double.py`: DQN/Double-DQN agent, dueling head, n-step replay
- `train_double_dqn.py`: training + plots + replay recording
- `evaluate_policy.py`: greedy evaluation for a saved checkpoint
- `batch_eval_run_gifs.py`: multi-method, multi-seed replay + GIF pipeline
- `seed_sweep_eval.py`: many-seed evaluation + plotting
- `assets/`: sprites (tiles/robot/plants)
- `runs/`, `replays/`, `gifs/`: generated outputs

## Troubleshooting

- **Pygame window doesn’t open**: try running locally (not over a headless SSH session) and ensure macOS permissions allow GUI apps.
- **`imageio` missing / GIF export fails**: ensure `imageio` + `Pillow` are installed (`pip install -r requirements.txt`).
- **Torch import errors**: install the correct PyTorch wheel for your OS/Python/CUDA combination.

