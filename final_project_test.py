"""
MICRO-515 Final Project — Compatibility Check Script
=====================================================
Use this script to verify that your evolved robot is compatible with the
grading pipeline before you submit.

What this script does
---------------------
1.  Loads your best genotype (x_best.npy) and robot XML from a checkpoint
    directory (or from manually specified paths — see Option B below).
2.  Runs the robot on the *dummy* evaluation terrain for N_EPISODES episodes.
3.  Saves a score file and a video to evaluation_output/.

Important: the score reported here is computed on a simplified dummy terrain.
It is NOT representative of the actual grading environment, which is hidden
and will only be used during the poster session.  A high score here does not
guarantee a high score on the leaderboard, and vice versa.

Quick-start (recommended)
--------------------------
    python final_project_test.py --best_dir_path results/final_mind_only
    python final_project_test.py --best_dir_path results/final_mind_body

The directory must contain x_best.npy and Robot.xml (from training checkpoints).
mind_only (280 genes @ hidden_size=8): also include fixed_body_genotype.npy if not using Option A
load_from_checkpoint (saved automatically during training).

Option B — supply files manually
----------------------------------
Set ROBOT_XML_PATH and GENOTYPE_PATH, then use world.geno2pheno to load controller weights.

Legacy Hebbian checkpoints (1120 genes): set MY_CONTROLLER to HebbianController below.

Submission reminder
--------------------
Always include in your zip:
  - final_project_train.py
  - x_best.npy
  - <your_robot>.xml
  - Your controller source file (even if unchanged from the default)
  - README.md (max 400 words)
"""

import argparse
import os

from evorob.utils.mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

import numpy as np

#os.environ.setdefault("MUJOCO_GL", "egl")

import evorob.world          # registers EvalEnv-v0
import gymnasium as gym

from evorob.world.eval_world import EvalWorld

# ===========================================================================
# STUDENT CONFIGURATION — edit this section
# ===========================================================================

# --- Controller ---
# Must match final_project_train.py (feedforward MLP 27→8→8, 280 genes).  None uses EvalWorld default.
from evorob.world.robot.controllers.mlp import NeuralNetworkController

MY_CONTROLLER = NeuralNetworkController(input_size=27, output_size=8, hidden_size=8)

# --- Paths ---
# Option A: training results directory (x_best.npy, Robot.xml, …)
CHECKPOINT_DIR = "results/final_mind_only"

# Option B: provide the robot XML and genotype as separate files
ROBOT_XML_PATH = None   # e.g. "/abs/path/to/Robot.xml"
GENOTYPE_PATH  = None   # e.g. "/abs/path/to/x_best.npy"

# --- Output ---
OUTPUT_DIR = "evaluation_output"
N_EPISODES = 64     # paired picker metric for the 24h sprint; bump to 256 for final submission
SEED       = 0      # fixed — do NOT change for a fair comparison
MAX_STEPS  = 1000   # fixed — do NOT change

# ===========================================================================


def _neutral_reward(info: dict) -> float:
    """Leaderboard reward: healthy_reward + x_position - ctrl_cost - cfrc_cost."""
    return (
        float(info.get("healthy_reward", 1.0))
        + float(info.get("x_position",   0.0))
        - float(info.get("ctrl_cost",     0.0))
        - float(info.get("cfrc_cost",     0.0))
    )


def _accumulate_y(info: dict, y_min: float, y_max: float, y_abs_max: float):
    y = float(info.get("y_position", float("nan")))
    if not np.isfinite(y):
        return y_min, y_max, y_abs_max
    return min(y_min, y), max(y_max, y), max(y_abs_max, abs(y))


def run_episodes(world: EvalWorld, n_episodes: int, seed: int):
    """Returns (rewards, y_stats) where y_stats has per-episode lateral diagnostics."""
    rng = np.random.default_rng(seed)
    env = gym.make("EvalEnv-v0", robot_path=world.world_file,
                   max_episode_steps=MAX_STEPS)
    rewards = []
    y_stats = []

    for ep in range(n_episodes):
        world.controller.reset_controller(batch_size=1)
        obs, info0 = env.reset(seed=int(rng.integers(0, 2 ** 31)))
        y0 = float(info0.get("y_position", float("nan")))
        y_min = y0 if np.isfinite(y0) else float("inf")
        y_max = y0 if np.isfinite(y0) else float("-inf")
        y_abs_max = abs(y0) if np.isfinite(y0) else 0.0
        y_last = y0 if np.isfinite(y0) else 0.0

        total, done = 0.0, False
        while not done:
            ctrl_obs = world.sensor_fn(obs) if world.sensor_fn is not None else obs
            action = world.controller.get_action(ctrl_obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, _, terminated, truncated, info = env.step(action)
            total += _neutral_reward(info)
            y_min, y_max, y_abs_max = _accumulate_y(info, y_min, y_max, y_abs_max)
            y_last = float(info.get("y_position", y_last))
            done = terminated or truncated
        rewards.append(total)
        if not np.isfinite(y_min):
            y_min = y_last
        if not np.isfinite(y_max):
            y_max = y_last
        y_stats.append({
            "y_final": y_last,
            "y_abs_max": y_abs_max,
            "y_min": y_min,
            "y_max": y_max,
        })
        print(f"  episode {ep + 1:3d}/{n_episodes}: reward={total:10.2f}  "
              f"|y|_max={y_abs_max:6.3f}  y_final={y_last:7.3f}")

    env.close()
    return rewards, y_stats


def record_video(world: EvalWorld, out_path: str, seed: int) -> None:
    try:
        import imageio
        env = gym.make("EvalEnv-v0", robot_path=world.world_file,
                       render_mode="rgb_array", max_episode_steps=MAX_STEPS)
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset(seed=seed)
        frames = []
        for _ in range(MAX_STEPS):
            frames.append(env.render())
            ctrl_obs = world.sensor_fn(obs) if world.sensor_fn is not None else obs
            action = world.controller.get_action(ctrl_obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()
        imageio.mimwrite(out_path, frames, fps=20)
        print(f"Video saved: {out_path}")
    except Exception as exc:
        print(f"Video skipped: {exc}")


def save_score(world: EvalWorld, rewards: list, output_dir: str,
               y_stats: list | None = None) -> None:
    arr = np.asarray(rewards, dtype=float)
    score_path = os.path.join(output_dir, "evaluation_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Final Project — Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Controller : {type(world.controller).__name__}"
                f"  ({world.controller.n_params} params)\n")
        f.write(f"Genotype   : {world.n_params} params  "
                f"(controller={world.n_weights}, body={world.n_body_params})\n")
        f.write(f"Reward     : healthy_reward + x_position - ctrl_cost - cfrc_cost\n\n")
        f.write(f"Mean  : {arr.mean():.2f}\n")
        f.write(f"Std   : {arr.std():.2f}\n")
        f.write(f"Best  : {arr.max():.2f}\n")
        f.write(f"Worst : {arr.min():.2f}\n\n")
        for i, r in enumerate(rewards):
            f.write(f"Episode {i + 1:3d}: {r:10.2f}\n")

        if y_stats is not None and len(y_stats) == len(rewards):
            abs_max = np.array([s["y_abs_max"] for s in y_stats], dtype=float)
            y_fin = np.array([s["y_final"] for s in y_stats], dtype=float)
            f.write("\n" + "=" * 60 + "\n")
            f.write("Lateral (y) diagnostics — torso/world y (m); not in reward\n")
            f.write("=" * 60 + "\n\n")
            f.write("Per episode: |y|_max = max |y| along trajectory; "
                    "y_final = y at last step.\n")
            f.write(f"{'Ep':>4}  {'|y|_max':>10}  {'y_final':>10}  {'y_min':>10}  {'y_max':>10}\n")
            for i, s in enumerate(y_stats):
                f.write(f"{i + 1:4d}  {s['y_abs_max']:10.4f}  {s['y_final']:10.4f}  "
                        f"{s['y_min']:10.4f}  {s['y_max']:10.4f}\n")
            f.write("\nSummary:\n")
            f.write(f"  mean |y|_max   : {float(abs_max.mean()):.4f} m\n")
            f.write(f"  max  |y|_max   : {float(abs_max.max()):.4f} m\n")
            f.write(f"  mean |y_final| : {float(np.abs(y_fin).mean()):.4f} m\n")

    print(f"Score saved: {score_path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MICRO-515 Final Project — compatibility check against the dummy evaluation terrain."
    )
    parser.add_argument(
        "--best_dir_path",
        default=None,
        metavar="DIR",
        help="Directory containing x_best.npy and Robot.xml. "
             "Overrides CHECKPOINT_DIR (e.g. results/final_mind_only).",
    )
    args = parser.parse_args()

    checkpoint_dir = args.best_dir_path if args.best_dir_path is not None else CHECKPOINT_DIR

    world = EvalWorld()

    if MY_CONTROLLER is not None:
        world.set_controller(MY_CONTROLLER)

    if ROBOT_XML_PATH is not None and GENOTYPE_PATH is not None:
        # Option B: student provides robot XML and genotype separately
        if not os.path.isfile(ROBOT_XML_PATH):
            raise FileNotFoundError(f"Robot XML not found: {ROBOT_XML_PATH}")
        if not os.path.isfile(GENOTYPE_PATH):
            raise FileNotFoundError(f"Genotype not found: {GENOTYPE_PATH}")
        world.update_robot_xml(ROBOT_XML_PATH)
        genotype = np.load(GENOTYPE_PATH, allow_pickle=True)
        world.geno2pheno(genotype)
        print(f"Robot  : {ROBOT_XML_PATH}")
        print(f"Geno   : {GENOTYPE_PATH}  shape={genotype.shape}"
              f"  (layout: controller={world.n_weights}, body={world.n_body_params})")
    else:
        # Option A (default): load everything from the checkpoint directory
        world.load_from_checkpoint(checkpoint_dir)

    print(f"\nRunning {N_EPISODES} episodes on the evaluation terrain  (seed={SEED}) …")
    rewards, y_stats = run_episodes(world, N_EPISODES, SEED)

    arr = np.asarray(rewards, dtype=float)
    abs_max = np.array([s["y_abs_max"] for s in y_stats], dtype=float)
    print(f"\nResults: mean={arr.mean():.2f} ± {arr.std():.2f}  "
          f"best={arr.max():.2f}  worst={arr.min():.2f}")
    print(f"Lateral: mean |y|_max={abs_max.mean():.3f} m  "
          f"worst |y|_max={abs_max.max():.3f} m  "
          f"(dummy bridge is ~3 m wide in y; half-width ≈1.5 m from x-axis)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_score(world, rewards, OUTPUT_DIR, y_stats=y_stats)
    record_video(world, os.path.join(OUTPUT_DIR, "evaluation_video.mp4"), SEED)
