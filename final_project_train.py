"""
MICRO-515 Final Project — Multi-task Robot Evolution
=====================================================
Evolve a legged robot to walk in the +x direction across three training
environments simultaneously, using a Hebbian MLP controller and NSGA-II.

Experiment modes (EVOLUTION_MODE / --mode):
  mind_only  — evolve Hebbian A,B,C,D rules only; fixed ant-like body
  mind_body  — co-evolve Hebbian rules and 8 leg-segment lengths

Training environments (3 objectives)
-------------------------------------
1  Flat  — standard ground, good friction  (FlatEnv-v0  / flat_world.xml)
2  Ice   — slippery ground, low friction   (IceEnv-v0   / ice_world.xml)
3  Hill  — procedural hilly terrain        (HillEnv-v0  / hill_world.xml)

The evaluation terrain is separate and fixed.  Students test their best
evolved robot on it using final_project_test.py — it is not trained on.

Mind-only runs save fixed_body_genotype.npy next to checkpoints for testing.
"""

import argparse
import os
import shutil
import xml.etree.ElementTree as xml
from os.path import join
from tempfile import TemporaryDirectory

from evorob.utils.mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

import gymnasium as gym
import numpy as np
import scipy.ndimage
from PIL import Image
from gymnasium.vector import AsyncVectorEnv

import evorob.world                         # registers EvalEnv-v0
from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root
from evorob.world.base import World
from evorob.world.robot.controllers.mlp_hebbian import HebbianController
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

ROOT_DIR = get_project_root()
_ASSETS  = join(ROOT_DIR, "evorob", "world", "robot", "assets")
MAX_EPISODE_STEPS = 1000  # fixed for leaderboard — do not change

# --- Experiment configuration ------------------------------------------------
# "mind_only" | "mind_body"  (overridable via --mode on the command line)
EVOLUTION_MODE = "mind_only"

# Ant-like fixed morphology for mind_only: ~0.2 m upper / ~0.4 m lower per leg
FIXED_BODY_GENOTYPE = np.array(
    [-0.6, 0.2, -0.6, 0.2, -0.6, 0.2, -0.6, 0.2], dtype=np.float64
)

RESULTS_DIRS = {
    "mind_only": join(ROOT_DIR, "results", "final_mind_only"),
    "mind_body": join(ROOT_DIR, "results", "final_mind_body"),
}


# ---------------------------------------------------------------------------
# FinalWorld — body + brain co-evolution across multiple terrains
# ---------------------------------------------------------------------------

class FinalWorld(World):
    """Translates a genotype into a robot phenotype and evaluates it.

    Genotype layout
    -----------------
    * mind_only:  [ Hebbian rule params (1120) ] — body from FIXED_BODY_GENOTYPE
    * mind_body:  [ Hebbian rule params (1120) | body params (8) ]

    Each call to evaluate_individual generates the robot body XML, injects it
    into every terrain template, then runs the controller in parallel episodes.
    """

    def __init__(self, evolution_mode: str | None = None):
        self.evolution_mode = evolution_mode or EVOLUTION_MODE
        if self.evolution_mode not in RESULTS_DIRS:
            raise ValueError(
                f"evolution_mode must be one of {list(RESULTS_DIRS)}; "
                f"got {self.evolution_mode!r}"
            )

        self.controller = HebbianController(
            input_size=27, output_size=8, hidden_size=8
        )

        if self.evolution_mode == "mind_only":
            self._fixed_body_genotype = FIXED_BODY_GENOTYPE.copy()
            self.n_weights = self.controller.n_params
            self.n_body_params = 0
        else:
            self._fixed_body_genotype = None
            self.n_weights = self.controller.n_params
            self.n_body_params = 8          # 4 legs × (upper + lower segment length)

        self.n_params = self.n_weights + self.n_body_params

        # Temporary directory holds AntRobot.xml + one combined world XML per terrain
        self.temp_dir        = TemporaryDirectory()
        self.flat_world_file = join(self.temp_dir.name, "WorldFlat.xml")
        self.ice_world_file  = join(self.temp_dir.name, "WorldIce.xml")
        self.hill_world_file = join(self.temp_dir.name, "WorldHill.xml")
        self.world_file      = self.hill_world_file  # default for visualisation

        # Joint geometry — matches the AntRobot topology
        self.joint_limits = [
            [-30, 30], [30, 70],
            [-30, 30], [-70, -30],
            [-30, 30], [-70, -30],
            [-30, 30], [30, 70],
        ]
        self.joint_axis = [
            [0, 0, 1], [-1, 1, 0],
            [0, 0, 1], [1, 1, 0],
            [0, 0, 1], [-1, 1, 0],
            [0, 0, 1], [1, 1, 0],
        ]

        # Custom sensor function — intercepts the raw env observation before it
        # reaches the controller.  Set to any callable obs -> obs' to filter,
        # augment, or reshape observations.  The controller input_size must match
        # the output of this function.
        #
        # Example — use only joint angles and velocities (14 values):
        #   self.sensor_fn = lambda obs: obs[:14]
        #   self.controller = NeuralNetworkController(input_size=14, ...)
        self.sensor_fn = None

        # When False, _run_env evaluates n_repeats episodes sequentially in a
        # single process (no AsyncVectorEnv subprocesses).  Set to False inside
        # multiprocessing.Pool workers, since daemonic pool workers cannot spawn
        # their own subprocesses.  Defaults to True for legacy single-process runs.
        self.use_async_vector_env = True

        self._create_terrain_file("terrain.png")

    # ------------------------------------------------------------------
    # Genotype → phenotype
    # ------------------------------------------------------------------

    def geno2pheno(self, genotype: np.ndarray):
        """Decode genotype into controller weights and body parameters.

        mind_only:  genotype is all Hebbian genes; body from _fixed_body_genotype.
        mind_body:  genotype[:n_weights] → rules, genotype[n_weights:] → body genes.

        Controller genes are scaled by 0.1 before HebbianController.geno2pheno.
        Body genes map to lengths via (g + 1) / 4 + 0.1.

        Returns (points, connectivity_mat) for AntRobot construction.
        """
        if self.n_body_params == 0:
            control_params = genotype * 0.1
            body_genes = self._fixed_body_genotype
        else:
            control_params = genotype[: self.n_weights] * 0.1
            body_genes = genotype[self.n_weights :]
        body_params = (body_genes + 1) / 4 + 0.1
        self.controller.geno2pheno(control_params)

        front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, back_left_leg, back_left_ankle, back_right_leg, back_right_ankle, = body_params

        # Define the 3D coordinates of the relative tree structure
        front_left_hip_xyz = np.array([0.2, 0.2, 0])
        front_left_knee_xyz = np.array([np.sqrt(0.5 * front_left_leg ** 2), np.sqrt(0.5 * front_left_leg ** 2), 0]) + front_left_hip_xyz
        front_left_toe_xyz = np.array([np.sqrt(0.5 * front_left_ankle ** 2), np.sqrt(0.5 * front_left_ankle ** 2), 0]) + front_left_knee_xyz

        front_right_hip_xyz = np.array([-0.2, 0.2, 0])
        front_right_knee_xyz = np.array([-np.sqrt(0.5 * front_right_leg ** 2), np.sqrt(0.5 * front_right_leg ** 2), 0]) + front_right_hip_xyz
        front_right_toe_xyz = np.array([-np.sqrt(0.5 * front_right_ankle ** 2), np.sqrt(0.5 * front_right_ankle ** 2), 0]) + front_right_knee_xyz

        back_left_hip_xyz = np.array([-0.2, -0.2, 0])
        back_left_knee_xyz = np.array([-np.sqrt(0.5 * back_left_leg ** 2), -np.sqrt(0.5 * back_left_leg ** 2), 0]) + back_left_hip_xyz
        back_left_toe_xyz = np.array([-np.sqrt(0.5 * back_left_ankle ** 2), -np.sqrt(0.5 * back_left_ankle ** 2), 0]) + back_left_knee_xyz

        back_right_hip_xyz = np.array([0.2, -0.2, 0])
        back_right_knee_xyz = np.array([np.sqrt(0.5 * back_right_leg ** 2), -np.sqrt(0.5 * back_right_leg ** 2), 0]) + back_right_hip_xyz
        back_right_toe_xyz = np.array([np.sqrt(0.5 * back_right_ankle ** 2), -np.sqrt(0.5 * back_right_ankle ** 2), 0]) + back_right_knee_xyz

        points = np.vstack([front_left_hip_xyz,
                            front_left_knee_xyz,
                            front_left_toe_xyz,
                            front_right_hip_xyz,
                            front_right_knee_xyz,
                            front_right_toe_xyz,
                            back_left_hip_xyz,
                            back_left_knee_xyz,
                            back_left_toe_xyz,
                            back_right_hip_xyz,
                            back_right_knee_xyz,
                            back_right_toe_xyz,
                            ])

        # define the type of connections [FIXED ARCHITECTURE]
        connectivity_mat = np.array(
            [[150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]
        )
        return points, connectivity_mat

    # ------------------------------------------------------------------
    # Robot XML generation
    # ------------------------------------------------------------------

    def update_robot_xml(self, genotype: np.ndarray) -> None:
        """Build robot body XML from genotype and inject into every terrain template.

        Writes AntRobot.xml to temp_dir, then creates one combined world XML per
        terrain (flat, ice, hill) by appending an <include> to the template.
        """
        points, connectivity_mat = self.geno2pheno(genotype)
        robot = AntRobot(
            points, connectivity_mat, self.joint_limits, self.joint_axis,
            name="Robot", verbose=False,
        )
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)          # → Robot.xml

        for template, world_file in [
            (join(_ASSETS, "flat_world.xml"), self.flat_world_file),
            (join(_ASSETS, "ice_world.xml"),  self.ice_world_file),
            (join(_ASSETS, "hill_world.xml"), self.hill_world_file),
        ]:
            tree = xml.parse(template)
            root = tree.getroot()
            root.append(xml.Element("include", attrib={"file": "Robot.xml"}))
            with open(world_file, "w") as f:
                f.write(xml.tostring(root, encoding="unicode"))

    def _create_terrain_file(self, filename: str, width: int = 200, depth: int = 400):
        """Hill terrain PNG: smooth start, bumpy middle, smooth end."""
        slope_deg = 5.0
        bump_scale = 0.08
        sigma = 4.0

        rise = np.tan(np.deg2rad(slope_deg))
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)

        # Smooth slope: starts flat, gradually rises
        slope_map = np.clip(X * rise, 0, 1)

        # Bell-shaped envelope: smooth at both ends, bumpy in the middle
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 1, (width, depth))
        bump_envelope = np.sin(np.pi * X)  # 0 at start, peaks at mid, 0 at end
        noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * bump_envelope
        noise_map = noise * bump_scale

        terrain = np.clip(slope_map + noise_map, 0, 1)
        terrain[-1, -1] = 1  # ensure max value for normalization

        img = Image.fromarray((terrain * 255).astype(np.uint8), mode="L")
        img.save(join(self.temp_dir.name, filename))

    # ------------------------------------------------------------------
    # Per-terrain evaluation
    # ------------------------------------------------------------------

    def _run_env(self, env_id: str, world_file: str, n_repeats: int, n_steps: int) -> float:
        """Run n_repeats episodes and return the mean total reward.

        If use_async_vector_env is True (default), n_repeats episodes run in
        parallel via gymnasium's AsyncVectorEnv.  If False, they run sequentially
        in this process (used inside multiprocessing.Pool workers, which cannot
        themselves spawn AsyncVectorEnv subprocesses).
        """
        if not self.use_async_vector_env:
            return self._run_env_serial(env_id, world_file, n_repeats, n_steps)

        envs = AsyncVectorEnv([
            (lambda eid, wf: lambda: gym.make(
                eid, robot_path=wf, max_episode_steps=n_steps
            ))(env_id, world_file)
            for _ in range(n_repeats)
        ])
        self.controller.reset_controller(batch_size=n_repeats)
        rewards = np.zeros((n_steps, n_repeats))
        obs, _ = envs.reset()
        if self.sensor_fn is not None:
            obs = self.sensor_fn(obs)
        done = np.zeros(n_repeats, dtype=bool)
        for t in range(n_steps):
            actions = np.where(done[:, None], 0, self.controller.get_action(obs))
            obs, r, terminated, truncated, _ = envs.step(actions)
            if self.sensor_fn is not None:
                obs = self.sensor_fn(obs)
            rewards[t, ~done] = r[~done]
            done |= terminated | truncated
            if done.all():
                break
        envs.close()
        return float(rewards.sum(axis=0).mean())

    def _run_env_serial(self, env_id: str, world_file: str, n_repeats: int, n_steps: int) -> float:
        """Run n_repeats episodes one after another in a single process.

        Used inside multiprocessing.Pool workers where AsyncVectorEnv cannot run.
        Outer-loop parallelism (across the population) gives the speedup instead.
        """
        env = gym.make(env_id, robot_path=world_file, max_episode_steps=n_steps)
        episode_totals = np.zeros(n_repeats)
        for r in range(n_repeats):
            self.controller.reset_controller(batch_size=1)
            obs, _ = env.reset()
            if self.sensor_fn is not None:
                obs = self.sensor_fn(obs)
            total = 0.0
            for t in range(n_steps):
                action = self.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, reward, terminated, truncated, _ = env.step(action)
                if self.sensor_fn is not None:
                    obs = self.sensor_fn(obs)
                total += float(reward)
                if terminated or truncated:
                    break
            episode_totals[r] = total
        env.close()
        return float(episode_totals.mean())

    def _eval_flat(self, n_repeats: int = 4, n_steps: int = 500) -> float:
        return self._run_env("FlatEnv-v0", self.flat_world_file, n_repeats, n_steps)

    def _eval_ice(self, n_repeats: int = 4, n_steps: int = 500) -> float:
        return self._run_env("IceEnv-v0", self.ice_world_file, n_repeats, n_steps)

    def _eval_hill(self, n_repeats: int = 4, n_steps: int = 500) -> float:
        return self._run_env("HillEnv-v0", self.hill_world_file, n_repeats, n_steps)

    def create_env(self, render_mode: str = "rgb_array", **kwargs):
        """Return a HillEnv-v0 instance (used for visualisation)."""
        return gym.make("HillEnv-v0", robot_path=self.hill_world_file,
                        render_mode=render_mode, **kwargs)

    # ------------------------------------------------------------------
    # Combined fitness for NSGA-II
    # ------------------------------------------------------------------

    def evaluate_individual(self, genotype: np.ndarray,
                            n_repeats: int = 4, n_steps: int = 500) -> np.ndarray:
        """Evaluate one genotype on all three training environments.

        Returns a 1-D array of three objective values: [flat, ice, hill].
        """
        self.update_robot_xml(genotype)
        return np.array([
            self._eval_flat(n_repeats, n_steps),
            self._eval_ice(n_repeats, n_steps),
            self._eval_hill(n_repeats, n_steps),
        ])


# ---------------------------------------------------------------------------
# Neutral leaderboard evaluation  (TA-graded — do not modify)
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
    n_episodes: int = 256,          # set to 256 for submission; lower for testing
) -> dict | None:
    """Evaluate the best genotype from a checkpoint on all three training terrains.

    Loads x_best.npy, evaluates it on flat, ice, and hill for n_episodes each,
    prints per-episode scores, records one video per terrain, and writes a score file.

    Args:
        checkpoint_dir: Path to your NSGA-II checkpoint folder.
        output_dir:     Where to save the score file and videos.
        n_episodes:     Episodes per terrain (256 for submission).
    """
    MAX_STEPS = MAX_EPISODE_STEPS   # DO NOT CHANGE
    SEED      = 0                   # DO NOT CHANGE

    # --- Locate checkpoint ---
    last_gen = get_last_checkpoint_dir(checkpoint_dir)

    def _load(fname):
        for d in ([last_gen] if last_gen else []) + [checkpoint_dir]:
            p = join(d, fname)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        return None

    x_best = _load("x_best.npy")
    if x_best is None:
        print(f"ERROR: x_best.npy not found in '{checkpoint_dir}'.")
        return None
    print(f"Loaded x_best  (shape: {x_best.shape})")

    fixed_body = _load("fixed_body_genotype.npy")
    x_size = int(np.asarray(x_best).size)
    if x_size == 1120:
        ckpt_mode = "mind_only"
    elif x_size == 1128:
        ckpt_mode = "mind_body"
    else:
        ckpt_mode = EVOLUTION_MODE
        print(f"WARNING: unexpected x_best size {x_size}; using EVOLUTION_MODE={ckpt_mode}")

    world = FinalWorld(evolution_mode=ckpt_mode)
    if fixed_body is not None and ckpt_mode == "mind_only":
        world._fixed_body_genotype = np.asarray(fixed_body, dtype=np.float64).reshape(-1)
    world.update_robot_xml(x_best)
    ctrl_name = type(world.controller).__name__
    print(f"Controller: {ctrl_name}  |  n_weights={world.n_weights}"
          f"  |  genotype size={world.n_params}\n")

    terrains = {
        "flat": ("FlatEnv-v0", world.flat_world_file),
        "ice":  ("IceEnv-v0",  world.ice_world_file),
        "hill": ("HillEnv-v0", world.hill_world_file),
    }

    def _neutral(info: dict) -> float:
        return (float(info.get("healthy_reward", 1.0))
                + float(info.get("x_position",   0.0))
                - float(info.get("ctrl_cost",     0.0))
                - float(info.get("cfrc_cost",     0.0)))

    def _stats(values: list) -> dict:
        arr = np.asarray(values)
        return dict(mean=float(arr.mean()), std=float(arr.std()),
                    best=float(arr.max()), worst=float(arr.min()), values=values)

    def _run(env_id: str, world_file: str) -> list:
        rng = np.random.default_rng(SEED)
        env = gym.make(env_id, robot_path=world_file, max_episode_steps=MAX_STEPS)
        rewards = []
        for ep in range(n_episodes):
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=int(rng.integers(0, 2 ** 31)))
            total, done = 0.0, False
            while not done:
                action = world.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, info = env.step(action)
                total += _neutral(info)
                done = terminated or truncated
            rewards.append(total)
        env.close()
        return rewards

    def _record(env_id: str, world_file: str, out_path: str) -> None:
        try:
            import imageio
            env = gym.make(env_id, robot_path=world_file,
                           render_mode="rgb_array", max_episode_steps=MAX_STEPS)
            world.controller.reset_controller(batch_size=1)
            obs, _ = env.reset(seed=SEED)
            frames = []
            for _ in range(MAX_STEPS):
                frames.append(env.render())
                action = world.controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            env.close()
            imageio.mimwrite(out_path, frames, fps=20)
            print(f"  Video: {out_path}")
        except Exception as exc:
            print(f"  Video skipped: {exc}")

    # --- Evaluate on each terrain ---
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for terrain_name, (env_id, world_file) in terrains.items():
        print(f"  Running {terrain_name}  ({n_episodes} episodes)...", flush=True)
        results[terrain_name] = _stats(_run(env_id, world_file))

    # Per-episode 3-column table
    t_names = list(results.keys())
    col_w = 12
    hdr = f"  {'Ep':>4}   " + "   ".join(f"{n.capitalize():>{col_w}}" for n in t_names)
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    for ep in range(n_episodes):
        row = f"  {ep + 1:>4}   " + "   ".join(
            f"{results[n]['values'][ep]:>{col_w}.2f}" for n in t_names
        )
        print(row)
    print(sep)
    print(f"  {'mean':>4}   " + "   ".join(
        f"{results[n]['mean']:>{col_w}.2f}" for n in t_names
    ))
    print(f"  {'std':>4}   " + "   ".join(
        f"{results[n]['std']:>{col_w}.2f}" for n in t_names
    ))
    print()

    # --- Record one video per terrain ---
    print("Recording videos...")
    for terrain_name, (env_id, world_file) in terrains.items():
        _record(env_id, world_file, join(output_dir, f"evaluation_{terrain_name}.mp4"))

    # --- Score file ---
    score_path = join(output_dir, "evaluation_score.txt")
    col = 60
    with open(score_path, "w") as f:
        f.write("=" * col + "\n")
        f.write("MICRO-515 Final Project — Evaluation Results\n")
        f.write("=" * col + "\n\n")
        f.write(f"Controller      : {ctrl_name} ({world.n_weights} params)\n")
        f.write(f"Genotype size   : {world.n_params}"
                f"  (controller={world.n_weights}, body={world.n_body_params})\n")
        f.write(f"Checkpoint      : {checkpoint_dir}\n")
        f.write(f"Episodes/terrain: {n_episodes}\n")
        f.write(f"Reward          : healthy_reward + x_position - ctrl_cost - cfrc_cost\n\n")

        f.write("=" * col + "\n")
        f.write("SUMMARY\n")
        f.write("=" * col + "\n")
        f.write(f"{'Terrain':<8} {'Mean':>9} {'Std':>8} {'Best':>9} {'Worst':>9}\n")
        f.write("-" * col + "\n")
        for terrain_name, r in results.items():
            f.write(f"{terrain_name:<8} {r['mean']:9.2f} {r['std']:8.2f}"
                    f" {r['best']:9.2f} {r['worst']:9.2f}\n")
        f.write("\n")

        for terrain_name, r in results.items():
            f.write("-" * 50 + "\n")
            f.write(f"{terrain_name.upper()} — Per-episode rewards\n")
            f.write("-" * 50 + "\n")
            for i, v in enumerate(r["values"]):
                f.write(f"  Episode {i + 1:3d}: {v:10.2f}\n")
            f.write("\n")

    print(f"\nScore saved to: {score_path}")
    print("=" * col)
    for terrain_name, r in results.items():
        print(f"  {terrain_name:<6}: {r['mean']:8.2f} ± {r['std']:7.2f}"
              f"  best={r['best']:.2f}  worst={r['worst']:.2f}")
    print("=" * col)
    return results


# ---------------------------------------------------------------------------
# Multiprocessing.Pool workers for population-parallel evaluation
# ---------------------------------------------------------------------------
#
# When run_multi_task_evolution is called with n_workers > 1, NSGA-II's
# population is dispatched across a Pool.  Each worker maintains a private
# FinalWorld instance (its own temp_dir, its own controller state) so workers
# never share mutable state.  Workers run n_repeats episodes sequentially via
# _run_env_serial; outer parallelism (across the population) gives the speedup.
#
# These must be defined at module level so they are picklable / importable when
# the Pool spawns child processes.

_worker_world: "FinalWorld | None" = None


def _worker_init(evolution_mode: str,
                 fixed_body_genotype: np.ndarray | None) -> None:
    """Per-worker initializer: build one FinalWorld in serial-eval mode."""
    global _worker_world
    _worker_world = FinalWorld(evolution_mode=evolution_mode)
    _worker_world.use_async_vector_env = False
    if fixed_body_genotype is not None:
        _worker_world._fixed_body_genotype = np.asarray(
            fixed_body_genotype, dtype=np.float64
        ).reshape(-1)


def _worker_eval(args):
    """Evaluate one individual; return (idx, fitness, robot_xml_content)."""
    idx, genotype, n_repeats, n_steps = args
    fitness = _worker_world.evaluate_individual(
        genotype, n_repeats=n_repeats, n_steps=n_steps
    )
    with open(join(_worker_world.temp_dir.name, "Robot.xml")) as f:
        robot_xml = f.read()
    return idx, fitness, robot_xml


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_multi_task_evolution(
    num_generations: int = 100,
    population_size: int = 100,
    n_parents:       int = 50,
    n_repeats:       int = 3,
    n_steps:         int = 500,
    mutation_prob:   float = 0.3,
    crossover_prob:  float = 0.5,
    bounds:          tuple = (-1, 1),
    ckpt_interval:   int = 10,
    results_dir:     str = None,
    random_seed:     int = 42,
    evolution_mode:  str | None = None,
    n_workers:       int = 1,
) -> None:
    np.random.seed(random_seed)

    mode = evolution_mode or EVOLUTION_MODE
    world = FinalWorld(evolution_mode=mode)
    print(f"Mode     : {world.evolution_mode}")
    print(f"Seed     : {random_seed}")
    print(f"Genotype : {world.n_params} params"
          f"  (controller={world.n_weights}, body={world.n_body_params})")

    if results_dir is None:
        results_dir = RESULTS_DIRS[mode]

    ea = NSGAII(
        population_size=population_size,
        n_opt_params=world.n_params,
        n_parents=n_parents,
        num_generations=num_generations,
        bounds=bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        output_dir=results_dir,
    )

    n_obj = 3
    print(f"\nRunning {num_generations} generations  pop={population_size}")
    print(f"Workers    : {n_workers}")
    print(f"Objectives : [flat, ice, hill]")
    print(f"Checkpoints: {results_dir}\n")

    os.makedirs(results_dir, exist_ok=True)
    if world.n_body_params == 0:
        np.save(join(results_dir, "fixed_body_genotype.npy"), world._fixed_body_genotype)

    _best_xml_stage = join(results_dir, "_best_robot.xml")  # staging copy of best robot
    _best_scalar = -np.inf

    # --- Optional Pool for population-parallel evaluation ---
    pool = None
    if n_workers > 1:
        import multiprocessing
        ctx = multiprocessing.get_context("fork")  # safe on Linux; cheap to spawn
        fixed_body_for_workers = (
            world._fixed_body_genotype if world.n_body_params == 0 else None
        )
        pool = ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(mode, fixed_body_for_workers),
        )

    try:
        for gen in range(num_generations):
            pop = ea.ask()
            fitnesses = np.empty((len(pop), n_obj))

            if pool is not None:
                # Population-parallel: workers evaluate genotypes in parallel.
                tasks = [
                    (idx, genotype, n_repeats, n_steps)
                    for idx, genotype in enumerate(pop)
                ]
                for idx, fitness, robot_xml in pool.imap_unordered(_worker_eval, tasks):
                    fitnesses[idx] = fitness
                    scalar = float(fitness.sum())
                    if scalar > _best_scalar:
                        _best_scalar = scalar
                        with open(_best_xml_stage, "w") as fh:
                            fh.write(robot_xml)
            else:
                # Serial path (legacy): one individual at a time in master process.
                for idx, genotype in enumerate(pop):
                    fitnesses[idx] = world.evaluate_individual(
                        genotype, n_repeats=n_repeats, n_steps=n_steps
                    )
                    scalar = float(fitnesses[idx].sum())
                    if scalar > _best_scalar:
                        _best_scalar = scalar
                        shutil.copy2(
                            join(world.temp_dir.name, "Robot.xml"),
                            _best_xml_stage,
                        )

            save_ckpt = (gen % ckpt_interval == 0)
            ea.tell(pop, fitnesses, save_checkpoint=save_ckpt)
            if save_ckpt:
                gen_dir = join(results_dir, str(gen))
                shutil.copy2(_best_xml_stage, join(gen_dir, "Robot.xml"))
                if world.n_body_params == 0:
                    np.save(
                        join(gen_dir, "fixed_body_genotype.npy"),
                        world._fixed_body_genotype,
                    )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # --- Training summary ---
    best_f = ea.f_best_so_far  # shape (3,) for NSGA-II
    score_path = join(results_dir, "training_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Final Project — Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Evolution mode  : {world.evolution_mode}\n")
        f.write(f"Random seed     : {random_seed}\n")
        f.write(f"Generations     : {num_generations}\n")
        f.write(f"Population size : {population_size}\n")
        f.write(f"n_repeats       : {n_repeats}\n")
        f.write(f"n_workers       : {n_workers}\n")
        f.write(f"Controller      : {type(world.controller).__name__}"
                f"  ({world.n_weights} params)\n")
        f.write(f"Genotype size   : {world.n_params}"
                f"  (controller={world.n_weights}, body={world.n_body_params})\n\n")
        f.write("Best individual (highest sum of objectives):\n")
        labels = ["flat", "ice", "hill"]
        for label, val in zip(labels, best_f):
            f.write(f"  {label:<6}: {float(val):10.2f}\n")
        f.write(f"  {'sum':<6}: {float(best_f.sum()):10.2f}\n")
    print(f"\nTraining summary saved to: {score_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task Hebbian evolution (NSGA-II)")
    parser.add_argument(
        "--mode",
        choices=("mind_only", "mind_body"),
        default=EVOLUTION_MODE,
        help="mind_only: evolve Hebbian rules only; mind_body: co-evolve rules + legs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for the EA. When set, checkpoints go to "
            "results/final_<mode>/seed_<n>/ so multiple seeds don't overwrite "
            "each other. Defaults to 42 with no subfolder (legacy behaviour)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes for population evaluation. "
            "1 (default) keeps the legacy single-process path with AsyncVectorEnv. "
            "On a SLURM cluster set this to $SLURM_CPUS_PER_TASK so the EA "
            "evaluates the population in parallel."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short run for pipeline check (few gens, small pop)",
    )
    args = parser.parse_args()

    # Per-seed results dir so seed sweeps don't overwrite each other.
    if args.seed is not None:
        normal_results_dir = join(RESULTS_DIRS[args.mode], f"seed_{args.seed}")
        smoke_results_dir = join(ROOT_DIR, "results", f"smoke_{args.mode}",
                                 f"seed_{args.seed}")
        seed_value = args.seed
    else:
        normal_results_dir = None  # falls back to RESULTS_DIRS[mode]
        smoke_results_dir = join(ROOT_DIR, "results", f"smoke_{args.mode}")
        seed_value = 42

    if args.smoke:
        run_multi_task_evolution(
            evolution_mode=args.mode,
            num_generations=2,
            population_size=4,
            n_parents=4,
            n_repeats=1,
            n_steps=50,
            ckpt_interval=1,
            results_dir=smoke_results_dir,
            random_seed=seed_value,
            n_workers=args.workers,
        )
    else:
        # pop=128, n_parents=64 → two clean waves of 64 on the cluster.
        run_multi_task_evolution(
            evolution_mode=args.mode,
            num_generations=100,
            population_size=128,
            n_parents=64,
            n_repeats=3,
            n_steps=1000,
            ckpt_interval=10,
            results_dir=normal_results_dir,
            random_seed=seed_value,
            n_workers=args.workers,
        )
