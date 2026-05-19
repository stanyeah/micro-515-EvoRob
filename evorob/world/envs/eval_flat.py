from os import path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 5.0}


class EvalFlatEnv(MujocoEnv, utils.EzPickle):
    """Flat terrain evaluation environment.

    Termination: robot is terminated when the torso height z falls outside
    [0.2, 1.0] m, the torso flips upside-down (R[2,2] < 0), the robot is not
    making forward/lateral progress (||v_xy|| < 1 cm/s for > 10 s), or state
    is non-finite. Upside-down and stuck checks are added for symmetry with
    EvalHillEnv — they kill the "stand still and farm healthy_reward" exploit
    and the "fall on belly and survive" exploit that the bare height check
    missed. The stuck check uses x-y velocity only so a robot that is purely
    falling/bouncing in z is still allowed (z-bound termination handles that).

    Training reward:  healthy_reward + x_velocity - ctrl_cost - cfrc_cost
    + upright_weight * R[2,2]  (torso upright bonus; training-only shaping)

    Termination uses R[2,2] < 0 (full flip), not a soft tilt cutoff — tilt pressure
    comes from the per-step upright bonus instead.

    Tune ctrl_cost_weight, cfrc_cost_weight, and upright_weight on the flat surface.

    The info dict always exposes the four keys required by the neutral
    leaderboard formula: healthy_reward, x_position, ctrl_cost, cfrc_cost.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        robot_path: str,
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        ctrl_cost_weight: float = 0.5,
        cfrc_cost_weight: float = 5e-4,
        upright_weight: float = 1.0,
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        xml_file_path = robot_path if path.isabs(robot_path) else path.join(
            path.dirname(path.realpath(__file__)), robot_path
        )

        utils.EzPickle.__init__(
            self, xml_file_path, frame_skip, default_camera_config,
            ctrl_cost_weight, cfrc_cost_weight, upright_weight, reset_noise_scale, **kwargs,
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._cfrc_cost_weight = cfrc_cost_weight
        self._upright_weight = upright_weight
        self._reset_noise_scale = reset_noise_scale
        self._stuck_count = 0

        MujocoEnv.__init__(
            self, xml_file_path, frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (self.data.qpos.size - 2) + self.data.qvel.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]

        x_velocity = (x_after - x_before) / self.dt
        x_position = float(x_after)
        xyz_velocity = self.data.qvel[:3].copy()
        healthy_reward = 1.0
        ctrl_cost = float(np.sum(action ** 2) * self._ctrl_cost_weight)
        cfrc_cost = float(np.sum(self.data.cfrc_ext[1:] ** 2) * self._cfrc_cost_weight)
        upright_bonus = self._torso_rzz()

        terminated = self._is_terminated(xyz_velocity)
        reward = (healthy_reward + x_velocity - ctrl_cost - cfrc_cost
                  + self._upright_weight * upright_bonus)

        info = {
            "healthy_reward": -10.0 if terminated else healthy_reward,
            "x_position": float(x_after),
            "y_position": float(self.data.body(1).xpos[1]),
            "ctrl_cost": ctrl_cost,
            "cfrc_cost": cfrc_cost,
            "upright_bonus": upright_bonus,
            "x_velocity": x_velocity,
        }

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info

    def _is_terminated(self, xyz_velocity: np.ndarray) -> bool:
        z = float(self.data.qpos[2])
        if not np.isfinite(self.state_vector()).all():
            return True
        if z < 0.2 or z > 1.0:
            return True
        if self._torso_upside_down():
            return True
        if np.linalg.norm(xyz_velocity[:2]) < 1e-2:
            self._stuck_count += 1
            if self._stuck_count > 10 / self.dt:
                return True
        else:
            self._stuck_count = 0
        return False

    def _torso_rzz(self) -> float:
        R = self.data.body(1).xmat.reshape(3, 3)
        return float(R[2, 2])

    def _torso_upside_down(self) -> bool:
        return self._torso_rzz() < 0.0

    def _get_obs(self):
        return np.concatenate((self.data.qpos.flat[2:], self.data.qvel.flat.copy()))

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + noise ** 2 * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        self._stuck_count = 0
        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": float(self.data.qpos[0]),
            "y_position": float(self.data.body(1).xpos[1]),
        }
