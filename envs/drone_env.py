import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import mujoco
except ImportError:
    mujoco = None


class CityDroneEnv(gym.Env):
    """
    ADAPT-X 3D City Drone Simulation Environment.
    Features a high-fidelity quadcopter UAV flying inside an urban city
    with skyscrapers, trees, roads, sidewalks, and crosswalks.
    Supports mid-air rotor loss and dynamic self-healing recovery.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path=None, target_pos=(0.0, 0.0, 4.0)):
        super().__init__()
        if mujoco is None:
            raise ImportError("MuJoCo is required. Install via `pip install mujoco`.")

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            self.model_path = os.path.join(curr_dir, "..", "models", "city_drone_env.xml")
        else:
            self.model_path = model_path

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.target_pos = np.array(target_pos, dtype=np.float32)

        # Drone physical parameters
        self.drone_body_id = self.model.body("drone").id
        self.mass = float(self.model.body_subtreemass[self.drone_body_id])
        self.gravity = 9.81
        self.hover_thrust_total = self.mass * self.gravity
        self.hover_thrust_per_rotor = self.hover_thrust_total / 4.0

        # Arm radius (distance from center to motor)
        self.arm_length = 0.18
        self.yaw_torque_coeff = 0.02

        # Rotor health state: [FR, FL, RR, RL] (1.0 = healthy, 0.0 = completely broken)
        self.rotor_health = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.fault_type = "none"
        self.fault_rotor = None

        # Action space: 4 thrust commands [0.0 to 12.0 Newtons]
        self.action_space = spaces.Box(
            low=0.0, high=12.0, shape=(4,), dtype=np.float32
        )

        # Observation space:
        # [pos_err (3), quat (4), lin_vel (3), ang_vel (3), rotor_health (4)] = 17 dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )

        # Step counter
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Initial drone state: pos (0, 0, target_z), orientation quat [1, 0, 0, 0]
        self.data.qpos[0:3] = self.target_pos
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0

        # Reset rotor health
        self.rotor_health = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.fault_type = "none"
        self.fault_rotor = None
        self.current_step = 0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def inject_rotor_fault(self, fault_type="dead_motor", rotor="fr"):
        """
        Injects a mid-air hardware failure on a specific rotor.
        rotor: 'fr', 'fl', 'rr', 'rl'
        fault_type: 'dead_motor' (0% thrust), 'damaged_blade' (50% thrust)
        """
        rotor_map = {"fr": 0, "fl": 1, "rr": 2, "rl": 3}
        idx = rotor_map.get(rotor.lower(), 0)

        self.fault_type = fault_type
        self.fault_rotor = rotor.upper()

        if fault_type == "dead_motor":
            self.rotor_health[idx] = 0.0
        elif fault_type == "damaged_blade":
            self.rotor_health[idx] = 0.5
        elif fault_type == "none":
            self.rotor_health[idx] = 1.0

    def step(self, action):
        self.current_step += 1

        # Clip action to physical limits
        action = np.clip(action, 0.0, 12.0)

        # Apply hardware degradation/damage
        actual_thrust = action * self.rotor_health
        self.data.ctrl[:] = actual_thrust

        # MuJoCo sub-steps to reach 50 Hz control rate (0.005s * 4 = 0.02s dt)
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Compute position error
        curr_pos = self.data.qpos[0:3]
        pos_err = np.linalg.norm(curr_pos - self.target_pos)

        # Reward: high reward for maintaining altitude and stability, small penalty for effort
        z_err = abs(curr_pos[2] - self.target_pos[2])
        upright = self.data.qpos[3]  # quat w (approx 1 when upright)
        reward = 10.0 - 5.0 * z_err - 2.0 * pos_err + 2.0 * upright - 0.05 * np.sum(action)

        # Terminate if crashed to ground (z < 0.3m) or flipped completely upside down
        terminated = bool(curr_pos[2] < 0.35)
        truncated = bool(self.current_step >= 1000)

        info = {
            "altitude": float(curr_pos[2]),
            "thrust": actual_thrust.tolist(),
            "health": self.rotor_health.tolist(),
            "fault": self.fault_type,
            "rotor": self.fault_rotor,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        curr_pos = self.data.qpos[0:3]
        pos_err = curr_pos - self.target_pos
        quat = self.data.qpos[3:7]
        lin_vel = self.data.qvel[0:3]
        ang_vel = self.data.qvel[3:6]
        return np.concatenate([pos_err, quat, lin_vel, ang_vel, self.rotor_health]).astype(np.float32)

    def get_euler_angles(self):
        """Converts drone quaternion to Roll, Pitch, Yaw in degrees."""
        w, x, y, z = self.data.qpos[3:7]
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
