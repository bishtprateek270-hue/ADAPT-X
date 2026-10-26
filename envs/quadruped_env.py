import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import mujoco
except ImportError:
    mujoco = None


class QuadrupedEnv(gym.Env):
    """
    12-DoF Quadruped Locomotion Environment for ADAPT-X.
    Built for robust Sim-to-Real RL and fault tolerance.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, config=None, model_path=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.config = config or {}

        # Resolve model path
        if model_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(curr_dir, "..", "models", "quadruped.xml")
        
        self.model_path = os.path.abspath(model_path)
        if mujoco is not None and os.path.exists(self.model_path):
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
        else:
            self.model = None
            self.data = None

        # Robot configuration
        self.num_joints = 12
        self.action_scale = float(self.config.get("robot", {}).get("action_scale", 0.4))
        self.kp = float(self.config.get("robot", {}).get("kp", 25.0))
        self.kd = float(self.config.get("robot", {}).get("kd", 0.8))
        self.torque_limit = float(self.config.get("robot", {}).get("torque_limit", 18.0))
        
        # Nominal default standing angles (FR, FL, RR, RL -> hip, thigh, calf)
        self.default_joint_pos = np.array([
            0.0, 0.6, -1.2,  # FR
            0.0, 0.6, -1.2,  # FL
            0.0, 0.6, -1.2,  # RR
            0.0, 0.6, -1.2   # RL
        ], dtype=np.float32)

        # Simulation timings
        self.dt = float(self.config.get("environment", {}).get("dt", 0.02))  # 50 Hz control
        self.sim_substeps = int(self.config.get("environment", {}).get("sim_substeps", 10))
        self.target_height = float(self.config.get("environment", {}).get("target_height", 0.28))
        self.max_episode_steps = int(1000)

        # Observation & Action Spaces
        # Obs: [base_ang_vel(3), proj_gravity(3), cmd(3), joint_pos_rel(12), joint_vel(12), prev_action(12), base_lin_vel(3)] = 48
        self.obs_dim = 48
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

        # State buffers
        self.current_step = 0
        self.prev_action = np.zeros(self.num_joints, dtype=np.float32)
        self.commands = np.zeros(3, dtype=np.float32)  # [vx_cmd, vy_cmd, wz_cmd]
        
        # Fault injection hook for Phase 2 readiness
        self.fault_config = {"type": "none", "leg": None, "joint_idx": None}

        # Renderer placeholder
        self.viewer = None

    def _sample_commands(self):
        """Sample random velocity commands for robust locomotion training."""
        ranges = self.config.get("environment", {}).get("command_ranges", {})
        vx_range = ranges.get("lin_vel_x", [0.4, 1.0])
        vy_range = ranges.get("lin_vel_y", [-0.2, 0.2])
        wz_range = ranges.get("ang_vel_yaw", [-0.5, 0.5])

        self.commands[0] = np.random.uniform(vx_range[0], vx_range[1])
        self.commands[1] = np.random.uniform(vy_range[0], vy_range[1])
        self.commands[2] = np.random.uniform(wz_range[0], wz_range[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_action.fill(0.0)
        self._sample_commands()

        if self.model is not None and self.data is not None:
            mujoco.mj_resetData(self.model, self.data)
            
            # Set base position and orientation (standing at nominal height)
            self.data.qpos[0:3] = [0.0, 0.0, 0.35]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quaternion (w, x, y, z)
            
            # Initial joint angles with small noise for exploration
            noise = np.random.uniform(-0.05, 0.05, size=self.num_joints)
            self.data.qpos[7:19] = self.default_joint_pos + noise
            self.data.qvel[:] = 0.0

            # Forward kinematics to update body positions
            mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def _get_obs(self):
        if self.model is None or self.data is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        # 1. Base orientation quaternion & Projected Gravity
        quat = self.data.qpos[3:7]  # [w, x, y, z]
        w, x, y, z = quat
        # Rotation matrix to project world gravity (0, 0, -1) into body frame:
        gx = 2.0 * (x * z - w * y)
        gy = 2.0 * (y * z + w * x)
        gz = 1.0 - 2.0 * (x * x + y * y)
        projected_gravity = np.array([-gx, -gy, -gz], dtype=np.float32)

        # 2. Base angular velocity in body frame (gyro)
        base_ang_vel = np.array(self.data.qvel[3:6], dtype=np.float32)

        # 3. Base linear velocity in body frame
        base_lin_vel = np.array(self.data.qvel[0:3], dtype=np.float32)

        # 4. Joint positions relative to default
        joint_pos_rel = (self.data.qpos[7:19] - self.default_joint_pos).astype(np.float32)

        # 5. Joint velocities
        joint_vel = self.data.qvel[6:18].astype(np.float32)

        obs = np.concatenate([
            base_ang_vel,          # 3
            projected_gravity,     # 3
            self.commands,         # 3
            joint_pos_rel,         # 12
            joint_vel,             # 12
            self.prev_action,      # 12
            base_lin_vel           # 3
        ]).astype(np.float32)

        return np.clip(obs, -50.0, 50.0)

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0)
        
        # Target angles via PD control: q_des = q_nominal + action * scale
        target_q = self.default_joint_pos + (action * self.action_scale)

        # Simulate physics substeps
        total_torque = 0.0
        if self.model is not None and self.data is not None:
            for _ in range(self.sim_substeps):
                q = self.data.qpos[7:19]
                dq = self.data.qvel[6:18]
                
                # Compute PD motor torques
                torques = self.kp * (target_q - q) - self.kd * dq
                torques = np.clip(torques, -self.torque_limit, self.torque_limit)

                # Fault injection check (for Phase 2)
                if self.fault_config["type"] == "dead_motor" and self.fault_config["joint_idx"] is not None:
                    torques[self.fault_config["joint_idx"]] = 0.0

                self.data.ctrl[:] = torques
                total_torque += np.sum(np.square(torques))
                mujoco.mj_step(self.model, self.data)

        # Compute observations, reward, termination
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action, total_torque / max(1, self.sim_substeps))
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps

        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, reward_info

    def _compute_reward(self, action, avg_torque_sq):
        if self.model is None:
            return 0.0, {}

        # Actual velocities
        lin_vel = self.data.qvel[0:3]
        ang_vel = self.data.qvel[3:6]
        base_z = self.data.qpos[2]

        # 1. Forward tracking reward: exp(-||v_xy - cmd||^2 / 0.25)
        vel_err = np.sum(np.square(lin_vel[0:2] - self.commands[0:2]))
        r_lin_vel = 2.5 * np.exp(-vel_err / 0.25)

        # 2. Yaw rate tracking reward: exp(-|w_z - cmd_wz|^2 / 0.25)
        yaw_err = np.square(ang_vel[2] - self.commands[2])
        r_ang_vel = 1.2 * np.exp(-yaw_err / 0.25)

        # 3. Base height penalty (keep around target_height)
        r_height = -15.0 * np.square(base_z - self.target_height)

        # 4. Vertical velocity penalty (prevent bouncing)
        r_lin_vel_z = -2.0 * np.square(lin_vel[2])

        # 5. Pitch and Roll angular velocity penalties (wobble minimization)
        r_ang_vel_xy = -0.05 * np.sum(np.square(ang_vel[0:2]))

        # 6. Action smoothness (difference from previous step)
        r_action_rate = -0.01 * np.sum(np.square(action - self.prev_action))

        # 7. Energy / torque penalty
        r_torque = -0.0001 * avg_torque_sq

        # 8. Alive bonus
        r_alive = 0.5

        total_reward = (
            r_lin_vel +
            r_ang_vel +
            r_height +
            r_lin_vel_z +
            r_ang_vel_xy +
            r_action_rate +
            r_torque +
            r_alive
        )

        info = {
            "r_lin_vel": r_lin_vel,
            "r_ang_vel": r_ang_vel,
            "r_height": r_height,
            "base_z": base_z,
            "vx": lin_vel[0]
        }
        return float(total_reward), info

    def _is_terminated(self):
        if self.model is None or self.data is None:
            return False
        
        # Terminate if base falls below 0.15m or exceeds 0.55m
        base_z = self.data.qpos[2]
        if base_z < 0.14 or base_z > 0.55:
            return True

        # Terminate if tilted beyond ~50 degrees (quaternion dot product with vertical)
        quat = self.data.qpos[3:7]
        w, x, y, z = quat
        gz = 1.0 - 2.0 * (x * x + y * y)
        if gz < 0.6:  # cos(53 deg) ~ 0.6
            return True

        return False

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
