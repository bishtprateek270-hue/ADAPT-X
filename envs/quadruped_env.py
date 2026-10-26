import os
import random
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
    Phase 2: Dynamic Failure Injection, Damage Space Modeling,
    and Asymmetric Gait Discovery.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    LEG_JOINTS = {
        "fr": [0, 1, 2],  # [hip, thigh, calf]
        "fl": [3, 4, 5],
        "rr": [6, 7, 8],
        "rl": [9, 10, 11]
    }
    LEGS = ["fr", "fl", "rr", "rl"]
    FAILURE_TYPES = ["none", "dead_motor", "locked_joint", "weakened_motor", "amputated_leg"]

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
        self.max_episode_steps = int(self.config.get("environment", {}).get("max_episode_steps", 1000))

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

        # Phase 2: Failure Space Configuration
        damage_cfg = self.config.get("damage_injection", {})
        self.damage_enabled = bool(damage_cfg.get("enabled", False))
        self.failure_prob = float(damage_cfg.get("failure_prob", 0.6))
        self.trigger_range = damage_cfg.get("trigger_step_range", [150, 400])
        self.allowed_failure_types = damage_cfg.get("failure_types", ["dead_motor", "locked_joint", "amputated_leg"])

        # Active fault state
        self.fault_active = False
        self.fault_type = "none"
        self.fault_leg = None
        self.fault_joints = []
        self.fault_trigger_step = -1
        self.locked_angles = {}

        self.viewer = None

    def inject_fault(self, fault_type, leg="fr", joint_subidx=None):
        """Programmatically trigger a specific hardware failure."""
        assert fault_type in self.FAILURE_TYPES, f"Unknown fault type {fault_type}"
        self.fault_active = (fault_type != "none")
        self.fault_type = fault_type
        self.fault_leg = leg
        
        if leg in self.LEG_JOINTS:
            if joint_subidx is not None:
                self.fault_joints = [self.LEG_JOINTS[leg][joint_subidx]]
            else:
                self.fault_joints = list(self.LEG_JOINTS[leg])
        else:
            self.fault_joints = []

        # If locked joint, capture current joint position to lock in place
        self.locked_angles = {}
        if fault_type == "locked_joint" and self.data is not None:
            for j in self.fault_joints:
                self.locked_angles[j] = float(self.data.qpos[7 + j])

    def _sample_commands(self):
        """Sample velocity commands (ensuring positive forward drive)."""
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

        # Reset fault states
        self.fault_active = False
        self.fault_type = "none"
        self.fault_leg = None
        self.fault_joints = []
        self.locked_angles = {}

        # Schedule random mid-episode failure if enabled
        if self.damage_enabled and (np.random.rand() < self.failure_prob):
            self.fault_trigger_step = np.random.randint(self.trigger_range[0], self.trigger_range[1])
            self._scheduled_fault_type = np.random.choice(self.allowed_failure_types)
            self._scheduled_leg = np.random.choice(self.LEGS)
        else:
            self.fault_trigger_step = -1

        if self.model is not None and self.data is not None:
            mujoco.mj_resetData(self.model, self.data)
            
            # Start slightly above ground in ready posture
            self.data.qpos[0:3] = [0.0, 0.0, 0.35]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quat w, x, y, z
            
            noise = np.random.uniform(-0.05, 0.05, size=self.num_joints)
            self.data.qpos[7:19] = self.default_joint_pos + noise
            self.data.qvel[:] = 0.0

            mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def _get_obs(self):
        if self.model is None or self.data is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        # 1. Orientation quaternion & Projected Gravity
        quat = self.data.qpos[3:7]
        w, x, y, z = quat
        gx = 2.0 * (x * z - w * y)
        gy = 2.0 * (y * z + w * x)
        gz = 1.0 - 2.0 * (x * x + y * y)
        projected_gravity = np.array([-gx, -gy, -gz], dtype=np.float32)

        # 2. Base angular velocity (IMU Gyro)
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

        # Trigger scheduled mid-episode failure
        if not self.fault_active and (self.current_step == self.fault_trigger_step):
            self.inject_fault(self._scheduled_fault_type, self._scheduled_leg)

        # Target angles via PD control: q_des = q_nominal + action * scale
        target_q = self.default_joint_pos + (action * self.action_scale)

        # Simulate physics substeps with damage modeling
        total_torque = 0.0
        if self.model is not None and self.data is not None:
            for _ in range(self.sim_substeps):
                q = self.data.qpos[7:19]
                dq = self.data.qvel[6:18]
                
                # Base PD torque computation
                torques = self.kp * (target_q - q) - self.kd * dq

                # === APPLY PHASE 2 HARDWARE DAMAGE INJECTION ===
                if self.fault_active:
                    if self.fault_type == "dead_motor":
                        for j in self.fault_joints:
                            torques[j] = 0.0  # Limp, passive swing

                    elif self.fault_type == "locked_joint":
                        for j in self.fault_joints:
                            lock_angle = self.locked_angles.get(j, self.default_joint_pos[j])
                            # Stiff locking spring-damper to hold frozen position
                            torques[j] = 120.0 * (lock_angle - q[j]) - 10.0 * dq[j]

                    elif self.fault_type == "weakened_motor":
                        for j in self.fault_joints:
                            torques[j] = np.clip(torques[j], -self.torque_limit * 0.25, self.torque_limit * 0.25)

                    elif self.fault_type == "amputated_leg":
                        for j in self.fault_joints:
                            torques[j] = 0.0  # Complete loss of limb power

                # Clamp torques to physical limits
                torques = np.clip(torques, -self.torque_limit, self.torque_limit)
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

        lin_vel = self.data.qvel[0:3]
        ang_vel = self.data.qvel[3:6]
        base_z = self.data.qpos[2]

        # 1. Forward Tracking: Combination of exponential alignment + direct forward velocity bonus
        vel_err = np.sum(np.square(lin_vel[0:2] - self.commands[0:2]))
        r_lin_vel = 2.0 * np.exp(-vel_err / 0.25)
        # Direct reward for forward velocity matching target
        r_forward_progress = 3.0 * min(lin_vel[0], self.commands[0]) if lin_vel[0] > 0 else 2.0 * lin_vel[0]

        # 2. Yaw rate tracking
        yaw_err = np.square(ang_vel[2] - self.commands[2])
        r_ang_vel = 1.0 * np.exp(-yaw_err / 0.25)

        # 3. Base height penalty (keep near nominal 0.28m)
        r_height = -15.0 * np.square(base_z - self.target_height)

        # 4. Vertical velocity penalty (prevent bouncing)
        r_lin_vel_z = -1.5 * np.square(lin_vel[2])

        # 5. Pitch and Roll angular velocity penalties (wobble minimization)
        r_ang_vel_xy = -0.05 * np.sum(np.square(ang_vel[0:2]))

        # 6. Action smoothness
        r_action_rate = -0.008 * np.sum(np.square(action - self.prev_action))

        # 7. Energy efficiency penalty
        r_torque = -0.00008 * avg_torque_sq

        # 8. Alive bonus (gated by being upright)
        r_alive = 0.5 if base_z > 0.20 else 0.0

        total_reward = (
            r_lin_vel +
            r_forward_progress +
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
            "r_forward": r_forward_progress,
            "base_z": base_z,
            "vx": lin_vel[0],
            "fault_active": self.fault_active,
            "fault_type": self.fault_type
        }
        return float(total_reward), info

    def _is_terminated(self):
        if self.model is None or self.data is None:
            return False
        
        base_z = self.data.qpos[2]
        if base_z < 0.13 or base_z > 0.55:
            return True

        # Terminate if torso pitch/roll exceeds ~55 degrees
        quat = self.data.qpos[3:7]
        w, x, y, z = quat
        gz = 1.0 - 2.0 * (x * x + y * y)
        if gz < 0.55:
            return True

        return False

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
