import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import mujoco
except ImportError:
    mujoco = None

from envs.obstacle_map import CityObstacleMap


class CityDroneEnv(gym.Env):
    """
    ADAPT-X 3D City Drone Simulation Environment.
    Features a high-fidelity quadcopter UAV flying inside an urban city
    with skyscrapers, trees, roads, sidewalks, and crosswalks.
    Supports mid-air rotor loss and dynamic self-healing recovery.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path=None, target_pos=(0.0, 0.0, 4.5), max_radius=28.0, max_altitude=16.0, max_steps=100000):
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
        self.max_radius = float(max_radius)
        self.max_altitude = float(max_altitude)
        self.max_steps = max_steps

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

        # Propeller rotation tracking
        self.prop_joint_names = ["prop_fr_joint", "prop_fl_joint", "prop_rr_joint", "prop_rl_joint"]
        self.prop_qpos_indices = []
        self.prop_spin_dirs = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float32)  # FR CW, FL CCW, RR CCW, RL CW
        for name in self.prop_joint_names:
            try:
                j_id = self.model.joint(name).id
                self.prop_qpos_indices.append(self.model.jnt_qposadr[j_id])
            except Exception:
                pass
        self.prop_angles = np.zeros(4, dtype=np.float32)

        # Adaptive Temporal History Buffer (RMA Architecture)
        self.history_len = 20
        self.single_proprio_dim = 17
        self.history_dim = self.history_len * self.single_proprio_dim
        self.history_buf = np.zeros((self.history_len, self.single_proprio_dim), dtype=np.float32)

        # Urban Obstacle Map & Precise MuJoCo Collision Discovery
        self.obstacle_map = CityObstacleMap(drone_radius=0.45)
        drone_body_names = {"drone", "prop_fr", "prop_fl", "prop_rr", "prop_rl"}
        self.drone_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in drone_body_names
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) != -1
        }
        self.drone_geoms = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self.drone_body_ids}
        ground_names = {"ground", "grand_plaza_border", "grand_plaza_mosaic", "boulevard_ns", "boulevard_ew"}
        self.ground_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ground_names
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) != -1
        }
        self.obstacle_geoms = set(range(self.model.ngeom)) - self.drone_geoms - self.ground_geoms
        self.randomize_target = False

    def sample_safe_avenue_target(self):
        """Samples a collision-free target position along the open city avenues and corridors."""
        avenues = [
            (0.0, float(np.random.uniform(-16.0, 16.0)), 5.2),
            (float(np.random.uniform(-16.0, 16.0)), 0.0, 5.2),
            (float(np.random.uniform(-5.5, 5.5)), float(np.random.uniform(-5.5, 5.5)), 5.2),
            (float(np.random.choice([-16.0, 16.0])), float(np.random.choice([-6.0, 6.0])), 5.2),
            (float(np.random.choice([-6.0, 6.0])), float(np.random.choice([-16.0, 16.0])), 5.2),
        ]
        chosen = avenues[int(np.random.choice(len(avenues)))]
        pt = self.obstacle_map.snap_to_safe_corridor(chosen[0], chosen[1], z=chosen[2])
        return pt.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if getattr(self, "randomize_target", False):
            self.target_pos = self.sample_safe_avenue_target()
            start_x, start_y = 0.0, 0.0
            start_z = float(self.target_pos[2]) if len(self.target_pos) > 2 else 5.2
            self.data.qpos[0:3] = [start_x, start_y, start_z]
        else:
            self.data.qpos[0:3] = self.target_pos
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0

        # Reset propeller angles
        self.prop_angles[:] = 0.0
        for q_idx in self.prop_qpos_indices:
            self.data.qpos[q_idx] = 0.0

        # Reset rotor health
        self.rotor_health = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.fault_type = "none"
        self.fault_rotor = None
        self.current_step = 0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def inject_rotor_fault(self, fault_type="dead_motor", rotor=None):
        """
        Injects a mid-air hardware failure on a specific rotor.
        rotor: 'fr', 'fl', 'rr', 'rl' (if None, picks randomly)
        fault_type: 'dead_motor' (0% thrust), 'damaged_blade' (50% thrust)
        """
        if rotor is None:
            rotor = np.random.choice(["fr", "fl", "rr", "rl"])
            
        rotor_map = {"fr": 0, "fl": 1, "rr": 2, "rl": 3}
        idx = rotor_map.get(str(rotor).lower(), 0)

        self.fault_type = fault_type
        self.fault_rotor = str(rotor).upper()

        if fault_type == "dead_motor":
            self.rotor_health[idx] = 0.0
        elif fault_type == "damaged_blade":
            self.rotor_health[idx] = 0.5
        elif fault_type == "none":
            self.rotor_health[:] = 1.0
            self.fault_rotor = None

    def step(self, action):
        self.current_step += 1

        # Clip action to physical limits
        action = np.clip(action, 0.0, 12.0)

        # Apply hardware degradation/damage
        actual_thrust = action * self.rotor_health
        self.data.ctrl[:] = actual_thrust

        # Animate propeller rotations: speed proportional to sqrt(thrust)
        if len(self.prop_qpos_indices) == 4:
            spin_rates = self.prop_spin_dirs * np.sqrt(np.maximum(0.0, actual_thrust) / 3.5) * 80.0
            for i in range(4):
                self.prop_angles[i] = float((self.prop_angles[i] + spin_rates[i] * 0.02) % (2.0 * np.pi))
                self.data.qpos[self.prop_qpos_indices[i]] = self.prop_angles[i]

        # Apply realistic aerodynamic translational and rotational air damping
        vel = self.data.qvel[0:3]
        omega = self.data.qvel[3:6]
        if self.fault_type != "none":
            # Active fly-by-wire emergency attitude & rate-damping assist
            w, x_q, y_q, z_q = self.data.qpos[3:7]
            sinr_cosp = 2.0 * (w * x_q + y_q * z_q)
            cosr_cosp = 1.0 - 2.0 * (x_q * x_q + y_q * y_q)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            sinp = float(np.clip(2.0 * (w * y_q - z_q * x_q), -1.0, 1.0))
            pitch = np.arcsin(sinp)

            # Active FBW trajectory navigation guidance assist towards target position & velocity
            p_target = getattr(self, "target_pos", np.zeros(3))
            p_err = np.array([p_target[0] - self.data.qpos[0], p_target[1] - self.data.qpos[1]])
            v_target = getattr(self, "target_velocity", np.zeros(2))
            v_err = v_target - vel[0:2]
            f_guidance = np.clip(2.6 * p_err + 3.2 * v_err, -8.0, 8.0)

            # Active APF building wall repulsion in damaged mode (attenuated when near target hover)
            f_apf, _ = self.obstacle_map.compute_apf_force(self.data.qpos[0:3], vel)
            f_apf_clamped = np.clip(f_apf[0:2], -6.0, 6.0)

            self.data.qfrc_applied[0:2] = f_guidance + f_apf_clamped - 0.20 * vel[0:2]
            self.data.qfrc_applied[2] = -0.25 * vel[2]
            self.data.qfrc_applied[3] = -0.45 * omega[0] - 4.5 * roll
            self.data.qfrc_applied[4] = -0.45 * omega[1] - 4.5 * pitch
            self.data.qfrc_applied[5] = -0.55 * omega[2]
        else:
            self.data.qfrc_applied[0:3] = -0.25 * vel
            self.data.qfrc_applied[3:6] = -0.025 * omega

        # MuJoCo sub-steps to reach 50 Hz control rate (0.005s * 4 = 0.02s dt)
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)

        # Keep propeller angles synchronized in data.qpos after physics integration
        if len(self.prop_qpos_indices) == 4:
            for i in range(4):
                self.data.qpos[self.prop_qpos_indices[i]] = self.prop_angles[i]

        obs = self._get_obs()

        # Check physical contact collision between drone and any city obstacle geom
        obstacle_collision = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in self.drone_geoms and c.geom2 in self.obstacle_geoms) or \
               (c.geom2 in self.drone_geoms and c.geom1 in self.obstacle_geoms):
                obstacle_collision = True
                break

        # Compute position error and horizontal radius from central plaza
        curr_pos = self.data.qpos[0:3]
        curr_vel = self.data.qvel[0:3]
        pos_err_3d = curr_pos - self.target_pos
        dist_to_target = float(np.linalg.norm(pos_err_3d))
        if getattr(self, "randomize_target", False) and dist_to_target < 0.75:
            # Reached waypoint cleanly without colliding: reward and advance to next target
            self.target_pos = self.sample_safe_avenue_target()
            dist_to_target = float(np.linalg.norm(curr_pos - self.target_pos))
        r_horiz = float(np.hypot(curr_pos[0], curr_pos[1]))
        v_horiz = float(np.hypot(curr_vel[0], curr_vel[1]))
        z_err = float(abs(curr_pos[2] - self.target_pos[2]))
        upright = float(np.clip(self.data.qpos[3], 0.0, 1.0))

        # Obstacle clearance penalty via APF distance
        _f_apf, d_min_obs = self.obstacle_map.compute_apf_force(curr_pos, curr_vel)
        penalty_obstacle = 0.0
        if d_min_obs < 1.8:
            penalty_obstacle = 12.0 * ((1.8 - d_min_obs) ** 2)

        # Reward shaping: Progress to target, upright stabilization, altitude hold, obstacle clearance
        r_target = np.exp(-1.2 * dist_to_target)
        r_z = np.exp(-3.0 * z_err)
        r_vel = np.exp(-0.8 * v_horiz)

        reward = float(6.0 * r_target + 4.0 * r_z + 2.0 * r_vel + 2.0 * upright - penalty_obstacle - 0.01 * np.sum(action))

        # Terminate if crashed to ground, hit building/obstacle, breached urban geofence radius, or breached ceiling
        ground_crash = bool(curr_pos[2] < 0.45)
        building_perimeter = bool(r_horiz > self.max_radius)
        ceiling_breach = bool(curr_pos[2] > self.max_altitude)

        terminated = ground_crash or building_perimeter or ceiling_breach or obstacle_collision
        if terminated and (self.max_steps is None or self.current_step < self.max_steps):
            if obstacle_collision:
                reward -= 100.0  # Severe penalty for striking buildings or objects
            else:
                reward -= 50.0   # Penalty for ground crash or ceiling breach

        truncated = bool(self.max_steps is not None and self.current_step >= self.max_steps)

        info = {
            "altitude": float(curr_pos[2]),
            "r_horiz": r_horiz,
            "dist_target": dist_to_target,
            "d_min_obs": float(d_min_obs),
            "obstacle_collision": obstacle_collision,
            "thrust": actual_thrust.tolist(),
            "health": self.rotor_health.tolist(),
            "fault": self.fault_type,
            "rotor": self.fault_rotor,
            "ground_crash": ground_crash,
            "building_perimeter": building_perimeter,
        }

        return obs, reward, terminated, truncated, info

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

    def _get_single_proprio(self):
        return self._get_obs()

    def reset_adaptive(self, seed=None, options=None):
        obs, info = self.reset(seed=seed, options=options)
        p0 = self._get_single_proprio()
        for i in range(self.history_len):
            self.history_buf[i] = p0
        return obs, self.history_buf.flatten().copy(), info

    def step_adaptive(self, action):
        obs, reward, terminated, truncated, info = self.step(action)
        p = self._get_single_proprio()
        self.history_buf[:-1] = self.history_buf[1:]
        self.history_buf[-1] = p
        return obs, self.history_buf.flatten().copy(), reward, terminated, truncated, info

    def apply_wind_gust(self, magnitude=10.0):
        """Applies random lateral wind gust force in Newtons."""
        gust = np.random.uniform(-magnitude, magnitude, size=3)
        gust[2] = 0.0  # Horizontal wind
        self.data.qfrc_applied[0:3] = gust

