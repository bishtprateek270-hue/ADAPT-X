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

    def __init__(self, model_path=None, target_pos=(0.0, 0.0, 4.5), max_radius=42.0, max_altitude=45.0, max_steps=100000):
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

        # Thrust site IDs for dynamic arm length scaling across drone classes
        self.thrust_site_names = ["thrust_site_fr", "thrust_site_fl", "thrust_site_rr", "thrust_site_rl"]
        self.thrust_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.thrust_site_names
        ]
        self.default_mass = float(self.mass)
        self.default_arm_length = 0.18
        self.twr = 2.4
        self.max_thrust_per_rotor = float(self.twr * self.hover_thrust_per_rotor)
        self.motor_tau = 0.035
        self.current_thrust = np.full(4, self.hover_thrust_per_rotor, dtype=np.float32)
        self.morphology_randomization = False
        self.current_profile = "standard"
        self.action_mode = "auto"

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
        self.prop_qvel_indices = []
        self.prop_spin_dirs = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float32)  # FR CW, FL CCW, RR CCW, RL CW
        for name in self.prop_joint_names:
            try:
                j_id = self.model.joint(name).id
                self.prop_qpos_indices.append(self.model.jnt_qposadr[j_id])
                self.prop_qvel_indices.append(self.model.jnt_dofadr[j_id])
            except Exception:
                pass
        self.prop_angles = np.zeros(4, dtype=np.float32)
        self.prop_blur_names = ["prop_blur_fr", "prop_blur_fl", "prop_blur_rr", "prop_blur_rl"]
        self.prop_blur_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.prop_blur_names
        ]

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

    def sample_universal_morphology(self, profile=None):
        """
        Samples drone physical morphology across the full spectrum of world quadrotors:
          1. 'nano_fpv': 0.10 - 0.45 kg, arm 0.06 - 0.13 m, TWR 2.8 - 4.5, tau 0.018 - 0.030 s
          2. 'compact_inspection': 0.45 - 2.50 kg, arm 0.14 - 0.28 m, TWR 2.0 - 3.0, tau 0.030 - 0.055 s
          3. 'commercial_enterprise': 2.50 - 7.50 kg, arm 0.28 - 0.48 m, TWR 1.7 - 2.5, tau 0.050 - 0.085 s
          4. 'heavy_industrial_barc': 7.50 - 20.00 kg, arm 0.48 - 0.75 m, TWR 1.5 - 2.2, tau 0.080 - 0.140 s
        """
        profiles = ["nano_fpv", "compact_inspection", "commercial_enterprise", "heavy_industrial_barc"]
        if profile is None or profile == "random" or profile == "all":
            profile = str(np.random.choice(profiles))

        if profile == "nano_fpv":
            mass = float(np.random.uniform(0.12, 0.45))
            arm = float(np.random.uniform(0.06, 0.13))
            twr = float(np.random.uniform(2.8, 4.5))
            tau = float(np.random.uniform(0.018, 0.030))
        elif profile == "compact_inspection":
            mass = float(np.random.uniform(0.45, 2.50))
            arm = float(np.random.uniform(0.14, 0.28))
            twr = float(np.random.uniform(2.0, 3.0))
            tau = float(np.random.uniform(0.030, 0.055))
        elif profile == "commercial_enterprise":
            mass = float(np.random.uniform(2.50, 7.50))
            arm = float(np.random.uniform(0.28, 0.48))
            twr = float(np.random.uniform(1.7, 2.5))
            tau = float(np.random.uniform(0.050, 0.085))
        else:  # heavy_industrial_barc
            mass = float(np.random.uniform(7.50, 20.00))
            arm = float(np.random.uniform(0.48, 0.75))
            twr = float(np.random.uniform(1.5, 2.2))
            tau = float(np.random.uniform(0.080, 0.140))

        self.set_drone_morphology(mass=mass, arm_length=arm, twr=twr, tau=tau, profile_name=profile)
        return self.get_true_morphology()

    def set_drone_morphology(self, mass=1.5, arm_length=0.18, twr=2.2, tau=0.035, profile_name="custom"):
        """Explicitly configures drone physical dimensions, mass, inertia, and actuator dynamics."""
        self.mass = float(mass)
        self.arm_length = float(arm_length)
        self.twr = float(twr)
        self.motor_tau = float(tau)
        self.current_profile = profile_name

        # Inertia tensor computation (rigid body quadrotor approximation)
        I_xx = float(0.45 * self.mass * (self.arm_length ** 2))
        I_yy = float(0.45 * self.mass * (self.arm_length ** 2))
        I_zz = float(I_xx + I_yy)

        # Update MuJoCo rigid body mass and inertia
        self.model.body_mass[self.drone_body_id] = self.mass
        self.model.body_inertia[self.drone_body_id] = [I_xx, I_yy, I_zz]

        # Update thruster moment arms in MuJoCo
        signs = [(1.0, -1.0), (1.0, 1.0), (-1.0, -1.0), (-1.0, 1.0)]
        for s_id, (sx, sy) in zip(self.thrust_site_ids, signs):
            if s_id != -1:
                self.model.site_pos[s_id] = [sx * self.arm_length, sy * self.arm_length, 0.045]

        # Update hover thrust and max thrust limits
        self.hover_thrust_total = float(self.mass * self.gravity)
        self.hover_thrust_per_rotor = float(self.hover_thrust_total / 4.0)
        self.max_thrust_per_rotor = float(self.twr * self.hover_thrust_per_rotor)
        self.model.actuator_ctrlrange[:, 1] = self.max_thrust_per_rotor
        self.current_thrust = np.full(4, self.hover_thrust_per_rotor, dtype=np.float32)

    def get_true_morphology(self):
        """Returns 7-dim ground truth morphology vector: [mass, arm, I_xx, I_yy, I_zz, hover_thrust, max_thrust]."""
        I = self.model.body_inertia[self.drone_body_id]
        return np.array([
            float(self.mass),
            float(self.arm_length),
            float(I[0]),
            float(I[1]),
            float(I[2]),
            float(self.hover_thrust_per_rotor),
            float(self.max_thrust_per_rotor)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if getattr(self, "morphology_randomization", False):
            self.sample_universal_morphology()

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
        self.current_thrust = np.full(4, self.hover_thrust_per_rotor, dtype=np.float32)

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

        action = np.asarray(action, dtype=np.float32)
        # Check whether action is normalized [-1, 1] or raw thrust Newtons
        mode = getattr(self, "action_mode", "auto")
        if mode == "normalized":
            is_normalized = True
        elif mode == "thrust":
            is_normalized = False
        else: # "auto"
            # Auto mode: RL policies output signed actions spanning [-1, 1] with negative values
            # Controllers sending raw thrust commands in Newtons are strictly non-negative [0, max_thrust]
            is_normalized = bool(np.any(action < -0.05) and np.all(action >= -1.5) and np.all(action <= 1.5))

        if is_normalized:
            cmd_thrust = np.zeros(4, dtype=np.float32)
            for i in range(4):
                u = float(np.clip(action[i], -1.0, 1.0))
                if u >= 0.0:
                    cmd_thrust[i] = self.hover_thrust_per_rotor + u * (self.max_thrust_per_rotor - self.hover_thrust_per_rotor)
                else:
                    cmd_thrust[i] = self.hover_thrust_per_rotor * (1.0 + u)
        else:
            cmd_thrust = np.clip(action, 0.0, getattr(self, "max_thrust_per_rotor", 12.0))

        # Apply hardware degradation/damage
        target_thrust = cmd_thrust * self.rotor_health

        # First-order motor time-constant lag (ESC dynamics)
        tau = getattr(self, "motor_tau", 0.035)
        dt = getattr(self, "dt", 0.005) * 4.0
        alpha = float(np.clip(dt / (tau + dt), 0.1, 1.0))
        if not hasattr(self, "current_thrust") or self.current_thrust is None:
            self.current_thrust = target_thrust.copy()
        else:
            self.current_thrust = (1.0 - alpha) * self.current_thrust + alpha * target_thrust

        actual_thrust = self.current_thrust
        self.data.ctrl[:] = actual_thrust

        # Animate realistic high-speed propeller spinning and dynamic motion blur
        if len(self.prop_qpos_indices) == 4:
            for i in range(4):
                thrust_i = float(actual_thrust[i])
                blur_gid = self.prop_blur_geom_ids[i] if i < len(self.prop_blur_geom_ids) else -1
                if thrust_i > 0.4:
                    # High-RPM realistic rotation rate (non-harmonic increment to prevent 50Hz stroboscopic aliasing)
                    spin_speed = self.prop_spin_dirs[i] * (np.sqrt(thrust_i / 3.5) * 190.0 + 65.0)
                    self.prop_angles[i] = float((self.prop_angles[i] + spin_speed * 0.02) % (2.0 * np.pi))
                    self.data.qpos[self.prop_qpos_indices[i]] = self.prop_angles[i]
                    # Dynamic motion blur rotor disc opacity proportional to throttle
                    if blur_gid != -1:
                        blur_alpha = float(np.clip(0.40 + 0.35 * (thrust_i / 3.5), 0.20, 0.78))
                        self.model.geom_rgba[blur_gid][3] = blur_alpha
                else:
                    # Motor cut: blade stopped, blur disc hidden
                    if blur_gid != -1:
                        self.model.geom_rgba[blur_gid][3] = 0.0

        # Scale aerodynamic damping and guidance forces dynamically with drone mass and inertia
        mass_scale = float(np.clip(self.mass / 1.50, 0.08, 14.0))
        inertia_val = float(self.model.body_inertia[self.drone_body_id][0])
        inertia_scale = float(np.clip(inertia_val / 0.027, 0.01, 25.0))

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
            f_guidance = np.clip(2.6 * p_err + 3.2 * v_err, -8.0, 8.0) * mass_scale

            # Active APF building wall repulsion in damaged mode (attenuated when near target hover)
            f_apf, _ = self.obstacle_map.compute_apf_force(self.data.qpos[0:3], vel)
            f_apf_clamped = np.clip(f_apf[0:2], -6.0, 6.0) * mass_scale

            self.data.qfrc_applied[0:2] = f_guidance + f_apf_clamped - (0.20 * vel[0:2] * mass_scale)
            self.data.qfrc_applied[2] = -0.25 * vel[2] * mass_scale
            self.data.qfrc_applied[3] = (-0.45 * omega[0] - 4.5 * roll) * inertia_scale
            self.data.qfrc_applied[4] = (-0.45 * omega[1] - 4.5 * pitch) * inertia_scale
            self.data.qfrc_applied[5] = (-0.55 * omega[2]) * inertia_scale
        else:
            self.data.qfrc_applied[0:3] = -0.25 * vel * mass_scale
            self.data.qfrc_applied[3:6] = -0.025 * omega * inertia_scale

        # MuJoCo sub-steps to reach 50 Hz control rate (0.005s * 4 = 0.02s dt)
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)

        # Keep propeller angles synchronized in data.qpos and zero rotational hinge drift
        if len(self.prop_qpos_indices) == 4:
            for i in range(4):
                self.data.qpos[self.prop_qpos_indices[i]] = self.prop_angles[i]
                if i < len(self.prop_qvel_indices):
                    self.data.qvel[self.prop_qvel_indices[i]] = 0.0

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

        norm_effort = 0.002 * np.sum(actual_thrust / max(0.1, self.hover_thrust_per_rotor))
        reward = float(6.0 * r_target + 4.0 * r_z + 2.0 * r_vel + 2.0 * upright - penalty_obstacle - norm_effort)

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
            "true_morphology": self.get_true_morphology().tolist(),
            "mass": float(self.mass),
            "arm_length": float(self.arm_length),
            "profile": self.current_profile,
            "hover_thrust": float(self.hover_thrust_per_rotor),
            "max_thrust": float(self.max_thrust_per_rotor),
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

