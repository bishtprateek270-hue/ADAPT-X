import numpy as np


class Sim2RealWrapper:
    """
    Sim-to-Real Reality Gap Bridging Utility for ADAPT-X.
    Injects realistic sensor noise, action latency, payload offsets,
    and friction domain randomization.
    """
    def __init__(
        self,
        action_delay_steps=1,
        joint_noise_std=0.008,
        gyro_noise_std=0.015,
        friction_range=(0.4, 1.3),
        mass_perturbation_pct=0.15
    ):
        self.action_delay_steps = action_delay_steps
        self.joint_noise_std = joint_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.friction_range = friction_range
        self.mass_perturbation_pct = mass_perturbation_pct

        # Delay buffer for hardware bus communication latency (10-30ms)
        self.action_buffer = []

    def reset_buffer(self, num_joints=12):
        self.action_buffer = [np.zeros(num_joints, dtype=np.float32) for _ in range(self.action_delay_steps + 1)]

    def apply_action_delay(self, current_action):
        """Simulates CAN bus / microcontroller transmission delay."""
        self.action_buffer.append(current_action.copy())
        if len(self.action_buffer) > self.action_delay_steps:
            return self.action_buffer.pop(0)
        return self.action_buffer[0]

    def add_sensor_noise(self, obs):
        """
        Injects realistic Gaussian white noise into observation channels:
        Quadruped: [ang_vel(3), proj_g(3), cmd(3), joint_pos(12), joint_vel(12), prev_act(12), lin_vel(3)]
        Drone: [pos_err(3), quat(4), lin_vel(3), ang_vel(3), rotor_health(4)]
        """
        noisy_obs = obs.copy()
        if len(obs) == 17:
            # Drone: IMU gyro, velocity estimate, position estimation noise
            noisy_obs[0:3] += np.random.normal(0.0, 0.01, size=3)   # Position sensor noise
            noisy_obs[7:10] += np.random.normal(0.0, 0.02, size=3)  # Velocity estimation noise
            noisy_obs[10:13] += np.random.normal(0.0, self.gyro_noise_std, size=3) # IMU Gyro
        else:
            # Quadruped
            noisy_obs[0:3] += np.random.normal(0.0, self.gyro_noise_std, size=3)
            if len(obs) >= 33:
                noisy_obs[9:21] += np.random.normal(0.0, self.joint_noise_std, size=12)
                noisy_obs[21:33] += np.random.normal(0.0, self.joint_noise_std * 2.0, size=12)

        return noisy_obs.astype(np.float32)

    def randomize_physics(self, model, data):
        """Randomizes ground plane friction and torso mass to bridge physical reality gap."""
        if model is None:
            return

        # Randomize ground friction
        rand_friction = np.random.uniform(self.friction_range[0], self.friction_range[1])
        model.geom_friction[0, 0] = rand_friction

        # Randomize torso mass (+/- 15%)
        base_mass = 5.0
        mass_scale = np.random.uniform(1.0 - self.mass_perturbation_pct, 1.0 + self.mass_perturbation_pct)
        model.body_mass[1] = base_mass * mass_scale
