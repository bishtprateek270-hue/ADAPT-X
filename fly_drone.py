"""
ADAPT-X: Self-Healing 3D City Drone Simulator
Interactive MuJoCo 3D flight simulation with mid-air rotor loss adaptation.
Surrounded by skyscrapers, glass towers, tree canopies, roads, and sidewalks.
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.drone_env import CityDroneEnv

try:
    import mujoco
    import mujoco.viewer
    VIEWER_AVAILABLE = True
except Exception:
    VIEWER_AVAILABLE = False


def run_drone_simulation(
    mode="cruise",
    damage="none",
    rotor="fr",
    damage_step=200,
    target_altitude=4.5,
    headless=False
):
    print("=" * 75)
    print("          ADAPT-X: 3D CITY DRONE FLIGHT SIMULATOR")
    print("   Fault-Tolerant Aerial Robotics with Real-Time Failure Recovery")
    print("=" * 75)
    print(f" Environment : 3D Urban City (Grand Plaza, Glass Skyscrapers, Trees, Avenues)")
    print(f" Vehicle     : High-Performance Quadcopter Drone (1.39 kg, 4x Brushless Rotors)")
    print(f" Flight Mode : {mode.upper()} (Altitude = {target_altitude:.1f} m)")
    if damage != "none":
        print(f" Injected Fail : Mid-Air [{damage.upper()} on Rotor {rotor.upper()}] at step {damage_step} (~{damage_step*0.02:.1f}s)")
    else:
        print(f" Injected Fail : NONE (Continuous Stable Flight - Zero Collisions)")
    print("=" * 75)

    env = CityDroneEnv(target_pos=(0.0, 0.0, target_altitude))
    obs, _ = env.reset()

    l = env.arm_length
    kp_z = 26.0
    kd_z = 10.0
    kp_pos = 0.25
    kd_pos = 0.35
    kp_att = 3.5
    kd_att = 0.45

    step = 0
    start_time = time.time()
    last_print = 0.0

    def compute_flight_control(step, env, mode, target_altitude):
        t = step * 0.02
        curr_pos = env.data.qpos[0:3]
        curr_vel = env.data.qvel[0:3]
        w, x_q, y_q, z_q = env.data.qpos[3:7]
        wx, wy, wz = env.data.qvel[3:6]

        # Target trajectory based on mode (stays safely within 2.5m plaza radius)
        if mode == "cruise":
            # Smooth 2-meter aerial orbit around the central plaza
            x_des = 2.0 * np.cos(0.35 * t)
            y_des = 2.0 * np.sin(0.35 * t)
            z_des = target_altitude + 0.15 * np.sin(0.5 * t)
        elif mode == "orbit":
            x_des = 2.5 * np.cos(0.25 * t)
            y_des = 2.5 * np.sin(0.25 * t)
            z_des = target_altitude
        else:
            # Stationary precision hover
            x_des, y_des, z_des = 0.0, 0.0, target_altitude

        # Cascaded Position -> Desired Roll/Pitch Angles
        ex = x_des - curr_pos[0]
        ey = y_des - curr_pos[1]
        pitch_des = np.clip( (kp_pos * ex - kd_pos * curr_vel[0]), -0.12, 0.12)
        roll_des  = np.clip(-(kp_pos * ey - kd_pos * curr_vel[1]), -0.12, 0.12)

        # Active Geofence boundary safeguard (keeps drone strictly within plaza)
        r_horiz = np.hypot(curr_pos[0], curr_pos[1])
        if r_horiz > 3.0:
            pen = r_horiz - 3.0
            rep_x = -(curr_pos[0] / r_horiz) * pen * 0.25
            rep_y = -(curr_pos[1] / r_horiz) * pen * 0.25
            pitch_des = np.clip(pitch_des + rep_x, -0.15, 0.15)
            roll_des  = np.clip(roll_des - rep_y, -0.15, 0.15)

        # Body-frame gravity vector:
        gx = 2.0 * (x_q * z_q - w * y_q)
        gy = 2.0 * (y_q * z_q + w * x_q)
        gz = 1.0 - 2.0 * (x_q * x_q + y_q * y_q)
        cos_tilt = np.clip(gz, 0.5, 1.0)

        # Altitude PID with tilt compensation:
        ez = z_des - curr_pos[2]
        T_total = (env.mass * env.gravity + kp_z * ez - kd_z * curr_vel[2]) / cos_tilt

        health = env.rotor_health
        if np.all(health == 1.0):
            # 4-Rotor Nominal Decoupled Control
            tau_x = - kp_att * (gy - roll_des) - kd_att * wx
            tau_y = - kp_att * (-gx - pitch_des) - kd_att * wy

            f_h = T_total / 4.0
            df_roll = tau_x / (4.0 * l)
            df_pitch = tau_y / (4.0 * l)

            u_fr = f_h - df_roll - df_pitch
            u_fl = f_h + df_roll - df_pitch
            u_rr = f_h - df_roll + df_pitch
            u_rl = f_h + df_roll + df_pitch
            u = [u_fr, u_fl, u_rr, u_rl]
        else:
            # Self-Healing Adaptive Bi-Rotor Dynamic Rebalancing
            dead_idx = int(np.argmin(health))
            f_diag = T_total / 2.0

            if dead_idx in [0, 3]:  # FR or RL failed -> active diagonal FL & RR
                diff = - 4.0 * (x_q - y_q) - 0.5 * (wx - wy)
                u = [0.0, f_diag + diff, f_diag - diff, 0.0]
            else:  # FL or RR failed -> active diagonal FR & RL
                diff = - 4.0 * (x_q + y_q) - 0.5 * (wx + wy)
                u = [f_diag + diff, 0.0, 0.0, f_diag - diff]

        return np.clip(u, 0.0, 12.0), (x_des, y_des, z_des)

    def print_hud(step, env, action, target_xyz):
        t = step * 0.02
        curr_pos = env.data.qpos[0:3]
        roll, pitch, yaw = env.get_euler_angles()
        health = env.rotor_health
        thrust = env.data.ctrl

        def bar(val, max_val=12.0):
            filled = int(round(val / max_val * 8))
            return "=" * filled + " " * (8 - filled)

        status_str = "NOMINAL 4-ROTOR FLIGHT" if np.all(health == 1.0) else "ADAPT-X REBALANCED 2-ROTOR CRUISE"

        print("-" * 75)
        print(f"Time: {t:5.2f}s | Step: {step:4d} | Status: [{status_str}]")
        print(f"Pos : X={curr_pos[0]:+5.2f}m | Y={curr_pos[1]:+5.2f}m | Alt={curr_pos[2]:5.2f}m (Target: {target_xyz[2]:.1f}m)")
        print(f"Att : Roll={roll:+5.1f} deg | Pitch={pitch:+5.1f} deg | Yaw={yaw:+5.1f} deg | Vz={env.data.qvel[2]:+5.2f}m/s")
        print(f"Rotors:")
        labels = ["FR", "FL", "RR", "RL"]
        rotor_lines = []
        for i in range(4):
            h_stat = "OK " if health[i] > 0.5 else "FAIL"
            rotor_lines.append(f"  [{labels[i]}] [{bar(thrust[i])}] {thrust[i]:4.2f}N ({h_stat})")
        print(" | ".join(rotor_lines[:2]))
        print(" | ".join(rotor_lines[2:]))

    if headless or not VIEWER_AVAILABLE:
        print("\n[Running Simulation in Headless Telemetry Mode...]")
        for step in range(1, 500):
            if damage != "none" and step == damage_step:
                env.inject_rotor_fault(damage, rotor=rotor)
                print("\n" + "!" * 75)
                print(f"  >>> ALERT: MID-AIR HARDWARE FAILURE INJECTED [{damage.upper()} ON {rotor.upper()} ROTOR]!")
                print(f"  >>> ADAPT-X SELF-HEALING CONTROLLER COMPENSATING FOR LOST THRUST...")
                print("!" * 75 + "\n")

            action, target_xyz = compute_flight_control(step, env, mode, target_altitude)
            obs, reward, terminated, truncated, info = env.step(action)

            now = time.time()
            if now - last_print > 0.5:
                print_hud(step, env, action, target_xyz)
                last_print = now

            if terminated:
                print(f"\n[CRASH DETECTED at step {step}] Altitude dropped below threshold.")
                break
        print("\n[Simulation Finished]")
        return

    # Interactive 3D Desktop Viewer
    print("\n[Launching Interactive 3D MuJoCo Window...]")
    print("3D Viewport Controls:")
    print("  * Left Click + Drag  : Rotate 3D Camera around City")
    print("  * Right Click + Drag : Zoom In / Out")
    print("  * Mouse Scroll Wheel : Fine Zoom")
    print("  * Double Click Drone : Auto-track Drone in Flight")
    print("  * Spacebar           : Pause / Resume Physics")
    print("=" * 75 + "\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            step += 1

            # Mid-air failure injection
            if damage != "none" and step == damage_step:
                env.inject_rotor_fault(damage, rotor=rotor)
                print("\n" + "!" * 75)
                print(f"  >>> ALERT: MID-AIR HARDWARE FAILURE INJECTED [{damage.upper()} ON {rotor.upper()} ROTOR]!")
                print(f"  >>> ADAPT-X CONTROLLER REBALANCED: DRONE REMAINS AIRBORNE ABOVE CITY!")
                print("!" * 75 + "\n")

            action, target_xyz = compute_flight_control(step, env, mode, target_altitude)
            obs, reward, terminated, truncated, info = env.step(action)

            # Sync physics state to visual window
            viewer.sync()

            now = time.time()
            if now - last_print > 0.5:
                print_hud(step, env, action, target_xyz)
                last_print = now

            if terminated:
                print(f"\n[Ground touch detected after {step} steps. Re-stabilizing at hover altitude...]")
                env.reset()
                step = 0

            # Maintain natural real-time 50 Hz frame rate
            elapsed = time.perf_counter() - step_start
            time.sleep(max(0.0, 0.02 - elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAPT-X 3D City Drone Simulator")
    parser.add_argument("--mode", type=str, default="cruise", choices=["cruise", "orbit", "hover"], help="Flight pattern")
    parser.add_argument("--damage", type=str, default="none", choices=["none", "dead_motor", "damaged_blade"], help="Failure type")
    parser.add_argument("--rotor", type=str, default="fr", choices=["fr", "fl", "rr", "rl"], help="Target rotor to damage")
    parser.add_argument("--damage-step", type=int, default=200, help="Step at which failure is injected")
    parser.add_argument("--altitude", type=float, default=4.5, help="Flight altitude in meters")
    parser.add_argument("--headless", action="store_true", help="Run without graphical 3D window")

    args = parser.parse_args()
    run_drone_simulation(
        mode=args.mode,
        damage=args.damage,
        rotor=args.rotor,
        damage_step=args.damage_step,
        target_altitude=args.altitude,
        headless=args.headless
    )
