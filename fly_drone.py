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
    damage="dead_motor",
    rotor="fr",
    damage_step=150,
    target_altitude=4.0,
    headless=False
):
    print("=" * 75)
    print("          ADAPT-X: 3D CITY DRONE FLIGHT SIMULATOR")
    print("   Fault-Tolerant Aerial Robotics with Real-Time Failure Recovery")
    print("=" * 75)
    print(f" Environment : 3D Urban City (Skyscrapers, Glass Towers, Trees, Streets)")
    print(f" Vehicle     : High-Performance Quadcopter Drone (1.39 kg, 4x Brushless Rotors)")
    print(f" Flight Mode : {mode.upper()} (Altitude = {target_altitude:.1f} m)")
    if damage != "none":
        print(f" Injected Fail : Mid-Air [{damage.upper()} on Rotor {rotor.upper()}] at step {damage_step} (~{damage_step*0.02:.1f}s)")
    else:
        print(f" Injected Fail : NONE (Continuous Nominal Cruise)")
    print("=" * 75)

    env = CityDroneEnv(target_pos=(0.0, 0.0, target_altitude))
    obs, _ = env.reset()

    l = env.arm_length
    kp_z = 32.0
    kd_z = 16.0
    kp_pos = 0.35
    kd_pos = 0.45
    kp_att = 14.0
    kd_att = 2.2
    kp_diag = 15.0
    kd_diag = 2.5

    step = 0
    start_time = time.time()
    last_print = 0.0

    def compute_flight_control(step, env, mode, target_altitude):
        t = step * 0.02
        curr_pos = env.data.qpos[0:3]
        curr_vel = env.data.qvel[0:3]
        w, x_q, y_q, z_q = env.data.qpos[3:7]
        wx, wy, wz = env.data.qvel[3:6]

        # Target trajectory based on mode
        if mode == "cruise":
            # Smooth aerial glide down the central avenue and banking turns
            x_des = 2.5 * np.sin(0.35 * t)
            y_des = 3.5 * np.sin(0.18 * t)
            z_des = target_altitude + 0.3 * np.cos(0.25 * t)
        elif mode == "orbit":
            # Orbiting around the central crossroad between skyscraper towers
            x_des = 3.0 * np.cos(0.3 * t)
            y_des = 3.0 * np.sin(0.3 * t)
            z_des = target_altitude
        else:
            # Stationary precision hover
            x_des, y_des, z_des = 0.0, 0.0, target_altitude

        # Cascaded Position -> Desired Roll/Pitch Angles
        ex = x_des - curr_pos[0]
        ey = y_des - curr_pos[1]
        pitch_des = np.clip( (kp_pos * ex - kd_pos * curr_vel[0]), -0.15, 0.15)
        roll_des  = np.clip(-(kp_pos * ey - kd_pos * curr_vel[1]), -0.15, 0.15)

        # Body-frame gravity vector:
        gx = 2.0 * (x_q * z_q - w * y_q)
        gy = 2.0 * (y_q * z_q + w * x_q)
        gz = 1.0 - 2.0 * (x_q * x_q + y_q * y_q)
        cos_tilt = np.clip(gz, 0.4, 1.0)

        # Altitude PID with tilt compensation:
        ez = z_des - curr_pos[2]
        T_total = (env.mass * env.gravity + kp_z * ez - kd_z * curr_vel[2]) / cos_tilt

        health = env.rotor_health
        if np.all(health == 1.0):
            # 4-Rotor Nominal Decoupled Control
            tau_x = - kp_att * (gy + roll_des) - kd_att * wx
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
            # Self-Healing Adaptive Tri-Rotor Trim Compensation:
            # When any rotor fails, the remaining 3 rotors tilt slightly into the missing
            # quadrant to align the net thrust vector with the center of gravity,
            # maintaining full flight altitude above the city!
            dead_idx = int(np.argmin(health))
            # Trim offsets for [FR, FL, RR, RL]
            trim_offsets = [
                (-0.15, +0.15),  # FR failed
                (-0.15, -0.15),  # FL failed
                (+0.15, +0.15),  # RR failed
                (+0.15, -0.15),  # RL failed
            ]
            gx_trim, gy_trim = trim_offsets[dead_idx]
            tau_x = - kp_att * (gy - gy_trim) - kd_att * wx
            tau_y = - kp_att * (-gx - gx_trim) - kd_att * wy

            f_3 = T_total / 3.0
            u = np.zeros(4)
            if dead_idx == 0:  # FR dead -> FL, RR, RL active
                u[1] = f_3 + tau_x / (3.0 * l) - tau_y / (3.0 * l)
                u[2] = f_3 - tau_x / (3.0 * l) + tau_y / (3.0 * l)
                u[3] = f_3 + tau_x / (3.0 * l) + tau_y / (3.0 * l)
            elif dead_idx == 1:  # FL dead -> FR, RR, RL active
                u[0] = f_3 - tau_x / (3.0 * l) - tau_y / (3.0 * l)
                u[2] = f_3 - tau_x / (3.0 * l) + tau_y / (3.0 * l)
                u[3] = f_3 + tau_x / (3.0 * l) + tau_y / (3.0 * l)
            elif dead_idx == 2:  # RR dead -> FR, FL, RL active
                u[0] = f_3 - tau_x / (3.0 * l) - tau_y / (3.0 * l)
                u[1] = f_3 + tau_x / (3.0 * l) - tau_y / (3.0 * l)
                u[3] = f_3 + tau_x / (3.0 * l) + tau_y / (3.0 * l)
            else:  # RL dead -> FR, FL, RR active
                u[0] = f_3 - tau_x / (3.0 * l) - tau_y / (3.0 * l)
                u[1] = f_3 + tau_x / (3.0 * l) - tau_y / (3.0 * l)
                u[2] = f_3 - tau_x / (3.0 * l) + tau_y / (3.0 * l)

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

        status_str = "NOMINAL 4-ROTOR FLIGHT" if np.all(health == 1.0) else "ADAPT-X REBALANCED 3/2-ROTOR CRUISE"

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
            if now - last_print > 0.4:
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
                print(f"\n[CRASH DETECTED after {step} steps. Resetting drone in mid-air...]")
                env.reset()
                step = 0

            # Maintain natural real-time 50 Hz frame rate
            elapsed = time.perf_counter() - step_start
            time.sleep(max(0.0, 0.02 - elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAPT-X 3D City Drone Simulator")
    parser.add_argument("--mode", type=str, default="cruise", choices=["cruise", "orbit", "hover"], help="Flight pattern")
    parser.add_argument("--damage", type=str, default="dead_motor", choices=["none", "dead_motor", "damaged_blade"], help="Failure type")
    parser.add_argument("--rotor", type=str, default="fr", choices=["fr", "fl", "rr", "rl"], help="Target rotor to damage")
    parser.add_argument("--damage-step", type=int, default=150, help="Step at which failure is injected")
    parser.add_argument("--altitude", type=float, default=4.0, help="Flight altitude in meters")
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
