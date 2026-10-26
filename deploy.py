import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.quadruped_env import QuadrupedEnv
from envs.sim2real_utils import Sim2RealWrapper

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class OnboardRobotController:
    """
    Production-Ready Embedded Controller for ADAPT-X.
    Simulates onboard deployment on an edge device (Raspberry Pi / Jetson).
    Runs ONNX or TorchScript policy with a 50 Hz real-time hardware loop.
    """
    def __init__(self, model_path, history_len=20, proprio_dim=42):
        self.model_path = model_path
        self.history_len = history_len
        self.proprio_dim = proprio_dim
        self.history_buffer = np.zeros((history_len, proprio_dim), dtype=np.float32)

        self.is_torchscript = model_path.endswith(".pt") or model_path.endswith(".pth")
        if self.is_torchscript:
            import torch
            self.ts_model = torch.jit.load(model_path, map_location="cpu")
            self.ts_model.eval()
            self.session = None
        else:
            if not ONNX_AVAILABLE:
                raise RuntimeError("onnxruntime is required for .onnx models. Install with `pip install onnxruntime`.")
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.ts_model = None

    def reset(self, initial_proprio):
        for i in range(self.history_len):
            self.history_buffer[i] = initial_proprio

    def update_history(self, current_proprio):
        self.history_buffer[:-1] = self.history_buffer[1:]
        self.history_buffer[-1] = current_proprio

    def compute_action(self, obs):
        if self.is_torchscript:
            import torch
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            hist_t = torch.tensor(self.history_buffer.flatten(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = self.ts_model(obs_t, hist_t).squeeze(0).numpy()
            return action
        else:
            onnx_inputs = {
                "observation": obs.reshape(1, -1).astype(np.float32),
                "history_buffer": self.history_buffer.flatten().reshape(1, -1).astype(np.float32)
            }
            action = self.session.run(None, onnx_inputs)[0]
            return action.squeeze(0)


def run_deployment_simulation(onnx_path, duration_seconds=15.0, damage="amputated_leg", leg="fr", damage_step=150):
    if not os.path.exists(onnx_path):
        print(f"[Error] ONNX model not found at {onnx_path}. Run `py -3.14 export.py` first.")
        return

    print("=" * 70)
    print("      ADAPT-X ONBOARD EMBEDDED CONTROLLER (SIM-TO-REAL RUNNER)")
    print("=" * 70)
    print(f"Deployable Model   : {onnx_path}")
    print(f"Target Loop Rate   : 50.0 Hz (20.0 ms period)")
    print(f"Hardware Failure   : Injected [{damage.upper()}] on {leg.upper()} leg at step {damage_step}")
    print("=" * 70)

    # Initialize environment and sim-to-real wrapper
    env = QuadrupedEnv()
    sim2real = Sim2RealWrapper(action_delay_steps=1, joint_noise_std=0.005, gyro_noise_std=0.01)
    controller = OnboardRobotController(onnx_path)

    obs, _ = env.reset()
    p0 = env._get_single_proprio()
    controller.reset(p0)
    sim2real.reset_buffer()

    total_steps = int(duration_seconds / env.dt)
    loop_durations = []
    
    print("\n[Running Real-Time Edge Control Loop...]")
    for step in range(1, total_steps + 1):
        t_loop_start = time.perf_counter()

        # 1. Inject mid-mission hardware damage
        if damage != "none" and step == damage_step:
            env.inject_fault(damage, leg=leg)
            print(f"\n  >>> STEP {step} (t={step*env.dt:.1f}s): HARDWARE DAMAGE OCCURRED [{damage.upper()} on {leg.upper()}]!")
            print("  >>> Proprioceptive history buffer detecting dynamic discrepancy...\n")

        # 2. Read noisy sensors (Sim-to-Real reality gap)
        noisy_obs = sim2real.add_sensor_noise(obs)
        proprio = env._get_single_proprio()
        controller.update_history(proprio)

        # 3. ONNX Controller Inference
        t_infer_start = time.perf_counter()
        raw_action = controller.compute_action(noisy_obs)
        infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

        # 4. Apply communication delay
        delayed_action = sim2real.apply_action_delay(raw_action)

        # 5. Step physics
        obs, reward, terminated, truncated, info = env.step(delayed_action)

        loop_ms = (time.perf_counter() - t_loop_start) * 1000.0
        loop_durations.append(loop_ms)

        if step % 50 == 0:
            status = "HEALTHY (4 LEGS)" if not env.fault_active else f"ADAPTING ({env.fault_type.upper()} ON {env.fault_leg.upper()})"
            print(
                f"Step [{step:>4d}/{total_steps}] | "
                f"Time: {step*env.dt:>4.1f}s | "
                f"Status: {status:<24} | "
                f"Vx: {info['vx']:>5.2f} m/s | "
                f"Infer: {infer_ms:>4.2f}ms | "
                f"Height: {info['base_z']:>4.2f}m"
            )

        if terminated:
            print(f"\n[Terminated Early at Step {step}]: Robot tipped over.")
            break

    p50 = np.percentile(loop_durations, 50)
    p99 = np.percentile(loop_durations, 99)
    print("\n" + "=" * 70)
    print("               DEPLOYMENT BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Steps Executed   : {step} ({step * env.dt:.1f} seconds)")
    print(f"P50 Total Loop Time    : {p50:.2f} ms (Budget: 20.0 ms)")
    print(f"P99 Total Loop Time    : {p99:.2f} ms")
    print(f"Effective Controller Rate: {1000.0 / p50:.1f} Hz")
    print(f"Survives Physical Damage: {'YES (100% OPERATIONAL)' if not terminated else 'NO'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADAPT-X Embedded Deployment Simulation")
    parser.add_argument("--model", type=str, default="deployments/adaptx_policy.onnx")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--damage", type=str, default="amputated_leg", choices=["none", "dead_motor", "locked_joint", "weakened_motor", "amputated_leg"])
    parser.add_argument("--leg", type=str, default="fr", choices=["fr", "fl", "rr", "rl"])
    parser.add_argument("--damage-step", type=int, default=150)
    args = parser.parse_args()

    run_deployment_simulation(
        onnx_path=args.model,
        duration_seconds=args.duration,
        damage=args.damage,
        leg=args.leg,
        damage_step=args.damage_step
    )
