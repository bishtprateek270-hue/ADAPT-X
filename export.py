import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.adaptive_ppo import AdaptiveActorCritic

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class DeployablePolicy(nn.Module):
    """
    Clean, zero-dependency deployment module for edge inference.
    Takes [obs, history] and directly produces target joint positions
    without critic networks or training graph overhead.
    """
    def __init__(self, history_encoder, actor_mean):
        super().__init__()
        self.history_encoder = history_encoder
        self.actor_mean = actor_mean

    def forward(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        # 1. Temporal history compression -> latent morphological state z_t
        z = self.history_encoder(history)
        
        # 2. Condition actor on [current_obs, latent_z]
        x = torch.cat([obs, z], dim=-1)
        
        # 3. Output deterministic action setpoints
        action = self.actor_mean(x)
        return action


def export_models(checkpoint_path, output_dir="./deployments", obs_dim=48, history_dim=840, act_dim=12, latent_dim=16):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")  # Export on CPU for universal edge portability

    # Load trained model
    full_model = AdaptiveActorCritic(
        obs_dim=obs_dim,
        history_dim=history_dim,
        act_dim=act_dim,
        latent_dim=latent_dim
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        full_model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Export] Loaded trained checkpoint from: {checkpoint_path}")
    else:
        print(f"[Export Warning] Checkpoint {checkpoint_path} not found. Exporting initialized weights.")

    full_model.eval()

    # Extract deployable sub-graph
    deployable_policy = DeployablePolicy(
        history_encoder=full_model.history_encoder,
        actor_mean=full_model.actor_mean
    ).to(device)
    deployable_policy.eval()

    # Dummy inputs for tracing
    dummy_obs = torch.randn(1, obs_dim, dtype=torch.float32, device=device)
    dummy_hist = torch.randn(1, history_dim, dtype=torch.float32, device=device)

    # 1. Export to TorchScript
    ts_path = os.path.join(output_dir, "adaptx_policy.torchscript.pt")
    traced_model = torch.jit.trace(deployable_policy, (dummy_obs, dummy_hist))
    traced_model.save(ts_path)
    print(f"[Export] Saved TorchScript model -> {ts_path} ({os.path.getsize(ts_path)/1024:.1f} KB)")

    # 2. Export to ONNX
    onnx_path = os.path.join(output_dir, "adaptx_policy.onnx")
    torch.onnx.export(
        deployable_policy,
        (dummy_obs, dummy_hist),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
        input_names=["observation", "history_buffer"],
        output_names=["target_joint_actions"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "history_buffer": {0: "batch_size"},
            "target_joint_actions": {0: "batch_size"}
        }
    )
    print(f"[Export] Saved ONNX model       -> {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")

    # 3. Numerical Parity Validation
    with torch.no_grad():
        torch_out = deployable_policy(dummy_obs, dummy_hist).numpy()

    if ONNX_AVAILABLE:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_inputs = {
            "observation": dummy_obs.numpy(),
            "history_buffer": dummy_hist.numpy()
        }
        onnx_out = session.run(None, onnx_inputs)[0]

        max_diff = np.max(np.abs(torch_out - onnx_out))
        print(f"[Validation] Max Numerical Parity Error: {max_diff:.8e}")
        assert max_diff < 1e-4, f"ONNX parity check failed! Diff: {max_diff}"
        print("[Validation] PASS: PyTorch and ONNX match with micro-precision!")

        # 4. Latency Benchmarking (1,000 iterations)
        print("\n[Benchmarking ONNX Edge Inference Latency (1,000 runs)...]")
        latencies = []
        for _ in range(1000):
            t0 = time.perf_counter()
            _ = session.run(None, onnx_inputs)
            latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        max_hz = 1000.0 / p50

        print("=" * 65)
        print(f"  P50 Inference Latency : {p50:.3f} ms ({p50*1000:.0f} microseconds)")
        print(f"  P95 Inference Latency : {p95:.3f} ms")
        print(f"  P99 Inference Latency : {p99:.3f} ms")
        print(f"  Theoretical Max Loop  : {max_hz:.0f} Hz (Required: 50 Hz)")
        print(f"  Headroom Margin       : {20.0 / p50:.1f}x faster than real-time deadline")
        print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Trained ADAPT-X Policy to ONNX and TorchScript")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase3/best_policy.pt")
    parser.add_argument("--output-dir", type=str, default="./deployments")
    args = parser.parse_args()

    export_models(args.checkpoint, args.output_dir)
