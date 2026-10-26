# ⚡ NVIDIA DGX & Isaac Lab Training Guide

This guide explains how to leverage your **NVIDIA DGX Cluster (A100 / H100 / SuperPOD)** to train the ADAPT-X Reinforcement Learning models with massive parallelism, and transfer the learned policies back to your laptop for real-time 3D visualization.

---

## 🏗️ Architecture: Hybrid Cloud / Cluster Workflow

```mermaid
graph TD
    A["🏢 NVIDIA DGX System<br/>(A100/H100/SuperPOD)"] -->|"Massive 128–4,096 Parallel Envs<br/>(40,000+ FPS Throughput)"| B["⚡ High-Speed Training<br/>(1M steps in ~90s)"]
    B -->|"Saves: dgx_drone_best_policy.pt<br/>or adaptx_policy.onnx"| C["📦 Checkpoint Export"]
    C -->|"git push or scp"| D["💻 Local Laptop (RTX 5050)<br/>(Interactive 3D Visualizer)"]
    D -->|"py -3.14 fly_drone.py"| E["🌆 Watch Flight in 3D City"]
```

---

## 🚀 Quick Start on NVIDIA DGX

### Step 1: Connect to your DGX Terminal & Clone
```bash
git clone https://github.com/bishtprateek270-hue/ADAPT-X.git
cd ADAPT-X
```

---

### Step 2: Choose Your Training Engine on DGX

#### 🟢 Option A: Vectorized DGX Trainer (Recommended, Zero Setup Overhead)
Scales training across **64 to 256 parallel environments** on DGX CPU cores and GPU tensor stream:
```bash
# Make script executable
chmod +x run_dgx.sh

# Run 1-click DGX launcher (128 parallel envs, 1,000,000 timesteps):
./run_dgx.sh

# Or run directly with custom parameters:
python3 train_dgx.py --env drone --num-envs 128 --timesteps 1000000 --device 0
```
* **Performance:** **$\sim 20,000 - 45,000\text{ FPS}$** (Completes 1 Million steps in **$\sim 45 - 90\text{ seconds}$**).
* **Output:** Saves trained weights to `checkpoints/dgx_drone_best_policy.pt`.

---

#### 🟣 Option B: NVIDIA Isaac Lab / Isaac Sim (Omniverse GPU Tensors)
If your DGX cluster has **NVIDIA Omniverse / Isaac Lab** installed:
```bash
# Run with Isaac Lab's Python launcher (4,096 parallel GPU actors):
./isaaclab.sh -p isaac_lab/train_isaac_lab.py --num_envs 4096 --robot drone --device cuda:0
```
* **Performance:** **$\sim 45,000+\text{ FPS}$** utilizing NVIDIA PhysX 5 GPU tensor streams.
* **Output:** Exports directly to `deployments/adaptx_isaac_policy.onnx`.

---

#### 🔵 Option C: Slurm Batch Scheduler (DGX SuperPOD / University Clusters)
If your DGX cluster uses Slurm for job scheduling:
```bash
# Submit batch job:
sbatch submit_slurm_dgx.sh

# Monitor status:
squeue -u $USER
tail -f logs/slurm_dgx_*.log
```

---

## 📥 Step 3: Copy Checkpoint to Your Laptop & Visualize

Once training completes on the DGX node, transfer the lightweight checkpoint file back to your laptop:

### Method 1: Via SCP / SFTP
```bash
# Run on your laptop terminal:
scp user@dgx-server:/path/to/ADAPT-X/checkpoints/dgx_drone_best_policy.pt ./checkpoints/
```

### Method 2: Via Git
On DGX:
```bash
git add checkpoints/dgx_drone_best_policy.pt deployments/*.onnx
git commit -m "feat: Add DGX-trained drone policy checkpoint"
git push origin main
```
On Laptop:
```bash
git pull origin main
```

---

## 🎮 Step 4: Run Interactive 3D Flight on Laptop
On your laptop, launch the 3D City visualizer using the newly trained DGX policy:

```powershell
py -3.14 fly_drone.py
```

Watch the drone glide through the 3D city with rock-solid stability and instant mid-air failure recovery learned on your NVIDIA DGX system!
