# ReinFlow BARN Navigation Challenge

Online RL fine-tuning of a flow matching navigation policy using ReinFlow,
built on top of the [ros_jackal](https://github.com/Daffan/ros_jackal) training infrastructure.

## Architecture

```
ConvNeXt1D encoder  (per-scan spatial features, 720 LiDAR rays)
      ↓
GRU (2-layer, hidden=256)  (temporal compression across 8 scans)
      ↓
CrossAttention  (goal queries LiDAR history)
      ↓
FlowMatchingHead  (Rectified Flow, 10-step ODE)
      ↓
(2,) [cmd_lin, cmd_ang]
```

**ReinFlow online fine-tuning** adds a small `ReinFlowNoiseNet` that converts
the deterministic flow ODE into a Markov chain, giving exact log-probabilities
without backpropagating through ODE steps. Uses PPO-style clipped updates.
The noise net is discarded at deployment — only the flow policy is used.

**Async actor-learner pipeline:**
- Actors (inside Singularity containers): collect transitions → write to `BUFFER_FOLDER`
- Learner (on host GPU): reads buffer → updates policy → writes `policy_latest.pt`

---

## File Structure

```
ros_jackal/
├── barn_nav_model.py          # Full model + ReinFlowNoiseNet + reinflow_forward()
├── actor.py                   # Rollout actor using gym env
├── train.py                   # Learner (GPU) — ReinFlow training loop
├── checkpoints/
│   └── iql_model_node_ready.pt  # Warm-start checkpoint (IQL or BC)
├── configs/
│   └── reinflow.yaml          # All hyperparameters
├── rl_algos/
│   └── reinflow.py            # update_critic, update_actor, ReplayBuffer
├── local_buffer/              # Runtime: policy_latest.pt + actor buffer files
└── logging/                   # TensorBoard logs
```

---

## Requirements

### Host machine (runs learner + TensorBoard)
- Python 3.8+
- PyTorch 2.x with CUDA
- `pip install -r requirements.txt`
- `pip install pyyaml tensorboard`

### Inside Singularity container (runs actors)
- PyTorch 1.10.1 already in image ✅
- All other deps already in image ✅

---

## Installation

### Step 1 — Clone and setup

```bash
git clone <your_repo_url>
cd ros_jackal
pip install -r requirements.txt
pip install pyyaml tensorboard
```

### Step 2 — Pull Singularity image (~3.5GB)

```bash
singularity pull --name local_buffer/nav_benchmark.sif \
    library://zifanxu/ros_jackal_image/image:latest
```

### Step 3 — Place warm-start checkpoint

```bash
mkdir -p checkpoints
cp /path/to/iql_model_node_ready.pt checkpoints/
```

### Step 4 — Set buffer path

```bash
export BUFFER_PATH=/absolute/path/to/ros_jackal/local_buffer
# Add to ~/.bashrc to make permanent:
echo 'export BUFFER_PATH=/absolute/path/to/ros_jackal/local_buffer' >> ~/.bashrc
```

### Step 5 — Verify imports inside container

```bash
./singularity_run.sh local_buffer/nav_benchmark.sif python3 test_imports.py
# All lines should print OK
```

---

## Running on Your Laptop (Testing Only)

Use this to verify the pipeline doesn't crash before running overnight on lab machine.
Requires ROS Melodic + Gazebo installed locally (not inside container).

```bash
# Terminal 1 — learner (will wait for actors to fill buffer)
export BUFFER_PATH=$(pwd)/local_buffer
python train.py --config configs/reinflow.yaml \
                --checkpoint checkpoints/iql_model_node_ready.pt

# Wait for: "[Learner] Initial policy saved — start actors now."

# Terminal 2 — single actor for testing
export BUFFER_PATH=$(pwd)/local_buffer
./singularity_run.sh local_buffer/nav_benchmark.sif \
    python3 actor.py --id 0 --num_actors 1

# Terminal 3 — monitor
tensorboard --logdir logging/
```

**What to check:**
- Actor prints episode results: `✓ SUCCESS` / `✗ COLLISION` / `⏱ TIMEOUT`
- Learner prints: `Step XXX | V=... Q=... actor_loss=...`
- TensorBoard shows `train/v_mean` increasing over time

**If actor crashes with AttributeError on `env.env.gazebo_sim`:**
```python
# In actor.py, change:
gazebo_sim = env.env.gazebo_sim
# To:
gazebo_sim = env.unwrapped.gazebo_sim
```

**If actor hangs during rollout (get_laser_scan deadlock):**
The env pauses Gazebo during step — `get_laser_scan()` may block waiting
for a scan that never comes because Gazebo is paused. Fix by getting the
scan from the obs instead of calling gazebo_sim directly during rollout:

```python
# In actor.py rollout loop, instead of:
lidar_buf.push_from_sim(gazebo_sim)
# Use the raw scan from the env's internal getter before step pauses:
# (see Known Issues section below)
```

---

## Running on Lab Machine (Full Training)

**Hardware:** AMD Threadripper 2990WX (32 cores), NVIDIA RTX 2080 Ti (11GB), 80GB RAM

```bash
# ── Terminal 1: Learner (GPU) ──────────────────────────────────
export BUFFER_PATH=/path/to/ros_jackal/local_buffer
python train.py --config configs/reinflow.yaml \
                --checkpoint checkpoints/iql_model_node_ready.pt

# Wait until you see:
# [Learner] Initial policy saved — start actors now.
# Then open terminals 2-11.

# ── Terminals 2-11: Actors (10 parallel Gazebo instances) ──────
export BUFFER_PATH=/path/to/ros_jackal/local_buffer

./singularity_run.sh local_buffer/nav_benchmark.sif python3 actor.py --id 0 --num_actors 10
./singularity_run.sh local_buffer/nav_benchmark.sif python3 actor.py --id 1 --num_actors 10
./singularity_run.sh local_buffer/nav_benchmark.sif python3 actor.py --id 2 --num_actors 10
./singularity_run.sh local_buffer/nav_benchmark.sif python3 actor.py --id 3 --num_actors 10
./singularity_run.sh local_buffer/nav_benchmark.sif python3 actor.py --id 4 --num_actors 10
./singularity_run.sh local_buffer/nav_benchmark.sif python3 actor.py --id 5 --num_actors 10
./singularity_run.sh local_buffer/nav_benchmark.sif python3 actor.py --id 6 --num_actors 10
./singularity_run.sh local_buffer/nav_benchmark.sif python3 actor.py --id 7 --num_actors 10
./singularity_run.sh local_buffer/nav_benchmark.sif python3 actor.py --id 8 --num_actors 10
./singularity_run.sh local_buffer/nav_benchmark.sif python3 actor.py --id 9 --num_actors 10

# ── Terminal 12: TensorBoard ────────────────────────────────────
tensorboard --logdir logging/
# Open browser: http://localhost:6006
```

---

## Resuming Training

If training is interrupted:

```bash
# Find latest checkpoint
ls local_buffer/train_ckpt_step*.pt | sort | tail -1

# Resume
python train.py --config configs/reinflow.yaml \
                --resume local_buffer/train_ckpt_step500000.pt
```

---

## Evaluation

After training, evaluate against the BARN challenge harness:

```bash
cd ~/AGV/the-barn-challenge

# Edit run.py section 1 to load your trained model
# Point model_path to local_buffer/train_ckpt_step<N>.pt

# Run on 50 test worlds
bash test.sh

# Report results
python report_test.py --out_path res/reinflow_out.txt
```

---

## TensorBoard Metrics

| Metric | Healthy sign |
|---|---|
| `train/v_mean` | Increases steadily |
| `train/q_mean` | Increases steadily |
| `train/actor_loss` | Decreases over time |
| `train/adv_mean` | Trends positive |
| `train/ratio_mean` | Stays near 1.0 |
| `train/ratio_max` | Stays below 5.0 |
| `buffer/size` | Grows as actors collect data |

**Warning signs:**
- `ratio_max` > 10 consistently → reduce `clip_eps` in `configs/reinflow.yaml`
- `v_mean` not increasing after 10k steps → reduce `lr_critic`
- `actor_loss` exploding → reduce `lr_actor`
- Buffer size not growing → actors are crashing, check actor terminal output

---

## Key Hyperparameters (`configs/reinflow.yaml`)

| Parameter | Default | Effect |
|---|---|---|
| `lr_actor` | 3e-5 | Low — backbone is warm-started |
| `lr_noise` | 3e-4 | Higher — noise net trains from scratch |
| `lr_critic` | 3e-4 | V and Q heads |
| `clip_eps` | 0.2 | PPO clipping — reduce if ratio_max too high |
| `flow_steps` | 10 | ODE steps during rollout |
| `flow_steps_train` | 4 | ODE steps during actor update (faster) |
| `critic_warmup_steps` | 5000 | Steps before actor update begins |
| `world_range` | [0, 250] | BARN worlds used for training |

---

## Known Issues and Fixes

### 1. `env.env.gazebo_sim` AttributeError
```python
# actor.py — try this if env.env doesn't work:
gazebo_sim = env.unwrapped.gazebo_sim
world_frame_goal = env.unwrapped.world_frame_goal
```

### 2. `get_laser_scan()` blocking during rollout
The gym env pauses Gazebo between steps. Calling `gazebo_sim.get_laser_scan()`
while Gazebo is paused will hang indefinitely. If this happens, unpause before
getting the scan:
```python
gazebo_sim.unpause()
lidar_buf.push_from_sim(gazebo_sim)
gazebo_sim.pause()
```

### 3. Actors can't find `barn_nav_model`
The container mounts repo at `/jackal_ws/src/ros_jackal`. If imports fail:
```python
# Add to top of actor.py:
sys.path.insert(0, '/jackal_ws/src/ros_jackal')
```
This is already in the file but verify it's the first sys.path insertion.

### 4. Buffer path mismatch
Learner and actors must use the same `BUFFER_PATH`. Always set:
```bash
export BUFFER_PATH=/absolute/path/to/ros_jackal/local_buffer
```
before running both learner and actors.

---

## Algorithm Reference

**ReinFlow** (https://reinflow.github.io):
- Injects learnable noise into flow ODE path
- Converts deterministic ODE → discrete Markov chain
- Gives exact log_prob without ODE backpropagation
- PPO-style clipped objective for stable updates
- Noise net discarded at deployment

**Critic** uses IQL-style updates (off-policy, replay buffer):
- V-head: expectile regression
- Q-head: twin Q Bellman backup
- Target networks: Polyak averaging (τ=0.005)
