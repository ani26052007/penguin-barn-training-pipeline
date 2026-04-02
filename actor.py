"""
actor.py
========
ReinFlow rollout actor using ros_jackal's gym environment.

The gym env (motion_control_continuous_laser-v0) handles everything:
  - ROS/Gazebo launch and reset
  - LiDAR observation
  - Collision detection via info["collision"]
  - Episode termination
  - ShapingRewardWrapper adds progress reward

This script only needs to:
  1. Wrap obs into (8, 720) rolling window + polar goal
  2. Sample action via ReinFlow noisy ODE (stores log_prob for PPO)
  3. Store transitions in replay buffer
  4. Flush to shared buffer folder periodically

Usage (inside Singularity container):
  python actor.py --id 0 --num_actors 10 --config configs/reinflow.yaml
"""

import os
import sys
import time
import argparse
import math
from collections import deque

import numpy as np
import torch
import yaml
import gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import envs.registration   # registers motion_control_continuous_laser-v0
from envs.wrappers import ShapingRewardWrapper

from barn_nav_model import (BARNNavModel, ReinFlowNoiseNet,
                             CONTEXT_DIM, reinflow_forward)
from rl_algos.reinflow import ReplayBuffer


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--id',          type=int, default=0)
    p.add_argument('--num_actors',  type=int, default=10)
    p.add_argument('--config',      type=str, default='configs/reinflow.yaml')
    p.add_argument('--buffer_path', type=str,
                   default=os.environ.get('BUFFER_FOLDER', 'local_buffer'))
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Rolling LiDAR window ──────────────────────────────────────────────────────

class LiDARBuffer:
    def __init__(self, seq_len=8, num_rays=720):
        self.seq_len  = seq_len
        self.num_rays = num_rays
        self.buf      = deque(maxlen=seq_len)

    def push(self, scan):
        s = np.array(scan, dtype=np.float32)
        s = np.clip(s, 0.0, 10.0)
        self.buf.append(s)

    def warmup(self):
        """Fill buffer by repeating first scan."""
        if len(self.buf) >= 1:
            first = self.buf[-1].copy()
            while len(self.buf) < self.seq_len:
                self.buf.appendleft(first)

    def get(self):
        """Returns (1, seq_len, num_rays) float32 numpy."""
        return np.stack(list(self.buf))[np.newaxis].astype(np.float32)

    def reset(self):
        self.buf.clear()


# ── Goal vector from env ──────────────────────────────────────────────────────

def get_polar_goal(env):
    """
    Returns [distance_m, bearing_rad] in robot frame.
    Uses Gazebo ground truth position — same as ros_bridge.py.
    """
    state = env.gazebo_sim.get_model_state()
    pos   = state.pose.position
    ori   = state.pose.orientation

    rx, ry = pos.x, pos.y
    ox, oy, oz, ow = ori.x, ori.y, ori.z, ori.w
    yaw = math.atan2(2*(ow*oz + ox*oy), 1 - 2*(oy*oy + oz*oz))

    # Goal in world frame
    gx = env.init_position[0] + env.goal_position[0]
    gy = env.init_position[1] + env.goal_position[1]

    dx   = gx - rx
    dy   = gy - ry
    dist = math.sqrt(dx**2 + dy**2)
    bear = math.atan2(dy, dx) - yaw
    bear = math.atan2(math.sin(bear), math.cos(bear))   # wrap to [-π,π]

    return np.array([dist, bear], dtype=np.float32)


# ── Policy I/O ────────────────────────────────────────────────────────────────

def load_policy(buffer_path, device='cpu'):
    path = os.path.join(buffer_path, 'policy_latest.pt')
    if not os.path.exists(path):
        return None, None
    try:
        ckpt      = torch.load(path, map_location=device)
        model     = BARNNavModel()
        noise_net = ReinFlowNoiseNet(context_dim=CONTEXT_DIM)
        model.load_state_dict(ckpt['model_state'])
        noise_net.load_state_dict(ckpt['noise_net_state'])
        model.to(device).eval()
        noise_net.to(device).eval()
        return model, noise_net
    except Exception as e:
        print(f"[Actor] Could not load policy: {e}")
        return None, None


# ── World assignment ──────────────────────────────────────────────────────────

def get_my_worlds(actor_id, num_actors, world_range):
    lo, hi       = world_range
    per_actor    = max(1, (hi - lo) // num_actors)
    start        = lo + actor_id * per_actor
    end          = min(hi, start + per_actor)
    worlds       = list(range(start, end))
    return worlds if worlds else list(range(lo, hi))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device('cpu')   # actors always on CPU inside container

    os.makedirs(args.buffer_path, exist_ok=True)

    num_rays   = cfg.get('num_rays', 720)
    seq_len    = cfg.get('seq_len', 8)
    flow_steps = cfg.get('flow_steps', 10)
    flush_every = cfg.get('flush_every', 200)
    v_max = cfg.get('v_max', 2.0);  v_min = cfg.get('v_min', -2.0)
    w_max = cfg.get('w_max', 2.0);  w_min = cfg.get('w_min', -2.0)

    my_worlds = get_my_worlds(args.id, args.num_actors,
                              cfg.get('world_range', [0, 250]))
    print(f"[Actor {args.id}] Worlds {my_worlds[0]}–{my_worlds[-1]}")

    # Wait for learner to publish initial policy
    model, noise_net = None, None
    while model is None:
        model, noise_net = load_policy(args.buffer_path, device)
        if model is None:
            print(f"[Actor {args.id}] Waiting for policy_latest.pt ...")
            time.sleep(5.0)
    print(f"[Actor {args.id}] Policy ready")

    local_buf  = ReplayBuffer(capacity=10_000, seq_len=seq_len,
                              num_rays=num_rays, device='cpu')
    lidar_buf  = LiDARBuffer(seq_len=seq_len, num_rays=num_rays)
    episode    = 0

    while True:
        world_idx  = my_worlds[episode % len(my_worlds)]
        world_name = f'BARN/world_{world_idx}.world'

        # ── Build env ─────────────────────────────────────────────────────────
        env = gym.make(
            id='motion_control_continuous_laser-v0',
            world_name=world_name,
            gui=False,
            init_position=[-2.25, 3, np.pi / 2],
            goal_position=[0, 10, 0],
            time_step=0.2,
            slack_reward=0,
            success_reward=float(cfg.get('r_goal', 20.0)),
            collision_reward=float(cfg.get('r_collision', -4.0)),
            failure_reward=0,
            max_collision=1,
        )
        env = ShapingRewardWrapper(env)

        obs = env.reset()
        lidar_buf.reset()

        # First scan — warm up window
        scan = obs[:num_rays].astype(np.float32)
        lidar_buf.push(scan)
        lidar_buf.warmup()

        done        = False
        ep_reward   = 0.0
        step_count  = 0
        ep_buf      = []

        # ── Rollout ───────────────────────────────────────────────────────────
        while not done:
            scan      = obs[:num_rays].astype(np.float32)
            lidar_buf.push(scan)

            lidar_seq = lidar_buf.get()              # (1, 8, 720)
            goal_vec  = get_polar_goal(env)           # (2,)
            goal_in   = goal_vec[np.newaxis]          # (1, 2)

            # ReinFlow action sampling
            with torch.no_grad():
                act_t, logp_t, _ = reinflow_forward(
                    model, noise_net,
                    torch.from_numpy(lidar_seq),
                    torch.from_numpy(goal_in),
                    n_steps=flow_steps
                )

            action   = act_t.numpy()[0]    # (2,)
            log_prob = logp_t.numpy()[0]   # (1,)

            action_clipped = np.array([
                np.clip(action[0], v_min, v_max),
                np.clip(action[1], w_min, w_max),
            ], dtype=np.float32)

            # Step
            next_obs, reward, done, info = env.step(action_clipped)

            # Next observation
            next_scan = next_obs[:num_rays].astype(np.float32)
            lidar_buf.push(next_scan)
            next_lidar_seq = lidar_buf.get()         # (1, 8, 720)
            next_goal_vec  = get_polar_goal(env)
            next_goal_in   = next_goal_vec[np.newaxis]

            ep_buf.append((
                lidar_seq[0],         # (8, 720)
                goal_in[0],           # (2,)
                action_clipped,       # (2,)
                next_lidar_seq[0],    # (8, 720)
                next_goal_in[0],      # (2,)
                np.float32(reward),
                np.float32(float(done)),
                log_prob,             # (1,)
            ))

            ep_reward  += reward
            step_count += 1
            obs         = next_obs

        # ── End of episode ────────────────────────────────────────────────────
        collision = info.get('collision', 0)
        status    = '✓ SUCCESS' if (not collision and done) else (
                    '✗ COLLISION' if collision else '⏱ TIMEOUT')
        print(f"[Actor {args.id}] Ep {episode:4d} | World {world_idx:3d} | "
              f"{step_count:3d} steps | R={ep_reward:+6.1f} | {status}")

        env.close()

        # Store transitions
        for (l, g, a, nl, ng, r, d, lp) in ep_buf:
            local_buf.add(l, g, a, nl, ng, r, d, lp)

        # Flush to shared disk
        if len(local_buf) >= flush_every:
            out = os.path.join(args.buffer_path,
                               f'actor_{args.id:02d}_ep{episode:05d}.pt')
            local_buf.save(out)
            local_buf = ReplayBuffer(capacity=10_000, seq_len=seq_len,
                                     num_rays=num_rays, device='cpu')
            print(f"[Actor {args.id}] Flushed → {out}")

        # Reload latest policy (async — fine to be a few episodes behind)
        new_model, new_noise = load_policy(args.buffer_path, device)
        if new_model is not None:
            model, noise_net = new_model, new_noise

        episode += 1


if __name__ == '__main__':
    main()
