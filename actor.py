"""
actor.py
========
ReinFlow rollout actor for ros_jackal.

Key design decisions:
  - Each actor is assigned ONE fixed world for its entire lifetime
  - env is created ONCE, reset() called between episodes (no close/recreate)
  - env obs is IGNORED for scan/goal — we call gazebo_sim directly for raw data
  - Flush to disk after EVERY episode (not batched)
  - unpause/pause around get_laser_scan() calls (env leaves Gazebo paused)

obs format (from jackal_gazebo_envs.py):
  - env obs[:720] normalized to (-1,1) with laser_clip=4  ← IGNORED
  - env obs[720:722] Cartesian goal normalized             ← IGNORED
  - We get raw [0,10]m scan and polar [dist,bearing] goal directly
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
import envs.registration
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


# ── LiDAR buffer ──────────────────────────────────────────────────────────────

class LiDARBuffer:
    def __init__(self, seq_len=8, num_rays=720):
        self.seq_len  = seq_len
        self.num_rays = num_rays
        self.buf      = deque(maxlen=seq_len)

    def push_from_sim(self, gazebo_sim):
        """Get raw scan — unpauses Gazebo to receive scan, then pauses again."""
        gazebo_sim.unpause()
        scan_msg = gazebo_sim.get_laser_scan()
        gazebo_sim.pause()
        scan = np.array(scan_msg.ranges, dtype=np.float32)
        scan = np.where(np.isfinite(scan), scan, 10.0)
        scan = np.clip(scan, 0.0, 10.0)
        if len(scan) != self.num_rays:
            scan = np.interp(
                np.linspace(0, len(scan)-1, self.num_rays),
                np.arange(len(scan)), scan
            ).astype(np.float32)
        self.buf.append(scan)

    def warmup(self):
        if len(self.buf) >= 1:
            first = self.buf[-1].copy()
            while len(self.buf) < self.seq_len:
                self.buf.appendleft(first)

    def get(self):
        return np.stack(list(self.buf))[np.newaxis].astype(np.float32)

    def reset(self):
        self.buf.clear()


# ── Goal vector ───────────────────────────────────────────────────────────────

def get_polar_goal(gazebo_sim, world_frame_goal):
    """[dist_m, bearing_rad] in robot frame. Matches ros_bridge.py exactly."""
    state = gazebo_sim.get_model_state()
    pos   = state.pose.position
    ori   = state.pose.orientation
    rx, ry = pos.x, pos.y
    ox, oy, oz, ow = ori.x, ori.y, ori.z, ori.w
    yaw  = math.atan2(2*(ow*oz + ox*oy), 1 - 2*(oy*oy + oz*oz))
    gx, gy = world_frame_goal
    dx, dy = gx - rx, gy - ry
    dist = math.sqrt(dx**2 + dy**2)
    bear = math.atan2(math.sin(math.atan2(dy, dx) - yaw),
                      math.cos(math.atan2(dy, dx) - yaw))
    return np.array([dist, bear], dtype=np.float32)


# ── Env ───────────────────────────────────────────────────────────────────────

def make_env(world_idx, cfg):
    env = gym.make(
        id='motion_control_continuous_laser-v0',
        world_name=f'BARN/world_{world_idx}.world',
        gui=False,
        init_position=[-2.25, 3, np.pi / 2],
        goal_position=[0, 10, 0],
        time_step=0.2,
        slack_reward=0,
        success_reward=float(cfg.get('r_goal', 20.0)),
        collision_reward=float(cfg.get('r_collision', -4.0)),
        failure_reward=0,
        goal_reward=1,
        max_collision=1,
    )
    return ShapingRewardWrapper(env)


# ── Policy ────────────────────────────────────────────────────────────────────

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

def get_my_world(actor_id, num_actors, world_range):
    """Each actor gets ONE fixed world for its entire lifetime."""
    lo, hi    = world_range
    per_actor = max(1, (hi - lo) // num_actors)
    world_idx = lo + actor_id * per_actor
    return min(world_idx, hi - 1)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device('cpu')

    os.makedirs(args.buffer_path, exist_ok=True)

    num_rays   = cfg.get('num_rays', 720)
    seq_len    = cfg.get('seq_len', 8)
    flow_steps = cfg.get('flow_steps', 10)
    v_max = cfg.get('v_max',  2.0);  v_min = cfg.get('v_min', -2.0)
    w_max = cfg.get('w_max', 4.0);  w_min = cfg.get('w_min', -4.0)
    # Fixed world for this actor's entire lifetime
    world_idx = get_my_world(args.id, args.num_actors,
                             cfg.get('world_range', [0, 250]))
    print(f"[Actor {args.id}] Fixed world: {world_idx}")

    # Wait for initial policy
    model, noise_net = None, None
    while model is None:
        model, noise_net = load_policy(args.buffer_path, device)
        if model is None:
            print(f"[Actor {args.id}] Waiting for policy_latest.pt ...")
            time.sleep(5.0)
    print(f"[Actor {args.id}] Policy ready")

    # Create env ONCE
    print(f"[Actor {args.id}] Loading world {world_idx}...")
    env              = make_env(world_idx, cfg)
    gazebo_sim       = env.env.gazebo_sim
    world_frame_goal = env.env.world_frame_goal

    lidar_buf = LiDARBuffer(seq_len=seq_len, num_rays=num_rays)
    episode   = 0

    while True:
        env.reset()
        lidar_buf.reset()
        lidar_buf.push_from_sim(gazebo_sim)
        lidar_buf.warmup()

        done       = False
        ep_reward  = 0.0
        step_count = 0
        ep_buf     = []

        while not done:
            lidar_buf.push_from_sim(gazebo_sim)
            lidar_seq = lidar_buf.get()
            goal_vec  = get_polar_goal(gazebo_sim, world_frame_goal)
            goal_in   = goal_vec[np.newaxis]

            with torch.no_grad():
                act_t, logp_t, _ = reinflow_forward(
                    model, noise_net,
                    torch.from_numpy(lidar_seq),
                    torch.from_numpy(goal_in),
                    n_steps=flow_steps
                )

            action   = act_t.numpy()[0]
            log_prob = logp_t.numpy()[0]

            action_clipped = np.array([
                np.clip(action[0], v_min, v_max),
                np.clip(action[1], w_min, w_max),
            ], dtype=np.float32)

            _, reward, done, info = env.step(action_clipped)

            lidar_buf.push_from_sim(gazebo_sim)
            next_lidar_seq = lidar_buf.get()
            next_goal_vec  = get_polar_goal(gazebo_sim, world_frame_goal)
            next_goal_in   = next_goal_vec[np.newaxis]

            ep_buf.append((
                lidar_seq[0], goal_in[0], action_clipped,
                next_lidar_seq[0], next_goal_in[0],
                np.float32(reward), np.float32(float(done)), log_prob,
            ))

            ep_reward  += reward
            step_count += 1

        collision = info.get('collided', False)
        success   = info.get('success', False)
        status    = ('SUCCESS' if success else 'COLLISION' if collision else 'TIMEOUT')
        print(f"[Actor {args.id}] Ep {episode:4d} | World {world_idx:3d} | "
              f"{step_count:3d} steps | R={ep_reward:+6.1f} | {status}")

        local_buf = ReplayBuffer(capacity=len(ep_buf)+1,
                                 seq_len=seq_len, num_rays=num_rays, device='cpu')
        for (l, g, a, nl, ng, r, d, lp) in ep_buf:
            local_buf.add(l, g, a, nl, ng, r, d, lp)

        out = os.path.join(args.buffer_path,
                           f'actor_{args.id:02d}_ep{episode:05d}.pt')
        local_buf.save(out)
        print(f"[Actor {args.id}] Flushed {len(ep_buf)} transitions -> {out}")

        new_model, new_noise = load_policy(args.buffer_path, device)
        if new_model is not None:
            model, noise_net = new_model, new_noise

        episode += 1


if __name__ == '__main__':
    main()
