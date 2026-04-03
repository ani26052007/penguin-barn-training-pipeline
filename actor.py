"""
actor.py
========
ReinFlow rollout actor for ros_jackal.

Critical obs format notes (from reading envs/jackal_gazebo_envs.py):
  - env obs[:720] is LiDAR normalized to (-1,1) with laser_clip=4
  - env obs[720:722] is Cartesian goal normalized, NOT polar
  - Our model expects raw metres [0,10] LiDAR and polar [dist, bearing] goal
  - Solution: IGNORE env obs for scan/goal. Use env.gazebo_sim directly.

env.step() is still used for:
  - reward (ShapingRewardWrapper adds y-progress shaping)
  - done flag
  - info dict (collision, success, time)

Env lifecycle:
  - env.close() kills all ROS processes (killall -9)
  - gym.make() takes ~10s to relaunch Gazebo
  - We create one env per world, reset() between episodes on same world
  - Close+recreate only when switching worlds
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


# ── Raw observation helpers ───────────────────────────────────────────────────

class LiDARBuffer:
    """
    Rolling window of raw LiDAR scans in metres.
    Reads directly from env.gazebo_sim — bypasses env's normalization.
    """
    def __init__(self, seq_len=8, num_rays=720):
        self.seq_len  = seq_len
        self.num_rays = num_rays
        self.buf      = deque(maxlen=seq_len)

    def push_from_sim(self, gazebo_sim):
        """
        Get raw scan from GazeboSimulation and push.
        Returns raw clipped scan (720,) in metres [0,10].
        """
        scan_msg = gazebo_sim.get_laser_scan()
        scan     = np.array(scan_msg.ranges, dtype=np.float32)
        scan     = np.where(np.isfinite(scan), scan, 10.0)
        scan     = np.clip(scan, 0.0, 10.0)
        if len(scan) != self.num_rays:
            scan = np.interp(
                np.linspace(0, len(scan)-1, self.num_rays),
                np.arange(len(scan)), scan
            ).astype(np.float32)
        self.buf.append(scan)
        return scan

    def warmup(self):
        """Fill buffer by repeating current scan."""
        if len(self.buf) >= 1:
            first = self.buf[-1].copy()
            while len(self.buf) < self.seq_len:
                self.buf.appendleft(first)

    def get(self):
        """Returns (1, seq_len, num_rays) float32."""
        return np.stack(list(self.buf))[np.newaxis].astype(np.float32)

    def reset(self):
        self.buf.clear()


def get_polar_goal(gazebo_sim, world_frame_goal):
    """
    Compute [dist_m, bearing_rad] in robot frame from Gazebo ground truth.
    Matches ros_bridge.py exactly — what the model was trained on.

    gazebo_sim.get_model_state() returns ground truth pose.
    world_frame_goal: (gx, gy) tuple in world frame.
    """
    state = gazebo_sim.get_model_state()
    pos   = state.pose.position
    ori   = state.pose.orientation

    rx, ry = pos.x, pos.y
    ox, oy, oz, ow = ori.x, ori.y, ori.z, ori.w
    yaw = math.atan2(2*(ow*oz + ox*oy), 1 - 2*(oy*oy + oz*oz))

    gx, gy = world_frame_goal
    dx     = gx - rx
    dy     = gy - ry
    dist   = math.sqrt(dx**2 + dy**2)
    bear   = math.atan2(dy, dx) - yaw
    bear   = math.atan2(math.sin(bear), math.cos(bear))   # wrap [-π, π]

    return np.array([dist, bear], dtype=np.float32)


# ── Env construction ──────────────────────────────────────────────────────────

def make_env(world_idx, cfg):
    """
    Creates and wraps the gym env for one BARN world.
    Blocks for ~10s while Gazebo launches.
    """
    world_name = f'BARN/world_{world_idx}.world'
    env = gym.make(
        id='motion_control_continuous_laser-v0',
        world_name=world_name,
        gui=False,
        init_position=[-2.25, 3, np.pi / 2],
        goal_position=[0, 10, 0],
        time_step=0.2,           # 5Hz — matches original paper
        slack_reward=0,
        success_reward=float(cfg.get('r_goal', 20.0)),
        collision_reward=float(cfg.get('r_collision', -4.0)),
        failure_reward=0,
        goal_reward=1,           # ShapingRewardWrapper adds y-progress on top
        max_collision=1,
    )
    env = ShapingRewardWrapper(env)
    return env


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
    lo, hi    = world_range
    per_actor = max(1, (hi - lo) // num_actors)
    start     = lo + actor_id * per_actor
    end       = min(hi, start + per_actor)
    worlds    = list(range(start, end))
    return worlds if worlds else list(range(lo, hi))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device('cpu')   # actors always CPU inside container

    os.makedirs(args.buffer_path, exist_ok=True)

    num_rays    = cfg.get('num_rays', 720)
    seq_len     = cfg.get('seq_len', 8)
    flow_steps  = cfg.get('flow_steps', 10)
    flush_every = cfg.get('flush_every', 200)
    v_max = cfg.get('v_max', 2.0);  v_min = cfg.get('v_min', -1.0)
    w_max = cfg.get('w_max', 3.14); w_min = cfg.get('w_min', -3.14)

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

    local_buf       = ReplayBuffer(capacity=10_000, seq_len=seq_len,
                                   num_rays=num_rays, device='cpu')
    lidar_buf       = LiDARBuffer(seq_len=seq_len, num_rays=num_rays)
    episode         = 0
    current_world   = None
    env             = None

    while True:
        world_idx = my_worlds[episode % len(my_worlds)]

        # Close and recreate env only when world changes
        if world_idx != current_world:
            if env is not None:
                env.close()
                time.sleep(2.0)
            print(f"[Actor {args.id}] Loading world {world_idx}...")
            env           = make_env(world_idx, cfg)
            current_world = world_idx
            # env.env is the unwrapped MotionControlContinuousLaser
            # gazebo_sim lives there
            gazebo_sim        = env.env.gazebo_sim
            world_frame_goal  = env.env.world_frame_goal  # (gx, gy) in world frame

        # ── Reset episode ──────────────────────────────────────────────────────
        env.reset()
        lidar_buf.reset()

        # First raw scan + warmup
        lidar_buf.push_from_sim(gazebo_sim)
        lidar_buf.warmup()

        done       = False
        ep_reward  = 0.0
        step_count = 0
        ep_buf     = []

        # ── Rollout ───────────────────────────────────────────────────────────
        while not done:
            # Get raw observations — bypassing env's normalization
            lidar_buf.push_from_sim(gazebo_sim)
            lidar_seq = lidar_buf.get()                           # (1, 8, 720) metres
            goal_vec  = get_polar_goal(gazebo_sim, world_frame_goal)  # (2,) [dist, bearing]
            goal_in   = goal_vec[np.newaxis]                      # (1, 2)

            # ReinFlow action sampling
            with torch.no_grad():
                act_t, logp_t, _ = reinflow_forward(
                    model, noise_net,
                    torch.from_numpy(lidar_seq),
                    torch.from_numpy(goal_in),
                    n_steps=flow_steps
                )

            action   = act_t.numpy()[0]   # (2,) [cmd_lin, cmd_ang]
            log_prob = logp_t.numpy()[0]  # (1,)

            action_clipped = np.array([
                np.clip(action[0], v_min, v_max),
                np.clip(action[1], w_min, w_max),
            ], dtype=np.float32)

            # Step — reward/done/info come from env, obs is ignored
            _, reward, done, info = env.step(action_clipped)

            # Next raw observations
            lidar_buf.push_from_sim(gazebo_sim)
            next_lidar_seq = lidar_buf.get()
            next_goal_vec  = get_polar_goal(gazebo_sim, world_frame_goal)
            next_goal_in   = next_goal_vec[np.newaxis]

            ep_buf.append((
                lidar_seq[0],           # (8, 720) — current
                goal_in[0],             # (2,)
                action_clipped,         # (2,)
                next_lidar_seq[0],      # (8, 720) — next
                next_goal_in[0],        # (2,)
                np.float32(reward),
                np.float32(float(done)),
                log_prob,               # (1,)
            ))

            ep_reward  += reward
            step_count += 1

        # ── End of episode ────────────────────────────────────────────────────
        collision = info.get('collided', False)
        success   = info.get('success', False)
        status    = ('✓ SUCCESS'   if success  else
                     '✗ COLLISION' if collision else
                     '⏱ TIMEOUT')
        print(f"[Actor {args.id}] Ep {episode:4d} | World {world_idx:3d} | "
              f"{step_count:3d} steps | R={ep_reward:+6.1f} | {status}")

        # Store transitions in local buffer
        for (l, g, a, nl, ng, r, d, lp) in ep_buf:
            local_buf.add(l, g, a, nl, ng, r, d, lp)

        # Flush to shared disk buffer
        if len(local_buf) >= flush_every:
            out = os.path.join(args.buffer_path,
                               f'actor_{args.id:02d}_ep{episode:05d}.pt')
            local_buf.save(out)
            local_buf = ReplayBuffer(capacity=10_000, seq_len=seq_len,
                                     num_rays=num_rays, device='cpu')
            print(f"[Actor {args.id}] Flushed → {out}")

        # Reload latest policy (async)
        new_model, new_noise = load_policy(args.buffer_path, device)
        if new_model is not None:
            model, noise_net = new_model, new_noise

        episode += 1


if __name__ == '__main__':
    main()
