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
  - Custom 6-component reward function replaces raw env reward
  - Running reward normalization: stats updated online, no hard-coded mean/std

obs format (from jackal_gazebo_envs.py):
  - env obs[:720] normalized to (-1,1) with laser_clip=4  ← IGNORED
  - env obs[720:722] Cartesian goal normalized             ← IGNORED
  - We get raw [0,10]m scan and polar [dist,bearing] goal directly

═══════════════════════════════════════════════════════════════════════════
CHANGELOG (BUG FIXES):
  FIX 3 — Reward normalization: online rewards are now normalized with a
           running mean/std computed from collected transitions.  The
           offline critic was trained on normalized rewards; pushing raw
           env rewards (magnitude ~20 on success, ~-4 on collision) into
           a critic calibrated for normalized targets produces completely
           wrong V/Q estimates and garbage advantages.

  NEW   — 6-component reward function with individual scaling via yaml:
           r_progress, r_collision, r_timeout, r_speed, r_obstacle, r_time
           Plus the unchanged terminal success bonus (r_goal).
═══════════════════════════════════════════════════════════════════════════
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


# ── Running Reward Normalizer ─────────────────────────────────────────────────

class RunningNormalizer:
    """
    Online Welford mean/variance estimator for reward normalization.
    Uses a fixed warm-up period before normalization kicks in.

    During warm-up (n < warmup_n) returns raw reward unchanged.
    After warm-up normalizes to approximately N(0,1) via:
        r_norm = (r - mean) / (std + eps)

    Clamps output to [-clip, clip] to prevent extreme advantages.
    """
    def __init__(self, warmup_n=500, clip=5.0, eps=1e-8):
        self.n       = 0
        self.mean    = 0.0
        self.M2      = 0.0        # sum of squared deviations
        self.warmup  = warmup_n
        self.clip    = clip
        self.eps     = eps

    def update(self, x: float):
        self.n  += 1
        delta    = x - self.mean
        self.mean += delta / self.n
        delta2   = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self):
        return self.M2 / max(self.n - 1, 1)

    @property
    def std(self):
        return math.sqrt(self.var + self.eps)

    def normalize(self, x: float) -> float:
        self.update(x)
        if self.n < self.warmup:
            return x          # raw during warm-up
        normed = (x - self.mean) / self.std
        return float(np.clip(normed, -self.clip, self.clip))


# ── 6-Component Reward Function ───────────────────────────────────────────────

def compute_shaped_reward(
    prev_goal_dist: float,
    curr_goal_dist: float,
    lidar_curr: np.ndarray,   # (720,) raw [0,10]m — latest scan
    action: np.ndarray,       # [cmd_lin, cmd_ang]
    info: dict,
    done: bool,
    cfg: dict,
) -> tuple:
    """
    6-component dense reward function.

    Returns:
        total   (float) — sum of all components
        parts   (dict)  — individual components for logging

    Components:
        1. progress  — reward proportional to distance closed toward goal
        2. collision — large penalty on contact (terminal)
        3. timeout   — moderate penalty on episode timeout (terminal)
        4. speed     — reward forward velocity, penalise reversing
        5. obstacle  — proximity penalty when within d_safe of nearest wall
        6. time      — small constant per-step penalty (encourages efficiency)
        + success    — large bonus on reaching goal (terminal, from config)
    """
    parts = {}

    # ── 1. Progress reward ────────────────────────────────────────────────────
    # Positive when robot moved toward goal; negative if it moved away.
    progress_m    = prev_goal_dist - curr_goal_dist   # metres closed
    parts['progress'] = cfg.get('r_progress', 1.0) * progress_m

    # ── 2. Collision penalty ──────────────────────────────────────────────────
    parts['collision'] = (
        cfg.get('r_collision', -4.0) if info.get('collided', False) else 0.0
    )

    # ── 3. Timeout penalty ────────────────────────────────────────────────────
    is_timeout = done and not info.get('success', False) \
                      and not info.get('collided', False)
    parts['timeout'] = cfg.get('r_timeout', -1.0) if is_timeout else 0.0

    # ── 4. Speed reward ───────────────────────────────────────────────────────
    # Only reward positive (forward) linear velocity.  Penalise reversing
    # slightly to discourage the robot from backing away from the goal.
    v_lin = float(action[0])
    if v_lin >= 0.0:
        parts['speed'] = cfg.get('r_speed', 0.05) * v_lin
    else:
        # Small penalty for reversing — magnitude capped at r_speed/2
        parts['speed'] = cfg.get('r_speed', 0.05) * 0.5 * v_lin

    # ── 5. Minimum-distance obstacle penalty ─────────────────────────────────
    # Smoothly penalise proximity to the nearest obstacle.
    # Activates only when d_min < d_safe (robot is in the danger zone).
    d_min  = float(np.min(lidar_curr))
    d_safe = cfg.get('d_safe', 0.5)          # metres — start of danger zone
    r_obs_k = cfg.get('r_obstacle_k', -0.5)  # penalty per metre inside zone
    if d_min < d_safe:
        parts['obstacle'] = r_obs_k * (d_safe - d_min)
    else:
        parts['obstacle'] = 0.0

    # ── 6. Time penalty ───────────────────────────────────────────────────────
    # Constant small negative reward — makes the robot prefer shorter paths.
    parts['time'] = cfg.get('r_time', -0.005)

    # ── Terminal success bonus ────────────────────────────────────────────────
    parts['success'] = (
        cfg.get('r_goal', 20.0) if info.get('success', False) else 0.0
    )

    total = sum(parts.values())
    return total, parts


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

    def latest_scan(self):
        """Returns most recent scan (720,) for reward computation."""
        return self.buf[-1].copy() if self.buf else np.full(self.num_rays, 10.0)

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
    """
    Create environment.  We set env-level rewards to 0 / neutral values
    because our custom compute_shaped_reward() replaces them entirely.
    The env reward is deliberately suppressed (slack_reward=0,
    collision_reward=0, success_reward=0) so we have full control.
    """
    env = gym.make(
        id='motion_control_continuous_laser-v0',
        world_name=f'BARN/world_{world_idx}.world',
        gui=False,
        init_position=[-2.25, 3, np.pi / 2],
        goal_position=[0, 10, 0],
        time_step=0.2,
        slack_reward=0,
        success_reward=0,    # We compute this in compute_shaped_reward()
        collision_reward=0,  # We compute this in compute_shaped_reward()
        failure_reward=0,
        goal_reward=0,       # We compute progress reward ourselves
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


# ── Diagnostic helpers ────────────────────────────────────────────────────────

def _diag_obs(lidar_seq, goal_vec, actor_id, episode):
    """Print observation stats every 10 episodes for sanity checking."""
    if episode % 10 != 0:
        return
    print(f"  [Diag Actor {actor_id}] ep={episode} "
          f"lidar=[{lidar_seq.min():.2f},{lidar_seq.max():.2f}] "
          f"goal_dist={goal_vec[0]:.2f}m bear={goal_vec[1]:.3f}rad")


def _diag_logprob(log_prob, actor_id, episode):
    if episode % 10 != 0:
        return
    lp = float(log_prob.item() if hasattr(log_prob, 'item') else log_prob)
    print(f"  [Diag Actor {actor_id}] ep={episode} log_prob={lp:.3f}  "
          f"(expected -20 to -5; NaN/±inf → sigma issue)")


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

    # Running reward normalizer — shared across all episodes
    # Warms up for 500 transitions before normalization kicks in.
    reward_normalizer = RunningNormalizer(
        warmup_n=cfg.get('reward_norm_warmup', 500),
        clip=cfg.get('reward_norm_clip', 5.0),
    )

    # Success-rate tracker (last 20 episodes)
    recent_outcomes = deque(maxlen=20)

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
        ep_parts   = {k: 0.0 for k in
                      ('progress','collision','timeout','speed',
                       'obstacle','time','success')}

        # Goal distance before first step (for progress reward)
        prev_goal_vec = get_polar_goal(gazebo_sim, world_frame_goal)

        while not done:
            lidar_buf.push_from_sim(gazebo_sim)
            lidar_seq = lidar_buf.get()          # (1, 8, 720)
            goal_vec  = get_polar_goal(gazebo_sim, world_frame_goal)
            goal_in   = goal_vec[np.newaxis]     # (1, 2)

            # Observation diagnostics (every 10 episodes)
            _diag_obs(lidar_seq, goal_vec, args.id, episode)

            with torch.no_grad():
                act_t, logp_t, _ = reinflow_forward(
                    model, noise_net,
                    torch.from_numpy(lidar_seq),
                    torch.from_numpy(goal_in),
                    n_steps=flow_steps
                )

            action   = act_t.numpy()[0]     # (2,)
            log_prob = logp_t.numpy()[0]    # (1,) or scalar

            _diag_logprob(log_prob, args.id, episode)

            action_clipped = np.array([
                np.clip(action[0], v_min, v_max),
                np.clip(action[1], w_min, w_max),
            ], dtype=np.float32)

            # Step the env — we DISCARD its reward; use our own below
            _, _env_reward, done, info = env.step(action_clipped)

            # Get next observation
            lidar_buf.push_from_sim(gazebo_sim)
            next_lidar_seq = lidar_buf.get()
            next_goal_vec  = get_polar_goal(gazebo_sim, world_frame_goal)
            next_goal_in   = next_goal_vec[np.newaxis]

            # ── Compute 6-component shaped reward ─────────────────────────────
            latest_scan = lidar_buf.latest_scan()   # (720,) raw metres
            raw_reward, parts = compute_shaped_reward(
                prev_goal_dist = float(prev_goal_vec[0]),
                curr_goal_dist = float(next_goal_vec[0]),
                lidar_curr     = latest_scan,
                action         = action_clipped,
                info           = info,
                done           = done,
                cfg            = cfg,
            )

            # ── FIX 3: Normalize reward before storing ─────────────────────────
            # Running normalizer keeps the reward scale consistent with what
            # the offline-pretrained V/Q critics expect.
            reward_norm = reward_normalizer.normalize(raw_reward)

            ep_buf.append((
                lidar_seq[0],       # (8, 720)
                goal_in[0],         # (2,)
                action_clipped,     # (2,)
                next_lidar_seq[0],  # (8, 720)
                next_goal_in[0],    # (2,)
                np.float32(reward_norm),
                np.float32(float(done)),
                log_prob,
            ))

            for k, v in parts.items():
                ep_parts[k] += v
            ep_reward  += raw_reward
            step_count += 1

            # Advance goal distance for next step's progress calculation
            prev_goal_vec = next_goal_vec

        # ── Episode done ──────────────────────────────────────────────────────
        collision = info.get('collided', False)
        success   = info.get('success',  False)
        status    = ('SUCCESS' if success else
                     'COLLISION' if collision else 'TIMEOUT')

        recent_outcomes.append(1 if success else 0)
        success_rate = sum(recent_outcomes) / len(recent_outcomes)

        # Per-component reward log
        parts_str = '  '.join(
            f"{k}={v:+.2f}" for k, v in ep_parts.items()
        )
        print(f"[Actor {args.id}] Ep {episode:4d} | World {world_idx:3d} | "
              f"{step_count:3d} steps | R_raw={ep_reward:+6.1f} | "
              f"SR={success_rate:.2f} | {status}")
        print(f"  Reward components: {parts_str}")
        print(f"  Normalizer: mean={reward_normalizer.mean:.3f} "
              f"std={reward_normalizer.std:.3f} n={reward_normalizer.n}")

        # Success-rate collapse detection
        if len(recent_outcomes) >= 20 and success_rate < 0.05:
            print(f"  ⚠ [Actor {args.id}] Low success rate ({success_rate:.2f}) "
                  f"over last 20 eps — check reward scale and lr_actor.")

        # ── Flush episode to disk ─────────────────────────────────────────────
        local_buf = ReplayBuffer(capacity=len(ep_buf)+1,
                                 seq_len=seq_len, num_rays=num_rays, device='cpu')
        for (l, g, a, nl, ng, r, d, lp) in ep_buf:
            local_buf.add(l, g, a, nl, ng, r, d, lp)

        out = os.path.join(args.buffer_path,
                           f'actor_{args.id:02d}_ep{episode:05d}.pt')
        local_buf.save(out)
        print(f"[Actor {args.id}] Flushed {len(ep_buf)} transitions → {out}")

        # ── Reload policy from learner ────────────────────────────────────────
        new_model, new_noise = load_policy(args.buffer_path, device)
        if new_model is not None:
            model, noise_net = new_model, new_noise

        episode += 1


if __name__ == '__main__':
    main()
