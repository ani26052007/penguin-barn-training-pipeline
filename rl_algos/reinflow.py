"""
rl_algos/reinflow.py
====================
ReinFlow online RL algorithm for fine-tuning flow matching policies.

FIXES APPLIED (vs previous version):
  FIX 1 [CRITICAL] update_actor: n_steps now reads `flow_steps` (not
         `flow_steps_train`) so the PPO ratio exp(logp_new - logp_old) is
         computed between log-probs from the SAME Markov-chain length.
         Mixing step counts (train=4, collect=10) gives invalid ratios
         and destroys the policy.

  FIX 2 [CRITICAL] update_critic: model.context() is now called inside
         torch.no_grad() so the backbone NEVER receives gradient from the
         critic loss.  Previously the backbone was silently updated during
         every critic step, corrupting the pre-trained representation.

Key insight:
  Standard online RL (SAC/TD3) needs actor_loss = -Q(s, actor(s))
  which requires backprop THROUGH the ODE steps → unstable.

  ReinFlow instead:
    1. Injects noise into ODE path → converts to Markov chain
    2. Gets exact log_prob from Gaussian transitions (no ODE backprop)
    3. Uses PPO-style clipped objective → stable updates

Three update functions (called in train.py):
  update_critic()  — V + twin-Q Bellman update  (same as IQL offline)
  update_actor()   — ReinFlow PPO update
  polyak_update()  — soft target network update

Reference: ReinFlow paper (https://reinflow.github.io)
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from barn_nav_model import reinflow_forward


# ── Hyperparameters (overridden by configs/reinflow.yaml) ─────────────────────
GAMMA        = 0.99    # discount factor
TAU          = 0.005   # polyak averaging for target nets
BETA         = 1.5     # advantage temperature (lower = more exploration)
CLIP_EPS     = 0.2     # PPO clip epsilon
Q_CLAMP      = 100.0   # max absolute Q value (prevents bootstrap explosion)
EXPECTILE    = 0.7     # IQL expectile for V update
FLOW_STEPS   = 10      # ODE steps — MUST be same in actor.py and update_actor


# ── Critic Update (V + twin Q) ────────────────────────────────────────────────

def update_critic(batch, model, v_head, q_head, v_targ, q_targ,
                  opt_v, opt_q, cfg):
    """
    IQL-style critic update — identical to offline training.
    Works off-policy from replay buffer.

    FIX 2: model.context() is called inside torch.no_grad() so the backbone
    receives NO gradient from the critic loss. The ctx tensor has
    requires_grad=False after this block, so the subsequent v_head / q_head
    calls also do not propagate into the backbone.

    V update: expectile regression
      L_V = E[ l_τ(Q_targ(s,a) - V(s)) ]
      where l_τ(u) = |τ - 1(u<0)| * u²

    Q update: Bellman backup
      L_Q = E[ (Q(s,a) - (r + γ*V_targ(s')))² ]
    """
    lidar, goal, action, next_lidar, next_goal, reward, done = (
        batch['lidar'], batch['goal'], batch['action'],
        batch['next_lidar'], batch['next_goal'],
        batch['reward'], batch['done']
    )

    # FIX 2: wrap ALL backbone calls in no_grad so critic loss never
    #         reaches the encoder / GRU / cross-attention weights.
    with torch.no_grad():
        ctx        = model.context(lidar, goal)           # (B, D) — no grad
        ctx_next   = model.context(next_lidar, next_goal) # (B, D) — no grad
        q_targ_val = q_targ.q_min(ctx, action).clamp(-Q_CLAMP, Q_CLAMP)
        v_next     = v_targ(ctx_next)
        td_target  = reward + cfg.get('gamma', GAMMA) * (1 - done) * v_next
        td_target  = td_target.clamp(-Q_CLAMP, Q_CLAMP)

    # ── V update ──────────────────────────────────────────────────────────────
    # ctx has requires_grad=False (came out of no_grad), so v_loss gradient
    # stays inside v_head only.
    opt_v.zero_grad()
    v_pred = v_head(ctx)
    diff   = q_targ_val - v_pred
    tau    = cfg.get('expectile', EXPECTILE)
    weight = torch.where(diff > 0,
                         torch.full_like(diff, tau),
                         torch.full_like(diff, 1.0 - tau))
    v_loss = (weight * diff.pow(2)).mean()
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(v_head.parameters(), 1.0)
    opt_v.step()

    # ── Q update ──────────────────────────────────────────────────────────────
    opt_q.zero_grad()
    q1, q2 = q_head(ctx, action)
    q_loss  = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(q_head.parameters(), 1.0)
    opt_q.step()

    return {
        'v_loss':        v_loss.item(),
        'q_loss':        q_loss.item(),
        'v_mean':        v_pred.mean().item(),
        'q_mean':        ((q1 + q2) / 2).mean().item(),
        'td_target_mean': td_target.mean().item(),
    }


# ── Actor Update (ReinFlow PPO) ───────────────────────────────────────────────

def update_actor(batch, model, noise_net, v_targ, q_targ,
                 opt_actor, opt_noise, cfg):
    """
    ReinFlow actor update.

    FIX 1: n_steps now reads cfg['flow_steps'] (the COLLECTION step count),
    NOT cfg['flow_steps_train'].  The PPO importance-sampling ratio
        ratio = exp(log_prob_new - log_prob_old)
    is only valid when new and old log_probs are computed on Markov chains
    of the same length.  Using different step counts makes the ratio
    meaningless — it diverges and destroys the policy within a few thousand
    steps.

    If you want faster training updates, reduce flow_steps EVERYWHERE
    (actor collection AND training) by setting:
        flow_steps: 4       # in reinflow.yaml (affects both)
    Do NOT set flow_steps_train to a different value than flow_steps.

    Steps:
      1. Run noisy forward ODE → get (action, log_prob_new, context)
      2. Compute advantage: A = Q(s,a) - V(s)
      3. PPO clipped surrogate:
           ratio = exp(log_prob_new - log_prob_old)
           loss  = -min(ratio*A, clip(ratio, 1±ε)*A)

    Both model (flow head + backbone) and noise_net are updated,
    subject to the backbone_freeze_steps setting in train.py.
    """
    lidar, goal  = batch['lidar'], batch['goal']
    old_log_prob = batch['log_prob']   # stored during rollout (same n_steps)

    opt_actor.zero_grad()
    opt_noise.zero_grad()

    # FIX 1: use flow_steps (collection count), NOT flow_steps_train
    n_steps = cfg.get('flow_steps', FLOW_STEPS)
    action, log_prob_new, ctx = reinflow_forward(
        model, noise_net, lidar, goal, n_steps=n_steps
    )

    # Advantage — detached, no gradient through critic
    with torch.no_grad():
        adv_raw      = q_targ.q_min(ctx.detach(), action.detach()) \
                       - v_targ(ctx.detach())
        adv_raw_mean = adv_raw.mean().item()
        adv_raw_std  = adv_raw.std().item()
        # Normalise for stable gradient scale regardless of Q/V magnitude
        adv = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)
        adv = adv.clamp(-3.0, 3.0)

    # PPO clipped surrogate
    eps   = cfg.get('clip_eps', CLIP_EPS)
    ratio = torch.exp(log_prob_new - old_log_prob.detach())
    ratio = ratio.clamp(0.0, 10.0)   # guard against inf on first updates
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - eps, 1 + eps) * adv
    actor_loss = -torch.min(surr1, surr2).mean()

    actor_loss.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    noise_grad_norm = torch.nn.utils.clip_grad_norm_(noise_net.parameters(), 1.0)
    opt_actor.step()
    opt_noise.step()

    return {
        'actor_loss':      actor_loss.item(),
        'ratio_mean':      ratio.mean().item(),
        'ratio_max':       ratio.max().item(),
        'actor_grad_norm': actor_grad_norm.item(),
        'noise_grad_norm': noise_grad_norm.item(),
        'adv_raw_mean':    adv_raw_mean,
        'adv_raw_std':     adv_raw_std,
        'log_prob_mean':   log_prob_new.mean().item(),
    }


# ── Target Network Update ─────────────────────────────────────────────────────

@torch.no_grad()
def polyak_update(source, target, tau=TAU):
    """Soft update: target = τ*source + (1-τ)*target"""
    for s_param, t_param in zip(source.parameters(), target.parameters()):
        t_param.data.mul_(1 - tau)
        t_param.data.add_(tau * s_param.data)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores transitions for off-policy critic updates.
    Stores full (8, 720) lidar windows per transition.

    Schema per transition:
      lidar      (8, 720)  float32  — current 8-frame window
      goal       (2,)      float32  — [dist, bearing]
      action     (2,)      float32  — [cmd_lin, cmd_ang]
      next_lidar (8, 720)  float32  — next 8-frame window
      next_goal  (2,)      float32  — next goal vector
      reward     (1,)      float32  — NORMALISED reward (actor.py normalises)
      done       (1,)      float32
      log_prob   (1,)      float32  — log prob from noisy ODE at collection time
    """
    def __init__(self, capacity=500_000, seq_len=8, num_rays=720, device='cpu'):
        self.capacity = capacity
        self.seq_len  = seq_len
        self.num_rays = num_rays
        self.device   = device
        self.ptr      = 0
        self.size     = 0

        # Pre-allocate on CPU — move batches to GPU at sample time
        self.lidar      = torch.zeros(capacity, seq_len, num_rays, dtype=torch.float32)
        self.goal       = torch.zeros(capacity, 2,       dtype=torch.float32)
        self.action     = torch.zeros(capacity, 2,       dtype=torch.float32)
        self.next_lidar = torch.zeros(capacity, seq_len, num_rays, dtype=torch.float32)
        self.next_goal  = torch.zeros(capacity, 2,       dtype=torch.float32)
        self.reward     = torch.zeros(capacity, 1,       dtype=torch.float32)
        self.done       = torch.zeros(capacity, 1,       dtype=torch.float32)
        self.log_prob   = torch.zeros(capacity, 1,       dtype=torch.float32)

    def add(self, lidar, goal, action, next_lidar, next_goal,
            reward, done, log_prob):
        """Add one transition. All inputs are numpy arrays."""
        i = self.ptr
        self.lidar[i]      = torch.from_numpy(np.asarray(lidar,      dtype=np.float32))
        self.goal[i]       = torch.from_numpy(np.asarray(goal,       dtype=np.float32))
        self.action[i]     = torch.from_numpy(np.asarray(action,     dtype=np.float32))
        self.next_lidar[i] = torch.from_numpy(np.asarray(next_lidar, dtype=np.float32))
        self.next_goal[i]  = torch.from_numpy(np.asarray(next_goal,  dtype=np.float32))
        self.reward[i]     = float(reward)
        self.done[i]       = float(done)
        self.log_prob[i]   = torch.from_numpy(np.array(log_prob, dtype=np.float32))

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device='cuda'):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'lidar':      self.lidar[idx].to(device),
            'goal':       self.goal[idx].to(device),
            'action':     self.action[idx].to(device),
            'next_lidar': self.next_lidar[idx].to(device),
            'next_goal':  self.next_goal[idx].to(device),
            'reward':     self.reward[idx].to(device),
            'done':       self.done[idx].to(device),
            'log_prob':   self.log_prob[idx].to(device),
        }

    def __len__(self):
        return self.size

    def save(self, path):
        torch.save({
            'lidar':      self.lidar[:self.size],
            'goal':       self.goal[:self.size],
            'action':     self.action[:self.size],
            'next_lidar': self.next_lidar[:self.size],
            'next_goal':  self.next_goal[:self.size],
            'reward':     self.reward[:self.size],
            'done':       self.done[:self.size],
            'log_prob':   self.log_prob[:self.size],
            'ptr':        self.ptr,
            'size':       self.size,
        }, path)

    def load(self, path):
        data = torch.load(path, map_location='cpu')
        n    = data['size']
        self.lidar[:n]      = data['lidar']
        self.goal[:n]       = data['goal']
        self.action[:n]     = data['action']
        self.next_lidar[:n] = data['next_lidar']
        self.next_goal[:n]  = data['next_goal']
        self.reward[:n]     = data['reward']
        self.done[:n]       = data['done']
        self.log_prob[:n]   = data['log_prob']
        self.ptr  = data['ptr']
        self.size = n
        print(f"[ReplayBuffer] Loaded {n} transitions from {path}")
