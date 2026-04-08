"""
barn_nav_model.py
=================
Complete model definition for BARN challenge online RL with ReinFlow.

Architecture:
  ConvNeXt1D encoder  (per-scan spatial features)
  GRU                 (temporal compression across 8 scans)
  CrossAttention      (goal queries LiDAR history)
  FlowMatchingHead    (Rectified Flow action head)
  ReinFlowNoiseNet    (NEW: noise injection for online RL — discarded at deployment)
  ValueHead           (V-function for IQL/ReinFlow critic)
  QHead               (twin Q-function for IQL/ReinFlow critic)

ReinFlow mechanics:
  - NoiseNet converts deterministic ODE → discrete Markov chain
  - Gives tractable log_prob without backpropping through ODE
  - Discarded after training; deployment uses standard flow.sample()

Usage:
  # Training
  model      = BARNNavModel()
  noise_net  = ReinFlowNoiseNet(context_dim=258)
  v_head     = ValueHead(context_dim=258)
  q_head     = QHead(context_dim=258)

  # Deployment (drop-in for model_node.py)
  model.load_state_dict(ckpt['model_state'])
  action = model.predict(lidar_seq_np, goal_np)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


# ── Constants ─────────────────────────────────────────────────────────────────
NUM_RAYS    = 720
SEQ_LEN     = 8
GRU_HIDDEN  = 256
GRU_LAYERS  = 2
ENCODER_DIM = 64
ATTN_HEADS  = 4
CONTEXT_DIM = GRU_HIDDEN + 2   # GRU output + goal (dist, bearing)
ACTION_DIM  = 2                 # (cmd_lin, cmd_ang)
FLOW_STEPS  = 10                # ODE integration steps


# ── ConvNeXt1D Encoder ────────────────────────────────────────────────────────

class ConvNeXt1DBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels,
                                kernel_size=7, padding=3, groups=channels)
        self.norm   = nn.LayerNorm(channels)
        self.mlp    = nn.Sequential(
            nn.Linear(channels, expansion * channels),
            nn.GELU(),
            nn.Linear(expansion * channels, channels),
        )

    def forward(self, x):
        r = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 2, 1)
        return x + r


class ConvNeXtEncoder(nn.Module):
    def __init__(self, out_dim=ENCODER_DIM):
        super().__init__()
        self.stem_conv = nn.Conv1d(1, 32, kernel_size=8, stride=8)
        self.stem_norm = nn.LayerNorm(32)
        self.stage1    = nn.Sequential(ConvNeXt1DBlock(32), ConvNeXt1DBlock(32))
        self.down1     = nn.Conv1d(32, 64, kernel_size=2, stride=2)
        self.stage2    = nn.Sequential(ConvNeXt1DBlock(64))
        self.pool      = nn.AdaptiveAvgPool1d(1)
        self.head      = nn.Linear(64, out_dim)

    def forward(self, x):
        # x: (B, 1, 720)
        x = self.stem_conv(x)
        x = x.permute(0, 2, 1)
        x = self.stem_norm(x)
        x = x.permute(0, 2, 1)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        return self.head(self.pool(x).squeeze(-1))  # (B, out_dim)


# ── Cross-Attention ────────────────────────────────────────────────────────────

class GoalLiDARCrossAttention(nn.Module):
    def __init__(self, goal_dim=2, gru_hidden=GRU_HIDDEN, num_heads=ATTN_HEADS):
        super().__init__()
        self.goal_proj = nn.Linear(goal_dim, gru_hidden)
        self.attn      = nn.MultiheadAttention(
            embed_dim=gru_hidden, num_heads=num_heads,
            dropout=0.1, batch_first=True)
        self.norm      = nn.LayerNorm(gru_hidden)

    def forward(self, goal, gru_out):
        # goal: (B, 2), gru_out: (B, GRU_HIDDEN)
        q      = self.goal_proj(goal).unsqueeze(1)
        kv     = gru_out.unsqueeze(1)
        out, _ = self.attn(q, kv, kv)
        return self.norm(out.squeeze(1) + gru_out)   # (B, GRU_HIDDEN)


# ── Flow Matching Head ─────────────────────────────────────────────────────────

class FlowMatchingHead(nn.Module):
    def __init__(self, context_dim=CONTEXT_DIM, action_dim=ACTION_DIM, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + context_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),                    nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),               nn.SiLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, a, ctx, t):
        # a: (B,2)  ctx: (B,D)  t: (B,1)
        return self.net(torch.cat([a, ctx, t], dim=-1))

    @torch.no_grad()
    def sample(self, ctx, n_steps=FLOW_STEPS):
        B, device = ctx.shape[0], ctx.device
        z  = torch.randn(B, ACTION_DIM, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B, 1), i * dt, device=device)
            z = z + dt * self.forward(z, ctx, t)
        return z   # (B, 2)


# ── ReinFlow Noise Network ─────────────────────────────────────────────────────

class ReinFlowNoiseNet(nn.Module):
    """
    Injects learnable Gaussian noise into the flow ODE path.
    Converts deterministic ODE → discrete-time Markov chain.
    Gives exact log_prob without backpropping through ODE steps.

    Architecture shares the context vector with FlowHead (same backbone).
    Small MLP on top — ~65k params.

    Input:  t       (B, 1)   flow time in [0,1]
            x_t     (B, 2)   current noisy action
            context (B, D)   goal-aware LiDAR context (from BARNNavModel)
    Output: sigma   (B, 2)   noise std at each action dimension (must be > 0)

    Discarded after training. Deployment uses standard flow.sample().
    """
    def __init__(self, context_dim=CONTEXT_DIM, flow_dim=ACTION_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + flow_dim + context_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, flow_dim),
            nn.Softplus(),   # output strictly positive (noise std)
        )
        # Bias toward small noise at init — avoids instability early in training
        with torch.no_grad():
            self.net[-2].bias.fill_(0.5)

    def forward(self, t, x_t, context):
        # t: (B,1)  x_t: (B,2)  context: (B,D)
        inp   = torch.cat([t, x_t, context], dim=-1)
        sigma = self.net(inp)
        # Clamp for numerical stability — sigma in [0.01, 2.0]
        return sigma.clamp(0.01, 2.0)


# ── Value and Q Heads ─────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """V(s) — scalar state value."""
    def __init__(self, context_dim=CONTEXT_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),      nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ctx):
        return self.net(ctx)   # (B, 1)


class QHead(nn.Module):
    """Twin Q(s,a) — returns min of two Q estimates for stability."""
    def __init__(self, context_dim=CONTEXT_DIM, action_dim=ACTION_DIM, hidden=256):
        super().__init__()
        inp = context_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(inp, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(inp, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ctx, action):
        x      = torch.cat([ctx, action], dim=-1)
        q1_val = self.q1(x)
        q2_val = self.q2(x)
        return q1_val, q2_val

    def q_min(self, ctx, action):
        q1, q2 = self.forward(ctx, action)
        return torch.min(q1, q2)


# ── Full Navigation Model ─────────────────────────────────────────────────────

class BARNNavModel(nn.Module):
    """
    Actor network — identical to offline training architecture.
    Warm-started from IQL checkpoint for online ReinFlow training.

    Forward pass (training):
      lidar_seq: (B, 8, 720)   8 consecutive LiDAR scans, raw metres [0,10]
      goal:      (B, 2)        [distance_m, bearing_rad] in robot frame
      a:         (B, 2)        noisy action for flow matching loss
      t:         (B, 1)        flow time in [0,1]
      → (B, 2) predicted velocity field

    Predict (deployment):
      lidar_np:  (1, 8, 720)   numpy
      goal_np:   (1, 2)        numpy
      → (1, 2)   numpy [cmd_lin, cmd_ang]
    """
    def __init__(self, seq_len=SEQ_LEN, gru_hidden=GRU_HIDDEN,
                 gru_layers=GRU_LAYERS, encoder_dim=ENCODER_DIM,
                 num_attn_heads=ATTN_HEADS):
        super().__init__()
        self.seq_len = seq_len
        self.encoder     = ConvNeXtEncoder(out_dim=encoder_dim)
        self.gru         = nn.GRU(encoder_dim, gru_hidden, gru_layers,
                                  batch_first=True,
                                  dropout=0.1 if gru_layers > 1 else 0.0)
        self.gru_dropout = nn.Dropout(0.1)
        self.cross_attn  = GoalLiDARCrossAttention(2, gru_hidden, num_attn_heads)
        self.flow        = FlowMatchingHead(gru_hidden + 2, ACTION_DIM, 256)

    def _encode(self, lidar_seq):
        B, T, L = lidar_seq.shape
        x       = self.encoder(
            lidar_seq.view(B * T, 1, L)
        ).view(B, T, -1)                   # (B, T, encoder_dim)
        _, h    = self.gru(x)              # h: (layers, B, hidden)
        return self.gru_dropout(h[-1])     # (B, hidden)

    def context(self, lidar_seq, goal):
        """Returns context vector for flow head and critic heads."""
        h_gru = self._encode(lidar_seq)              # (B, GRU_HIDDEN)
        h_att = self.cross_attn(goal, h_gru)         # (B, GRU_HIDDEN)
        return torch.cat([h_att, goal], dim=-1)       # (B, GRU_HIDDEN+2)

    def forward(self, lidar_seq, goal, a, t):
        """Training forward — returns velocity field prediction."""
        return self.flow(a, self.context(lidar_seq, goal), t)

    @torch.no_grad()
    def predict(self, lidar_np, goal_np, n_steps=FLOW_STEPS):
        """
        Deployment wrapper.
        lidar_np: (1, 8, 720) float32 numpy
        goal_np:  (1, 2)      float32 numpy
        returns:  (1, 2)      float32 numpy [cmd_lin, cmd_ang]
        """
        device = next(self.parameters()).device
        lidar  = torch.from_numpy(lidar_np).to(device)
        goal   = torch.from_numpy(goal_np).to(device)
        ctx    = self.context(lidar, goal)
        return self.flow.sample(ctx, n_steps).cpu().numpy()


# ── ReinFlow Forward Pass (used in rl_algos/reinflow.py) ──────────────────────

def reinflow_forward(model, noise_net, lidar_seq, goal, n_steps=FLOW_STEPS):
    """
    Forward pass through noisy flow ODE.
    Returns (action, log_prob, context).

    The noisy ODE is:
      x_{t+1} = x_t + v(x_t, ctx, t)*dt + σ(x_t, ctx, t)*√dt * ε
    where ε ~ N(0, I).

    log_prob is sum of Gaussian log-probs across all steps — exact and
    tractable because each transition is a Gaussian given (x_t, ctx, t).

    Args:
        model:     BARNNavModel
        noise_net: ReinFlowNoiseNet
        lidar_seq: (B, 8, 720)
        goal:      (B, 2)
        n_steps:   number of ODE steps

    Returns:
        action:   (B, 2)  final action after ODE
        log_prob: (B, 1)  sum of log probs across steps
        ctx:      (B, D)  context vector (for critic)
    """
    ctx = model.context(lidar_seq, goal)   # (B, D)
    B   = ctx.shape[0]
    device = ctx.device

    x        = torch.randn(B, ACTION_DIM, device=device)
    dt       = 1.0 / n_steps
    log_prob = torch.zeros(B, 1, device=device)

    for i in range(n_steps):
        t     = torch.full((B, 1), i * dt, device=device)
        v     = model.flow(x, ctx, t)                    # velocity field
        sigma = noise_net(t, x, ctx)                     # noise std
        noise = torch.randn_like(x)

        x_next = x + v * dt + sigma * (dt ** 0.5) * noise

        # Log prob of this Gaussian transition (exact)
        dist      = Normal(x + v * dt, sigma * (dt ** 0.5) + 1e-8)
        step_logp = dist.log_prob(x_next).sum(-1, keepdim=True)  # (B, 1)
        log_prob  = log_prob + step_logp

        x = x_next

    return x, log_prob, ctx


# ── Utility: load checkpoint ───────────────────────────────────────────────────

def load_barn_checkpoint(path, device='cpu'):
    """
    Loads IQL or BC checkpoint into BARNNavModel.
    Returns model with backbone weights loaded, ready for online training.
    """
    ckpt  = torch.load(path, map_location=device)
    model = BARNNavModel()

    key   = 'model_state' if 'model_state' in ckpt else 'actor_state'
    state = {k.replace('module.', ''): v for k, v in ckpt[key].items()}

    # Filter to only backbone keys (encoder + gru + cross_attn + flow)
    # This allows loading even if checkpoint has V/Q head keys
    model_keys = set(model.state_dict().keys())
    filtered   = {k: v for k, v in state.items() if k in model_keys}

    missing = model_keys - set(filtered.keys())
    if missing:
        print(f"[load_barn_checkpoint] WARNING: {len(missing)} keys missing: {list(missing)[:5]}")

    model.load_state_dict(filtered, strict=False)
    model.to(device)
    print(f"[load_barn_checkpoint] Loaded {len(filtered)}/{len(model_keys)} keys from {path}")
    print(f"  epoch: {ckpt.get('epoch', '?')}  val_loss: {ckpt.get('val_loss', '?')}")
    return model


if __name__ == '__main__':
    # Sanity check — all shapes correct
    B = 4
    lidar = torch.randn(B, SEQ_LEN, NUM_RAYS)
    goal  = torch.randn(B, 2)
    a     = torch.randn(B, ACTION_DIM)
    t     = torch.rand(B, 1)

    model     = BARNNavModel()
    noise_net = ReinFlowNoiseNet(context_dim=CONTEXT_DIM)
    v_head    = ValueHead(context_dim=CONTEXT_DIM)
    q_head    = QHead(context_dim=CONTEXT_DIM)

    # Training forward
    vf  = model(lidar, goal, a, t)
    ctx = model.context(lidar, goal)
    V   = v_head(ctx)
    Q1, Q2 = q_head(ctx, a)
    sigma   = noise_net(t, a, ctx)

    # ReinFlow forward
    action, log_prob, ctx2 = reinflow_forward(model, noise_net, lidar, goal)

    print("All shapes:")
    print(f"  velocity field : {vf.shape}")       # (4, 2)
    print(f"  context        : {ctx.shape}")       # (4, 258)
    print(f"  V(s)           : {V.shape}")         # (4, 1)
    print(f"  Q1(s,a)        : {Q1.shape}")        # (4, 1)
    print(f"  sigma          : {sigma.shape}")     # (4, 2)
    print(f"  action         : {action.shape}")    # (4, 2)
    print(f"  log_prob       : {log_prob.shape}")  # (4, 1)

    total = sum(p.numel() for p in model.parameters())
    noise = sum(p.numel() for p in noise_net.parameters())
    print(f"\nParams — model: {total:,}  noise_net: {noise:,}")
