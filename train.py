"""
train.py
========
ReinFlow learner — runs on GPU (lab machine), reads transitions from
shared buffer folder written by actor.py processes.

Async actor-learner architecture:
  Actors (in Singularity): collect transitions → save to BUFFER_FOLDER/*.pt
  Learner (this script):   read buffer files → update policy → save to
                           BUFFER_FOLDER/policy_latest.pt

Usage:
  # Warm-start from IQL/BC checkpoint:
  python train.py --config configs/reinflow.yaml \
                  --checkpoint checkpoints/iql_model_node_ready.pt

  # Resume interrupted run:
  python train.py --config configs/reinflow.yaml \
                  --resume checkpoints/train_ckpt_step50000.pt

═══════════════════════════════════════════════════════════════════════════
CHANGELOG:
  NEW — backbone_freeze_steps phase in Phase 2:
        For the first N steps of full training, only the flow head and
        noise_net are updated; the ConvNeXt encoder, GRU, and cross-
        attention are frozen.  This prevents catastrophic forgetting of
        the pre-trained navigation prior during the first noisy critic
        bootstrap.  Controlled by backbone_freeze_steps in the yaml.

  NOTE — The update_critic backbone-detach bug (FIX 2) is handled in
         rl_algos/reinflow.py — no changes needed here for that fix.
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import copy
import time
import glob
import argparse
import yaml
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barn_nav_model import (BARNNavModel, ReinFlowNoiseNet,
                             ValueHead, QHead, CONTEXT_DIM,
                             load_barn_checkpoint)
from rl_algos.reinflow import (update_critic, update_actor,
                                polyak_update, ReplayBuffer)


_loaded_files = set()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',      type=str, default='configs/reinflow.yaml')
    p.add_argument('--checkpoint',  type=str, default=None,
                   help='Path to IQL/BC checkpoint for warm-starting backbone')
    p.add_argument('--resume',      type=str, default=None,
                   help='Path to previous train.py checkpoint to resume from')
    p.add_argument('--buffer_path', type=str,
                   default=os.environ.get('BUFFER_FOLDER', 'local_buffer'))
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def wait_for_buffer(buffer_path, min_transitions=1000, poll_interval=10.0):
    """Block until enough transitions are in the buffer."""
    print(f"[Learner] Waiting for {min_transitions} transitions in buffer...")
    while True:
        buf_files = glob.glob(os.path.join(buffer_path, 'actor_*.pt'))
        estimated = len(buf_files) * 80
        if estimated >= min_transitions:
            print(f"[Learner] Buffer ready: ~{estimated} transitions "
                  f"({len(buf_files)} files)")
            return
        print(f"[Learner] Buffer: {len(buf_files)} files "
              f"(~{estimated}/{min_transitions})... waiting")
        time.sleep(poll_interval)


def load_all_buffer_files(buffer_path, replay_buffer, max_files=500):
    global _loaded_files
    buf_files = sorted(glob.glob(os.path.join(buffer_path, 'actor_*.pt')))
    # Keep only the most recent files — stale-policy data is low value
    buf_files = buf_files[-max_files:]
    new_files = [f for f in buf_files if f not in _loaded_files]
    if not new_files:
        return 0
    loaded = 0
    for f in new_files:
        try:
            data = torch.load(f, map_location='cpu')
            n = data['reward'].shape[0]
            for i in range(n):
                replay_buffer.add(
                    data['lidar'][i].numpy(),
                    data['goal'][i].numpy(),
                    data['action'][i].numpy(),
                    data['next_lidar'][i].numpy(),
                    data['next_goal'][i].numpy(),
                    data['reward'][i].numpy(),
                    data['done'][i].numpy(),
                    data['log_prob'][i].numpy(),
                )
            _loaded_files.add(f)
            loaded += n
        except Exception as e:
            print(f"[Learner] Warning: could not load {f}: {e}")
    if loaded:
        print(f"[ReplayBuffer] Added {loaded} transitions, "
              f"buffer size={len(replay_buffer)}")
    return loaded


def save_policy(model, noise_net, buffer_path, step, cfg, extra=None):
    """Save policy_latest.pt for actors + optionally a full checkpoint."""
    policy_path = os.path.join(buffer_path, 'policy_latest.pt')
    torch.save({
        'model_state':     model.state_dict(),
        'noise_net_state': noise_net.state_dict(),
        'step':            step,
    }, policy_path)

    if extra is not None:
        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = os.path.join('checkpoints', f'train_ckpt_step{step}.pt')
        torch.save(extra, ckpt_path)
        print(f"[Learner] Saved full checkpoint → {ckpt_path}")


def freeze_backbone(model):
    """Freeze ConvNeXt encoder, GRU, and cross-attention — leave flow head free."""
    for p in model.encoder.parameters():    p.requires_grad = False
    for p in model.gru.parameters():        p.requires_grad = False
    for p in model.cross_attn.parameters(): p.requires_grad = False
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_free   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Learner] Backbone FROZEN: {n_frozen:,} params frozen, "
          f"{n_free:,} params free (flow head only)")


def unfreeze_backbone(model):
    """Unfreeze all backbone parameters for full fine-tuning."""
    for p in model.parameters(): p.requires_grad = True
    n = sum(p.numel() for p in model.parameters())
    print(f"[Learner] Backbone UNFROZEN: all {n:,} params now trainable")


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Learner] Device: {device}")

    os.makedirs(args.buffer_path, exist_ok=True)
    os.makedirs('logging', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer   = SummaryWriter(f'logging/reinflow_{run_name}')

    # ── Build models ──────────────────────────────────────────────────────────
    model     = BARNNavModel().to(device)
    noise_net = ReinFlowNoiseNet(context_dim=CONTEXT_DIM).to(device)
    v_head    = ValueHead(context_dim=CONTEXT_DIM).to(device)
    q_head    = QHead(context_dim=CONTEXT_DIM).to(device)

    # Target networks (EMA copies — not updated by optimizer)
    v_targ = copy.deepcopy(v_head).to(device)
    q_targ = copy.deepcopy(q_head).to(device)
    for p in v_targ.parameters(): p.requires_grad = False
    for p in q_targ.parameters(): p.requires_grad = False

    # ── Warm-start from IQL/BC checkpoint ─────────────────────────────────────
    start_step = 0
    if args.resume:
        print(f"[Learner] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        noise_net.load_state_dict(ckpt['noise_net_state'])
        v_head.load_state_dict(ckpt['v_head_state'])
        q_head.load_state_dict(ckpt['q_head_state'])
        v_targ.load_state_dict(ckpt['v_targ_state'])
        q_targ.load_state_dict(ckpt['q_targ_state'])
        start_step = ckpt.get('step', 0)
        print(f"[Learner] Resumed at step {start_step}")
    elif args.checkpoint:
        print(f"[Learner] Warm-starting backbone from {args.checkpoint}")
        model = load_barn_checkpoint(args.checkpoint, device)
        print("[Learner] V/Q heads: random init (trained from online data)")

    # ── Optimizers ────────────────────────────────────────────────────────────
    lr_actor  = cfg.get('lr_actor',  3e-5)
    lr_critic = cfg.get('lr_critic', 3e-4)
    lr_noise  = cfg.get('lr_noise',  3e-4)

    opt_actor = torch.optim.AdamW(model.parameters(),     lr=lr_actor,  weight_decay=1e-4)
    opt_noise = torch.optim.AdamW(noise_net.parameters(), lr=lr_noise,  weight_decay=1e-4)
    opt_v     = torch.optim.AdamW(v_head.parameters(),    lr=lr_critic, weight_decay=1e-4)
    opt_q     = torch.optim.AdamW(q_head.parameters(),    lr=lr_critic, weight_decay=1e-4)

    # ── Replay buffer ──────────────────────────────────────────────────────────
    replay_buffer = ReplayBuffer(
        capacity=cfg.get('buffer_capacity', 300_000),
        device='cpu'
    )

    # ── Training config ────────────────────────────────────────────────────────
    batch_size      = cfg.get('batch_size', 256)
    critic_warmup   = cfg.get('critic_warmup_steps', 5_000)
    total_steps     = cfg.get('total_steps', 500_000)
    save_every      = cfg.get('save_every', 2_500)
    log_every       = cfg.get('log_every', 50)
    buffer_refresh  = cfg.get('buffer_refresh_steps', 500)
    actor_updates   = cfg.get('actor_updates_per_critic', 1)
    # NEW: backbone freeze for the first N steps of Phase 2
    backbone_freeze = cfg.get('backbone_freeze_steps', 20_000)

    # ── Bootstrap: save initial policy so actors can start ────────────────────
    initial_policy_path = os.path.join(args.buffer_path, 'policy_latest.pt')
    if not os.path.exists(initial_policy_path):
        print('[Learner] Saving initial policy for actors to bootstrap...')
        torch.save({
            'model_state':     model.state_dict(),
            'noise_net_state': noise_net.state_dict(),
            'step': 0,
        }, initial_policy_path)
        print('[Learner] Initial policy saved — start actors now.')

    wait_for_buffer(args.buffer_path,
                    min_transitions=cfg.get('warmup_transitions', 2_000))
    load_all_buffer_files(args.buffer_path, replay_buffer)

    # ── Phase 1: Critic warmup ─────────────────────────────────────────────────
    # Train V+Q only (backbone and noise_net frozen) until critics are calibrated.
    print(f"\n[Learner] === PHASE 1: Critic warmup ({critic_warmup} steps) ===")
    for p in model.parameters():     p.requires_grad = False
    for p in noise_net.parameters(): p.requires_grad = False

    for step in range(1, critic_warmup + 1):
        if len(replay_buffer) < batch_size:
            time.sleep(1.0)
            load_all_buffer_files(args.buffer_path, replay_buffer)
            continue

        batch  = replay_buffer.sample(batch_size, device=device)
        c_logs = update_critic(batch, model, v_head, q_head,
                               v_targ, q_targ, opt_v, opt_q, cfg)
        polyak_update(v_head, v_targ, tau=cfg.get('tau', 0.005))
        polyak_update(q_head, q_targ, tau=cfg.get('tau', 0.005))

        if step % log_every == 0:
            for k, v in c_logs.items():
                writer.add_scalar(f'warmup/{k}', v, step)
            print(f"  Warmup {step:5d}/{critic_warmup} | "
                  f"V={c_logs['v_mean']:+.3f} Q={c_logs['q_mean']:+.3f} "
                  f"V_loss={c_logs['v_loss']:.4f} Q_loss={c_logs['q_loss']:.4f} "
                  f"TD={c_logs['td_target_mean']:+.3f}")

            # Sanity check: V should be in reasonable range for normalized rewards
            if abs(c_logs['v_mean']) > 50:
                print(f"  ⚠ V_mean={c_logs['v_mean']:.1f} is large — "
                      "check reward normalization in actor.py")

        if step % buffer_refresh == 0:
            load_all_buffer_files(args.buffer_path, replay_buffer)

    # Unfreeze for Phase 2
    for p in model.parameters():     p.requires_grad = True
    for p in noise_net.parameters(): p.requires_grad = True

    # ── Phase 2: Full ReinFlow training ───────────────────────────────────────
    print(f"\n[Learner] === PHASE 2: ReinFlow training ({total_steps} steps) ===")
    if backbone_freeze > 0:
        print(f"[Learner] Backbone will be frozen for first "
              f"{backbone_freeze} steps of Phase 2.")
        freeze_backbone(model)

    step             = start_step
    last_buffer_load = time.time()
    backbone_frozen  = backbone_freeze > 0

    while step < total_steps:
        # ── Unfreeze backbone after freeze window ──────────────────────────────
        if backbone_frozen and step >= backbone_freeze:
            unfreeze_backbone(model)
            backbone_frozen = False
            # Rebuild optimizer with new lr after backbone thaws
            opt_actor = torch.optim.AdamW(
                model.parameters(), lr=lr_actor, weight_decay=1e-4
            )
            print(f"[Learner] Step {step}: backbone unfrozen, optimizer rebuilt")

        # ── Periodic buffer reload ─────────────────────────────────────────────
        if time.time() - last_buffer_load > 30.0:
            loaded = load_all_buffer_files(args.buffer_path, replay_buffer)
            last_buffer_load = time.time()
            writer.add_scalar('buffer/size', len(replay_buffer), step)

        if len(replay_buffer) < batch_size:
            time.sleep(2.0)
            continue

        # ── Critic update ──────────────────────────────────────────────────────
        batch  = replay_buffer.sample(batch_size, device=device)
        c_logs = update_critic(batch, model, v_head, q_head,
                               v_targ, q_targ, opt_v, opt_q, cfg)
        polyak_update(v_head, v_targ, tau=cfg.get('tau', 0.005))
        polyak_update(q_head, q_targ, tau=cfg.get('tau', 0.005))

        # ── Actor update (ReinFlow PPO) ────────────────────────────────────────
        for _ in range(actor_updates):
            batch  = replay_buffer.sample(batch_size, device=device)
            a_logs = update_actor(batch, model, noise_net,
                                  v_targ, q_targ,
                                  opt_actor, opt_noise, cfg)

        step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if step % log_every == 0:
            all_logs = {**c_logs, **a_logs}
            for k, v_val in all_logs.items():
                writer.add_scalar(f'train/{k}', v_val, step)
            writer.add_scalar('train/backbone_frozen', int(backbone_frozen), step)

            print(f"  Step {step:6d} | "
                  f"V={c_logs['v_mean']:+.3f} "
                  f"Q={c_logs['q_mean']:+.3f} "
                  f"actor_loss={a_logs['actor_loss']:+.4f} "
                  f"ratio={a_logs['ratio_mean']:.3f} "
                  f"ratio_max={a_logs['ratio_max']:.2f} "
                  f"adv_raw={a_logs['adv_raw_mean']:+.3f}±{a_logs['adv_raw_std']:.3f} "
                  f"gnorm={a_logs['actor_grad_norm']:.2f} "
                  f"logp={a_logs['log_prob_mean']:.2f} "
                  f"buf={len(replay_buffer):,} "
                  f"{'[FROZEN]' if backbone_frozen else '[FREE]'}")

            # PPO ratio sanity check
            if a_logs['ratio_mean'] > 3.0 or a_logs['ratio_mean'] < 0.3:
                print(f"  ⚠ PPO ratio={a_logs['ratio_mean']:.2f} out of range — "
                      "flow_steps in actor vs learner may be mismatched!")

        # ── Save policy for actors ─────────────────────────────────────────────
        if step % save_every == 0:
            extra = {
                'model_state':     model.state_dict(),
                'noise_net_state': noise_net.state_dict(),
                'v_head_state':    v_head.state_dict(),
                'q_head_state':    q_head.state_dict(),
                'v_targ_state':    v_targ.state_dict(),
                'q_targ_state':    q_targ.state_dict(),
                'step':            step,
            }
            save_policy(model, noise_net, args.buffer_path, step, cfg, extra)
            writer.add_scalar('train/step_saved', step, step)

    print(f"\n[Learner] Training complete at step {step}")
    writer.close()


if __name__ == '__main__':
    main()
