import sys
sys.path.insert(0, '/jackal_ws/src/ros_jackal')

print("Testing imports...")

try:
    import torch
    print(f"  torch        : {torch.__version__}")
except Exception as e:
    print(f"  torch FAILED : {e}")

try:
    import yaml
    print(f"  yaml         : OK")
except Exception as e:
    print(f"  yaml FAILED  : {e}")

try:
    from barn_nav_model import BARNNavModel, ReinFlowNoiseNet, CONTEXT_DIM, reinflow_forward
    print(f"  barn_nav_model : OK")
except Exception as e:
    print(f"  barn_nav_model FAILED : {e}")

try:
    from rl_algos.reinflow import ReplayBuffer, update_critic, update_actor, polyak_update
    print(f"  rl_algos.reinflow : OK")
except Exception as e:
    print(f"  rl_algos.reinflow FAILED : {e}")

try:
    import gym
    import envs.registration
    print(f"  envs.registration : OK")
except Exception as e:
    print(f"  envs.registration FAILED : {e}")

try:
    from envs.wrappers import ShapingRewardWrapper
    print(f"  wrappers : OK")
except Exception as e:
    print(f"  wrappers FAILED : {e}")

print("\nDone.")
