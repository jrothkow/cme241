"""
Train PPO on the Multi-Capital Life MDP using Stable Baselines 3.

Usage
-----
  python train_ppo.py
  python train_ppo.py --timesteps 1000000
  python train_ppo.py --check-env
  python train_ppo.py --neighborhood 2 --output ppo_nbhd2_v6 --check-env

Outputs
-------
  <output>.zip   : trained SB3 PPO model
  ./ppo_logs/    : TensorBoard logs (optional; view with `tensorboard --logdir ppo_logs`)
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure current directory is on path when called from a parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from gym_env_v5 import LifeGymEnv5 as LifeGymEnv


def make_env(neighborhood: int):
    """Factory for Monitor-wrapped LifeGymEnv."""
    def _init():
        return Monitor(LifeGymEnv(neighborhood=neighborhood))
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on the Multi-Capital Life MDP (v3)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total environment steps to train for (default: 500,000)",
    )
    parser.add_argument(
        "--neighborhood",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Neighborhood tier: 0=budget, 1=mid-tier, 2=premium (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ppo_life_mdp_v3",
        help="Output model filename without extension (default: ppo_life_mdp_v3)",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run gymnasium env checker before training",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    args = parser.parse_args()

    # --- Optional environment sanity check ---
    if args.check_env:
        from gymnasium.utils.env_checker import check_env as gymnasium_check_env

        print("Running gymnasium environment checker...")
        test_env = LifeGymEnv(neighborhood=args.neighborhood)
        gymnasium_check_env(test_env, warn=True, skip_render_check=True)
        test_env.close()
        print("Environment check passed!\n")

    # --- Build environment ---
    env = Monitor(LifeGymEnv(neighborhood=args.neighborhood))

    # --- PPO configuration ---
    #
    # Observation is now a 97-dimensional one-hot vector:
    #   cash(21) + assets(31) + work_intensity(3) + energy(3) + fitness(3) + time(36)
    #
    # n_steps = 1800: collect 50 complete episodes per rollout (50 x 36 steps).
    #   This keeps rollout boundaries aligned with the 36-step episode length.
    #
    # gamma = 0.99: matches the MDP discount factor so learned values are
    #   comparable to the exact backward-induction solution.
    #
    # net_arch = [256, 256]: somewhat larger MLP is appropriate now that the
    #   observation is higher-dimensional and the state space is much larger.
    #
    tb_log = None if args.no_tensorboard else "./ppo_logs/"

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1800,
        batch_size=72,                              # 1800 / 72 = 25 exact minibatches
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,                                # small entropy bonus encourages exploration early on
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        tensorboard_log=tb_log,
    )

    print(
        f"Training PPO for {args.timesteps:,} timesteps "
        f"(neighborhood={args.neighborhood}, seed={args.seed})..."
    )
    model.learn(total_timesteps=args.timesteps)

    model.save(args.output)
    print(f"\nModel saved to {args.output}.zip")
    if tb_log:
        print(f"TensorBoard logs: {tb_log}  (run: tensorboard --logdir {tb_log})")


if __name__ == "__main__":
    main()