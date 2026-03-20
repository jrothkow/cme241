"""
Evaluate a trained PPO model against the exact optimal policy (backward induction).

Usage
-----
  python evaluate_ppo_v5.py
  python evaluate_ppo_v5.py --model ppo_nbhd2_v6 --neighborhood 2
  python evaluate_ppo_v5.py --n-episodes 1000 --neighborhood 1

Metrics reported
----------------
  - Mean ± std total reward: PPO vs exact optimal
  - Optimality gap (absolute and %)
  - Policy agreement rate: % of time steps where PPO's greedy action
    matches the exact optimal action
  - Per-action-dimension agreement: invest / mode / volume / consumption
  - Final cash, asset, strength, work_cap, endurance, load, and injury
    distributions (PPO vs optimal)
"""

from __future__ import annotations

import argparse
import os
import sys

# Add phase3/ (parent of v5/) to path so that `v5.*` imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from gym_env_v5 import LifeGymEnv5
from multi_capital_mdp_v6 import (
    MultiCapitalMDPv5,
    solve_mdp_fast_v5,
    policy_at_v5,
)


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_ppo_episodes(
    model: PPO,
    env: LifeGymEnv5,
    n_episodes: int,
    seed: int = 0,
) -> tuple[list[float], list[int], list[int], list[int], list[int], list[int], list[int], list[int]]:
    """
    Run n_episodes with the PPO policy (deterministic greedy).

    Returns
    -------
    rewards        : total reward per episode
    final_cash     : cash bin at end of episode
    final_assets   : asset bin at end of episode
    final_strength : strength level at end of episode
    final_work_cap : work_cap level at end of episode
    final_endurance: endurance level at end of episode
    final_load     : load level at end of episode
    final_injury   : injury level at end of episode
    """
    rewards, final_cash, final_assets = [], [], []
    final_strength, final_work_cap, final_endurance, final_load, final_injury = [], [], [], [], []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(float(total_reward))
        final_cash.append(env._state.cash)
        final_assets.append(env._state.assets)
        final_strength.append(env._state.strength)
        final_work_cap.append(env._state.work_cap)
        final_endurance.append(env._state.endurance)
        final_load.append(env._state.load)
        final_injury.append(env._state.injury)

    return rewards, final_cash, final_assets, final_strength, final_work_cap, final_endurance, final_load, final_injury


def run_optimal_episodes(
    PI: np.ndarray,
    env: LifeGymEnv5,
    n_episodes: int,
    seed: int = 0,
) -> tuple[list[float], list[int], list[int], list[int], list[int], list[int], list[int], list[int]]:
    """
    Run n_episodes following the exact optimal policy from backward induction.

    Returns
    -------
    rewards        : total reward per episode
    final_cash     : cash bin at end of episode
    final_assets   : asset bin at end of episode
    final_strength : strength level at end of episode
    final_work_cap : work_cap level at end of episode
    final_endurance: endurance level at end of episode
    final_load     : load level at end of episode
    final_injury   : injury level at end of episode
    """
    rewards, final_cash, final_assets = [], [], []
    final_strength, final_work_cap, final_endurance, final_load, final_injury = [], [], [], [], []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total_reward = 0.0
        done = False

        while not done:
            opt_action = policy_at_v5(PI, env._state)
            action = np.array(
                [opt_action.invest, opt_action.mode, opt_action.volume, opt_action.consumption],
                dtype=np.int64,
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(float(total_reward))
        final_cash.append(env._state.cash)
        final_assets.append(env._state.assets)
        final_strength.append(env._state.strength)
        final_work_cap.append(env._state.work_cap)
        final_endurance.append(env._state.endurance)
        final_load.append(env._state.load)
        final_injury.append(env._state.injury)

    return rewards, final_cash, final_assets, final_strength, final_work_cap, final_endurance, final_load, final_injury


def compute_policy_agreement(
    model: PPO,
    PI: np.ndarray,
    env: LifeGymEnv5,
    n_episodes: int = 200,
    seed: int = 0,
) -> tuple[float, float, float, float, float]:
    """
    Run n_episodes with PPO and return:
      - full action-tuple agreement
      - invest agreement
      - mode agreement
      - volume agreement
      - consumption agreement
    """
    full_agreements:        list[bool] = []
    invest_agreements:      list[bool] = []
    mode_agreements:        list[bool] = []
    volume_agreements:      list[bool] = []
    consumption_agreements: list[bool] = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False

        while not done:
            opt_action = policy_at_v5(PI, env._state)
            ppo_action, _ = model.predict(obs, deterministic=True)

            ppo_invest      = int(ppo_action[0])
            ppo_mode        = int(ppo_action[1])
            ppo_volume      = int(ppo_action[2])
            ppo_consumption = int(ppo_action[3])

            invest_match      = ppo_invest      == opt_action.invest
            mode_match        = ppo_mode        == opt_action.mode
            volume_match      = ppo_volume      == opt_action.volume
            consumption_match = ppo_consumption == opt_action.consumption
            full_match = invest_match and mode_match and volume_match and consumption_match

            full_agreements.append(full_match)
            invest_agreements.append(invest_match)
            mode_agreements.append(mode_match)
            volume_agreements.append(volume_match)
            consumption_agreements.append(consumption_match)

            obs, _, terminated, truncated, _ = env.step(ppo_action)
            done = terminated or truncated

    return (
        float(np.mean(full_agreements)),
        float(np.mean(invest_agreements)),
        float(np.mean(mode_agreements)),
        float(np.mean(volume_agreements)),
        float(np.mean(consumption_agreements)),
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _dist_str(values: list[int], n_levels: int) -> str:
    """Return a compact distribution string, e.g. '0:12%  1:45%  2:43%'."""
    counts = np.bincount(values, minlength=n_levels)
    pcts = 100.0 * counts / len(values)
    return "  ".join(f"{lvl}:{pct:.0f}%" for lvl, pct in enumerate(pcts))


def print_results(
    ppo_rewards: list[float],
    opt_rewards: list[float],
    ppo_cash: list[int],
    opt_cash: list[int],
    ppo_assets: list[int],
    opt_assets: list[int],
    ppo_strength: list[int],
    opt_strength: list[int],
    ppo_work_cap: list[int],
    opt_work_cap: list[int],
    ppo_endurance: list[int],
    opt_endurance: list[int],
    ppo_load: list[int],
    opt_load: list[int],
    ppo_injury: list[int],
    opt_injury: list[int],
    full_agreement: float,
    invest_agreement: float,
    mode_agreement: float,
    volume_agreement: float,
    consumption_agreement: float,
) -> None:
    ppo_mean, ppo_std = np.mean(ppo_rewards), np.std(ppo_rewards)
    opt_mean, opt_std = np.mean(opt_rewards), np.std(opt_rewards)
    gap_abs = opt_mean - ppo_mean
    gap_pct = 100.0 * gap_abs / abs(opt_mean) if opt_mean != 0 else float("nan")

    width = 64
    print("\n" + "=" * width)
    print("  EVALUATION RESULTS (V5)")
    print("=" * width)
    print(f"  {'PPO total reward':<30}: {ppo_mean:8.2f} ± {ppo_std:.2f}")
    print(f"  {'Optimal total reward':<30}: {opt_mean:8.2f} ± {opt_std:.2f}")
    print(f"  {'Optimality gap':<30}: {gap_abs:+.2f}  ({gap_pct:.1f}%)")
    print(f"  {'Full policy agreement':<30}: {100*full_agreement:.1f}% of steps")
    print(f"  {'Invest agreement':<30}: {100*invest_agreement:.1f}% of steps")
    print(f"  {'Mode agreement':<30}: {100*mode_agreement:.1f}% of steps")
    print(f"  {'Volume agreement':<30}: {100*volume_agreement:.1f}% of steps")
    print(f"  {'Consumption agreement':<30}: {100*consumption_agreement:.1f}% of steps")
    print("-" * width)
    print("  Final cash distribution (bins 0–20):")
    print(f"    PPO    : {_dist_str(ppo_cash, 21)}")
    print(f"    Optimal: {_dist_str(opt_cash, 21)}")
    print("  Final asset distribution (bins 0–30):")
    print(f"    PPO    : {_dist_str(ppo_assets, 31)}")
    print(f"    Optimal: {_dist_str(opt_assets, 31)}")
    print("  Final strength distribution (0=detrained, 3=high):")
    print(f"    PPO    : {_dist_str(ppo_strength, 4)}")
    print(f"    Optimal: {_dist_str(opt_strength, 4)}")
    print("  Final work capacity distribution (0=detrained, 3=high):")
    print(f"    PPO    : {_dist_str(ppo_work_cap, 4)}")
    print(f"    Optimal: {_dist_str(opt_work_cap, 4)}")
    print("  Final endurance distribution (0=detrained, 3=high):")
    print(f"    PPO    : {_dist_str(ppo_endurance, 4)}")
    print(f"    Optimal: {_dist_str(opt_endurance, 4)}")
    print("  Final training load distribution (0=fresh, 4=overloaded):")
    print(f"    PPO    : {_dist_str(ppo_load, 5)}")
    print(f"    Optimal: {_dist_str(opt_load, 5)}")
    print("  Final injury distribution (0=healthy, 1=minor, 2=major):")
    print(f"    PPO    : {_dist_str(ppo_injury, 3)}")
    print(f"    Optimal: {_dist_str(opt_injury, 3)}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO vs exact optimal policy (v5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ppo_life_mdp_v5",
        help="Path to saved model (without .zip extension; default: ppo_life_mdp_v5)",
    )
    parser.add_argument(
        "--neighborhood",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Neighborhood tier matching the trained model (default: 1)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=500,
        help="Number of evaluation episodes per policy (default: 500)",
    )
    parser.add_argument(
        "--agreement-episodes",
        type=int,
        default=200,
        help="Episodes for policy agreement calculation (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Base seed for evaluation episodes (default: 100)",
    )
    args = parser.parse_args()

    # Load model
    model_path = args.model if args.model.endswith(".zip") else args.model + ".zip"
    if not os.path.exists(model_path):
        print(f"Error: model file '{model_path}' not found.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = PPO.load(args.model)

    # Build env and exact solution
    env = LifeGymEnv5(neighborhood=args.neighborhood)
    mdp = MultiCapitalMDPv5(neighborhood=args.neighborhood)

    print("Solving MDP exactly (backward value iteration)...")
    _, PI = solve_mdp_fast_v5(mdp)

    # Run evaluations
    print(f"\nRunning {args.n_episodes} PPO episodes (deterministic)...")
    ppo_rewards, ppo_cash, ppo_assets, ppo_str, ppo_wc, ppo_end, ppo_load, ppo_inj = run_ppo_episodes(
        model, env, args.n_episodes, seed=args.seed
    )

    print(f"Running {args.n_episodes} optimal-policy episodes...")
    opt_rewards, opt_cash, opt_assets, opt_str, opt_wc, opt_end, opt_load, opt_inj = run_optimal_episodes(
        PI, env, args.n_episodes, seed=args.seed
    )

    print(f"Computing policy agreement over {args.agreement_episodes} episodes...")
    (
        full_agreement,
        invest_agreement,
        mode_agreement,
        volume_agreement,
        consumption_agreement,
    ) = compute_policy_agreement(
        model, PI, env, n_episodes=args.agreement_episodes, seed=args.seed
    )

    print_results(
        ppo_rewards, opt_rewards,
        ppo_cash, opt_cash,
        ppo_assets, opt_assets,
        ppo_str, opt_str,
        ppo_wc, opt_wc,
        ppo_end, opt_end,
        ppo_load, opt_load,
        ppo_inj, opt_inj,
        full_agreement,
        invest_agreement,
        mode_agreement,
        volume_agreement,
        consumption_agreement,
    )


if __name__ == "__main__":
    main()
